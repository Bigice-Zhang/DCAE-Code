import torch
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.distributions import Categorical

# helpers
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

#PreNorm
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    
#FeedForward
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# patch-based image to token embedder
class ImageEmbedder(nn.Module):
    def __init__(
        self,
        *,
        dim,
        image_size,
        patch_size,
        image_channel,
        dropout = 0.
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        patch_dim = image_channel * patch_size ** 2

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        return self.dropout(x)

# heterogeneous feature fusion (HFF) strategy
class FeatureFusionTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

# distributional attention
class DistributionAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, entropy, kv_include_self = False):
        entropy = rearrange(entropy, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=5, p2=5)
        b, n, _, h = *x.shape, self.heads
        entropy = default(entropy, x)
        
        if kv_include_self:
            entropy = torch.cat((x, entropy), dim=1)  # cross attention requires CLS token includes itself as key / value

        # Raw data generates Q, entropy generates K and V
        qkv = (self.to_q(x), *self.to_kv(entropy).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        
        dots = einsum('b h i d, b h j d -> b h i j', q, k)

        '''
        Take the entropy value of data distribution information as a new weight 
        and multiply it with the original attention weight to obtain the new attention weight.
        The new attention weights will be more inclined to focus on data with different labels,
        that is, areas with uneven data distribution.
        '''
        attn = self.attend(dots) 
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# multi-head data distributional attention(MDDA)
class DistributionTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, DistributionAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x, entropy):
        for attn, ff in self.layers:
            x = attn(x, entropy=entropy) + x
            x = ff(x) + x
        return self.norm(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        h = self.heads

        qkv = (self.to_q(x), *self.to_kv(x).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k)  # self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# distributional convolutional autoencoder(DCAE) for two data distribution
class FoxEar(nn.Module):
    def __init__(self,
                 dim=150,
                 depth=3,
                 heads=6, 
                 dim_head=64, 
                 dropout=0.1
                ):
        super().__init__()

        self.ear = nn.ModuleDict({
            # Encoder block
            'e1': nn.Conv2d(6, 18, 3, 1, 1), 
            'e2': nn.Conv2d(18, 54, 3, 2, 0),
            'e3': nn.Conv2d(54, 150, 3, 2, 0), 
            'norm1': nn.BatchNorm2d(150), 

            # multi-head data distributional attention(MDDA)
            'attend': DistributionTransformer(dim=dim, 
                                depth=depth, 
                                heads=heads, 
                                dim_head=dim_head, 
                                mlp_dim=dim*4, 
                                dropout=dropout),

            # Decoder block
            'd1': nn.ConvTranspose2d(150, 54, 3, 2, 0),
            'd2': nn.ConvTranspose2d(54, 18, 3, 2, 0),
            'd3': nn.ConvTranspose2d(18, 6, 3, 1, 1),  
            'norm2': nn.BatchNorm2d(6)
        })
        self.act = nn.ReLU()

    def forward(self, x, entropy):
        x1 = self.act(self.ear['e1'](x))
        x2 = self.act(self.ear['e2'](x1))
        x3 = self.ear['norm1'](self.act(self.ear['e3'](x2)))

        x3 = rearrange(x3, 'b c h w -> b (h w) c')
        x3 = self.ear['attend'](x3, entropy)
        x3 = rearrange(x3, 'b (h w) c -> b c h w', h=3)

        x4 = self.act(self.ear['d1'](x3))
        x5 = self.act(self.ear['d2'](x4))
        x6 = self.ear['norm2'](self.act(self.ear['d3'](x5)))
        return x6


class FoxFace(nn.Module):
    def __init__(self,
                 dim=300, 
                 image_size=15, 
                 image_channel=6,
                 patch_size=5,
                 depth=3, 
                 heads=6, 
                 dim_head=64,
                 dropout=0.1,

                ) -> None:
        super().__init__()
        self.image_embedder = ImageEmbedder(dim = dim,
                                            image_size = image_size, 
                                            patch_size = patch_size, 
                                            image_channel = image_channel, 
                                            dropout = dropout)

        self.nose = FeatureFusionTransformer(dim=dim, 
                                depth=depth, 
                                heads=heads, 
                                dim_head=dim_head, 
                                mlp_dim=dim*4, 
                                dropout=dropout)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
    def forward(self, x1, x2):
        x1 = self.image_embedder(x1)
        x2 = self.image_embedder(x2)
        x = torch.cat((x1, x2), dim=1)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        en_x = self.nose(x)
        return en_x

class FoxHead(nn.Module):
    def __init__(self,
                 num_classes=15,
                 ear_dim=150,
                 face_dim=300, 
                 image_size=15, 
                 image_channel=6,
                 patch_size=5,
                 depth=3, 
                 heads=6, 
                 dim_head=64,
                 dropout=0.1 ) -> None:
        super().__init__()

        self.left_ear = FoxEar(dim=ear_dim,
                          depth=depth,
                          heads=heads,
                          dim_head=dim_head,
                          dropout=dropout)
        
        self.right_ear = FoxEar(dim=ear_dim,
                          depth=depth,
                          heads=heads,
                          dim_head=dim_head,
                          dropout=dropout)
        
        self.face = FoxFace(dim=face_dim,
                            image_size=image_size, 
                            image_channel=image_channel,
                            patch_size=patch_size,
                            depth=depth, 
                            heads=heads, 
                            dim_head=dim_head,
                            dropout=dropout,
                            )

        self.mlp_head = nn.Sequential(nn.LayerNorm(face_dim), nn.Linear(face_dim, num_classes))

    def forward(self, X):
        # obtain Gamma distribution data and Gaussian distribution data
        gm_x = X[::,0:6,...] #(b,6,15,15)
        gu_x = X[::,6:12,...] #(b,6,15,15)

        # Calculate the entropy value of data distribution
        gm_x_entropy = Categorical(logits=gm_x).entropy().unsqueeze(2).repeat(1,1,X.shape[2],1)
        gu_x_entropy = Categorical(logits=gu_x).entropy().unsqueeze(2).repeat(1,1,X.shape[2],1)

        # Characterizing and learning Gamma distribution data and Gaussian distribution data
        gm_x_hat = self.left_ear(gm_x, gm_x_entropy)
        gu_x_hat = self.right_ear(gu_x, gu_x_entropy)

        # Heterogeneous feature fusion module
        en_x = self.face(gm_x_hat, gu_x_hat)

        # Perform final classification through the classification header
        x_cls = en_x[:, 0, :]
        y_hat = self.mlp_head(x_cls)

        return y_hat, gm_x_hat, gu_x_hat