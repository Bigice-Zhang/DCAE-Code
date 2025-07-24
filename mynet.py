import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# pre-layernorm

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# feedforward

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
    
# external convolution token
    
class ExConvToken(nn.Module):
    '''
    convolution layers used to extract featuers which form a feature pooling 
    '''
    def __init__(self) -> torch.tensor:
        super().__init__()
        self.c1 = nn.Conv2d(9, 45, 3, 1, 1) #(45,15,15)
        self.p1 = nn.MaxPool2d(3, 2, 0) #(45,7,7)
        self.c2 = nn.Conv2d(45, 81, 3, 1, 1) #(81,7,7)
        self.p2 = nn.MaxPool2d(3, 1, 0) #(81,5,5)
        self.c3 = nn.Conv2d(81, 225, 3, 1, 1) #(225,5,5)
        self.p3 = nn.MaxPool2d(3, 1, 0) #(225,3,3)
        self.act = nn.ReLU()

    def forward(self, x):
        b = x.shape[0]
        x1 = self.act(self.p1(self.c1(x))) #(b,45,7,7)
        x2 = self.act(self.p2(self.c2(x1))) #(b,81,5,5) 小尺寸分支
        x3 = self.act(self.p3(self.c3(x2))) #(b,225,3,3) 大尺寸分支
        x2 = rearrange(x2, 'b c h w -> b (h w) c') #(b, 25, 81)
        x3 = rearrange(x3, 'b c h w -> b (h w) c') #(b, 9, 255)
        return x2, x3
    

# attention

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

    def forward(self, x, prev, context = None, kv_include_self = False):
        b, n, _, h = *x.shape, self.heads
        context = default(context, x)

        if kv_include_self:
            context = torch.cat((x, context), dim = 1) # cross attention requires CLS token includes itself as key / value

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if prev is not None:
            dots += prev #将前一层的自注意力结果加到这一层

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), dots

# transformer encoder, for small and large patches

class Transformer(nn.Module):
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
    
# RealFormerEncoderLayer
class RealFormerEncoderLayer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads, dim_head, dropout),
                nn.LayerNorm(dim),
                FeedForward(dim, mlp_dim, dropout),
                nn.LayerNorm(dim)
            ]))
        

    def forward(self, x, prev=None):
        for attn, norm1, ff, norm2 in self.layers:
            residual = x
            x, prev = attn(x, prev)
            x = norm1(x+residual)
            residual = x
            x = ff(x)
            out = norm2(x+residual)
        # residual = x
        # x, prev = self.attn(x, prev)
        # x = self.norm1(x + residual)
        # residual = x
        # x = self.ff(x)
        # out = self.norm2(x + residual)
        return out, prev

# projecting CLS tokens, in the case that small and large patch tokens have different dimensions

class ProjectInOut(nn.Module):
    def __init__(self, dim_in, dim_out, fn):
        super().__init__()
        self.fn = fn

        need_projection = dim_in != dim_out
        self.project_in = nn.Linear(dim_in, dim_out) if need_projection else nn.Identity()
        self.project_out = nn.Linear(dim_out, dim_in) if need_projection else nn.Identity()

    def forward(self, x, prev=None, *args, **kwargs):
        x = self.project_in(x)
        x = self.fn(x, prev, *args, **kwargs)
        x = self.project_out(x)
        return x, prev

# cross attention transformer

class CrossTransformer(nn.Module):
    def __init__(self, sm_dim, lg_dim, depth, heads, dim_head, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # ProjectInOut(sm_dim, lg_dim, PreNorm(lg_dim, Attention(lg_dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                # ProjectInOut(lg_dim, sm_dim, PreNorm(sm_dim, Attention(sm_dim, heads = heads, dim_head = dim_head, dropout = dropout)))
                ProjectInOut(sm_dim, lg_dim, RealFormerEncoderLayer(lg_dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                ProjectInOut(lg_dim, sm_dim, RealFormerEncoderLayer(sm_dim, heads = heads, dim_head = dim_head, dropout = dropout))
            ]))

    def forward(self, sm_tokens, lg_tokens):
        (sm_cls, sm_patch_tokens), (lg_cls, lg_patch_tokens) = map(lambda t: (t[:, :1], t[:, 1:]), (sm_tokens, lg_tokens))

        sm_prev, lg_prev = None, None
        for sm_attend_lg, lg_attend_sm in self.layers:
            sm_cls, sm_prev = sm_attend_lg(sm_cls, sm_prev, context = lg_patch_tokens, kv_include_self = True) + sm_cls
            lg_cls, lg_prev = lg_attend_sm(lg_cls, lg_prev, context = sm_patch_tokens, kv_include_self = True) + lg_cls

        sm_tokens = torch.cat((sm_cls, sm_patch_tokens), dim = 1)
        lg_tokens = torch.cat((lg_cls, lg_patch_tokens), dim = 1)
        return sm_tokens, lg_tokens

# multi-scale encoder

class MultiScaleEncoder(nn.Module):
    def __init__(
        self,
        *,
        depth,
        sm_dim,
        lg_dim,
        sm_enc_params,
        lg_enc_params,
        cross_attn_heads,
        cross_attn_depth,
        cross_attn_dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                RealFormerEncoderLayer(dim = sm_dim, dropout = dropout, **sm_enc_params),
                RealFormerEncoderLayer(dim = lg_dim, dropout = dropout, **lg_enc_params),
                CrossTransformer(sm_dim = sm_dim, lg_dim = lg_dim, depth = cross_attn_depth, heads = cross_attn_heads, dim_head = cross_attn_dim_head, dropout = dropout)
            ]))

    def forward(self, sm_tokens, lg_tokens):
        for sm_enc, lg_enc, cross_attend in self.layers:
            sm_tokens, lg_tokens = sm_enc(sm_tokens), lg_enc(lg_tokens)
            sm_tokens, lg_tokens = cross_attend(sm_tokens, lg_tokens)

        return sm_tokens, lg_tokens

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
        num_patches = (image_size // patch_size) ** 2
        patch_dim = image_channel * patch_size ** 2

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape #(b,n,d)
        # b, n = img.shape[0], img.shape[1]

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1) #(b,n+1,d)
        x += self.pos_embedding[:, :(n + 1)] #维度不变

        return self.dropout(x)

# cross ViT class

class CrossViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        image_channel,
        num_classes,
        sm_dim,
        lg_dim,
        sm_patch_size = 12,
        sm_enc_depth = 1,
        sm_enc_heads = 8,
        sm_enc_mlp_dim = 2048,
        sm_enc_dim_head = 64,
        lg_patch_size = 16,
        lg_enc_depth = 4,
        lg_enc_heads = 8,
        lg_enc_mlp_dim = 2048,
        lg_enc_dim_head = 64,
        cross_attn_depth = 2,
        cross_attn_heads = 8,
        cross_attn_dim_head = 64,
        depth = 3,
        dropout = 0.1,
        emb_dropout = 0.1
    ):
        super().__init__()
        self.sm_image_embedder = ImageEmbedder(dim = sm_dim, image_size = image_size, patch_size = sm_patch_size, image_channel = image_channel, dropout = emb_dropout)
        self.lg_image_embedder = ImageEmbedder(dim = lg_dim, image_size = image_size, patch_size = lg_patch_size, image_channel = image_channel, dropout = emb_dropout)

        # self.ex_conv_token = ExConvToken() #卷积外部令牌

        #gamma数据分布卷积自编码器
        # self.gamma_cae = ConvAutoEncoder()
        # self.gamma_cad = ConvAutoDecoder()
        # #gauss数据分布卷积自编码器
        # self.gauss_cae = ConvAutoEncoder()
        # self.gauss_cad = ConvAutoDecoder()


        self.multi_scale_encoder = MultiScaleEncoder(
            depth = depth,
            sm_dim = sm_dim,
            lg_dim = lg_dim,
            cross_attn_heads = cross_attn_heads,
            cross_attn_dim_head = cross_attn_dim_head,
            cross_attn_depth = cross_attn_depth,
            sm_enc_params = dict(
                depth = sm_enc_depth,
                heads = sm_enc_heads,
                mlp_dim = sm_enc_mlp_dim,
                dim_head = sm_enc_dim_head
            ),
            lg_enc_params = dict(
                depth = lg_enc_depth,
                heads = lg_enc_heads,
                mlp_dim = lg_enc_mlp_dim,
                dim_head = lg_enc_dim_head
            ),
            dropout = dropout
        )

        self.sm_mlp_head = nn.Sequential(nn.LayerNorm(sm_dim), nn.Linear(sm_dim, num_classes))
        self.lg_mlp_head = nn.Sequential(nn.LayerNorm(lg_dim), nn.Linear(lg_dim, num_classes))
        

    def forward(self, img):
        sm_tokens = self.sm_image_embedder(img) #(b,25+1,81)
        lg_tokens = self.lg_image_embedder(img) #(b,9+1,225)

        ex_sm, ex_lg = self.ex_conv_token(img) #(b,25,81) #(b,9,255)

        gamma_X_hat = self.gamma_cad(self.gamma_cae(img))
        gauss_X_hat = self.gauss_cad(self.gauss_cae(img))

        #将卷积得到的特征图作为令牌直接进行训练
        # sm_tokens = self.sm_image_embedder(ex_sm) #(b,25+1,81)
        # lg_tokens = self.lg_image_embedder(ex_lg) #(b,9+1,225)

        sm_tokens = torch.cat((ex_sm, sm_tokens), dim=1) #(b,25+1+25,81)
        lg_tokens = torch.cat((ex_lg, lg_tokens), dim=1) #(b,9+1+9,225)

        sm_tokens, lg_tokens = self.multi_scale_encoder(sm_tokens, lg_tokens)

        sm_cls, lg_cls = map(lambda t: t[:, 0], (sm_tokens, lg_tokens))

        sm_logits = self.sm_mlp_head(sm_cls)
        lg_logits = self.lg_mlp_head(lg_cls)

        return sm_logits + lg_logits, gamma_X_hat, gauss_X_hat
