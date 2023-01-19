import torch
from torch import nn,einsum

from einops import rearrange,repeat
from einops.layers.torch import Rearrange

def pair(t):
    return t if isinstance(t,tuple) else (t,t)

#归一化
class PreNorm(nn.Module):
    def __init__(self,dim,fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
    def forward(self,x):
        return self.fn(self.norm(x))

#MLP层
class FeedForward(nn.Module):
    def __init__(self,dim,hidden_dim,dropout=0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim,hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim,dim),
            nn.Dropout(dropout)
        )
    def forward(self,x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self,dim,heads=8,dim_head=64,dropout=0):
        super().__init__()
        inner_dim = dim_head*heads
        project_out = not(heads==1 and dim_head==dim)  #not(True)时维度和输入一致

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)    #行归一化
        self.to_qkv = nn.Linear(dim,inner_dim*3,bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim,dim),
            nn.Dropout(dropout)
        )if project_out else nn.Identity()

    def forward(self,x):
        b,n,_,h = *x.shape,self.heads
        qkv = self.to_qkv(x).chunk(3,dim = -1)
        q,k,v = map(lambda t:rearrange(t,'b n (h d) -> b h n d',h=h),qkv)

        dots = einsum('b h i d,b h j d->b h i j',q,k)*self.scale
        attn = self.attend(dots)

        out = einsum('b h i j,b h j d->b h i d',attn,v)
        out = rearrange(out,'b h n d->b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self,depth,dim,heads,dim_head,dropout,hidden_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim=dim,fn=Attention(dim=dim,heads=heads,dim_head=dim_head,dropout=dropout)),
                PreNorm(dim=dim,fn=FeedForward(dim=dim,hidden_dim=hidden_dim,dropout=dropout))
            ]))
    def forward(self,x):
        for attn,ff in self.layers:
            x = attn(x)+x
            x = ff(x)+x
        return x

class ViT(nn.Module):
    def __init__(self,image_size,path_size,num_classes,dim,depth,heads,mlp_dim,pool='cls',channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height,image_width = pair(image_size)
        patch_height,patch_width = pair(path_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0
        assert pool in {'cls', 'mean'}

        num_patches = (image_height//patch_height)*(image_width//patch_width)
        patch_dim = channels*patch_height*patch_width

        #pach展平并映射到目标维度
        self.to_patch_embding = nn.Sequential(
            Rearrange('b c (h p1) (w p2)-> b (h w) (p1 p2 c)',p1=patch_height,p2 = patch_width),
            nn.Linear(patch_dim,dim)
        )

        #位置编码
        self.pos_embding = nn.Parameter(torch.randn(1,num_patches+1,dim))
        #信息融合向量
        self.cls_token = nn.Parameter(torch.randn(1,1,dim))
        #编码时dropout
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(depth=depth,dim=dim,heads=heads,dim_head=dim_head,dropout=dropout,hidden_dim=mlp_dim)

        self.pool = pool    #最终分类选择向量方式
        #最终输出层
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim,num_classes)
        )
    def forward(self,img):
        x = self.to_patch_embding(img)
        b,n,_ = x.shape
        cls_tokens = repeat(self.cls_token,'() n d -> b n d',b=b)
        x = torch.cat((cls_tokens,x),dim=1)
        x+= self.pos_embding
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool=='mean' else x[:,0]

        return self.mlp_head(x)

my_vit = ViT(
    image_size = 256,
    path_size = 32,
    num_classes = 10,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    pool='cls',
    channels=3,
    dim_head=64,
    dropout=0.1,
    emb_dropout=0.1
)

img = torch.randn(16, 3, 256, 256)

preds = my_vit(img)

print(preds.shape)  # (16, 1000)


