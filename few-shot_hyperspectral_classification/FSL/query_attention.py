import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        # 这个LayerNorm是对输入矩阵的最后一位进行归一化，输入的dim参数要跟矩阵最后一维的大小一致
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, y,**kwargs):
        return self.fn(self.norm(x), self.norm(y), **kwargs)

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
    def forward(self, x, y):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.q_matrix = nn.Linear(dim, inner_dim, bias=False)
        self.k_matrix = nn.Linear(dim, inner_dim, bias=False)
        self.v_matrix = nn.Linear(dim, inner_dim, bias=False)

        #self.norm = nn.LayerNorm(dim)


    def forward(self, x, y):
        #x = self.norm(x)
        #y = self.norm(y)

        # 这个操作是将x与Q,K,V进行运算，然后将各个运算结果分开来
        # 其实这里的运算是通过x与一个|QKV|三个矩阵拼接的矩阵进行相乘实现的，运算完成之后，按列分割成3块，就
        # 分别得到了各个运算结果
        # qkv = self.to_qkv(x).chunk(3, dim = -1)
        q = self.q_matrix(x) # (240, 160)
        k = self.k_matrix(y)
        v = self.v_matrix(y)

        # 这里的这个rearrange就是根据'b n (h d) -> b h n d'指定的规则，来调整输入张量的维度大小，
        # 这里是将 b n m规格的矩阵，调整为 b h n d，其中，h即为多头注意力机制中的多头的数目，n为输入矩阵的行数
        # d为变换之后的列数，也即向量的维度大小，这里输出的q,k,v矩阵规格为： b,8,n,64
        # 这里的qkv其实是包含了三个矩阵，所以这里的map就是对每个矩阵进行操作
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        # q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)
        # k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        # v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)
        q = rearrange(q, 'n (h d) -> h n d', h = self.heads)
        k = rearrange(k, 'n (h d) -> h n d', h=self.heads)
        v = rearrange(v, 'n (h d) -> h n d', h=self.heads)
        #这个操作是执行 qk'/根号K的运算，也就是transformer的自注意力机制运算中的一步
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # 这里的attend函数就是对多头注意力机制所形成的每个注意力对应的矩阵（规格为b,8,n,n）进行softmax运算
        attn = self.attend(dots)
        # 这里的操作就是利用注意力机制形成的系数，来乘以value矩阵，获得每头注意力下面的值向量
        # 由 b,8,n,n 维度的矩阵，得到 b,8,n,64规格的矩阵

        out = torch.matmul(attn, v)
        # 这里的这个操作是将b,8,n,64规格的矩阵变换为b,n,(8*64)规则的矩阵，也就是将每头注意力得到的向量进行拼接合并
        # out = rearrange(out, 'b h n d -> b n (h d)')
        out = rearrange(out, 'h n d -> n (h d)')
        # 这里是将多头注意力机制的运算结果再经过一个线性变换，变换到以前的维度，就是将b,n,(8*64)规格变换为b,n,64规格
        #return self.to_out(out)+x
        return self.to_out(out)




class Transformer(nn.Module):
    # dim: 输入的每个词的向量的维度大小
    # depth： 叠加的编码模块的数量
    # heads: 多头注意力机制的头的个数
    # dim_head: 每头注意力机制的输出的向量的大小
    # mlp_dim: 编码模块中的前馈子模块的多层感知机隐藏层的大小
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x, y):
        x.to(device)
        y.to(device)
        for attn, ff in self.layers:
            x = attn(x, y) + x #在多头注意力机制之后，先进行normalize，然后再求和
            x = ff(x, y) + x #同样是再前向传播的过程后，先进行normalize，然后再求和
        return x
