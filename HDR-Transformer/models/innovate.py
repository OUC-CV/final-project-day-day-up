import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm
from einops import rearrange
import numbers
from timm.models.layers import DropPath, trunc_normal_, to_2tuple


# 多层感知机（MLP）带有两个线性层，激活函数和dropout
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# 初始化网络权重
@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


# 通过堆叠相同的块来创建层
def make_layer(basic_block, num_basic_block, **kwarg):
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


# 残差块
class ResidualBlockNoBN(nn.Module):
    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


# 上采样模块
class Upsample(nn.Sequential):
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

# 无偏置的层归一化
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


# 有偏置的层归一化
class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


# 选择层归一化类型
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

# 互注意力机制
class Mutual_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Mutual_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        assert x.shape == y.shape
        b, c, h, w = x.shape

        q = self.q(x)  # 计算查询向量
        k = self.k(y)  # 计算键向量
        v = self.v(y)  # 计算值向量

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature  # 计算注意力权重
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

# Change模块，包括归一化、互注意力和MLP
class Change(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias'):
        super(Change, self).__init__()

        # 初始化归一化层
        self.norm1_x = LayerNorm(dim, LayerNorm_type) 
        self.norm1_y = LayerNorm(dim, LayerNorm_type)  
        self.attn = Mutual_Attention(dim, num_heads, bias)  # 相互注意力机制

        self.norm2 = nn.LayerNorm(dim) 
        mlp_hidden_dim = int(dim * ffn_expansion_factor)  # MLP隐藏层维度
        self.ffn = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)  # MLP

    def forward(self, x, y, x_size):
        b, n, c = x.shape 
        h, w = x_size  
        assert n == h * w 

        # 将x和y调整为4维张量
        x = x.permute(0, 2, 1).view(b, c, h, w)  
        y = y.permute(0, 2, 1).view(b, c, h, w)  
        b, c, h, w = x.shape 

        # 互注意力机制
        fused = x + self.attn(self.norm1_x(x), self.norm1_y(y))  # 计算互注意力并进行融合

        # MLP
        fused = to_3d(fused)  # 将fused转换为3维张量
        fused = fused + self.ffn(self.norm2(fused))  # 通过MLP并进行残差连接

        return fused


# 注意力机制
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads  # 每个头的维度
        self.scale = qk_scale or head_dim ** -0.5  # 缩放因子

        self.q = nn.Linear(dim, dim, bias=qkv_bias)  # 查询向量的全连接层
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)  # 键和值向量的全连接层
        self.attn_drop = nn.Dropout(attn_drop)  # 注意力权重的dropout
        self.proj = nn.Linear(dim, dim)  # 输出的全连接层
        self.proj_drop = nn.Dropout(proj_drop)  # 输出的dropout

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:  # 如果sr_ratio大于1，则使用空间缩减
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)  # 空间缩减的卷积层
            self.norm = nn.LayerNorm(dim)  # 层归一化

    def forward(self, x, y, H=None, W=None):
        assert x.dim() == 3, x.shape
        assert x.shape == y.shape 
        B, N, C = x.shape  # 获取批次大小，序列长度和通道数

        # 计算查询向量
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:  # 如果sr_ratio大于1，则对y进行空间缩减
            y_ = y.permute(0, 2, 1).reshape(B, C, H, W)  
            y_ = self.sr(y_).reshape(B, C, -1).permute(0, 2, 1) 
            y_ = self.norm(y_)  # 归一化
            kv = self.kv(y_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(y).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # 分离键和值向量

        # 计算注意力权重
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)  # 对权重进行softmax
        attn = self.attn_drop(attn)  # 对权重进行dropout

        # 计算注意力输出
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)  # 输出层
        x = self.proj_drop(x)  # 输出dropout

        return x


# 合并注意力机制
class Merge_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Merge_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.q2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.conv = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.project_out1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.project_out2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.convout = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        assert x.shape == y.shape
        b, c, h, w = x.shape

        q1 = self.q1(x)
        k1 = self.k1(x)
        v1 = self.v1(x)

        q2 = self.q1(y)
        k2 = self.k1(y)
        v2 = self.v1(y)

        q = self.conv(q1 + q2)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        k1 = rearrange(k1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v1 = rearrange(v1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k2 = rearrange(k2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v2 = rearrange(v2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k1 = torch.nn.functional.normalize(k1, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)

        attn1 = (q @ k1.transpose(-2, -1)) * self.temperature
        attn1 = attn1.softmax(dim=-1)
        out1 = (attn1 @ v1)
        out1 = rearrange(out1, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out1 = self.project_out1(out1)

        attn2 = (q @ k2.transpose(-2, -1)) * self.temperature
        attn2 = attn2.softmax(dim=-1)
        out2 = (attn2 @ v2)
        out2 = rearrange(out2, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out2 = self.project_out2(out2)
        out = self.convout(out1 + out2 + x + y)
        return out