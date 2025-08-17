import torch
from torch import Tensor, nn
from torch.nn import functional as F
from abc import abstractmethod
'''
    代码讲解视频：https://www.bilibili.com/video/BV1B6mtYKECo/


    缝合一：
    论文地址：https://arxiv.org/pdf/2404.14757
    论文题目：SST: Multi-Scale Hybrid Mamba - Transformer Experts for Long - Short Range Time Series Forecasting
    中文题目：SST：用于长短期时间序列预测的多尺度混合Mamba - Transformer专家模型
        系列讲解视频：https://www.bilibili.com/video/BV1cVmBYvEQL/
    缝合二：
    论文地址：https://dl.acm.org/doi/abs/10.1145/3664647.3680650
    论文题目：Multi-Scale and Detail-Enhanced Segment Anything Model for Salient Object Detection（A会议 2024）
    中文题目：差分边缘增强模块（A会议 2024）
        系列讲解视频：https://www.bilibili.com/video/BV1PTDoYAEhx/
'''
def silu(x):
    return x * F.sigmoid(x)

class RMSNorm(nn.Module):
    """
        Gated Root Mean Square Layer Normalization
        Paper: https://arxiv.org/abs/1910.07467
    """
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x, z):
        x = x * silu(z)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class Mamba2(nn.Module):
    def __init__(self, d_model: int,  # model dimension (D)
                 n_layer: int = 24,  # number of Mamba-2 layers in the language model
                 d_state: int = 128,  # state dimension (N)
                 d_conv: int = 4,  # convolution kernel size
                 expand: int = 2,  # expansion factor (E)
                 headdim: int = 64,  # head dimension (P)
                 chunk_size: int = 64,  # matrix partition size (Q)
                 ):
        super().__init__()
        self.n_layer = n_layer
        self.d_state = d_state
        self.headdim = headdim
        # self.chunk_size = torch.tensor(chunk_size, dtype=torch.int32)
        self.chunk_size = chunk_size

        self.d_inner = expand * d_model
        assert self.d_inner % self.headdim == 0, "self.d_inner must be divisible by self.headdim"
        self.nheads = self.d_inner // self.headdim

        d_in_proj = 2 * self.d_inner + 2 * self.d_state + self.nheads
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False)

        conv_dim = self.d_inner + 2 * d_state
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, d_conv, groups=conv_dim, padding=d_conv - 1, )
        self.dt_bias = nn.Parameter(torch.empty(self.nheads, ))
        self.A_log = nn.Parameter(torch.empty(self.nheads, ))
        self.D = nn.Parameter(torch.empty(self.nheads, ))
        self.norm = RMSNorm(self.d_inner, )
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False, )

    def forward(self, u: Tensor):
        # 计算负指数的 A_log，用于参数化的状态空间模型
        A = -torch.exp(self.A_log)  # (nheads,)

        # 输入投影，将输入 u 映射到更高维度的特征空间
        zxbcdt = self.in_proj(u)  # (batch, seqlen, d_in_proj)

        """
            局部特征提取
        """
        # 将投影结果分割为 z, xBC 和 dt
        z, xBC, dt = torch.split(
            zxbcdt,
            [
                self.d_inner,
                self.d_inner + 2 * self.d_state,
                self.nheads,
            ],
            dim=-1,
        )

        # 通过 softplus 激活函数处理 dt，并加上偏置
        dt = F.softplus(dt + self.dt_bias)  # (batch, seqlen, nheads)

        # 通过一维卷积处理 xBC，捕获局部上下文信息
        xBC = silu(
            self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, : u.shape[1], :]
        )  # (batch, seqlen, d_inner + 2 * d_state)

        # 将卷积结果分割为 x, B, C
        x, B, C = torch.split(
            xBC, [self.d_inner, self.d_state, self.d_state], dim=-1
        )


        """
            多头机制
            重塑 x 的形状以适应多头机制
        """
        _b, _l, _hp = x.shape
        _h = _hp // self.headdim
        _p = self.headdim
        x = x.reshape(_b, _l, _h, _p)

        """
            # 使用 ssd 函数进行复杂的序列状态计算
            函数用于处理复杂的序列状态更新，结合参数化的状态空间模型，捕获长序列中的依赖关系。
        """
        y = self.ssd(x * dt.unsqueeze(-1),
                                     A * dt,
                                     B.unsqueeze(2),
                                     C.unsqueeze(2), )

        # 将计算结果与输入 x 结合，应用可学习参数 D
        y = y + x * self.D.unsqueeze(-1)

        # 将 y 重塑回原始形状
        _b, _l, _h, _p = y.shape
        y = y.reshape(_b, _l, _h * _p)

        # 应用 RMSNorm 进行归一化，并使用 z 进行缩放
        y = self.norm(y, z)

        # 通过输出投影层将特征维度调整回输入维度
        y = self.out_proj(y)

        return y

    def segsum(self, x: Tensor) -> Tensor:
        T = x.size(-1)
        device = x.device
        x = x[..., None].repeat(1, 1, 1, 1, T)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=-1)
        x = x.masked_fill(~mask, 0)
        x_segsum = torch.cumsum(x, dim=-2)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=0)
        x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
        return x_segsum

    def ssd(self, x, A, B, C):
        chunk_size = self.chunk_size
        # if x.shape[1] % chunk_size == 0:
        #
        x = x.reshape(x.shape[0], x.shape[1] // chunk_size, chunk_size, x.shape[2], x.shape[3], )
        B = B.reshape(B.shape[0], B.shape[1] // chunk_size, chunk_size, B.shape[2], B.shape[3], )
        C = C.reshape(C.shape[0], C.shape[1] // chunk_size, chunk_size, C.shape[2], C.shape[3], )
        A = A.reshape(A.shape[0], A.shape[1] // chunk_size, chunk_size, A.shape[2])
        A = A.permute(0, 3, 1, 2)
        A_cumsum = torch.cumsum(A, dim=-1)

        # 1. Compute the output for each intra-chunk (diagonal blocks)
        L = torch.exp(self.segsum(A))
        Y_diag = torch.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C, B, L, x)

        # 2. Compute the state for each intra-chunk
        # (right term of low-rank factorization of off-diagonal blocks; B terms)
        decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
        states = torch.einsum("bclhn, bhcl, bclhp -> bchpn", B, decay_states, x)

        # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
        # (middle term of factorization of off-diag blocks; A terms)

        initial_states = torch.zeros_like(states[:, :1])
        states = torch.cat([initial_states, states], dim=1)

        decay_chunk = torch.exp(self.segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))[0]
        new_states = torch.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)
        states = new_states[:, :-1]

        # 4. Compute state -> output conversion per chunk
        # (left term of low-rank factorization of off-diagonal blocks; C terms)
        state_decay_out = torch.exp(A_cumsum)
        Y_off = torch.einsum("bclhn, bchpn, bhcl -> bclhp", C, states, state_decay_out)

        # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
        # Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
        Y = Y_diag + Y_off
        Y = Y.reshape(Y.shape[0], Y.shape[1] * Y.shape[2], Y.shape[3], Y.shape[4], )

        return Y

class _BiMamba2(nn.Module):
    def __init__(self,
                 cin: int,
                 cout: int,
                 d_model: int,  # model dimension (D)
                 n_layer: int = 24,  # number of Mamba-2 layers in the language model
                 d_state: int = 128,  # state dimension (N)
                 d_conv: int = 4,  # convolution kernel size
                 expand: int = 2,  # expansion factor (E)
                 headdim: int = 64,  # head dimension (P)
                 chunk_size: int = 64,  # matrix partition size (Q)
                 ):
        super().__init__()
        self.fc_in = nn.Linear(cin, d_model, bias=False)  # 调整通道数到cmid
        self.mamba2_for = Mamba2(d_model, n_layer, d_state, d_conv, expand, headdim, chunk_size, )  # 正向
        self.mamba2_back = Mamba2(d_model, n_layer, d_state, d_conv, expand, headdim, chunk_size, )  # 负向
        self.fc_out = nn.Linear(d_model, cout, bias=False)  # 调整通道数到cout
        self.chunk_size = chunk_size

    @abstractmethod
    def forward(self, x):
        pass

class BiMamba2_2D(_BiMamba2):
    def __init__(self, cin, cout, d_model, **mamba2_args):
        super().__init__(cin, cout, d_model, **mamba2_args)
        self.fc_in = torch.nn.Linear(cin, d_model)
        self.fc_out = torch.nn.Linear(d_model, cout)

    def forward(self, x):
        h, w = x.shape[2:]
        x = F.pad(x, (0, (8 - x.shape[3] % 8) % 8,
                      0, (8 - x.shape[2] % 8) % 8))  # 将 h , w  pad到8的倍数, [b, c64, h8, w8]
        _b, _c, _h, _w = x.shape

        x = x.permute(0, 2, 3, 1).reshape(_b, _h * _w, _c)

        x = self.fc_in(x)  # 调整通道数为目标通道数
        x1 = self.mamba2_for(x)
        x2 = self.mamba2_back(x.flip(1)).flip(1)
        x = x1 + x2
        x = self.fc_out(x)  # 调整通道数为目标通道数

        x = x.reshape(_b, _h, _w, -1).permute(0, 3, 1, 2)  # 恢复到 (batch, channel, height, width)
        x = x[:, :, :h, :w]  # 截取原图大小
        return x

class Mamba_Enhancement_Fusion_Module(nn.Module):
    def __init__(self, img_dim, feature_dim, norm = nn.BatchNorm2d, act=nn.ReLU):
        super().__init__()
        self.img_in_conv = nn.Sequential(
            nn.Conv2d(img_dim,feature_dim, 3, padding=1, bias=False),
            norm(feature_dim),
            act()
        )

        # 缝合 https://www.bilibili.com/video/BV1cVmBYvEQL/
        self.img_er = BiMamba2_2D(128, 128, 64)

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(feature_dim *2, feature_dim, 3, padding=1, bias=False),
            norm(feature_dim),
            act(),
            nn.Conv2d(feature_dim, feature_dim * 2, 3, padding=1, bias=False),
            norm(feature_dim * 2),
            act(),
        )

        self.out_conv = nn.Conv2d(feature_dim * 2, img_dim, 1)

        self.feature_upsample = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, 3, padding=1, bias=False),
            norm(feature_dim),
            act(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False),
            norm(feature_dim),
            act(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False),
            norm(feature_dim),
            act(),
        )

    def forward(self, img, feature, b_feature):
        """
           源头：发现问题：
                https://www.bilibili.com/video/BV1cVmBYvEQL/
                尽管时间序列预测取得了显著进展，但现有的预测器忽视长短期时间序列之间的异质性，导致在实际应用中性能下降。
                长范围时间序列应关注全局模式，短范围时间序列应强调局部变化，但二者区分标准不明确，且现有方法难以分别有效处理长、短范围时间序列并整合其依赖关系。
           进阶思路：
                尽管XXX取得了显著进展，但现有的预测器忽视局部、全局之间的异质性，导致在实际应用中性能下降。
                长范围时间序列应关注全局模式，短范围时间序列应强调局部变化，但二者区分标准不明确，且现有方法难以分别有效处理全局、局部特征并整合其依赖关系。
        """
        # #torch.Size([1, 128, 128, 128])
        img_feature = self.img_in_conv(img)

        # 故事1：全局特征
        img_feature = self.img_er(img_feature) + img_feature

        # 故事2：局部特征
        feature = torch.cat([feature, b_feature], dim=1)
        feature = self.feature_upsample(feature)

        # 故事3：全局-局部特征融合
        out_feature = torch.cat([feature, img_feature], dim=1)
        out_feature = self.fusion_conv(out_feature)
        out = self.out_conv(out_feature)
        return out

if __name__ == "__main__":

    MSDE =Mamba_Enhancement_Fusion_Module(img_dim=64,feature_dim=128)

    input = torch.randn(1, 64, 128, 128)

    feature = torch.randn(1, 128, 32, 32)
    b_feature = torch.randn(1, 128, 32, 32)

    output= MSDE(input,feature,b_feature)

    print('Input size:', input.size())
    print('Output size:', output.size())
    print('---------------------------------')