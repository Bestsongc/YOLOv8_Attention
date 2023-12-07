import torch
from torch import nn, Tensor

__all__ = ["SimAM", "BiLevelRoutingAttention"]


# SimAM Attention Module
# START
class SimAM(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "lambda=%f)" % self.e_lambda
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = (
            x_minus_mu_square
            / (
                4
                * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)
            )
            + 0.5
        )

        return x * self.activaton(y)


# END OF SimAm Attention Module


# BiLevelRouting Attention Module
# START
class TopkRouting(nn.Module):
    """
    differentiable topk routing with scaling
    Args:
        qk_dim: int, feature dimension of query and key
        topk: int, the 'topk'
        qk_scale: int or None, temperature (multiply) of softmax activation
        with_param: bool, wether inorporate learnable params in routing unit
        diff_routing: bool, wether make routing differentiable
        soft_routing: bool, wether make output value multiplied by routing weights
    """

    def __init__(
        self, qk_dim, topk=4, qk_scale=None, param_routing=False, diff_routing=False
    ):
        super().__init__()
        self.topk = topk
        self.qk_dim = qk_dim
        self.scale = qk_scale or qk_dim**-0.5
        self.diff_routing = diff_routing
        # TODO: norm layer before/after linear?
        self.emb = nn.Linear(qk_dim, qk_dim) if param_routing else nn.Identity()
        # routing activation
        self.routing_act = nn.Softmax(dim=-1)

    def forward(self, query: Tensor, key: Tensor) -> Tuple[Tensor]:
        """
        Args:
            q, k: (n, p^2, c) tensor
        Return:
            r_weight, topk_index: (n, p^2, topk) tensor
        """
        if not self.diff_routing:
            query, key = query.detach(), key.detach()
        query_hat, key_hat = self.emb(query), self.emb(
            key
        )  # per-window pooling -> (n, p^2, c)
        attn_logit = (query_hat * self.scale) @ key_hat.transpose(
            -2, -1
        )  # (n, p^2, p^2)
        topk_attn_logit, topk_index = torch.topk(
            attn_logit, k=self.topk, dim=-1
        )  # (n, p^2, k), (n, p^2, k)
        r_weight = self.routing_act(topk_attn_logit)  # (n, p^2, k)

        return r_weight, topk_index


class KVGather(nn.Module):
    def __init__(self, mul_weight="none"):
        super().__init__()
        assert mul_weight in ["none", "soft", "hard"]
        self.mul_weight = mul_weight

    def forward(self, r_idx: Tensor, r_weight: Tensor, kv: Tensor):
        """
        r_idx: (n, p^2, topk) tensor
        r_weight: (n, p^2, topk) tensor
        kv: (n, p^2, w^2, c_kq+c_v)

        Return:
            (n, p^2, topk, w^2, c_kq+c_v) tensor
        """
        # select kv according to routing index
        n, p2, w2, c_kv = kv.size()
        topk = r_idx.size(-1)
        # print(r_idx.size(), r_weight.size())
        # FIXME: gather consumes much memory (topk times redundancy), write cuda kernel?
        topk_kv = torch.gather(
            kv.view(n, 1, p2, w2, c_kv).expand(
                -1, p2, -1, -1, -1
            ),  # (n, p^2, p^2, w^2, c_kv) without mem cpy
            dim=2,
            index=r_idx.view(n, p2, topk, 1, 1).expand(
                -1, -1, -1, w2, c_kv
            ),  # (n, p^2, k, w^2, c_kv)
        )

        if self.mul_weight == "soft":
            topk_kv = (
                r_weight.view(n, p2, topk, 1, 1) * topk_kv
            )  # (n, p^2, k, w^2, c_kv)
        elif self.mul_weight == "hard":
            raise NotImplementedError("differentiable hard routing TBA")
        # else: #'none'
        #     topk_kv = topk_kv # do nothing

        return topk_kv


class QKVLinear(nn.Module):
    def __init__(self, dim, qk_dim, bias=True):
        super().__init__()
        self.dim = dim
        self.qk_dim = qk_dim
        self.qkv = nn.Linear(dim, qk_dim + qk_dim + dim, bias=bias)

    def forward(self, x):
        q, kv = self.qkv(x).split([self.qk_dim, self.qk_dim + self.dim], dim=-1)
        return q, kv


class BiLevelRoutingAttention(nn.Module):
    """
    n_win: number of windows in one side (so the actual number of windows is n_win*n_win)
    kv_per_win: for kv_downsample_mode='ada_xxxpool' only, number of key/values per window. Similar to n_win, the actual number is kv_per_win*kv_per_win.
    topk: topk for window filtering
    param_attention: 'qkvo'-linear for q,k,v and o, 'none': param free attention
    param_routing: extra linear for routing
    diff_routing: wether to set routing differentiable
    soft_routing: wether to multiply soft routing weights
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        n_win=7,
        qk_dim=None,
        qk_scale=None,
        kv_per_win=4,
        kv_downsample_ratio=4,
        kv_downsample_kernel=None,
        kv_downsample_mode="identity",
        topk=4,
        param_attention="qkvo",
        param_routing=False,
        diff_routing=False,
        soft_routing=False,
        side_dwconv=3,
        auto_pad=True,
    ):
        super().__init__()
        # local attention setting
        self.dim = dim
        self.n_win = n_win  # Wh, Ww
        self.num_heads = num_heads
        self.qk_dim = qk_dim or dim
        assert (
            self.qk_dim % num_heads == 0 and self.dim % num_heads == 0
        ), "qk_dim and dim must be divisible by num_heads!"
        self.scale = qk_scale or self.qk_dim**-0.5

        ################side_dwconv (i.e. LCE in ShuntedTransformer)###########
        self.lepe = (
            nn.Conv2d(
                dim,
                dim,
                kernel_size=side_dwconv,
                stride=1,
                padding=side_dwconv // 2,
                groups=dim,
            )
            if side_dwconv > 0
            else lambda x: torch.zeros_like(x)
        )

        ################ global routing setting #################
        self.topk = topk
        self.param_routing = param_routing
        self.diff_routing = diff_routing
        self.soft_routing = soft_routing
        # router
        assert not (
            self.param_routing and not self.diff_routing
        )  # cannot be with_param=True and diff_routing=False
        self.router = TopkRouting(
            qk_dim=self.qk_dim,
            qk_scale=self.scale,
            topk=self.topk,
            diff_routing=self.diff_routing,
            param_routing=self.param_routing,
        )
        if self.soft_routing:  # soft routing, always diffrentiable (if no detach)
            mul_weight = "soft"
        elif self.diff_routing:  # hard differentiable routing
            mul_weight = "hard"
        else:  # hard non-differentiable routing
            mul_weight = "none"
        self.kv_gather = KVGather(mul_weight=mul_weight)

        # qkv mapping (shared by both global routing and local attention)
        self.param_attention = param_attention
        if self.param_attention == "qkvo":
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Linear(dim, dim)
        elif self.param_attention == "qkv":
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Identity()
        else:
            raise ValueError(
                f"param_attention mode {self.param_attention} is not surpported!"
            )

        self.kv_downsample_mode = kv_downsample_mode
        self.kv_per_win = kv_per_win
        self.kv_downsample_ratio = kv_downsample_ratio
        self.kv_downsample_kenel = kv_downsample_kernel
        if self.kv_downsample_mode == "ada_avgpool":
            assert self.kv_per_win is not None
            self.kv_down = nn.AdaptiveAvgPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == "ada_maxpool":
            assert self.kv_per_win is not None
            self.kv_down = nn.AdaptiveMaxPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == "maxpool":
            assert self.kv_downsample_ratio is not None
            self.kv_down = (
                nn.MaxPool2d(self.kv_downsample_ratio)
                if self.kv_downsample_ratio > 1
                else nn.Identity()
            )
        elif self.kv_downsample_mode == "avgpool":
            assert self.kv_downsample_ratio is not None
            self.kv_down = (
                nn.AvgPool2d(self.kv_downsample_ratio)
                if self.kv_downsample_ratio > 1
                else nn.Identity()
            )
        elif self.kv_downsample_mode == "identity":  # no kv downsampling
            self.kv_down = nn.Identity()
        elif self.kv_downsample_mode == "fracpool":
            # assert self.kv_downsample_ratio is not None
            # assert self.kv_downsample_kenel is not None
            # TODO: fracpool
            # 1. kernel size should be input size dependent
            # 2. there is a random factor, need to avoid independent sampling for k and v
            raise NotImplementedError("fracpool policy is not implemented yet!")
        elif kv_downsample_mode == "conv":
            # TODO: need to consider the case where k != v so that need two downsample modules
            raise NotImplementedError("conv policy is not implemented yet!")
        else:
            raise ValueError(
                f"kv_down_sample_mode {self.kv_downsaple_mode} is not surpported!"
            )

        # softmax for local attention
        self.attn_act = nn.Softmax(dim=-1)

        self.auto_pad = auto_pad


# END OF BiLevelRouting Attention Module
