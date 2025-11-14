
import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


from mpd.models.layers.layers import TimeEncoder

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)



class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class PlaningDiT(nn.Module):
    def __init__(
        self, 
        state_dim=2,
        hidden_size=64, 
        num_heads=8, 
        mlp_ratio=4.0,
        depth=28,
        output_size=2,
    ):
        super(PlaningDiT, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.depth = depth
        self.state_dim = state_dim

        self.time_mlp = TimeEncoder(32, 64)

        self.x_proj = nn.Linear(2, 64)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])

        self.x_reproj = nn.Linear(hidden_size, output_size)

    
    def forward(self, x, time, context):
        x = self.x_proj(x)
        time = self.time_mlp(time)
        context = context + time
        # print(f"dit model x shape: {x.shape}", f"context shape: {context.shape}")
        for block in self.blocks:
            x = block(x, context)
        x = self.x_reproj(x)
        return x


if __name__ == "__main__":
    """
    DiTBlock forward 方法期望的输入：
        x: (batch_size, seq_len, hidden_size) - 序列数据
        c: (batch_size, hidden_size) - 条件向量，维度必须等于 hidden_size
    output:
        x: (batch_size, seq_len, hidden_size)
    """

    batch_size = 16
    seq_len = 16  # 序列长度（token 数量）
    hidden_size = 2  # 特征维度
    
    # x 的形状应该是 (batch_size, seq_len, hidden_size)
    x = torch.randn(batch_size, seq_len, hidden_size)    # 根据
    # context 的形状应该是 (batch_size, hidden_size)，维度必须等于 hidden_size
    context = torch.randn(batch_size, 64)
    time = torch.randn(batch_size, )
    print(f"输入 x 的形状: {x.shape}")
    print(f"输入 context 的形状: {context.shape}")
    
    model = PlaningDiT(hidden_size=64, num_heads=8, mlp_ratio=4.0, depth=28, output_size=2)

    output = model(x, time, context)
    print(f"输出形状: {output.shape}")