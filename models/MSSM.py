import torch
import torch.nn as nn
from einops import rearrange

# Ensure mamba_ssm is installed
from mamba_ssm.modules.mamba_simple import Mamba

class MSSM(nn.Module):
    """
    Motion-aware State Space Model (MSSM)
    Definitive Stable Version: Corrected dimensions for gating.
    """
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        conv_bias=True,
        bias=False,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.d_model = d_model
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.d_conv = d_conv
        self.d_state = d_state

        # --- FINAL CORRECTION 1: Adjust projection size ---
        # We need two main paths (fa, fm) and one final gate (fg).
        # Total projection size should be d_inner * 3.
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 3, bias=bias, **factory_kwargs)

        # Layers for the two branches
        self.conv1d_multi = nn.Conv1d(
            in_channels=self.d_inner, out_channels=self.d_inner, bias=conv_bias,
            kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1, **factory_kwargs
        )
        self.conv1d_single = nn.Conv1d(
            in_channels=self.d_inner, out_channels=self.d_inner, bias=conv_bias,
            kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1, **factory_kwargs
        )
        # Core SSM module
        self.ssm = Mamba(
            d_model=self.d_inner, d_state=self.d_state,
            d_conv=self.d_conv, expand=1,
        )
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        
        # Add LayerNorm for stability before the fusion step
        self.norm_fusion = nn.LayerNorm(self.d_inner)


    def forward(self, x, F=4):
        B, N, C = x.shape
        Np = N // F
        assert N % F == 0, f"Total sequence length {N} must be divisible by {F}."

        # --- FINAL CORRECTION 2: Correctly chunk the projection ---
        # Project and split into three parts: fa_proj, fm_proj, and the final gate fg
        x_proj = self.in_proj(x)
        fa_proj, fm_proj, fg = x_proj.chunk(3, dim=-1)

        # --- Appearance Branch (fa) ---
        fa = rearrange(fa_proj, 'b (f n) c -> (b f) n c', f=F, n=Np).contiguous()
        fa = rearrange(fa, 'bf n c -> bf c n').contiguous()
        fa = self.conv1d_single(fa)[:, :, :Np]
        fa = rearrange(fa, 'bf c n -> bf n c').contiguous()
        fa = rearrange(fa, '(b f) n c -> b (f n) c', f=F, n=Np).contiguous()

        # --- Motion Branch (fm) ---
        fm = rearrange(fm_proj, 'b n c -> b c n').contiguous()
        fm = self.conv1d_multi(fm)[:, :, :N]
        fm = rearrange(fm, 'b c n -> b n c').contiguous()
        
        # --- Stable Fusion: ADDITION instead of multiplication ---
        # We add the two feature streams. This is the most stable operation.
        fused_features = fa + fm
        
        # Apply LayerNorm before the SSM for extra stability
        fused_features_norm = self.norm_fusion(fused_features)
        
        # --- SSM and Final Gating ---
        ssm_out = self.ssm(fused_features_norm)
        
        # The final gate `fg` now has the correct dimension (d_inner)
        output = ssm_out * self.act(fg)
        output = self.out_proj(output)
        
        return output

# --- 使用示例 (Example Usage) ---
if __name__ == '__main__':
    # Simulation Parameters
    batch_size = 2
    num_scans = 4      # F, number of aggregated scans
    points_per_scan = 1024
    d_model = 128      # Feature dimension

    # Instantiate the corrected MSSM module
    mssm_block = MSSM(d_model=d_model).cuda()
    print("✅ MSSM Block Instantiated Successfully!")

    # Create a dummy input tensor. The sequence length N is the total number of points.
    total_points = num_scans * points_per_scan
    input_features = torch.randn(batch_size, total_points, d_model).cuda()

    # Perform a forward pass
    print(f"\nRunning forward pass with input shape: {input_features.shape}")
    try:
        output_features = mssm_block(input_features, F=num_scans)
        # Verify that the output shape is consistent with the input shape
        assert input_features.shape == output_features.shape
        print(f"✅ Forward pass successful!")
        print(f"   Input shape:  {input_features.shape}")
        print(f"   Output shape: {output_features.shape}")
    except Exception as e:
        print(f"❌ An error occurred during the forward pass: {e}")