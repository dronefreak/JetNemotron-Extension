import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryPositionEmbedding:
    """Implements Rotary Positional Embedding (RoPE) for applying position-dependent
    rotations to query and key vectors.

    This encodes relative positions into the Q/K dot products.
    """

    def __init__(self, dim):
        # RoPE pre-computes cosine and sine frequency for each pair of dimensions.
        # We use base 10000 and pair each two adjacent dimensions.
        half = dim // 2
        # Prepare frequencies for each position dimension (half of dim if dim is even).
        freq_exponents = torch.arange(0, half, dtype=torch.float32) / float(half)
        freq_rates = 10000.0 ** (-2 * freq_exponents)  # shape [half]
        self.freq_rates = (
            freq_rates  # will be used to compute cos/sin for given positions
        )

    def apply_rotary(self, Q: torch.Tensor, K: torch.Tensor):
        """Apply rotary embedding to the last dimension of Q and K.

        Expects Q, K shape: [batch, seq, heads, dim] (dim is even).
        """
        if Q.shape[-1] % 2 != 0:
            raise ValueError("RoPE dimension must be even.")
        half = Q.shape[-1] // 2
        # positions from 0 to seq-1
        seq_len = Q.shape[1]
        pos = torch.arange(seq_len, dtype=torch.float32, device=Q.device)
        # Compute angles: outer product of positions and frequency rates
        angles = torch.outer(pos, self.freq_rates.to(Q.device))  # [seq, half]
        # Get cosine and sine values for each position and dimension pair
        cos = torch.cos(angles)  # [seq, half]
        sin = torch.sin(angles)  # [seq, half]
        # Expand cos/sin to [batch, seq, heads, half] for broadcasting
        cos = cos[None, :, None, :].expand_as(Q[..., :half])
        sin = sin[None, :, None, :].expand_as(Q[..., :half])
        # Apply rotation to each pair of dimensions:
        # Split Q, K into two halves
        Q1, Q2 = Q[..., :half], Q[..., half:]
        K1, K2 = K[..., :half], K[..., half:]
        # Rotate
        Q_rotated = torch.cat([Q1 * cos - Q2 * sin, Q1 * sin + Q2 * cos], dim=-1)
        K_rotated = torch.cat([K1 * cos - K2 * sin, K1 * sin + K2 * cos], dim=-1)
        return Q_rotated, K_rotated


class JetBlockAttention(nn.Module):
    def __init__(
        self, hidden_size, n_heads, qk_dim, v_dim, kernel_size=4, use_rope=True
    ):
        super().__init__()
        self.n_heads, self.qk_dim, self.v_dim = n_heads, qk_dim, v_dim
        self.kernel_size = kernel_size

        # Q/K/V projections
        self.W_q = nn.Linear(hidden_size, n_heads * qk_dim, bias=True)
        self.W_k = nn.Linear(hidden_size, n_heads * qk_dim, bias=True)
        self.W_v = nn.Linear(hidden_size, n_heads * v_dim, bias=True)

        # kernel generator shares the SAME INPUT as Q/K/V (x), with reduction ratio 8 ---
        red = max(1, hidden_size // 8)  # reduction ratio 8 as per paper
        self.kernel_gen = nn.Sequential(
            nn.Linear(hidden_size, red, bias=True),
            nn.SiLU(),
            nn.Linear(red, n_heads * kernel_size, bias=True),
        )

        self.W_out = nn.Linear(n_heads * v_dim, hidden_size, bias=True)
        self.gate_linear = nn.Linear(
            qk_dim, 2, bias=True
        )  # gates from Q (data-dependent gating)
        self.rope = RotaryPositionEmbedding(qk_dim) if use_rope else None

    def forward(self, x):
        B, T, _ = x.shape
        # Standard projections
        Q = self.W_q(x).view(B, T, self.n_heads, self.qk_dim)
        K = self.W_k(x).view(B, T, self.n_heads, self.qk_dim)
        V = self.W_v(x).view(B, T, self.n_heads, self.v_dim)
        if self.rope is not None:
            Q, K = self.rope.apply_rotary(Q, K)

        # per-position, per-head dynamic kernels from block input x (shared with Q/K/V) ---
        # shape -> [B, T, n_heads, kernel_size]
        kernels = self.kernel_gen(x).view(B, T, self.n_heads, self.kernel_size)

        # causal depthwise dynamic conv over V (values only)
        out = torch.zeros(
            B, T, self.n_heads, self.v_dim, device=x.device, dtype=x.dtype
        )  # stores the per-time-step JetBlock output before the final linear projection.
        # Shape [B, T, nH, v_dim].
        ctx = torch.zeros(
            B, self.n_heads, self.v_dim, device=x.device, dtype=x.dtype
        )  # the recurrent state (JetBlock’s running “context”), one vector per head.
        # Shape [B, nH, v_dim]. It’s the linear-attention “memory” that gets updated at each token.
        pad = self.kernel_size - 1
        Vpad = (
            F.pad(V, (0, 0, 0, 0, pad, 0)) if pad > 0 else V
        )  # left-padded values for causal convolution. Original V is [B, T, nH, v_dim].
        # With K-1 zeros on the left along time, Vpad becomes [B, T+K-1, nH, v_dim].

        for t in range(T):
            # gather last kernel_size V's (causal)
            Vwin = Vpad[:, t : t + self.kernel_size].permute(
                0, 2, 1, 3
            )  # [B, nH, K, v_dim] From the query at time t (Q[:, t] → [B, nH, qk_dim])
            # we compute 2 gates per head and position → [B, nH, 2], then sigmoid to keep them in [0,1].
            w = kernels[:, t, :, :].unsqueeze(-1)  # [B, nH, K, 1]
            Vmix = (Vwin * w).sum(dim=2)  # [B, nH, v_dim]

            # Gated DeltaNet-style time mixing (the recurrent update)
            g = torch.sigmoid(self.gate_linear(Q[:, t]))  # [B, nH, 2]
            u, v = (
                g[..., 0:1],
                g[..., 1:2],
            )  # [B, nH, 1] u (forget/retain) gate: how much of previous ctx to keep.,
            # v (input/update) gate: how much of new mixed value Vmix to integrate.
            ctx = ctx * u + Vmix * v  # recurrent state update
            out[:, t] = ctx

            # This turns the attention accumulation into an RNN-like update:
            # no quadratic score matrix; we just carry a small per-head state forward.

        out = out.reshape(B, T, self.n_heads * self.v_dim)
        return self.W_out(out)