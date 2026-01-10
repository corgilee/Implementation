import torch
import torch.nn as nn
import torch.nn.functional as F


class SessionEncoder(nn.Module):
    """
    Lightweight Transformer-based session encoder for next-video recommendation.

    Inputs it typically consumes:
      - Session sequence embeddings:  [B, N, D]
      - Current video embedding:      [B, D]  (used as query / anchor)
      - Optional per-step session signals (e.g., dwell bucket, autoplay flag): [B, N, S]
      - Optional global context:      [B, C]  (device/time/network), injected as bias

    Output:
      - Session intent vector h_s:    [B, D]
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 64,
        step_signal_dim: int = 0,   # e.g., dwell/autoplay signals per history item
        context_dim: int = 0,       # e.g., device/time/network
        use_cls_token: bool = False # alternative to query-attention pooling
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.use_cls_token = use_cls_token

        # Optional projection for per-step signals appended to each history item
        if step_signal_dim > 0:
            self.step_proj = nn.Linear(d_model + step_signal_dim, d_model)
        else:
            self.step_proj = None

        # Positional embedding for short sessions (N typically 3-10, but allow larger)
        self.pos_emb = nn.Embedding(max_seq_len + (1 if use_cls_token else 0), d_model)

        # Optional CLS token (if you want transformer to produce a pooled representation)
        if use_cls_token:
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls, std=0.02)

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Candidate-conditioned pooling (DIN-style): current video as query, attends to history
        self.attn_pool = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Optional context -> bias added to pooled representation
        if context_dim > 0:
            self.ctx_proj = nn.Sequential(
                nn.Linear(context_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
        else:
            self.ctx_proj = None

        self.out_norm = nn.LayerNorm(d_model)
        self.out_drop = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(
        self,
        hist_emb: torch.Tensor,                 # [B, N, D]
        current_emb: torch.Tensor,              # [B, D]
        hist_mask: torch.Tensor | None = None,  # [B, N] True=valid, False=pad
        step_signals: torch.Tensor | None = None,  # [B, N, S]
        context: torch.Tensor | None = None,        # [B, C]
    ) -> torch.Tensor:
        """
        Returns:
          h_s: [B, D]
        """
        B, N, D = hist_emb.shape
        assert D == self.d_model, f"hist_emb last dim {D} must equal d_model {self.d_model}"
        assert current_emb.shape == (B, D)

        # Build sequence (optionally prepend CLS)
        x = hist_emb  # [B, N, D]

        # Optional per-step signals: concat then project back to d_model
        if self.step_proj is not None:
            if step_signals is None:
                raise ValueError("step_signals must be provided when step_signal_dim>0")
            x = torch.cat([x, step_signals], dim=-1)  # [B, N, D+S]
            x = self.step_proj(x)                     # [B, N, D]

        # Optional CLS token
        if self.use_cls_token:
            cls = self.cls.expand(B, -1, -1)          # [B, 1, D]
            x = torch.cat([cls, x], dim=1)            # [B, 1+N, D]

            # If we use CLS, extend mask accordingly (CLS is always valid)
            if hist_mask is not None:
                cls_mask = torch.ones(B, 1, device=hist_mask.device, dtype=hist_mask.dtype)
                hist_mask = torch.cat([cls_mask, hist_mask], dim=1)  # [B, 1+N]
            N = N + 1

        # Positional embedding
        pos_ids = torch.arange(N, device=x.device).unsqueeze(0).expand(B, N)  # [B, N]
        x = x + self.pos_emb(pos_ids)  # [B, N, D]

        # Transformer expects a padding mask: True means "ignore"
        # We'll use key_padding_mask=True for padded positions.
        if hist_mask is None:
            key_padding_mask = None
        else:
            # hist_mask True=valid -> padding_mask False for valid, True for pad
            key_padding_mask = ~hist_mask.bool()  # [B, N]

        # Encode sequence
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)  # [B, N, D]

        # Pooling to produce a single session intent vector h_s:
        if self.use_cls_token:
            # CLS representation (position 0)
            pooled = h[:, 0, :]  # [B, D]
        else:
            # Candidate-conditioned attention pooling (DIN-style):
            # Query = current_emb, Keys/Values = encoded history tokens
            q = current_emb.unsqueeze(1)  # [B, 1, D]
            # MultiheadAttention uses key_padding_mask with True=ignore
            pooled, _ = self.attn_pool(q, h, h, key_padding_mask=key_padding_mask)  # [B, 1, D]
            pooled = pooled.squeeze(1)  # [B, D]

        # Optional context injection (global conditioning)
        if self.ctx_proj is not None and context is not None:
            pooled = pooled + self.ctx_proj(context)  # [B, D]

        pooled = self.out_norm(pooled)
        pooled = self.out_drop(pooled)
        return pooled


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    B, N, D = 2, 6, 64
    S, C = 3, 8

    hist_emb = torch.randn(B, N, D)
    current_emb = torch.randn(B, D)

    # Mask: first sample has all valid, second sample has last 2 padded
    hist_mask = torch.tensor([[1, 1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 0, 0]], dtype=torch.bool)

    step_signals = torch.randn(B, N, S)   # e.g., [autoplay_flag, dwell_bucket, position_bucket]
    context = torch.randn(B, C)           # e.g., [device/time/network features]

    encoder = SessionEncoder(
        d_model=D,
        n_heads=4,
        n_layers=2,
        dropout=0.1,
        max_seq_len=64,
        step_signal_dim=S,
        context_dim=C,
        use_cls_token=False,  # try True as an alternative pooling method
    )

    h_s = encoder(
        hist_emb=hist_emb,
        current_emb=current_emb,
        hist_mask=hist_mask,
        step_signals=step_signals,
        context=context,
    )
    print("h_s shape:", h_s.shape)  # [B, D]
