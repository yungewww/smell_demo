import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Positional encoding ----------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)


# ---------- Transformer ----------
class Transformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        num_classes: int,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_positional_encoding: bool = True,
        use_cls_token: bool = False,
        pool: str = "mean",  # 'mean' or 'cls'
    ):
        super().__init__()
        assert pool in ("mean", "cls"), "pool must be 'mean' or 'cls'"

        # Project input features to model dimension
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.LayerNorm(model_dim),
        )

        self.pos = (
            SinusoidalPositionalEncoding(model_dim)
            if use_positional_encoding
            else nn.Identity()
        )
        self.use_cls_token = use_cls_token
        self.pool = pool

        enc_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=4 * model_dim,
            dropout=dropout,
            batch_first=True,
            activation=activation,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
            nn.init.normal_(self.cls_token, std=0.02)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, num_classes),
        )

    def _key_padding_mask(self, lengths: torch.Tensor, T: int) -> torch.Tensor:
        # True = PAD position
        device = lengths.device
        rng = torch.arange(T, device=device).unsqueeze(0)
        return rng >= lengths.unsqueeze(1)

    def forward_features(
        self, x: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        x: (B, T, F), lengths: (B,) actual lengths (optional)
        returns: (B, D)
        """
        x = self.input_proj(x)
        x = self.pos(x)

        key_padding_mask = None
        if lengths is not None:
            key_padding_mask = self._key_padding_mask(lengths, x.size(1))

        if self.use_cls_token:
            cls = self.cls_token.expand(x.size(0), -1, -1)
            x = torch.cat([cls, x], dim=1)
            if key_padding_mask is not None:
                pad0 = torch.zeros(
                    (key_padding_mask.size(0), 1), dtype=torch.bool, device=x.device
                )
                key_padding_mask = torch.cat([pad0, key_padding_mask], dim=1)

        h = self.transformer(x, src_key_padding_mask=key_padding_mask)

        if self.use_cls_token and self.pool == "cls":
            feat = h[:, 0]
        else:
            tokens = h[:, 1:] if self.use_cls_token else h
            if key_padding_mask is None:
                feat = tokens.mean(dim=1)
            else:
                mask = (~key_padding_mask).float()
                if self.use_cls_token:
                    mask = mask[:, 1:]
                denom = mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
                feat = (tokens * mask.unsqueeze(-1)).sum(dim=1) / denom

        return feat

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        feat = self.forward_features(x, lengths)
        feat = self.dropout(feat)
        return self.classifier(feat)


if __name__ == "__main__":
    model = Transformer(
        input_dim=12,  # 每个时间点的特征数（你的传感器列数）
        model_dim=128,
        num_classes=3,  # 类别数量（例如 chienan, rose, orange）
        num_heads=4,
        num_layers=3,
    )

    x = torch.randn(8, 100, 12)  # batch=8, window=100, channels=12
    out = model(x)
    print(out.shape)  # torch.Size([8, 3])
