import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    def __init__(self, context_length, prediction_length, hidden_size, num_layers, dropout, num_quantiles=3):
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.num_quantiles = num_quantiles
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        nn.init.constant_(self.gate_mlp[2].bias, -1.0)

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            batch_first=True
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, prediction_length * num_quantiles),
        )

    def _encode_sequence(self, context):
        x = context.unsqueeze(-1)
        out, (h, _) = self.lstm(x)
        return out, h[-1]

    def forward(self, context, raf_context=None, raf_mask=None, **kwargs):
        local_out, local_h = self._encode_sequence(context)
        B = context.size(0)

        if raf_context is not None and raf_context.dim() == 3:
            K = raf_context.size(1)
            L_ret = raf_context.size(2)

            raf_folded = raf_context.view(B * K, L_ret)
            _, raf_h_folded = self._encode_sequence(raf_folded)
            raf_h = raf_h_folded.view(B, K, self.hidden_size)

            local_summary_expanded = local_h.unsqueeze(1).expand(-1, K, -1)
            gate_input = torch.cat([local_summary_expanded, raf_h], dim=-1)
            gate_score = self.gate_mlp(gate_input)

            filtered_raf_h = raf_h * gate_score

            attn_out, _ = self.attn(
                query=local_h.unsqueeze(1),
                key=filtered_raf_h,
                value=filtered_raf_h,
            )
        else:
            attn_out, _ = self.attn(
                query=local_h.unsqueeze(1),
                key=local_out,
                value=local_out,
            )

        combined_features = torch.cat([local_h, attn_out.squeeze(1)], dim=-1)
        out = self.head(combined_features)
        return out.view(-1, self.num_quantiles, self.prediction_length)
