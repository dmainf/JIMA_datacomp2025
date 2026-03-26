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
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, prediction_length * num_quantiles),
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
            raf_out_folded, _ = self._encode_sequence(raf_folded)
            raf_out = raf_out_folded.view(B, K * L_ret, self.hidden_size)

            local_summary_expanded = local_h.unsqueeze(1).expand(-1, K * L_ret, -1)
            gate_input = torch.cat([local_summary_expanded, raf_out], dim=-1)
            gate_score = self.gate_mlp(gate_input)

            filtered_raf = raf_out * gate_score
            combined_seq = torch.cat([local_out, filtered_raf], dim=1)

            if raf_mask is not None:
                raf_mask_flat = raf_mask.view(B, K * L_ret)
                local_mask = torch.ones(B, local_out.size(1), dtype=torch.bool, device=context.device)
                combined_mask = torch.cat([local_mask, raf_mask_flat], dim=1)
                key_padding_mask = ~combined_mask
            else:
                key_padding_mask = None
        else:
            combined_seq = local_out
            key_padding_mask = None

        query = local_h.unsqueeze(1)
        attn_out, _ = self.attn(
            query=query,
            key=combined_seq,
            value=combined_seq,
            key_padding_mask=key_padding_mask
        )

        out_features = attn_out.squeeze(1)
        out = self.head(out_features)
        return out.view(-1, self.num_quantiles, self.prediction_length)
