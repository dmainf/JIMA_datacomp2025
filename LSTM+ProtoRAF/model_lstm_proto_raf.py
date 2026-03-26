import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    def __init__(self, context_length, prediction_length, hidden_size, num_layers, dropout, num_quantiles=3):
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.num_quantiles = num_quantiles

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, prediction_length * num_quantiles),
        )

    def forward(self, context, raf_context=None, raf_mask=None, **kwargs):
        if raf_context is not None and raf_context.dim() == 3:
            B, K, L_ret = raf_context.shape
            raf_flat = raf_context.view(B, K * L_ret)
            extended = torch.cat([raf_flat, context], dim=1)
        else:
            extended = context

        x = extended.unsqueeze(-1)
        _, (h, _) = self.lstm(x)
        out = self.head(h[-1])
        return out.view(-1, self.num_quantiles, self.prediction_length)
