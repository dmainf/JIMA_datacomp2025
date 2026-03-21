import torch
import torch.nn as nn
from typing import Optional
from chronos.chronos_bolt import ChronosBoltModelForForecasting, ChronosBoltOutput


class ChronosBoltFiDModel(ChronosBoltModelForForecasting):
    def __init__(self, config):
        super().__init__(config)
        d_model = config.d_model

        self.scale_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )

        self.gate_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

    def _init_weights(self, module):
        if module in [self.scale_mlp, self.gate_mlp]:
            return
        for sub in [self.scale_mlp, self.gate_mlp]:
            if module in list(sub.modules()):
                if isinstance(module, nn.Linear):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                    if module.bias is not None:
                        if sub is self.gate_mlp and module is self.gate_mlp[2]:
                            module.bias.data.fill_(-1.0)
                        else:
                            module.bias.data.zero_()
                return
        super()._init_weights(module)

    def forward(
        self,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        raf_context: Optional[torch.Tensor] = None,
        raf_mask: Optional[torch.Tensor] = None,
        raf_scale_ratio: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> ChronosBoltOutput:
        """
        Args:
            context: [B, L_ctx] - Local context
            mask: [B, L_ctx] - Local context mask
            target: [B, pred_len] - Target for training
            target_mask: [B, pred_len] - Target mask
            raf_context: [B, K, L_ret] - K retrieved waveforms
            raf_mask: [B, K, L_ret] - Mask for retrieved waveforms
            raf_scale_ratio: [B, K] - Scale ratio for each retrieved waveform
        """
        batch_size = context.size(0)

        local_hidden, loc_scale_raw, input_embeds, attention_mask = self.encode(
            context=context, mask=mask
        )

        loc, scale = loc_scale_raw
        scale = torch.clamp(scale, min=1.0)
        loc_scale = (loc, scale)

        hidden_states = local_hidden

        if raf_context is not None and raf_context.dim() == 3:
            has_valid = raf_mask.any(dim=-1).any(dim=-1) if raf_mask is not None else ~torch.isnan(raf_context).all(dim=-1).all(dim=-1)

            if has_valid.any():
                # Concatenate K streams along time axis into 1 (no Multi-Stream)
                raf_context_cat = raf_context.view(batch_size, K * L_ret)  # [B, K*L_ret]
                raf_mask_cat = raf_mask.view(batch_size, K * L_ret) if raf_mask is not None else None

                raf_hidden, _, _, raf_attn_mask = self.encode(
                    context=raf_context_cat, mask=raf_mask_cat
                )  # raf_hidden: [B, K*L_patches, D]

                if raf_scale_ratio is not None:
                    L_patches = raf_hidden.size(1) // K
                    scale_input = torch.log(raf_scale_ratio.clamp(min=1e-6)).unsqueeze(-1).to(dtype=raf_hidden.dtype, device=raf_hidden.device)  # [B, K, 1]
                    scale_emb = self.scale_mlp(scale_input)  # [B, K, D]
                    scale_emb_expanded = scale_emb.unsqueeze(2).expand(-1, -1, L_patches, -1).reshape(batch_size, K * L_patches, -1)
                    raf_hidden = raf_hidden * torch.sigmoid(scale_emb_expanded)

                attn_mask_expanded = attention_mask.unsqueeze(-1).to(local_hidden.dtype)
                local_summary = (local_hidden * attn_mask_expanded).sum(dim=1, keepdim=True) / (attn_mask_expanded.sum(dim=1, keepdim=True) + 1e-9)

                local_summary_expanded = local_summary.expand(-1, raf_hidden.size(1), -1)
                gate_input = torch.cat([local_summary_expanded, raf_hidden], dim=-1)
                gate_score = self.gate_mlp(gate_input)
                filtered_raf = raf_hidden * gate_score

                hidden_states = torch.cat([local_hidden, filtered_raf], dim=1)
                attention_mask = torch.cat([attention_mask, raf_attn_mask], dim=1)

        sequence_output = self.decode(input_embeds, attention_mask, hidden_states)

        quantile_preds_shape = (
            batch_size,
            self.num_quantiles,
            self.chronos_config.prediction_length,
        )
        quantile_preds = self.output_patch_embedding(sequence_output).view(
            *quantile_preds_shape
        )

        loss = None
        if target is not None:
            target, _ = self.instance_norm(target, loc_scale)
            target = target.unsqueeze(1)

            target = target.to(quantile_preds.device)
            target_mask = (
                target_mask.unsqueeze(1).to(quantile_preds.device)
                if target_mask is not None
                else ~torch.isnan(target)
            )
            target[~target_mask] = 0.0

            if self.chronos_config.prediction_length > target.shape[-1]:
                padding_shape = (
                    *target.shape[:-1],
                    self.chronos_config.prediction_length - target.shape[-1],
                )
                target = torch.cat(
                    [target, torch.zeros(padding_shape).to(target)], dim=-1
                )
                target_mask = torch.cat(
                    [target_mask, torch.zeros(padding_shape).to(target_mask)], dim=-1
                )

            loss = (
                2
                * torch.abs(
                    (target - quantile_preds)
                    * (
                        (target <= quantile_preds).float()
                        - self.quantiles.view(1, self.num_quantiles, 1)
                    )
                )
                * target_mask.float()
            )
            loss = loss.mean(dim=-2)
            loss = loss.mean(dim=-1)
            loss = loss.mean()

        quantile_preds = self.instance_norm.inverse(
            quantile_preds.view(batch_size, -1), loc_scale
        ).view(*quantile_preds_shape)

        return ChronosBoltOutput(loss=loss, quantile_preds=quantile_preds)
