import torch
import torch.nn as nn
from typing import Optional
from chronos.chronos_bolt import ChronosBoltModelForForecasting, ChronosBoltOutput


class ChronosBoltFiDModel(ChronosBoltModelForForecasting):
    def __init__(self, config):
        super().__init__(config)

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
        Context Extension版 RAF実装:
        検索データ(raf_context)を直近データ(context)の前に結合し、
        一つの長い文脈としてエンコードする。

        Args:
            context: [B, L_ctx]
            raf_context: [B, K, L_ret]
        """
        if raf_context is not None:
            batch_size = context.shape[0]

            raf_context_flat = raf_context.view(batch_size, -1)

            if raf_mask is not None:
                raf_mask_flat = raf_mask.view(batch_size, -1)
            else:
                raf_mask_flat = torch.ones_like(raf_context_flat, dtype=torch.bool)

            context = torch.cat([raf_context_flat, context], dim=1)

            if mask is not None:
                mask = torch.cat([raf_mask_flat, mask], dim=1)
            else:
                ctx_mask = torch.ones(
                    batch_size, context.shape[1] - raf_context_flat.shape[1],
                    dtype=torch.bool, device=context.device
                )
                mask = torch.cat([raf_mask_flat, ctx_mask], dim=1)

        hidden_states, loc_scale, input_embeds, attention_mask = self.encode(
            context=context, mask=mask
        )

        sequence_output = self.decode(input_embeds, attention_mask, hidden_states)

        quantile_preds_shape = (
            context.size(0),
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
            loss = loss.mean()

        quantile_preds = self.instance_norm.inverse(
            quantile_preds.view(context.size(0), -1), loc_scale
        ).view(*quantile_preds_shape)

        return ChronosBoltOutput(loss=loss, quantile_preds=quantile_preds)
