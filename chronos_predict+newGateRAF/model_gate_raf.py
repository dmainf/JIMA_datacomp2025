import torch
import torch.nn as nn
from typing import Optional
from chronos.chronos_bolt import ChronosBoltModelForForecasting, ChronosBoltOutput


class ChronosBoltFiDModel(ChronosBoltModelForForecasting):
    def __init__(self, config):
        super().__init__(config)
        d_model = config.d_model
        pred_len = self.chronos_config.prediction_length

        self.gate_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, pred_len),
        )

    def encode_prenormalized(self, context: torch.Tensor, mask: torch.Tensor = None):
        """instance_normをスキップして既に正規化済みの系列をエンコードする"""
        batch_size, _ = context.shape
        mask = mask.to(context.dtype) if mask is not None else (~torch.isnan(context)).to(context.dtype)
        context = context.to(self.dtype)
        mask = mask.to(self.dtype)

        patched_context = self.patch(context)
        patched_mask = torch.nan_to_num(self.patch(mask), nan=0.0)
        patched_context = torch.where(patched_mask > 0.0, patched_context, 0.0)
        patched_context = torch.cat([patched_context, patched_mask], dim=-1)

        attention_mask = patched_mask.sum(dim=-1) > 0
        input_embeds = self.input_patch_embedding(patched_context)

        if self.chronos_config.use_reg_token:
            reg_input_ids = torch.full(
                (batch_size, 1), self.config.reg_token_id, device=input_embeds.device
            )
            reg_embeds = self.shared(reg_input_ids)
            input_embeds = torch.cat([input_embeds, reg_embeds], dim=-2)
            attention_mask = torch.cat(
                [attention_mask.to(self.dtype), torch.ones_like(reg_input_ids).to(self.dtype)],
                dim=-1,
            )

        encoder_outputs = self.encoder(attention_mask=attention_mask, inputs_embeds=input_embeds)
        return encoder_outputs[0], attention_mask

    def _init_weights(self, module):
        if module in [self.gate_mlp]:
            return
        for sub in [self.gate_mlp]:
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
        raf_future: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> ChronosBoltOutput:
        """
        Args:
            context:     [B, L_ctx]       local context
            mask:        [B, L_ctx]       local context mask
            target:      [B, pred_len]    training target
            raf_context: [B, K, L_ret]    retrieved context (z-score normalized with retrieved ctx stats)
            raf_mask:    [B, K, L_ret]    retrieved context mask
            raf_future:  [B, K, pred_len] retrieved future (raw values)
        """
        batch_size = context.size(0)
        pred_len = self.chronos_config.prediction_length

        # Step 1: local contextをエンコード → loc_scaleを取得
        local_hidden, loc_scale_raw, input_embeds, attention_mask = self.encode(
            context=context, mask=mask
        )
        loc, scale = loc_scale_raw
        scale = torch.clamp(scale, min=1.0)
        loc_scale = (loc, scale)

        # Step 2: local contextのみでデコード → 正規化空間での予測
        sequence_output = self.decode(input_embeds, attention_mask, local_hidden)
        quantile_preds_shape = (batch_size, self.num_quantiles, pred_len)
        model_pred_norm = self.output_patch_embedding(sequence_output).view(*quantile_preds_shape)

        # Step 3: retrieved contextからgateを計算し，retrieved futureをブレンド
        quantile_preds = model_pred_norm
        if raf_context is not None and raf_context.dim() == 3 and raf_future is not None:
            K = raf_context.size(1)
            L_ret = raf_context.size(2)
            has_valid = raf_mask.any(dim=-1).any(dim=-1) if raf_mask is not None else \
                torch.ones(batch_size, dtype=torch.bool, device=context.device)

            if has_valid.any():
                # retrieved contextをエンコード（内部instance_normスキップ）
                raf_ctx_folded = raf_context.view(batch_size * K, L_ret)
                raf_mask_folded = raf_mask.view(batch_size * K, L_ret) if raf_mask is not None else None
                raf_hidden_folded, raf_attn_folded = self.encode_prenormalized(
                    raf_ctx_folded, raf_mask_folded
                )
                L_patches = raf_hidden_folded.size(1)
                D = raf_hidden_folded.size(2)
                raf_hidden = raf_hidden_folded.view(batch_size, K, L_patches, D)
                raf_attn = raf_attn_folded.view(batch_size, K, L_patches)

                # retrieved contextのsummary [B, K, D]
                raf_attn_exp = raf_attn.unsqueeze(-1).to(raf_hidden.dtype)
                raf_summary = (raf_hidden * raf_attn_exp).sum(dim=2) / (raf_attn_exp.sum(dim=2) + 1e-9)

                # local contextのsummary [B, D]
                attn_exp = attention_mask.unsqueeze(-1).to(local_hidden.dtype)
                local_summary = (local_hidden * attn_exp).sum(dim=1) / (attn_exp.sum(dim=1) + 1e-9)
                local_summary_exp = local_summary.unsqueeze(1).expand(-1, K, -1)  # [B, K, D]

                # gate: [B, K, pred_len]
                gate_input = torch.cat([local_summary_exp, raf_summary], dim=-1)  # [B, K, D*2]
                gate_logits = self.gate_mlp(gate_input)  # [B, K, pred_len]

                # K方向のsoftmax（どのretrieved itemを重視するか）
                weights = torch.softmax(gate_logits, dim=1)  # [B, K, pred_len]
                # 全体的な信頼度（retrievalをどれだけ使うか）
                alpha_global = torch.sigmoid(gate_logits.max(dim=1)[0])  # [B, pred_len]

                # raw raf_futureをloc_scaleで正規化
                raf_future_norm = (raf_future - loc.unsqueeze(1)) / scale.unsqueeze(1)  # [B, K, pred_len]
                raf_blend = (weights * raf_future_norm).sum(dim=1)  # [B, pred_len]

                # 正規化空間でブレンド
                alpha = alpha_global.unsqueeze(1)  # [B, 1, pred_len]
                quantile_preds = (1 - alpha) * model_pred_norm + alpha * raf_blend.unsqueeze(1)

        # Loss（正規化空間で計算）
        loss = None
        if target is not None:
            target_norm, _ = self.instance_norm(target, loc_scale)
            target_norm = target_norm.unsqueeze(1).to(quantile_preds.device)
            target_mask_t = (
                target_mask.unsqueeze(1).to(quantile_preds.device)
                if target_mask is not None
                else ~torch.isnan(target_norm)
            )
            target_norm[~target_mask_t] = 0.0

            if pred_len > target_norm.shape[-1]:
                pad_shape = (*target_norm.shape[:-1], pred_len - target_norm.shape[-1])
                target_norm = torch.cat([target_norm, torch.zeros(pad_shape).to(target_norm)], dim=-1)
                target_mask_t = torch.cat([target_mask_t, torch.zeros(pad_shape).to(target_mask_t)], dim=-1)

            loss = (
                2 * torch.abs(
                    (target_norm - quantile_preds)
                    * ((target_norm <= quantile_preds).float() - self.quantiles.view(1, self.num_quantiles, 1))
                )
                * target_mask_t.float()
            )
            loss = loss.mean(dim=-2).mean(dim=-1).mean()

        # 逆正規化（一度だけ）
        quantile_preds = self.instance_norm.inverse(
            quantile_preds.view(batch_size, -1), loc_scale
        ).view(*quantile_preds_shape)

        return ChronosBoltOutput(loss=loss, quantile_preds=quantile_preds)
