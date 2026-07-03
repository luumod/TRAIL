import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inverse_softplus(x: float) -> torch.Tensor:
    x = float(max(x, 1e-8))
    return torch.tensor(math.log(math.expm1(x)), dtype=torch.float32)


class CounterfactualTrajectoryCausalReview(nn.Module):
    """
    Trajectory-aware evidence-guided logit refinement (TELR).

    This module implements the method formulas:
    - Eq. (36): log-sum-exp evidence aggregation.
    - Eq. (37): current D/P -> medication positive and negative evidence.
    - Eq. (38): counterfactual-route-weighted historical evidence.
    - Eq. (39): retrieved global evidence.
    - Eq. (40)-(42): bounded logit refinement Delta.
    """

    def __init__(
        self,
        causal_graph,
        num_med,
        temperature=0.1,
        max_delta=0.5,
        lambda_med=0.5,
        init_w_cur=1.0,
        init_w_hist=1.0,
        init_w_global=1.0,
        history_topk=3,
        device="cuda:0",
        tau_d=None,
        tau_q=None,
        eps_c=1e-8,
    ):
        super().__init__()

        self.num_med = int(num_med)
        self.device = torch.device(device)
        self.tau_agg = float(max(temperature, 1e-6))
        self.tau_d = float(max(temperature if tau_d is None else tau_d, 1e-6))
        self.tau_q = float(max(temperature if tau_q is None else tau_q, 1e-6))
        self.eps_c = float(max(eps_c, 1e-12))
        self.lambda_med = float(min(max(lambda_med, 0.0), 1.0))
        self.history_topk = int(history_topk)

        dm_effect = torch.tensor(
            causal_graph.dm_effect.values,
            dtype=torch.float32,
            device=self.device,
        )
        pm_effect = torch.tensor(
            causal_graph.pm_effect.values,
            dtype=torch.float32,
            device=self.device,
        )

        self.register_buffer("dm_effect", self._clean_effect(dm_effect))
        self.register_buffer("pm_effect", self._clean_effect(pm_effect))
        self.register_buffer("dm_effect_pos", torch.clamp(self.dm_effect, min=0.0))
        self.register_buffer("dm_effect_neg", torch.clamp(-self.dm_effect, min=0.0))
        self.register_buffer("pm_effect_pos", torch.clamp(self.pm_effect, min=0.0))
        self.register_buffer("pm_effect_neg", torch.clamp(-self.pm_effect, min=0.0))

        init_weights = torch.tensor(
            [
                max(float(init_w_cur), 1e-8),
                max(float(init_w_hist), 1e-8),
                max(float(init_w_global), 1e-8),
            ],
            dtype=torch.float32,
        )
        self.evidence_weight_logits = nn.Parameter(torch.log(init_weights))
        self.raw_max_delta = nn.Parameter(_inverse_softplus(max_delta))

    @property
    def evidence_weights(self):
        return F.softmax(self.evidence_weight_logits, dim=0)

    @property
    def w_cur(self):
        return self.evidence_weights[0]

    @property
    def w_hist(self):
        return self.evidence_weights[1]

    @property
    def w_global(self):
        return self.evidence_weights[2]

    @property
    def max_delta(self):
        return F.softplus(self.raw_max_delta)

    def _clean_effect(self, effect: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(effect, nan=0.0, posinf=0.0, neginf=0.0)

    def _zero_evidence(self, reference: torch.Tensor | None = None) -> torch.Tensor:
        if reference is None:
            reference = self.dm_effect
        return torch.zeros(
            self.num_med,
            dtype=reference.dtype,
            device=reference.device,
        )

    def _to_long_tensor(self, ids, max_size: int, device: torch.device) -> torch.Tensor:
        if ids is None or len(ids) == 0:
            return torch.empty(0, dtype=torch.long, device=device)
        idx = torch.as_tensor(ids, dtype=torch.long, device=device).reshape(-1)
        return idx[(idx >= 0) & (idx < max_size)]

    def _agg(self, values: torch.Tensor) -> torch.Tensor:
        """Eq. (36): tau * log(mean(exp(q / tau))); empty sets return zero."""
        if values.numel() == 0:
            return self._zero_evidence(values)
        return self.tau_agg * (
            torch.logsumexp(values / self.tau_agg, dim=0)
            - math.log(float(values.size(0)))
        )

    def _visit_entity_evidence(self, diags, procs, positive=True):
        """Aggregate D->M and P->M evidence for one visit."""
        evidence_list = []
        device = self.dm_effect.device

        dm_matrix = self.dm_effect_pos if positive else self.dm_effect_neg
        pm_matrix = self.pm_effect_pos if positive else self.pm_effect_neg

        d_idx = self._to_long_tensor(diags, dm_matrix.size(0), device)
        if d_idx.numel() > 0:
            evidence_list.append(dm_matrix[d_idx])

        p_idx = self._to_long_tensor(procs, pm_matrix.size(0), device)
        if p_idx.numel() > 0:
            evidence_list.append(pm_matrix[p_idx])

        if len(evidence_list) == 0:
            return self._zero_evidence()

        return self._agg(torch.cat(evidence_list, dim=0))

    def _current_evidence(self, diags, procs):
        return (
            self._visit_entity_evidence(diags, procs, positive=True),
            self._visit_entity_evidence(diags, procs, positive=False),
        )

    def _prepare_history(
        self,
        history_diags,
        history_procs,
        route_weights,
        history_indices=None,
    ):
        if history_diags is None or len(history_diags) == 0:
            return [], [], self.dm_effect.new_empty(0), [], []

        hist_len = len(history_diags)
        history_diags = list(history_diags)
        if history_procs is None:
            history_procs = [[] for _ in range(hist_len)]
        else:
            history_procs = list(history_procs[:hist_len])
            if len(history_procs) < hist_len:
                history_procs += [[] for _ in range(hist_len - len(history_procs))]

        if history_indices is None:
            history_indices = list(range(hist_len))
        else:
            history_indices = [int(i) for i in list(history_indices)[:hist_len]]
            if len(history_indices) < hist_len:
                history_indices += list(range(len(history_indices), hist_len))

        if route_weights is None:
            weights = self.dm_effect.new_ones(hist_len)
        else:
            weights = route_weights.to(self.dm_effect.device).float().reshape(-1)[:hist_len]
            if weights.numel() < hist_len:
                weights = torch.cat(
                    [weights, weights.new_zeros(hist_len - weights.numel())],
                    dim=0,
                )
        weights = torch.clamp(weights, min=0.0)

        effective_k = hist_len if self.history_topk <= 0 else min(self.history_topk, hist_len)
        local_positions = list(range(hist_len))
        if effective_k < hist_len:
            if float(weights.sum().item()) <= 1e-8:
                chosen = list(range(hist_len - effective_k, hist_len))
            else:
                chosen = torch.topk(weights, k=effective_k, largest=True).indices.tolist()
                chosen = sorted(int(i) for i in chosen)
            local_positions = chosen
            history_diags = [history_diags[i] for i in chosen]
            history_procs = [history_procs[i] for i in chosen]
            history_indices = [history_indices[i] for i in chosen]
            weights = weights[torch.as_tensor(chosen, device=weights.device)]

        if weights.numel() > 0:
            weight_sum = weights.sum()
            if weight_sum <= 1e-8:
                weights = torch.ones_like(weights) / weights.numel()
            else:
                weights = weights / weight_sum

        return history_diags, history_procs, weights, history_indices, local_positions

    def _align_ce(self, ce_tensor, hist_len, device, dtype):
        if ce_tensor is None:
            return torch.zeros(hist_len, self.num_med, device=device, dtype=dtype)
        ce = ce_tensor.to(device=device, dtype=dtype)
        if ce.dim() == 1:
            ce = ce.unsqueeze(0)
        ce = ce[:hist_len, :self.num_med]
        if ce.size(0) < hist_len:
            pad = torch.zeros(hist_len - ce.size(0), self.num_med, device=device, dtype=dtype)
            ce = torch.cat([ce, pad], dim=0)
        if ce.size(1) < self.num_med:
            pad = torch.zeros(ce.size(0), self.num_med - ce.size(1), device=device, dtype=dtype)
            ce = torch.cat([ce, pad], dim=1)
        return ce

    def _historical_evidence(
        self,
        history_diags,
        history_procs,
        route_weights,
        ce_tensor=None,
        history_indices=None,
    ):
        history_diags, history_procs, weights, used_indices, local_positions = self._prepare_history(
            history_diags,
            history_procs,
            route_weights,
            history_indices,
        )
        hist_len = len(history_diags)
        if hist_len == 0:
            return self._zero_evidence(), self._zero_evidence(), used_indices, weights

        ce_source_len = max(local_positions) + 1 if local_positions else hist_len
        ce = self._align_ce(ce_tensor, ce_source_len, self.dm_effect.device, self.dm_effect.dtype)
        ce = ce[torch.as_tensor(local_positions, device=ce.device, dtype=torch.long)]
        ce_pos = torch.clamp(ce, min=0.0)
        ce_neg = torch.clamp(-ce, min=0.0)

        hist_pos = self._zero_evidence()
        hist_neg = self._zero_evidence()
        for k, (d_list, p_list) in enumerate(zip(history_diags, history_procs)):
            visit_pos = self._visit_entity_evidence(d_list, p_list, positive=True)
            visit_neg = self._visit_entity_evidence(d_list, p_list, positive=False)
            hist_pos = hist_pos + weights[k] * ce_pos[k] * visit_pos
            hist_neg = hist_neg + weights[k] * ce_neg[k] * visit_neg

        return hist_pos, hist_neg, used_indices, weights

    def _normalize_global_weights(self, global_weights, K: int) -> torch.Tensor:
        if K <= 0:
            return self.dm_effect.new_empty(0)
        if global_weights is None:
            return torch.ones(K, device=self.dm_effect.device) / K

        weights = global_weights.to(self.dm_effect.device).float().reshape(-1)[:K]
        if weights.numel() < K:
            pad = torch.zeros(K - weights.numel(), device=self.dm_effect.device)
            weights = torch.cat([weights, pad], dim=0)

        weights = torch.clamp(weights, min=0.0)
        weight_sum = weights.sum()
        if weight_sum <= 1e-8:
            return torch.ones(K, device=self.dm_effect.device) / K
        return weights / weight_sum

    def _global_evidence(
        self,
        global_diags=None,
        global_procs=None,
        global_meds=None,
        global_weights=None,
    ):
        if global_diags is None or len(global_diags) == 0:
            return self._zero_evidence(), self._zero_evidence()

        K = len(global_diags)
        if global_procs is None:
            global_procs = [[] for _ in range(K)]
        if global_meds is None:
            global_meds = [[] for _ in range(K)]

        weights = self._normalize_global_weights(global_weights, K)
        global_pos = self._zero_evidence()
        global_neg = self._zero_evidence()

        for j in range(K):
            entity_pos = self._visit_entity_evidence(global_diags[j], global_procs[j], positive=True)
            entity_neg = self._visit_entity_evidence(global_diags[j], global_procs[j], positive=False)

            med_exp = self._zero_evidence()
            med_idx = self._to_long_tensor(global_meds[j], self.num_med, self.dm_effect.device)
            if med_idx.numel() > 0:
                med_exp[med_idx] = 1.0

            global_pos = global_pos + weights[j] * (
                self.lambda_med * med_exp + (1.0 - self.lambda_med) * entity_pos
            )
            global_neg = global_neg + weights[j] * (1.0 - self.lambda_med) * entity_neg

        return global_pos, global_neg

    def forward(
        self,
        logits,
        cur_diags,
        cur_procs,
        history_diags=None,
        history_procs=None,
        route_weights=None,
        history_indices=None,
        a_hist=None,
        global_diags=None,
        global_procs=None,
        global_meds=None,
        global_weights=None,
        ce_tensor=None,
        calibration_progress=1.0,
    ):
        if a_hist is None:
            a_hist = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        elif not torch.is_tensor(a_hist):
            a_hist = torch.tensor(float(a_hist), device=logits.device, dtype=logits.dtype)
        else:
            a_hist = a_hist.to(device=logits.device, dtype=logits.dtype)
        a_hist = torch.clamp(a_hist.reshape(1), min=0.0, max=1.0)

        s_cur_pos, s_cur_neg = self._current_evidence(cur_diags, cur_procs)
        s_hist_pos, s_hist_neg, used_history_indices, used_route_weights = self._historical_evidence(
            history_diags,
            history_procs,
            route_weights,
            ce_tensor=ce_tensor,
            history_indices=history_indices,
        )
        s_global_pos, s_global_neg = self._global_evidence(
            global_diags,
            global_procs,
            global_meds,
            global_weights,
        )

        s_cur_pos = s_cur_pos.to(logits.device, logits.dtype)
        s_cur_neg = s_cur_neg.to(logits.device, logits.dtype)
        s_hist_pos = s_hist_pos.to(logits.device, logits.dtype)
        s_hist_neg = s_hist_neg.to(logits.device, logits.dtype)
        s_global_pos = s_global_pos.to(logits.device, logits.dtype)
        s_global_neg = s_global_neg.to(logits.device, logits.dtype)

        weights = self.evidence_weights.to(logits.device, logits.dtype)
        w_cur, w_hist, w_global = weights[0], weights[1], weights[2]

        S_pos = (
            w_cur * s_cur_pos
            + w_hist * a_hist * s_hist_pos
            + w_global * (1.0 - a_hist) * s_global_pos
        )
        S_neg = (
            w_cur * s_cur_neg
            + w_hist * a_hist * s_hist_neg
            + w_global * (1.0 - a_hist) * s_global_neg
        )

        diff = S_pos - S_neg
        evidence_sum = S_pos + S_neg
        direction = torch.tanh(diff / self.tau_d)
        strength = (
            1.0 - torch.exp(-evidence_sum / self.tau_q)
        ) * diff.abs() / (evidence_sum + self.eps_c)
        strength = torch.nan_to_num(strength, nan=0.0, posinf=1.0, neginf=0.0)
        strength = torch.clamp(strength, min=0.0, max=1.0)

        if torch.is_tensor(calibration_progress):
            warmup = calibration_progress.to(logits.device, logits.dtype)
        else:
            warmup = torch.tensor(float(calibration_progress), device=logits.device, dtype=logits.dtype)
        warmup = torch.clamp(warmup.reshape(1), min=0.0, max=1.0)

        delta = self.max_delta.to(logits.device, logits.dtype) * warmup * direction * strength
        reviewed_logits = logits + delta.unsqueeze(0)

        aux_info = {
            "s_cur_pos": s_cur_pos.detach(),
            "s_cur_neg": s_cur_neg.detach(),
            "s_hist_pos": s_hist_pos.detach(),
            "s_hist_neg": s_hist_neg.detach(),
            "s_global_pos": s_global_pos.detach(),
            "s_global_neg": s_global_neg.detach(),
            "S_pos": S_pos.detach(),
            "S_neg": S_neg.detach(),
            "delta": delta.detach(),
            "direction": direction.detach(),
            "strength": strength.detach(),
            "g": strength.detach(),
            "w_cur": w_cur.detach(),
            "w_hist": w_hist.detach(),
            "w_global": w_global.detach(),
            "max_delta": self.max_delta.detach(),
            "calibration_progress": warmup.detach(),
            "used_history_indices": used_history_indices,
            "used_route_weights": used_route_weights.detach(),
        }

        return reviewed_logits, aux_info
