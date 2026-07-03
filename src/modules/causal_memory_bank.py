import torch
import torch.nn.functional as F
from tqdm import tqdm


class CausalVisitMemoryBank:
    """
    Counterfactual Causal Memory Bank.

    存储训练集中所有就诊的双空间 key:
    1. key_raw: 原始语义空间，用于冷启动和训练早期检索
    2. key_causal: 反事实因果轨迹空间，用于训练稳定后的因果响应检索

    检索时:
        sim = rho * cos(q_causal, key_causal) + (1 - rho) * cos(q_raw, key_raw)
    """

    def __init__(
        self,
        device,
        emb_dim,
        topk=10,
        momentum=0.7,
        exclude_same_patient=True
    ):
        self.device = device
        self.emb_dim = emb_dim
        self.topk = topk
        self.momentum = momentum
        self.exclude_same_patient = exclude_same_patient

        self.key_raw = None       # [N, 3 * emb_dim]
        self.key_causal = None    # [N, 3 * emb_dim]
        self.values = None        # [N, 3 * emb_dim]

        self.patient_ids = None   # [N]
        self.global_ids = None    # [N]

        self.diags = []
        self.procs = []
        self.meds = []
        self.a_hists = None       # [N, 1]
        self.selected_history_indices = []
        self.selected_history_counts = None

    @staticmethod
    def _normalize(x):
        return F.normalize(x, p=2, dim=-1)

    @torch.no_grad()
    def refresh(self, data_train, model, epoch=0, use_momentum=True):
        """
        用当前模型快照重新编码整个训练集，刷新 memory bank。

        注意：
        1. refresh 过程不参与梯度更新；
        2. 一个 epoch 内 memory bank 固定；
        3. epoch 结束后再 refresh。
        """
        was_training = model.training
        model.eval()

        new_key_raw = []
        new_key_causal = []
        new_values = []

        patient_ids = []
        global_ids = []

        diags, procs, meds = [], [], []
        a_hists = []
        selected_history_indices = []
        selected_history_counts = []

        global_idx = 0

        for p_id, patient in tqdm(enumerate(data_train), total=len(data_train), desc=f"Refreshing Causal Memory Bank @ epoch {epoch}"):
            seq_input = patient
            adm = patient[-1]

            enc_info = model.encode_patient_trajectory(seq_input)

            # [1, 3 * emb_dim]
            key_raw = enc_info["h_raw"].detach()
            key_causal = enc_info["h_causal"].detach()
            a_hist = enc_info["a_hist"].detach()

            # value 建议使用因果化后的 h_causal
            # 因为最终你要聚合的是“因果响应一致”的全局经验
            value = key_causal

            new_key_raw.append(key_raw.squeeze(0))
            new_key_causal.append(key_causal.squeeze(0))
            new_values.append(value.squeeze(0))

            patient_ids.append(p_id)
            global_ids.append(global_idx)

            diags.append(list(adm[0]))
            procs.append(list(adm[1]))
            meds.append(list(adm[2]))
            a_hists.append(a_hist.view(1))
            selected = list(enc_info.get("selected_history_indices", []))
            selected_history_indices.append(selected)
            selected_history_counts.append(len(selected))

            global_idx += 1

        new_key_raw = self._normalize(torch.stack(new_key_raw).to(self.device)) # [N, 3 * emb_dim]
        new_key_causal = self._normalize(torch.stack(new_key_causal).to(self.device)) # [N, 3 * emb_dim]
        new_values = torch.stack(new_values).to(self.device) # [N, 3 * emb_dim]
        new_a_hists = torch.stack(a_hists).to(self.device) # [N, 1]

        if (self.key_raw is not None
            and use_momentum
            and self.momentum > 0
            and self.key_raw.shape == new_key_raw.shape
        ):
            self.key_raw = self._normalize(
                self.momentum * self.key_raw + (1 - self.momentum) * new_key_raw
            )
            self.key_causal = self._normalize(
                self.momentum * self.key_causal + (1 - self.momentum) * new_key_causal
            )
            self.values = (
                self.momentum * self.values + (1 - self.momentum) * new_values
            )
        else:
            self.key_raw = new_key_raw
            self.key_causal = new_key_causal
            self.values = new_values

        self.patient_ids = torch.LongTensor(patient_ids).to(self.device)
        self.global_ids = torch.LongTensor(global_ids).to(self.device)

        self.diags = diags
        self.procs = procs
        self.meds = meds
        self.a_hists = new_a_hists
        self.selected_history_indices = selected_history_indices
        self.selected_history_counts = torch.as_tensor(
            selected_history_counts, dtype=torch.long, device=self.device
        )

        model.train(was_training)

    @torch.no_grad()
    def search(
        self,
        q_raw,
        q_causal,
        rho,
        query_patient_id=None,
        query_global_id=None,
        topk=None
    ):
        """
        q_raw: [1, 3 * emb_dim]
        q_causal: [1, 3 * emb_dim]
        rho: float or scalar tensor, 表示 causal similarity 的占比
        """
        if topk is None:
            topk = self.topk

        if self.key_raw is None or self.key_causal is None:
            return None

        q_raw = self._normalize(q_raw.detach())
        q_causal = self._normalize(q_causal.detach())

        sim_raw = torch.matmul(self.key_raw, q_raw.squeeze(0))          # [N]
        sim_causal = torch.matmul(self.key_causal, q_causal.squeeze(0)) # [N]

        if torch.is_tensor(rho):
            rho = float(rho.detach().cpu().item())

        rho = max(0.0, min(1.0, rho))

        sim = rho * sim_causal + (1.0 - rho) * sim_raw # equ

        valid_mask = torch.ones_like(sim, dtype=torch.bool)

        # 训练阶段必须排除同一个患者，避免检索到自己或自己的未来就诊
        if self.exclude_same_patient and query_patient_id is not None:
            valid_mask = valid_mask & (self.patient_ids != int(query_patient_id))

        # 保险起见，也排除当前 global_id
        if query_global_id is not None:
            valid_mask = valid_mask & (self.global_ids != int(query_global_id))

        valid_count = int(valid_mask.sum().item())

        if valid_count == 0:
            return None

        sim = sim.masked_fill(~valid_mask, -1e9)

        real_topk = min(topk, valid_count)
        top_scores, top_indices = torch.topk(sim, k=real_topk, dim=0)

        weights = F.softmax(top_scores, dim=0)  # [K]

        top_indices_list = top_indices.detach().cpu().tolist()

        return {
            "indices": top_indices,
            "scores": top_scores,
            "weights": weights,
            "values": self.values[top_indices],        # [K, 3 * emb_dim]
            "diags": [self.diags[i] for i in top_indices_list],
            "procs": [self.procs[i] for i in top_indices_list],
            "meds": [self.meds[i] for i in top_indices_list],
            "patient_ids": self.patient_ids[top_indices],
            "a_hists": self.a_hists[top_indices],
            "selected_history_indices": [
                self.selected_history_indices[i] for i in top_indices_list
            ],
            "selected_history_counts": self.selected_history_counts[top_indices],
        }