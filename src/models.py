import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from contextlib import contextmanager
from modules.homo_relation_graph import homo_relation_graph
from modules.causal_trajectory_review import CounterfactualTrajectoryCausalReview
from modules.layers import *


class BasicModel(nn.Module):
    def __init__(
            self,
            vocab_size,
            ddi_adj,
            emb_dim=64,
            device=torch.device("cuda:0"),
    ):
        super(BasicModel, self).__init__()

        self.device = device
        self.emb_dim = emb_dim

        # pre-embedding
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i], emb_dim) for i in range(3)]
        )
        self.emb_fuse_weight = nn.Embedding(3, 1)
        self.dropout = nn.Dropout(p=0.5)
        self.query = nn.Linear(3 * emb_dim, vocab_size[2])

        self.init_weights()

    def forward(self, patient):
        def sum_embedding(embedding):
            return embedding.sum(dim=1).unsqueeze(dim=0)  # (1,1,dim)

        adm = patient[-1]
        i1 = sum_embedding(
            self.dropout(
                self.embeddings[0](
                    torch.LongTensor(adm[0]).unsqueeze(dim=0).to(self.device)
                )
            )
        )  # (1,1,dim)
        i2 = sum_embedding(
            self.dropout(
                self.embeddings[1](
                    torch.LongTensor(adm[1]).unsqueeze(dim=0).to(self.device)
                )
            )
        )

        # Med
        if adm == patient[0]:
            i3 = torch.zeros(1, 1, self.emb_dim).to(self.device)
        else:
            adm_last = patient[-2]
            i3 = sum_embedding(self.dropout(self.embeddings[2](torch.LongTensor(adm_last[2]).unsqueeze(dim=0).to(self.device))))

        emb_fuse_weight = self.emb_fuse_weight(torch.tensor([0, 1, 2]).to(self.device))
        patient_representations = torch.cat([i1 * emb_fuse_weight[0], i2 * emb_fuse_weight[1], i3 * emb_fuse_weight[2]],dim=-1).squeeze(0)
        result = self.query(patient_representations)  # (1, dim)

        return result

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)

class EnhancedFeatureFusion(nn.Module):
    """
    Query-aware multi-view drug feature fusion.

    Inputs:
        query:    [1, emb_dim], patient representation
        base_emb: [num_med, emb_dim], molecular/base drug embedding
        ehr_emb:  [num_med, emb_dim], EHR graph-enhanced drug embedding
        ddi_emb:  [num_med, emb_dim], DDI graph-enhanced risk embedding

    Output:
        enhanced_emb:    [num_med, emb_dim]
        view_importance: [3], average importance of [base, EHR, -DDI]
    """
    def __init__(self, emb_dim, num_views=3, dropout=0.1):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_views = num_views

        self.cross_view_proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.view_query = nn.Linear(emb_dim, emb_dim)
        self.view_key = nn.Linear(emb_dim, emb_dim)
        self.view_value = nn.Linear(emb_dim, emb_dim)

        self.fusion_norm = nn.LayerNorm(emb_dim)
        self.fusion_proj = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(emb_dim)

    def forward(self, query, base_emb, ehr_emb, ddi_emb):
        # [num_med, 3, emb_dim]. The third view is negative DDI evidence.
        view_features = torch.stack([base_emb, ehr_emb, -ddi_emb], dim=1)
        if view_features.size(1) != self.num_views:
            raise ValueError(
                f"EnhancedFeatureFusion expected {self.num_views} views, "
                f"but got {view_features.size(1)}."
            )

        view_features = view_features + self.dropout(self.cross_view_proj(view_features))

        # Query-aware view attention for each drug.
        q = self.view_query(query)                         # [1, emb_dim]
        k = self.view_key(view_features)                   # [num_med, 3, emb_dim]
        v = self.view_value(view_features)                 # [num_med, 3, emb_dim]

        attn_scores = torch.einsum('bd,mvd->bmv', q, k) / math.sqrt(self.emb_dim)  # [1, num_med, 3]
        attn_weights = torch.softmax(attn_scores, dim=-1)
        view_importance = attn_weights.mean(dim=(0, 1))    # [3]

        fused_features = torch.einsum('bmv,mvd->bmd', attn_weights, v).squeeze(0)  # [num_med, emb_dim]
        fused_features = self.fusion_norm(
            fused_features + self.dropout(self.fusion_proj(fused_features))
        )

        # Keep the user's query-aware drug reweighting idea, but apply it as a gate
        # over drug embeddings. This preserves the semantics of drug-wise relevance.
        drug_attn = torch.softmax(
            torch.matmul(fused_features, query.t()) / math.sqrt(self.emb_dim),
            dim=0
        )  # [num_med, 1]
        enhanced_emb = drug_attn * fused_features

        return self.output_norm(enhanced_emb), view_importance


class GCN(nn.Module):
    """GCN module used by the original GAMENet drug memory."""
    def __init__(self, voc_size, emb_dim, adj, device=torch.device('cpu:0')):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device

        adj = self.normalize(np.asarray(adj, dtype=np.float32) + np.eye(adj.shape[0], dtype=np.float32))
        self.register_buffer("adj", torch.FloatTensor(adj).to(device))
        self.register_buffer("x", torch.eye(voc_size).to(device))

        self.gcn1 = GraphConvolution(voc_size, emb_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.gcn2 = GraphConvolution(emb_dim, emb_dim)

    def forward(self):
        node_embedding = self.gcn1(self.x, self.adj)
        node_embedding = F.relu(node_embedding)
        node_embedding = self.dropout(node_embedding)
        node_embedding = self.gcn2(node_embedding, self.adj)
        return node_embedding

    def normalize(self, mx):
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.0
        r_mat_inv = np.diagflat(r_inv)
        return r_mat_inv.dot(mx)


class GAMENet(nn.Module):
    """
    Original GAMENet backbone with a visit_mask extension used only for
    inexpensive approximate history screening. Exact Top-K interventions are
    performed by physically deleting admissions before calling this backbone.

    - input: List[admission], each admission is [diag_ids, proc_ids, med_ids, optional_graph_id]
    - visit_mask: Bool tensor/list with shape [seq] or [1, seq]. False means the visit is zeroed.
    - output: logits with shape [1, med_vocab_size]. In training mode this keeps the original
      GAMENet behavior and returns (logits, ddi_penalty).
    """
    def __init__(self, vocab_size, ehr_adj, ddi_adj, emb_dim=64,
                 device=torch.device('cpu:0'), ddi_in_memory=True):
        super(GAMENet, self).__init__()
        K = len(vocab_size)
        self.K = K
        self.vocab_size = vocab_size
        self.device = device
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.ddi_in_memory = ddi_in_memory

        # GAMENet only embeds diagnosis and procedure sequences; medications are used as
        # dynamic memory values.
        self.embeddings = nn.ModuleList([nn.Embedding(vocab_size[i], emb_dim) for i in range(K - 1)])
        self.dropout = nn.Dropout(p=0.4)

        self.encoders = nn.ModuleList([
            nn.GRU(emb_dim, emb_dim * 2, batch_first=True) for _ in range(K - 1)
        ])

        self.query = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 4, emb_dim),
        )

        self.ehr_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ehr_adj, device=device)
        self.ddi_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ddi_adj, device=device)
        self.inter = nn.Parameter(torch.FloatTensor(1))

        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 3, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, vocab_size[2])
        )

        self.init_weights()

    def _normalize_visit_mask(self, visit_mask, seq_len):
        """Return a bool mask with shape [batch, seq_len]."""
        if visit_mask is None:
            return torch.ones(1, seq_len, dtype=torch.bool, device=self.device)
        if not torch.is_tensor(visit_mask):
            visit_mask = torch.tensor(visit_mask, dtype=torch.bool, device=self.device)
        visit_mask = visit_mask.to(device=self.device, dtype=torch.bool)
        if visit_mask.dim() == 1:
            visit_mask = visit_mask.unsqueeze(0)
        elif visit_mask.dim() != 2:
            raise ValueError(f"visit_mask must have shape [seq] or [batch, seq], but got {tuple(visit_mask.shape)}.")
        if visit_mask.size(1) != seq_len:
            raise ValueError(f"visit_mask length must be {seq_len}, but got {visit_mask.size(1)}.")
        return visit_mask

    def _mean_embedding(self, embedding_layer, ids, batch_size, active_mask):
        if ids is None or len(ids) == 0:
            base = torch.zeros(1, 1, embedding_layer.embedding_dim, device=self.device)
        else:
            ids_tensor = torch.as_tensor(ids, dtype=torch.long, device=self.device).unsqueeze(dim=0)
            base = self.dropout(embedding_layer(ids_tensor)).mean(dim=1).unsqueeze(dim=0)
        base = base.expand(batch_size, -1, -1)
        return base * active_mask.view(batch_size, 1, 1).to(dtype=base.dtype)

    def forward(self, input, visit_mask=None):
        seq_len = len(input)
        if seq_len == 0:
            raise ValueError("GAMENet received an empty patient sequence.")

        visit_mask = self._normalize_visit_mask(visit_mask, seq_len)
        batch_size = visit_mask.size(0)

        i1_seq = []
        i2_seq = []
        for adm_idx, adm in enumerate(input):
            active = visit_mask[:, adm_idx]
            i1 = self._mean_embedding(self.embeddings[0], adm[0], batch_size, active)
            i2 = self._mean_embedding(self.embeddings[1], adm[1], batch_size, active)
            i1_seq.append(i1)
            i2_seq.append(i2)

        i1_seq = torch.cat(i1_seq, dim=1)  # [batch, seq, emb]
        i2_seq = torch.cat(i2_seq, dim=1)  # [batch, seq, emb]

        o1, _ = self.encoders[0](i1_seq)
        o2, _ = self.encoders[1](i2_seq)
        patient_representations = torch.cat([o1, o2], dim=-1)  # [batch, seq, 4*emb]
        queries = self.query(patient_representations)           # [batch, seq, emb]
        query = queries[:, -1, :]                               # [batch, emb]

        if self.ddi_in_memory:
            drug_memory = self.ehr_gcn() - self.ddi_gcn() * self.inter
        else:
            drug_memory = self.ehr_gcn()

        key_weights1 = F.softmax(torch.mm(query, drug_memory.t()), dim=-1)
        fact1 = torch.mm(key_weights1, drug_memory)

        if seq_len > 1:
            history_keys = queries[:, :seq_len - 1, :]
            history_mask = visit_mask[:, :seq_len - 1]

            active_history = history_mask.any(dim=1)
            if bool(active_history.any().item()):
                history_values = torch.zeros(
                    batch_size, seq_len - 1, self.vocab_size[2],
                    dtype=torch.float32, device=self.device
                )
                for idx, adm in enumerate(input[:-1]):
                    if adm[2]:
                        med_ids = torch.as_tensor(adm[2], dtype=torch.long, device=self.device)
                        history_values[:, idx, med_ids] = history_mask[:, idx].float().unsqueeze(1)

                visit_logits = torch.bmm(history_keys, query.unsqueeze(-1)).squeeze(-1)
                visit_logits = visit_logits.masked_fill(~history_mask, -1e9)
                visit_weight = F.softmax(visit_logits, dim=-1)
                visit_weight = torch.where(
                    active_history.unsqueeze(1),
                    visit_weight,
                    torch.zeros_like(visit_weight)
                )
                weighted_values = torch.bmm(visit_weight.unsqueeze(1), history_values).squeeze(1)
                fact2 = torch.mm(weighted_values, drug_memory)
                fact2 = torch.where(active_history.unsqueeze(1), fact2, fact1)
            else:
                fact2 = fact1
        else:
            fact2 = fact1

        output = self.output(torch.cat([query, fact1, fact2], dim=-1))

        if self.training:
            neg_pred_prob = torch.sigmoid(output)
            neg_pred_prob = torch.bmm(neg_pred_prob.unsqueeze(2), neg_pred_prob.unsqueeze(1))
            batch_neg = neg_pred_prob.mul(self.tensor_ddi_adj.unsqueeze(0)).mean()
            return output, batch_neg
        return output

    def init_weights(self):
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)
        self.inter.data.uniform_(-initrange, initrange)


class GAMENetProxyAdapter(nn.Module):
    """
    Adapter that makes a pretrained GAMENet usable as CounterfactualCausalRouter.proxy_predictor.
    The adapter always returns logits only, regardless of GAMENet train/eval return format.
    """
    input_mode = "sequence"

    def __init__(self, vocab_size, ehr_adj, ddi_adj, emb_dim=64, device=torch.device('cpu:0'),
                 pretrained_path=None, freeze=True, strict_load=False, ddi_in_memory=True):
        super().__init__()
        self.device = device
        self.freeze = bool(freeze)
        self.model = GAMENet(
            vocab_size=vocab_size,
            ehr_adj=ehr_adj,
            ddi_adj=ddi_adj,
            emb_dim=emb_dim,
            device=device,
            ddi_in_memory=ddi_in_memory
        )

        if pretrained_path:
            self.load_pretrained(pretrained_path, strict=strict_load)
        else:
            print("[GAMENetProxyAdapter] No pretrained_path was provided; GAMENet proxy is randomly initialized.")

        self.set_frozen(self.freeze)

    def _extract_state_dict(self, checkpoint):
        if isinstance(checkpoint, nn.Module):
            return checkpoint.state_dict()
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model_state_dict", "net", "model"):
                if key in checkpoint:
                    value = checkpoint[key]
                    return value.state_dict() if isinstance(value, nn.Module) else value
            return checkpoint
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")

    def load_pretrained(self, pretrained_path, strict=False):
        try:
            checkpoint = torch.load(pretrained_path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(pretrained_path, map_location=self.device)

        state_dict = self._extract_state_dict(checkpoint)
        clean_state = {}
        for key, value in state_dict.items():
            new_key = key
            # Strip common wrappers. Use a loop so keys such as
            # "module.proxy_predictor.model.embeddings.0.weight" become
            # "embeddings.0.weight".
            changed = True
            while changed:
                changed = False
                for prefix in ("module.", "proxy_predictor.model.", "proxy_predictor.", "model.", "gamenet."):
                    if new_key.startswith(prefix):
                        new_key = new_key[len(prefix):]
                        changed = True
            clean_state[new_key] = value

        missing, unexpected = self.model.load_state_dict(clean_state, strict=strict)
        print(
            f"[GAMENetProxyAdapter] Loaded pretrained GAMENet from {pretrained_path}. "
            f"missing={len(missing)}, unexpected={len(unexpected)}, strict={strict}"
        )
        if len(missing) > 0:
            print(f"[GAMENetProxyAdapter] missing keys sample: {missing[:5]}")
        if len(unexpected) > 0:
            print(f"[GAMENetProxyAdapter] unexpected keys sample: {unexpected[:5]}")

    def set_frozen(self, freeze=True):
        self.freeze = bool(freeze)
        for param in self.model.parameters():
            param.requires_grad = not self.freeze
        if self.freeze:
            self.model.eval()
        return self

    def train(self, mode=True):
        super().train(mode)
        # A frozen pretrained GAMENet should remain deterministic; keep dropout off even when
        # CounterfactualCausalRouter.train() is called every iteration.
        if self.freeze:
            self.model.eval()
        else:
            self.model.train(mode)
        return self

    def forward(self, patient, visit_mask=None):
        if self.freeze:
            with torch.no_grad():
                out = self.model(patient, visit_mask=visit_mask)
        else:
            out = self.model(patient, visit_mask=visit_mask)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out

    def counterfactual_logits(self, patient, base_mask=None, candidate_indices=None):
        """Batch cheap post-input masking for approximate history screening.

        candidate_indices normally contains only historical visits. The returned
        tensor contains the observational logits in row 0 and one masked result
        for each candidate afterwards. This method is intentionally approximate;
        exact Top-K interventions are performed by physically deleting admissions
        and re-running the compact trajectory in CounterfactualCausalRouter.
        """
        seq_len = len(patient)
        if base_mask is None:
            base_mask = torch.ones(seq_len, dtype=torch.bool, device=self.device)
        elif not torch.is_tensor(base_mask):
            base_mask = torch.tensor(base_mask, dtype=torch.bool, device=self.device)
        base_mask = base_mask.to(device=self.device, dtype=torch.bool).view(-1)
        if base_mask.numel() != seq_len:
            raise ValueError(f"base_mask length must be {seq_len}, got {base_mask.numel()}.")

        if candidate_indices is None:
            candidate_indices = list(range(seq_len))
        candidate_indices = [int(i) for i in candidate_indices]
        if any(i < 0 or i >= seq_len for i in candidate_indices):
            raise IndexError("candidate_indices contains an out-of-range visit index.")

        masks = base_mask.unsqueeze(0).repeat(len(candidate_indices) + 1, 1)
        if candidate_indices:
            row_ids = torch.arange(1, len(candidate_indices) + 1, device=self.device)
            col_ids = torch.as_tensor(candidate_indices, dtype=torch.long, device=self.device)
            masks[row_ids, col_ids] = False
        return self.forward(patient, visit_mask=masks)


class SafeDrugModel(nn.Module):
    """
    SafeDrug backbone adapted for use as a sequence proxy.

    The visit_mask interface is used for inexpensive approximate screening.
    Exact Top-K interventions physically delete the selected admission and rebuild
    the compact trajectory before this model is called.
    """
    def __init__(
        self,
        vocab_size,
        ddi_adj,
        ddi_mask_H,
        MPNNSet,
        N_fingerprints,
        average_projection,
        emb_dim=64,
        device=torch.device("cpu:0"),
    ):
        super(SafeDrugModel, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.device = device

        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size[i], emb_dim) for i in range(2)
        ])
        self.dropout = nn.Dropout(p=0.5)
        self.encoders = nn.ModuleList([
            nn.GRU(emb_dim, emb_dim, batch_first=True) for _ in range(2)
        ])
        self.query = nn.Sequential(nn.ReLU(), nn.Linear(2 * emb_dim, emb_dim))

        self.bipartite_transform = nn.Sequential(
            nn.Linear(emb_dim, ddi_mask_H.shape[1])
        )
        self.bipartite_output = MaskLinear(ddi_mask_H.shape[1], vocab_size[2], False)

        self.MPNN_molecule_Set = list(zip(*MPNNSet))
        mpnn_encoder = MolecularGraphNeuralNetwork(
            N_fingerprints, emb_dim, layer_hidden=2, device=device
        )
        with torch.no_grad():
            mpnn_emb = mpnn_encoder(self.MPNN_molecule_Set)
            mpnn_emb = torch.mm(
                average_projection.to(device=self.device),
                mpnn_emb.to(device=self.device),
            ).detach()
        # Fixed molecular features should move with the model but need not be
        # serialized because they are deterministically rebuilt from MPNNSet.
        self.register_buffer("MPNN_emb", mpnn_emb, persistent=False)

        self.MPNN_output = nn.Linear(vocab_size[2], vocab_size[2])
        self.MPNN_layernorm = nn.LayerNorm(vocab_size[2])

        self.register_buffer("tensor_ddi_adj", torch.FloatTensor(ddi_adj).to(device))
        self.register_buffer("tensor_ddi_mask_H", torch.FloatTensor(ddi_mask_H).to(device))
        self.init_weights()

    def _normalize_visit_mask(self, visit_mask, seq_len):
        """Return a bool mask with shape [batch, seq_len]."""
        if visit_mask is None:
            return torch.ones(1, seq_len, dtype=torch.bool, device=self.device)
        if not torch.is_tensor(visit_mask):
            visit_mask = torch.tensor(visit_mask, dtype=torch.bool, device=self.device)
        visit_mask = visit_mask.to(device=self.device, dtype=torch.bool)
        if visit_mask.dim() == 1:
            visit_mask = visit_mask.unsqueeze(0)
        elif visit_mask.dim() != 2:
            raise ValueError(f"visit_mask must have shape [seq] or [batch, seq], but got {tuple(visit_mask.shape)}.")
        if visit_mask.size(1) != seq_len:
            raise ValueError(f"visit_mask length must be {seq_len}, but got {visit_mask.size(1)}.")
        return visit_mask

    def _sum_embedding(self, embedding_layer, ids, batch_size, active_mask):
        if ids is None or len(ids) == 0:
            base = torch.zeros(1, 1, embedding_layer.embedding_dim, device=self.device)
        else:
            ids_tensor = torch.as_tensor(ids, dtype=torch.long, device=self.device).unsqueeze(dim=0)
            emb = self.dropout(embedding_layer(ids_tensor))
            base = emb.sum(dim=1).unsqueeze(dim=0)
        base = base.expand(batch_size, -1, -1)
        return base * active_mask.view(batch_size, 1, 1).to(dtype=base.dtype)

    def forward(self, input, visit_mask=None):
        seq_len = len(input)
        if seq_len == 0:
            raise ValueError("SafeDrugModel received an empty patient sequence.")

        visit_mask = self._normalize_visit_mask(visit_mask, seq_len)
        batch_size = visit_mask.size(0)

        i1_seq = []
        i2_seq = []
        for adm_idx, adm in enumerate(input):
            active = visit_mask[:, adm_idx]
            i1 = self._sum_embedding(self.embeddings[0], adm[0], batch_size, active)
            i2 = self._sum_embedding(self.embeddings[1], adm[1], batch_size, active)
            i1_seq.append(i1)
            i2_seq.append(i2)

        i1_seq = torch.cat(i1_seq, dim=1)  # [batch, seq, emb]
        i2_seq = torch.cat(i2_seq, dim=1)  # [batch, seq, emb]

        o1, _ = self.encoders[0](i1_seq)
        o2, _ = self.encoders[1](i2_seq)
        patient_representations = torch.cat([o1, o2], dim=-1)  # [batch, seq, 2*emb]
        query = self.query(patient_representations)[:, -1, :]  # [batch, emb]

        MPNN_match = torch.sigmoid(torch.mm(query, self.MPNN_emb.t()))
        MPNN_att = self.MPNN_layernorm(MPNN_match + self.MPNN_output(MPNN_match))

        bipartite_emb = self.bipartite_output(
            torch.sigmoid(self.bipartite_transform(query)),
            self.tensor_ddi_mask_H.t()
        )

        result = torch.mul(bipartite_emb, MPNN_att)

        neg_pred_prob = torch.sigmoid(result)
        neg_pred_prob = torch.bmm(neg_pred_prob.unsqueeze(2), neg_pred_prob.unsqueeze(1))
        batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj.unsqueeze(0)).sum()

        return result, batch_neg

    def init_weights(self):
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)


class SafeDrugProxyAdapter(nn.Module):
    """
    Adapter that makes a pretrained SafeDrug model usable as CounterfactualCausalRouter.proxy_predictor.
    It mirrors GAMENetProxyAdapter: accepts a raw patient sequence plus visit_mask and always returns logits only.
    """
    input_mode = "sequence"

    def __init__(self, vocab_size, ddi_adj, ddi_mask_H, MPNNSet, N_fingerprints,
                 average_projection, emb_dim=64, device=torch.device('cpu:0'),
                 pretrained_path=None, freeze=True, strict_load=False):
        super().__init__()
        self.device = device
        self.freeze = bool(freeze)
        self.model = SafeDrugModel(
            vocab_size=vocab_size,
            ddi_adj=ddi_adj,
            ddi_mask_H=ddi_mask_H,
            MPNNSet=MPNNSet,
            N_fingerprints=N_fingerprints,
            average_projection=average_projection,
            emb_dim=emb_dim,
            device=device,
        )

        if pretrained_path:
            self.load_pretrained(pretrained_path, strict=strict_load)
        else:
            print("[SafeDrugProxyAdapter] No pretrained_path was provided; SafeDrug proxy is randomly initialized.")

        self.set_frozen(self.freeze)

    def _extract_state_dict(self, checkpoint):
        if isinstance(checkpoint, nn.Module):
            return checkpoint.state_dict()
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model_state_dict", "net", "model"):
                if key in checkpoint:
                    value = checkpoint[key]
                    return value.state_dict() if isinstance(value, nn.Module) else value
            return checkpoint
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")

    def load_pretrained(self, pretrained_path, strict=False):
        try:
            checkpoint = torch.load(pretrained_path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(pretrained_path, map_location=self.device)

        state_dict = self._extract_state_dict(checkpoint)
        clean_state = {}
        for key, value in state_dict.items():
            new_key = key
            changed = True
            while changed:
                changed = False
                for prefix in (
                    "module.", "proxy_predictor.model.", "proxy_predictor.",
                    "model.", "safedrug.", "safe_drug.", "safedrug_model."
                ):
                    if new_key.startswith(prefix):
                        new_key = new_key[len(prefix):]
                        changed = True
            clean_state[new_key] = value

        missing, unexpected = self.model.load_state_dict(clean_state, strict=strict)
        print(
            f"[SafeDrugProxyAdapter] Loaded pretrained SafeDrug from {pretrained_path}. "
            f"missing={len(missing)}, unexpected={len(unexpected)}, strict={strict}"
        )
        if len(missing) > 0:
            print(f"[SafeDrugProxyAdapter] missing keys sample: {missing[:5]}")
        if len(unexpected) > 0:
            print(f"[SafeDrugProxyAdapter] unexpected keys sample: {unexpected[:5]}")

    def set_frozen(self, freeze=True):
        self.freeze = bool(freeze)
        for param in self.model.parameters():
            param.requires_grad = not self.freeze
        if self.freeze:
            self.model.eval()
        return self

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.model.eval()
        else:
            self.model.train(mode)
        return self

    def forward(self, patient, visit_mask=None):
        if self.freeze:
            with torch.no_grad():
                out = self.model(patient, visit_mask=visit_mask)
        else:
            out = self.model(patient, visit_mask=visit_mask)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out

    def counterfactual_logits(self, patient, base_mask=None, candidate_indices=None):
        """Batch cheap post-input masking for approximate history screening.

        candidate_indices normally contains only historical visits. The returned
        tensor contains the observational logits in row 0 and one masked result
        for each candidate afterwards. This method is intentionally approximate;
        exact Top-K interventions are performed by physically deleting admissions
        and re-running the compact trajectory in CounterfactualCausalRouter.
        """
        seq_len = len(patient)
        if base_mask is None:
            base_mask = torch.ones(seq_len, dtype=torch.bool, device=self.device)
        elif not torch.is_tensor(base_mask):
            base_mask = torch.tensor(base_mask, dtype=torch.bool, device=self.device)
        base_mask = base_mask.to(device=self.device, dtype=torch.bool).view(-1)
        if base_mask.numel() != seq_len:
            raise ValueError(f"base_mask length must be {seq_len}, got {base_mask.numel()}.")

        if candidate_indices is None:
            candidate_indices = list(range(seq_len))
        candidate_indices = [int(i) for i in candidate_indices]
        if any(i < 0 or i >= seq_len for i in candidate_indices):
            raise IndexError("candidate_indices contains an out-of-range visit index.")

        masks = base_mask.unsqueeze(0).repeat(len(candidate_indices) + 1, 1)
        if candidate_indices:
            row_ids = torch.arange(1, len(candidate_indices) + 1, device=self.device)
            col_ids = torch.as_tensor(candidate_indices, dtype=torch.long, device=self.device)
            masks[row_ids, col_ids] = False
        return self.forward(patient, visit_mask=masks)


class ARMRMambaModelArgs:
    def __init__(self, d_model, d_state=16, expand=2, dt_rank='auto', d_conv=4,
                 conv_bias=True, bias=False):
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.expand = int(expand)
        self.dt_rank = dt_rank
        self.d_conv = int(d_conv)
        self.conv_bias = bool(conv_bias)
        self.bias = bool(bias)
        self.d_inner = int(self.expand * self.d_model)
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
        self.dt_rank = int(self.dt_rank)


class ARMRMambaBlock(nn.Module):
    """Mamba-style block copied from ARMR/MyNet, implemented without external einops dependency."""
    def __init__(self, args: ARMRMambaModelArgs):
        super().__init__()
        self.args = args
        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)
        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)
        A = torch.arange(1, args.d_state + 1).repeat(args.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A.float()))
        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        b, l, _ = x.shape
        x_and_res = self.in_proj(x)
        x, res = x_and_res.split([self.args.d_inner, self.args.d_inner], dim=-1)
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :l]
        x = x.transpose(1, 2)
        x = F.silu(x)
        y = self.ssm(x)
        y = y * F.silu(res)
        return self.out_proj(y)

    def ssm(self, x):
        d_in, n = self.A_log.shape
        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        x_dbl = self.x_proj(x)
        delta, B, C = x_dbl.split([self.args.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        return self.selective_scan(x, delta, A, B, C, D)

    def selective_scan(self, u, delta, A, B, C, D):
        b, l, d_in = u.shape
        n = A.shape[1]
        deltaA = torch.exp(torch.einsum('bld,dn->bldn', delta, A))
        deltaB_u = torch.einsum('bld,bln,bld->bldn', delta, B, u)
        x = torch.zeros((b, d_in, n), device=u.device, dtype=u.dtype)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = torch.einsum('bdn,bn->bd', x, C[:, i, :])
            ys.append(y)
        y = torch.stack(ys, dim=1)
        return y + u * D


class ARMRGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        return output + self.bias if self.bias is not None else output


class ARMRGCN(nn.Module):
    """GCN module used inside ARMR/MyNet."""
    def __init__(self, voc_size, emb_dim, ehr_adj, ddi_adj, device=torch.device('cpu:0')):
        super().__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        ehr_adj = torch.as_tensor(ehr_adj, dtype=torch.float32, device=device)
        ddi_adj = torch.as_tensor(ddi_adj, dtype=torch.float32, device=device)
        eye = torch.eye(ehr_adj.shape[0], device=device)
        self.register_buffer('ehr_adj', self.normalize(ehr_adj + eye))
        self.register_buffer('ddi_adj', self.normalize(ddi_adj + eye))
        self.register_buffer('x', torch.eye(voc_size, device=device))
        self.gcn1 = ARMRGraphConvolution(voc_size, emb_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.gcn2 = ARMRGraphConvolution(emb_dim, emb_dim)
        self.gcn3 = ARMRGraphConvolution(emb_dim, emb_dim)

    def forward(self):
        ehr_node_embedding = self.gcn1(self.x, self.ehr_adj)
        ehr_node_embedding = F.relu(ehr_node_embedding)
        ehr_node_embedding = self.dropout(ehr_node_embedding)
        ehr_node_embedding = self.gcn2(ehr_node_embedding, self.ehr_adj)

        ddi_node_embedding = self.gcn1(self.x, self.ddi_adj)
        ddi_node_embedding = F.relu(ddi_node_embedding)
        ddi_node_embedding = self.dropout(ddi_node_embedding)
        ddi_node_embedding = self.gcn3(ddi_node_embedding, self.ddi_adj)
        return ehr_node_embedding, ddi_node_embedding

    def normalize(self, mx):
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1).flatten()
        r_inv[torch.isinf(r_inv)] = 0.0
        r_mat_inv = torch.diag(r_inv)
        return torch.mm(r_mat_inv, mx)


class ARMRSimpleNetV(nn.Module):
    """Auxiliary SimpleNetV kept so pretrained MyNet checkpoints load with the original key names."""
    def __init__(self, emb_dim, visit_num, voc_size):
        super().__init__()
        self.visit_num = int(visit_num)
        self.diag_encoder = nn.Linear(voc_size[0], emb_dim)
        self.proc_encoder = nn.Linear(voc_size[1], emb_dim)
        self.med_encoder = nn.Linear(voc_size[2], emb_dim)
        self.output_layer = nn.Linear(emb_dim * self.visit_num, voc_size[2])

    def forward(self, diags, procs, meds):
        seq_len = diags.size(1)
        if seq_len < self.visit_num:
            pad_len = self.visit_num - seq_len
            diags = torch.cat((diags, torch.zeros(diags.shape[0], pad_len, diags.shape[2], device=diags.device)), dim=1)
            procs = torch.cat((procs, torch.zeros(procs.shape[0], pad_len, procs.shape[2], device=procs.device)), dim=1)
            meds = torch.cat((meds, torch.zeros(meds.shape[0], pad_len, meds.shape[2], device=meds.device)), dim=1)
        else:
            diags = diags[:, :self.visit_num, :]
            procs = procs[:, :self.visit_num, :]
            meds = meds[:, :self.visit_num, :]
        visits_emb = self.diag_encoder(diags) + self.proc_encoder(procs) + self.med_encoder(meds)
        return self.output_layer(visits_emb.reshape(visits_emb.shape[0], -1))


class ARMRPiecewiseTSL(nn.Module):
    def __init__(self, emb_dim, k):
        super().__init__()
        self.k = int(k)
        self.mamba = ARMRMambaBlock(ARMRMambaModelArgs(d_model=emb_dim))
        self.lin = nn.Linear(self.k * emb_dim, self.k * emb_dim)
        self.norm = nn.LayerNorm(self.k * emb_dim)

    def forward(self, seq):
        batch, seq_len, emb_dim = seq.shape
        if seq_len < self.k:
            pad_len = self.k - seq_len
            seq = torch.cat((seq, torch.zeros(batch, pad_len, emb_dim, device=seq.device, dtype=seq.dtype)), dim=1)
        near_seq = seq[:, :self.k, :]
        near_h = self.lin(self.norm(near_seq.reshape(batch, -1))).view(batch, self.k, emb_dim) + near_seq
        far_h = torch.zeros(batch, self.k, emb_dim, device=seq.device, dtype=seq.dtype)
        if seq_len > self.k:
            far_seq = self.mamba(torch.flip(seq[:, self.k:, :], [1]))
            q = near_h
            kv = far_seq
            attn_scores = torch.bmm(q, kv.transpose(-2, -1))
            attn_scores = F.softmax(attn_scores, dim=-1)
            far_h = torch.bmm(attn_scores, kv)
        return torch.cat((near_h, far_h), dim=1)


class ARMRPatientRepLearn(nn.Module):
    def __init__(self, emb_dim, k, d_voc_size, p_voc_size, m_voc_size):
        super().__init__()
        self.emb_dim = emb_dim
        self.tsl_d = ARMRPiecewiseTSL(emb_dim, k)
        self.tsl_p = ARMRPiecewiseTSL(emb_dim, k)
        self.d_lin = nn.Linear(d_voc_size, emb_dim)
        self.p_lin = nn.Linear(p_voc_size, emb_dim)
        self.m_lin = nn.Linear(m_voc_size, emb_dim)

    def forward(self, diags, procs, meds):
        e_d = self.d_lin(diags)
        e_p = self.p_lin(procs)
        e_m = self.m_lin(meds)
        e_h = e_d + e_p + e_m
        h_d = self.tsl_d(e_d)
        h_p = self.tsl_p(e_p)
        if diags.size(1) < h_d.size(1):
            pad_len = h_d.size(1) - diags.size(1)
            e_h = torch.cat((e_h, torch.zeros(e_h.shape[0], pad_len, e_h.shape[2], device=e_h.device, dtype=e_h.dtype)), dim=1)
        else:
            e_h = e_h[:, :h_d.size(1), :]
        return torch.cat((e_h, h_d + h_p), dim=1)


class ARMRMedRepLearn(nn.Module):
    def __init__(self, emb_dim, k, m_voc_size):
        super().__init__()
        self.emb_dim = emb_dim
        self.m_voc_size = m_voc_size
        self.tsl_old = ARMRPiecewiseTSL(emb_dim, k)
        self.tsl_new = ARMRPiecewiseTSL(emb_dim, k)
        self.m_embs = nn.Embedding(m_voc_size, emb_dim)
        self.lin_expand = nn.Linear(emb_dim, 2 * k * emb_dim)

    def forward(self, meds):
        batch = meds.shape[0]
        history = torch.cat((
            torch.zeros(batch, 1, self.m_voc_size, device=meds.device, dtype=meds.dtype),
            (torch.cumsum(meds, dim=1) > 0).float()[:, :-1, :]
        ), dim=1)
        old = meds * history
        new = meds - old
        e_old = torch.matmul(old, self.m_embs.weight)
        e_new = torch.matmul(new, self.m_embs.weight)
        masks = (meds.sum(dim=1) > 0).float().unsqueeze(2).repeat(1, 1, self.emb_dim)
        old_m_embs = self.m_embs.weight.unsqueeze(0).repeat(batch, 1, 1) * masks
        new_m_embs = self.m_embs.weight.unsqueeze(0).repeat(batch, 1, 1) * (1 - masks)
        h_old = self.tsl_old(e_old).reshape(batch, -1)
        h_new = self.tsl_new(e_new).reshape(batch, -1)
        scores_old = torch.cosine_similarity(h_old.unsqueeze(1), self.lin_expand(old_m_embs), dim=2)
        scores_new = torch.cosine_similarity(h_new.unsqueeze(1), self.lin_expand(new_m_embs), dim=2)
        total_scores = scores_old + scores_new
        return self.m_embs.weight.unsqueeze(0).repeat(batch, 1, 1) * total_scores.unsqueeze(2)


class ARMRMyNet(nn.Module):
    """
    ARMR/MyNet backbone.

    It expects multi-hot tensors generated from the original patient sequence:
    diags: [batch, seq_len, diag_vocab]
    procs: [batch, seq_len, proc_vocab]
    meds:  [batch, seq_len, med_vocab]

    ARMR's data loader places the current visit at index 0 and sets the current visit's
    medication vector to zero. The adapter below reproduces that conversion from the
    CEGMed patient-list format.
    """
    def __init__(self, emb_dim, k, voc_size, ehr_adj, ddi_adj, device=torch.device('cpu:0')):
        super().__init__()
        self.d_voc_size = voc_size[0]
        self.p_voc_size = voc_size[1]
        self.m_voc_size = voc_size[2]
        self.emb_dim = emb_dim
        self.k = int(k)
        self.medrep = ARMRMedRepLearn(emb_dim, self.k, self.m_voc_size)
        self.patrep = ARMRPatientRepLearn(emb_dim, self.k, self.d_voc_size, self.p_voc_size, self.m_voc_size)
        self.lin1 = nn.Linear(2 * self.k * emb_dim, self.m_voc_size)
        self.lin2 = nn.Linear(2 * self.k * emb_dim, 2 * self.k * emb_dim)
        self.norm = nn.LayerNorm(2 * self.k * emb_dim)
        self.gcn = ARMRGCN(self.m_voc_size, emb_dim, ehr_adj, ddi_adj, device=device)
        self.sm = ARMRSimpleNetV(emb_dim, self.k, voc_size)
        self.lin_med_expand = nn.Linear(emb_dim, 2 * self.k * emb_dim)
        self.w = 0.7

    def forward(self, diags, procs, meds):
        # Preserve the original ARMR/MyNet behavior: procedure features are explicitly zeroed.
        procs = torch.zeros_like(procs).to(procs.device)
        batch = diags.shape[0]
        pat = self.patrep(diags, procs, meds)
        h_patient = pat[:, 2 * self.k:, :].reshape(batch, -1)
        q = h_patient + self.lin2(self.norm(h_patient))
        ehr_meds, ddi_meds = self.gcn()
        h_meds = self.medrep(meds)
        h_meds = h_meds + ehr_meds.unsqueeze(0) * 1.0
        h_meds = h_meds + ddi_meds.unsqueeze(0) * 0.5
        h_meds = self.lin_med_expand(h_meds)
        o_1 = self.lin1(pat[:, :2 * self.k, :].reshape(batch, -1))
        o_2 = torch.cosine_similarity(q.unsqueeze(1), h_meds, dim=2)
        return o_1 * self.w + o_2 * (1.0 - self.w)


class ARMRProxyAdapter(nn.Module):
    """
    Adapter that makes a pretrained ARMR/MyNet usable as CounterfactualCausalRouter.proxy_predictor.

    The adapter converts a CEGMed patient list into ARMR's reversed multi-hot tensor format and
    accepts visit_mask for approximate history screening; exact Top-K routing
    physically deletes admissions before tensor conversion.
    """
    input_mode = "sequence"

    def __init__(self, vocab_size, ehr_adj, ddi_adj, emb_dim=256, visit_num=3,
                 device=torch.device('cpu:0'), pretrained_path=None, freeze=True,
                 strict_load=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.device = device
        self.freeze = bool(freeze)
        self.model = ARMRMyNet(
            emb_dim=emb_dim,
            k=visit_num,
            voc_size=vocab_size,
            ehr_adj=ehr_adj,
            ddi_adj=ddi_adj,
            device=device,
        )
        if pretrained_path:
            self.load_pretrained(pretrained_path, strict=strict_load)
        else:
            print("[ARMRProxyAdapter] No pretrained_path was provided; ARMR/MyNet proxy is randomly initialized.")
        self.set_frozen(self.freeze)

    def _extract_state_dict(self, checkpoint):
        if isinstance(checkpoint, nn.Module):
            return checkpoint.state_dict()
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model_state_dict", "net", "model", "best_model_params"):
                if key in checkpoint:
                    value = checkpoint[key]
                    return value.state_dict() if isinstance(value, nn.Module) else value
            return checkpoint
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")

    def load_pretrained(self, pretrained_path, strict=False):
        try:
            checkpoint = torch.load(pretrained_path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(pretrained_path, map_location=self.device)
        state_dict = self._extract_state_dict(checkpoint)
        clean_state = {}
        for key, value in state_dict.items():
            new_key = key
            changed = True
            while changed:
                changed = False
                for prefix in (
                    "module.", "proxy_predictor.model.", "proxy_predictor.",
                    "model.", "armr.", "mynet.", "my_net.", "net."
                ):
                    if new_key.startswith(prefix):
                        new_key = new_key[len(prefix):]
                        changed = True
            clean_state[new_key] = value
        missing, unexpected = self.model.load_state_dict(clean_state, strict=strict)
        print(
            f"[ARMRProxyAdapter] Loaded pretrained ARMR/MyNet from {pretrained_path}. "
            f"missing={len(missing)}, unexpected={len(unexpected)}, strict={strict}"
        )
        if len(missing) > 0:
            print(f"[ARMRProxyAdapter] missing keys sample: {missing[:5]}")
        if len(unexpected) > 0:
            print(f"[ARMRProxyAdapter] unexpected keys sample: {unexpected[:5]}")

    def set_frozen(self, freeze=True):
        self.freeze = bool(freeze)
        for param in self.model.parameters():
            param.requires_grad = not self.freeze
        if self.freeze:
            self.model.eval()
        return self

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.model.eval()
        else:
            self.model.train(mode)
        return self

    def _normalize_visit_mask(self, visit_mask, seq_len):
        """Return a bool mask with shape [batch, seq_len]."""
        if visit_mask is None:
            return torch.ones(1, seq_len, dtype=torch.bool, device=self.device)
        if not torch.is_tensor(visit_mask):
            visit_mask = torch.tensor(visit_mask, dtype=torch.bool, device=self.device)
        visit_mask = visit_mask.to(device=self.device, dtype=torch.bool)
        if visit_mask.dim() == 1:
            visit_mask = visit_mask.unsqueeze(0)
        elif visit_mask.dim() != 2:
            raise ValueError(f"visit_mask must have shape [seq] or [batch, seq], but got {tuple(visit_mask.shape)}.")
        if visit_mask.size(1) != seq_len:
            raise ValueError(f"visit_mask length must be {seq_len}, but got {visit_mask.size(1)}.")
        return visit_mask

    def _patient_to_armr_tensors(self, patient, visit_mask=None):
        seq_len = len(patient)
        if seq_len == 0:
            raise ValueError("ARMRProxyAdapter received an empty patient sequence.")
        visit_mask = self._normalize_visit_mask(visit_mask, seq_len)
        batch_size = visit_mask.size(0)
        diag_size, proc_size, med_size = self.vocab_size

        diags = torch.zeros(batch_size, seq_len, diag_size, dtype=torch.float32, device=self.device)
        procs = torch.zeros(batch_size, seq_len, proc_size, dtype=torch.float32, device=self.device)
        meds = torch.zeros(batch_size, seq_len, med_size, dtype=torch.float32, device=self.device)

        # Fill chronological tensors, then flip them to ARMR order: current -> history.
        for adm_idx, adm in enumerate(patient):
            active = visit_mask[:, adm_idx].float()
            if adm[0]:
                diag_ids = torch.as_tensor(adm[0], dtype=torch.long, device=self.device)
                diags[:, adm_idx, diag_ids] = active.unsqueeze(1)
            if adm[1]:
                proc_ids = torch.as_tensor(adm[1], dtype=torch.long, device=self.device)
                procs[:, adm_idx, proc_ids] = active.unsqueeze(1)
            # ARMR uses the current visit's medications as labels, not as input features.
            if adm_idx != seq_len - 1 and adm[2]:
                med_ids = torch.as_tensor(adm[2], dtype=torch.long, device=self.device)
                meds[:, adm_idx, med_ids] = active.unsqueeze(1)

        return diags.flip(1), procs.flip(1), meds.flip(1)

    def forward(self, patient, visit_mask=None):
        diags, procs, meds = self._patient_to_armr_tensors(patient, visit_mask=visit_mask)
        if self.freeze:
            with torch.no_grad():
                return self.model(diags, procs, meds)
        return self.model(diags, procs, meds)

    def counterfactual_logits(self, patient, base_mask=None, candidate_indices=None):
        """Batch cheap post-input masking for approximate history screening.

        candidate_indices normally contains only historical visits. The returned
        tensor contains the observational logits in row 0 and one masked result
        for each candidate afterwards. This method is intentionally approximate;
        exact Top-K interventions are performed by physically deleting admissions
        and re-running the compact trajectory in CounterfactualCausalRouter.
        """
        seq_len = len(patient)
        if base_mask is None:
            base_mask = torch.ones(seq_len, dtype=torch.bool, device=self.device)
        elif not torch.is_tensor(base_mask):
            base_mask = torch.tensor(base_mask, dtype=torch.bool, device=self.device)
        base_mask = base_mask.to(device=self.device, dtype=torch.bool).view(-1)
        if base_mask.numel() != seq_len:
            raise ValueError(f"base_mask length must be {seq_len}, got {base_mask.numel()}.")

        if candidate_indices is None:
            candidate_indices = list(range(seq_len))
        candidate_indices = [int(i) for i in candidate_indices]
        if any(i < 0 or i >= seq_len for i in candidate_indices):
            raise IndexError("candidate_indices contains an out-of-range visit index.")

        masks = base_mask.unsqueeze(0).repeat(len(candidate_indices) + 1, 1)
        if candidate_indices:
            row_ids = torch.arange(1, len(candidate_indices) + 1, device=self.device)
            col_ids = torch.as_tensor(candidate_indices, dtype=torch.long, device=self.device)
            masks[row_ids, col_ids] = False
        return self.forward(patient, visit_mask=masks)

class CounterfactualCausalRouter(nn.Module):
    def __init__(self,
                 ddi_adj,
                 ehr_adj,
                 ddi_mask_H,
                 MPNNSet,
                 N_fingerprints,
                 average_projection,
                 causal_graph,
                 vocab_size,
                 pretrained_embeddings=None,
                 emb_dim=64,
                 device='cuda:0',
                 model_config=None):
        super(CounterfactualCausalRouter, self).__init__()
        cfg = {
            'use_causal_graph': True,
            'use_causal_routing': True,
            'use_ctcr': True,
            'no_causal_routing_a_hist': 0.5,
            'use_memory_bank': True,
            'dropout': 0.3,
            'att_tau': 20.0,
            'cross_att_heads': 4,
            'routing_hidden_dim': 64,
            'routing_mid_dim': 32,
            'hist_hidden_dim': 32,
            'query_hidden_mult': 3,
            'ctcr_temperature': 0.1,
            'ctcr_max_delta': 0.1,
            'ctcr_lambda_med': 0.5,
            'ctcr_w_cur': 1.0,
            'ctcr_w_hist': 1.0,
            'ctcr_w_global': 1.0,
            'ddi_loss_weight': 0.0005,
            'proxy_dropout': 0.3,
            'cross_att_dropout': 0.1,
            'mpnn_layer_hidden': 2,
            'gcn_layers': 2,
            'proxy_model_type': 'mlp',
            'gamenet_pretrained_path': None,
            'freeze_proxy': True,
            'gamenet_strict_load': False,
            'gamenet_ddi_in_memory': True,
            'safedrug_pretrained_path': None,
            'safedrug_strict_load': False,
            'safedrug_emb_dim': None,
            'armr_pretrained_path': None,
            'armr_strict_load': False,
            'armr_emb_dim': None,
            'armr_visit_num': 3,
            'history_topk': 3,
        }
        if model_config:
            cfg.update({key: value for key, value in model_config.items() if value is not None})

        use_causal_graph = cfg['use_causal_graph']
        use_causal_routing = cfg['use_causal_routing']
        use_ctcr = cfg['use_ctcr']
        no_causal_routing_a_hist = cfg['no_causal_routing_a_hist']
        use_memory_bank = cfg['use_memory_bank']
        dropout = cfg['dropout']
        att_tau = cfg['att_tau']
        cross_att_heads = cfg['cross_att_heads']
        routing_hidden_dim = cfg['routing_hidden_dim']
        routing_mid_dim = cfg['routing_mid_dim']
        hist_hidden_dim = cfg['hist_hidden_dim']
        query_hidden_mult = cfg['query_hidden_mult']
        ctcr_temperature = cfg['ctcr_temperature']
        ctcr_max_delta = cfg['ctcr_max_delta']
        ctcr_lambda_med = cfg['ctcr_lambda_med']
        ctcr_w_cur = cfg['ctcr_w_cur']
        ctcr_w_hist = cfg['ctcr_w_hist']
        ctcr_w_global = cfg['ctcr_w_global']
        ddi_loss_weight = cfg['ddi_loss_weight']
        proxy_dropout = cfg['proxy_dropout']
        cross_att_dropout = cfg['cross_att_dropout']
        mpnn_layer_hidden = cfg['mpnn_layer_hidden']
        gcn_layers = cfg['gcn_layers']
        proxy_model_type = cfg['proxy_model_type']
        gamenet_pretrained_path = cfg['gamenet_pretrained_path']
        freeze_proxy = cfg['freeze_proxy']
        gamenet_strict_load = cfg['gamenet_strict_load']
        gamenet_ddi_in_memory = cfg['gamenet_ddi_in_memory']
        safedrug_pretrained_path = cfg['safedrug_pretrained_path']
        safedrug_strict_load = cfg['safedrug_strict_load']
        safedrug_emb_dim = cfg['safedrug_emb_dim']
        armr_pretrained_path = cfg['armr_pretrained_path']
        armr_strict_load = cfg['armr_strict_load']
        armr_emb_dim = cfg['armr_emb_dim']
        armr_visit_num = cfg['armr_visit_num']
        history_topk = cfg['history_topk']
        self.emb_dim = emb_dim
        self.med_vocab_size = vocab_size[2]
        self.device = device
        self.causal_graph = causal_graph
        self.use_causal_graph = bool(use_causal_graph and causal_graph is not None)
        self.use_causal_routing = bool(use_causal_routing)
        self.use_ctcr = bool(use_ctcr and self.use_causal_graph)
        self.no_causal_routing_a_hist = float(no_causal_routing_a_hist)
        self.use_memory_bank = bool(use_memory_bank)
        self.dropout_rate = float(dropout)
        self.ddi_loss_weight = float(ddi_loss_weight)
        self.gcn_layers = max(1, int(gcn_layers))
        self.proxy_model_type = str(proxy_model_type).lower()
        self.freeze_proxy = bool(freeze_proxy)
        # Number of historical admissions retained after approximate screening.
        # A non-positive value keeps all history and is useful only for ablation.
        self.history_topk = int(history_topk)

        if pretrained_embeddings is not None:
            print("Using pretrained embeddings.")

        self.embeddings = nn.ModuleList(
            pretrained_embeddings if pretrained_embeddings is not None else [nn.Embedding(vocab_size[i], emb_dim) for i in range(3)]
        )
        self.homo_graph = nn.ModuleList([
            homo_relation_graph(emb_dim, device), # d
            homo_relation_graph(emb_dim, device), # p
            homo_relation_graph(emb_dim, device), # m
        ])
        self.dropout = nn.Dropout(p=self.dropout_rate)

        self.raw_encoders = nn.ModuleList(
            [
                nn.GRU(
                    input_size=self.emb_dim,
                    hidden_size=self.emb_dim,
                    batch_first=True,
                )
                for _ in range(3)
            ]
        )

        # causal graph embedding 对应的三个 GRU
        self.causal_encoders = nn.ModuleList(
            [
                nn.GRU(
                    input_size=self.emb_dim,
                    hidden_size=self.emb_dim,
                    batch_first=True,
                )
                for _ in range(3)
            ]
        )

        raw_ehr_adj = np.asarray(ehr_adj, dtype=np.float32)
        raw_ddi_adj = np.asarray(ddi_adj, dtype=np.float32)

        ehr_adj_norm = self.normalize(raw_ehr_adj + np.eye(raw_ehr_adj.shape[0], dtype=np.float32))
        ddi_adj_norm = self.normalize(raw_ddi_adj + np.eye(raw_ddi_adj.shape[0], dtype=np.float32))

        # Graph convolution uses normalized adjacency with self-loops.
        self.register_buffer("ehr_adj", torch.FloatTensor(ehr_adj_norm).to(self.device))
        self.register_buffer("ddi_adj", torch.FloatTensor(ddi_adj_norm).to(self.device))

        # DDI loss should use the original DDI adjacency without self-loops/normalization.
        ddi_penalty_adj = raw_ddi_adj.copy()
        np.fill_diagonal(ddi_penalty_adj, 0.0)
        self.register_buffer("tensor_ddi_adj", torch.FloatTensor(ddi_penalty_adj).to(self.device))
        self.register_buffer("tensor_ehr_adj", torch.FloatTensor(raw_ehr_adj).to(self.device))
        self.register_buffer("tensor_ddi_mask_H", torch.FloatTensor(ddi_mask_H).to(self.device))

        self.ehr_gcn = GraphConvolution(emb_dim, emb_dim, bias=False).to(device)
        self.ddi_gcn = GraphConvolution(emb_dim, emb_dim, bias=False).to(device)

        self.bipartite_transform = nn.Sequential(
            nn.Linear(emb_dim, ddi_mask_H.shape[1])
        )
        self.bipartite_output = MaskLinear(ddi_mask_H.shape[1], vocab_size[2], False)

        self.MPNN_molecule_Set = list(zip(*MPNNSet))
        self.MPNN_emb = MolecularGraphNeuralNetwork(
            N_fingerprints, emb_dim, layer_hidden=max(1, int(mpnn_layer_hidden)), device=device
        ).forward(self.MPNN_molecule_Set)
        self.MPNN_emb = torch.mm(
            average_projection.to(device=self.device),
            self.MPNN_emb.to(device=self.device),
        ).to(device=self.device)

        self.MPNN_layernorm = nn.LayerNorm(vocab_size[2])
        self.MPNN_output = nn.Linear(vocab_size[2], vocab_size[2])
        
        # ---------------------
        # long
        self.att_tau = float(att_tau)
        self.linear_layer = nn.Linear(3 * emb_dim, 3 * emb_dim)
        # ---------------------

        # ---------------------
        # heng
        self.cross_att = ScaledDotProductAttention(
            3 * emb_dim,
            3 * emb_dim,
            emb_dim,
            cross_att_heads,
            dropout=cross_att_dropout
        )
        self.drug_output = nn.Linear(emb_dim, emb_dim)
        self.drug_layernorm = nn.LayerNorm(emb_dim)
        # ---------------------

        
        # 2. 代理药物预测器 (Proxy Predictor)
        # - mlp: original h_obs -> logits proxy.
        # - gamenet/safedrug/armr: visit_mask is used only for cheap screening;
        #   exact Top-K interventions receive physically compacted trajectories.
        # - armr: pretrained ARMR/MyNet sequence -> logits proxy, with ARMR tensor-format conversion.
        if self.proxy_model_type in ('mlp', 'proxy', 'proxymodel'):
            self.proxy_model_type = 'mlp'
            self.proxy_predictor = ProxyModel(vocab_size, emb_dim, device, dropout=proxy_dropout)
        elif self.proxy_model_type in ('gamenet', 'game_net'):
            self.proxy_model_type = 'gamenet'
            self.proxy_predictor = GAMENetProxyAdapter(
                vocab_size=vocab_size,
                ehr_adj=raw_ehr_adj,
                ddi_adj=raw_ddi_adj,
                emb_dim=emb_dim,
                device=device,
                pretrained_path=gamenet_pretrained_path,
                freeze=freeze_proxy,
                strict_load=gamenet_strict_load,
                ddi_in_memory=gamenet_ddi_in_memory
            )
        elif self.proxy_model_type in ('safedrug', 'safe_drug', 'safe-drug'):
            self.proxy_model_type = 'safedrug'
            proxy_emb_dim = emb_dim if safedrug_emb_dim is None else int(safedrug_emb_dim)
            self.proxy_predictor = SafeDrugProxyAdapter(
                vocab_size=vocab_size,
                ddi_adj=raw_ddi_adj,
                ddi_mask_H=ddi_mask_H,
                MPNNSet=MPNNSet,
                N_fingerprints=N_fingerprints,
                average_projection=average_projection,
                emb_dim=proxy_emb_dim,
                device=device,
                pretrained_path=safedrug_pretrained_path,
                freeze=freeze_proxy,
                strict_load=safedrug_strict_load
            )
        elif self.proxy_model_type in ('armr', 'mynet', 'my_net'):
            self.proxy_model_type = 'armr'
            proxy_emb_dim = emb_dim if armr_emb_dim is None else int(armr_emb_dim)
            self.proxy_predictor = ARMRProxyAdapter(
                vocab_size=vocab_size,
                ehr_adj=raw_ehr_adj,
                ddi_adj=raw_ddi_adj,
                emb_dim=proxy_emb_dim,
                visit_num=armr_visit_num,
                device=device,
                pretrained_path=armr_pretrained_path,
                freeze=freeze_proxy,
                strict_load=armr_strict_load
            )
        else:
            raise ValueError(f"Unknown proxy_model_type: {proxy_model_type}")

        # 3. 因果路由门控网络 (Adaptive Causal Routing)
        # 输入是因果效应差值 CE_k (维度: med_vocab_size)，输出为该病程块的初始标量权重
        self.routing_mlp = nn.Sequential(
            nn.Linear(self.med_vocab_size, routing_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(routing_hidden_dim, routing_mid_dim),
            nn.ReLU(),
            nn.Linear(routing_mid_dim, 1)
        )
        
        # 4. 历史总体贡献评估模块
        # 将所有病程的 CE 聚合后，映射为一个 0~1 的值 a_hist
        self.hist_contribution_mlp = nn.Sequential(
            nn.Linear(self.med_vocab_size, hist_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hist_hidden_dim, 1),
            nn.Sigmoid() # 保证输出在 (0, 1) 之间
        )

        self.base_query = nn.Sequential(
            nn.Linear(3 * emb_dim, 2 * emb_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(2 * emb_dim, emb_dim)
        )

        self.causal_adapter = nn.Sequential(
            nn.Linear(7 * emb_dim, 2 * emb_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(2 * emb_dim, emb_dim)
        )

        # 初始 gate 很小，避免因果模块一开始破坏 backbone
        self.raw_causal_gate = nn.Parameter(torch.tensor(-5.0))

        query_hidden_dim = int(query_hidden_mult * emb_dim)
        self.query = nn.Sequential(
            nn.Linear(7 * emb_dim, query_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(query_hidden_dim, 3 * emb_dim),
            nn.ReLU(),
            nn.Linear(3 * emb_dim, emb_dim)
        )

        self.fusion_layer = EnhancedFeatureFusion(emb_dim=self.emb_dim, num_views=3, dropout=0.1).to(self.device)

        if self.use_ctcr:
            self.ctcr_start_epoch = 10
            self.ctcr_ramp_epochs = 10
            self.causal_review = CounterfactualTrajectoryCausalReview(
                causal_graph=causal_graph,
                num_med=vocab_size[2],
                temperature=ctcr_temperature,
                max_delta=ctcr_max_delta,
                lambda_med=ctcr_lambda_med,
                init_w_cur=ctcr_w_cur,
                init_w_hist=ctcr_w_hist,
                init_w_global=ctcr_w_global,
                history_topk=self.history_topk,
                device=device
            )
        else:
            self.causal_review = None

    def train(self, mode=True):
        super().train(mode)
        # A proxy is considered frozen only after its parameters are explicitly
        # marked requires_grad=False. This keeps the MLP proxy trainable during
        # warm-up even when --freeze_proxy means "freeze after warm-up".
        proxy_is_frozen = all(
            not parameter.requires_grad
            for parameter in self.proxy_predictor.parameters()
        )
        if proxy_is_frozen:
            self.proxy_predictor.eval()
        # Any switch back to train mode means parameters may change afterwards,
        # so cached eval-only drug views must be discarded.
        if mode:
            self._cached_drug_views = None
        return self

    def _ordered_homo_subgraph(self, graph, graph_type, ids):
        """
        Reorder the homogeneous causal subgraph so that graph.nodes()[i] matches
        the i-th embedding in ids. This avoids node-feature mismatch in NetworkX.
        """
        node_names = [f"{graph_type}_{int(i)}" for i in ids]
        node_set = set(node_names)
        ordered_graph = graph.__class__()
        ordered_graph.add_nodes_from(node_names)

        if graph is not None:
            for u, v, data in graph.edges(data=True):
                if u in node_set and v in node_set:
                    ordered_graph.add_edge(u, v, **data)

        return ordered_graph

    def _cross_visit_attention_weights(self, visit_embs):
        """Return normalized current-to-visit relevance weights only.

        ``visit_embs[-1]`` is the terminal state of the forward GRU and is the
        trajectory representation used by both the MLP proxy and the downstream
        backbone. Attention is retained solely as a cheap *screening/routing*
        signal; it is intentionally not used to create a second observational
        patient representation.
        """
        if visit_embs.ndim != 2 or visit_embs.size(0) == 0:
            raise ValueError(
                "visit_embs must have shape [seq_len, hidden_dim] with seq_len > 0, "
                f"got {tuple(visit_embs.shape)}."
            )
        weights, _ = self.calc_cross_visit_scores(visit_embs)
        return weights

    def _compute_drug_views(self):
        """Compute molecular, EHR-enhanced, and DDI-enhanced drug embeddings.

        During evaluation / memory refresh the model parameters are fixed, so this
        cache avoids recomputing the same drug graph views for every visit. In
        training mode the cache is never reused because parameters change after
        each optimizer step.
        """
        if not self.training and hasattr(self, "_cached_drug_views") and self._cached_drug_views is not None:
            return self._cached_drug_views

        base_emb = self.MPNN_emb

        ehr_emb = base_emb
        for _ in range(self.gcn_layers):
            ehr_emb = F.leaky_relu(self.ehr_gcn(ehr_emb, self.ehr_adj))

        ddi_emb = base_emb
        for _ in range(self.gcn_layers):
            ddi_emb = F.leaky_relu(self.ddi_gcn(ddi_emb, self.ddi_adj))

        views = (base_emb, ehr_emb, ddi_emb)
        if not self.training:
            self._cached_drug_views = views
        else:
            self._cached_drug_views = None
        return views

    def encode_proxy_training_representation(self, patient):
        """Representation used for proxy warm-up without invoking the proxy itself."""
        patient_rep_obs = self.encode_patient_observational_representation(patient)
        return patient_rep_obs


    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def calc_cross_visit_scores(self, embedding, mask=None):
        """
        embedding: (seq, 3 * emb) 
        mask: (1, seq)
        """

        # Extract the current att value when calculating attention
        diag_keys = embedding[:, :] # (seq, 3 * dim) key: past visits and current visit 
        diag_query = embedding[-1:,:] # (1, 3 * dim) query: current visit
        diag_scores = torch.mm(self.linear_layer(diag_query), diag_keys.transpose(0, 1)) / math.sqrt(
            diag_query.size(-1))  # (1, seq) attention weight
        diag_scores_encoder = diag_scores

        if mask is not None:
            diag_scores = diag_scores.masked_fill(mask == 0, -1e9)
            diag_scores_encoder = diag_scores_encoder.masked_fill(mask == 0, -1e9)

        scores = F.softmax(diag_scores / self.att_tau, dim=-1)
        scores_encoder = F.softmax(diag_scores_encoder / self.att_tau, dim=-1)
        return scores, scores_encoder

    def _safe_sum_embedding(self, embedding_layer, ids, apply_dropout=True):
        """Sum a code set into [1, 1, emb_dim] while preserving device/dtype."""
        if ids is None or len(ids) == 0:
            return embedding_layer.weight.new_zeros(1, 1, self.emb_dim)

        ids_tensor = torch.as_tensor(
            ids, dtype=torch.long, device=embedding_layer.weight.device
        ).reshape(-1)
        emb = embedding_layer(ids_tensor).unsqueeze(0)
        if apply_dropout:
            emb = self.dropout(emb)
        return emb.sum(dim=1, keepdim=True)

    def _encode_entity_set(
        self,
        entity_type,
        graph_id,
        ids,
        graph_type=None,
        apply_dropout=True,
    ):
        """Encode one diagnosis/procedure/medication set."""
        embedding_layer = self.embeddings[entity_type]
        if ids is None or len(ids) == 0:
            return embedding_layer.weight.new_zeros(1, 1, self.emb_dim)

        ids_tensor = torch.as_tensor(
            ids, dtype=torch.long, device=embedding_layer.weight.device
        ).reshape(1, -1)
        emb = embedding_layer(ids_tensor)
        if apply_dropout:
            emb = self.dropout(emb)

        if self.use_causal_graph:
            graph = self.causal_graph.get_graph(graph_id, graph_type)
            graph = self._ordered_homo_subgraph(graph, graph_type, ids)
            emb = self.homo_graph[entity_type](graph, emb)

        return emb.sum(dim=1, keepdim=True)

    def _encode_visit_sequences(self, patient, apply_dropout=True, include_raw=True):
        """Encode one trajectory into raw and causal visit-level GRU states.

        Args:
            patient: chronological admissions. The final item is the current visit.
            apply_dropout: whether to apply entity-input dropout before the GRUs.
            include_raw: whether to also compute the raw-GRU branch. Screening and
                strict MLP counterfactuals only need causal states, so setting this
                to ``False`` avoids redundant raw embedding/GRU computation.

        The previous-medication input is rebuilt from the supplied trajectory. As
        a result, physically deleting a visit also removes its medication from the
        next retained visit, which is required for strict Top-K interventions.
        """
        if patient is None or len(patient) == 0:
            raise ValueError("patient trajectory must contain at least one admission.")

        if include_raw:
            raw_diag_seq, raw_proc_seq, raw_med_seq = [], [], []
        causal_diag_seq, causal_proc_seq, causal_med_seq = [], [], []

        for adm_id, adm in enumerate(patient):
            if len(adm) < 4:
                raise ValueError(
                    f"Admission {adm_id} must contain [diags, procs, meds, graph_id]."
                )

            diags, procs = adm[0], adm[1]
            if include_raw:
                raw_diag_seq.append(
                    self._safe_sum_embedding(self.embeddings[0], diags, apply_dropout)
                )
                raw_proc_seq.append(
                    self._safe_sum_embedding(self.embeddings[1], procs, apply_dropout)
                )

            causal_diag_seq.append(
                self._encode_entity_set(0, adm[3], diags, "Diag", apply_dropout)
            )
            causal_proc_seq.append(
                self._encode_entity_set(1, adm[3], procs, "Proc", apply_dropout)
            )

            if adm_id == 0:
                causal_med = self.embeddings[2].weight.new_zeros(
                    1, 1, self.emb_dim
                )
                if include_raw:
                    raw_med = causal_med
            else:
                previous_adm = patient[adm_id - 1]
                previous_meds = previous_adm[2]
                if include_raw:
                    raw_med = self._safe_sum_embedding(
                        self.embeddings[2], previous_meds, apply_dropout
                    )
                causal_med = self._encode_entity_set(
                    2,
                    previous_adm[3],
                    previous_meds,
                    "Med",
                    apply_dropout,
                )

            if include_raw:
                raw_med_seq.append(raw_med)
            causal_med_seq.append(causal_med)

        if include_raw:
            raw_inputs = [
                torch.cat(raw_diag_seq, dim=1),
                torch.cat(raw_proc_seq, dim=1),
                torch.cat(raw_med_seq, dim=1),
            ]
            raw_outputs = [
                encoder(seq)[0]
                for encoder, seq in zip(self.raw_encoders, raw_inputs)
            ]
            raw_visit_embs = torch.cat(raw_outputs, dim=-1).squeeze(0)
        else:
            raw_visit_embs = None

        causal_inputs = [
            torch.cat(causal_diag_seq, dim=1),
            torch.cat(causal_proc_seq, dim=1),
            torch.cat(causal_med_seq, dim=1),
        ]
        causal_outputs = [
            encoder(seq)[0]
            for encoder, seq in zip(self.causal_encoders, causal_inputs)
        ]
        causal_visit_embs = torch.cat(causal_outputs, dim=-1).squeeze(0)
        return raw_visit_embs, causal_visit_embs

    def _effective_history_topk(self, history_len):
        if history_len <= 0:
            return 0
        if self.history_topk <= 0:
            return history_len
        return min(self.history_topk, history_len)

    @staticmethod
    def _compact_patient(patient, selected_history_indices):
        """Keep selected historical admissions in chronological order plus current."""
        history_len = max(len(patient) - 1, 0)
        selected = sorted(
            {int(i) for i in selected_history_indices if 0 <= int(i) < history_len}
        )
        return [patient[i] for i in selected] + [patient[-1]], selected

    def _select_history_indices_from_scores(self, scores, history_len):
        """Return chronological Top-K original history indices."""
        k = self._effective_history_topk(history_len)
        if k == 0:
            return []
        scores = scores.reshape(-1)[:history_len]
        if scores.numel() != history_len:
            raise ValueError(
                f"history score length must be {history_len}, got {scores.numel()}."
            )
        top_indices = torch.topk(scores, k=k, largest=True).indices.tolist()
        return sorted(int(i) for i in top_indices)

    def _attention_screen_history(self, patient):
        """Lightweight relevance-only Top-K screening for an MLP proxy.

        The full trajectory is encoded exactly once, without stochastic input
        dropout. The current visit attends to the historical GRU states, and the
        resulting relevance weights are used only to choose Top-K candidates.
        No patient-level attention pooling is constructed here.
        """
        _, full_causal_visit_embs = self._encode_visit_sequences(
            patient, apply_dropout=False, include_raw=False
        )
        history_len = max(full_causal_visit_embs.size(0) - 1, 0)
        if history_len == 0:
            return [], full_causal_visit_embs.new_empty(0)

        history_scores = self._cross_visit_attention_weights(full_causal_visit_embs).squeeze(0)[:history_len]

        selected = self._select_history_indices_from_scores(history_scores, history_len)
        return selected, history_scores

    def encode_patient_observational_representation(self, patient):
        """Encode the screened compact trajectory for proxy warm-up.

        This method deliberately stops after light Top-K screening. Warm-up is
        not an attribution step, so it must not trigger strict counterfactual
        re-encoding.
        """
        with torch.no_grad():
            selected_indices, _ = self._attention_screen_history(patient)
        compact_patient, _ = self._compact_patient(patient, selected_indices)
        _, compact_causal_visit_embs = self._encode_visit_sequences(
            compact_patient, apply_dropout=False, include_raw=False
        )
        return compact_causal_visit_embs[-1:, :]

    @contextmanager
    def _deterministic_proxy(self):
        """Temporarily disable proxy dropout during attribution computation."""
        was_training = self.proxy_predictor.training
        self.proxy_predictor.eval()
        try:
            yield
        finally:
            proxy_is_frozen = all(
                not parameter.requires_grad
                for parameter in self.proxy_predictor.parameters()
            )
            if was_training and not proxy_is_frozen:
                self.proxy_predictor.train(True)

    def _proxy_logits(self, patient=None, h_obs=None, visit_mask=None):
        """Return proxy logits while hiding whether the proxy is MLP, GAMENet, or SafeDrug."""
        if self.proxy_model_type in ('gamenet', 'safedrug', 'armr'):
            if patient is None:
                raise ValueError(f"{self.proxy_model_type} proxy requires the raw patient visit sequence.")
            return self.proxy_predictor(patient, visit_mask=visit_mask)

        if h_obs is None:
            raise ValueError("MLP ProxyModel requires h_obs representation.")
        return self.proxy_predictor(h_obs)

    def _proxy_probability(self, patient, causal_visit_embs=None):
        """Deterministic proxy probability for one compact trajectory."""
        if self.proxy_model_type in ("gamenet", "safedrug", "armr"):
            logits = self._proxy_logits(patient=patient)
        else:
            if causal_visit_embs is None:
                _, causal_visit_embs = self._encode_visit_sequences(
                    patient, apply_dropout=False, include_raw=False
                )
            # The terminal forward-GRU state summarizes the compact trajectory.
            # It is the same observational representation used by the main path.
            h_obs = causal_visit_embs[-1:, :]
            logits = self._proxy_logits(h_obs=h_obs)
        return torch.sigmoid(logits)

    def _approximate_history_screen(self, patient):
        """Run exactly one lightweight full-history screening pass.

        * MLP proxy: rank histories by current-to-history relevance in the
          causal GRU state space. This is intentionally only a candidate filter.
        * Sequence proxy: use the adapter's batched visit-mask pass to obtain a
          cheap approximate effect for every historical visit.

        Strict counterfactual effects are never computed here; they are reserved
        for the selected compact Top-K trajectory.
        """
        history_len = max(len(patient) - 1, 0)
        reference = self.embeddings[0].weight
        if history_len == 0:
            return (
                [],
                reference.new_empty(0, self.med_vocab_size),
                reference.new_empty(0),
            )

        if (
            self.proxy_model_type in ("gamenet", "safedrug", "armr")
            and hasattr(self.proxy_predictor, "counterfactual_logits")
        ):
            with torch.no_grad(), self._deterministic_proxy():
                base_mask = torch.ones(
                    len(patient), dtype=torch.bool, device=reference.device
                )
                logits = self.proxy_predictor.counterfactual_logits(
                    patient,
                    base_mask=base_mask,
                    candidate_indices=list(range(history_len)),
                )
                probs = torch.sigmoid(logits)
                approximate_ce = probs[:1] - probs[1:]
                screen_scores = approximate_ce.abs().mean(dim=-1)

            invalid = ~torch.isfinite(screen_scores)
            if bool(invalid.any().item()) or float(screen_scores.abs().max().item()) <= 1e-12:
                # Fall back to the same deterministic relevance screen used by
                # the MLP proxy; this also keeps selection well-defined when a
                # frozen sequence proxy is uninformative early in training.
                _, fallback_scores = self._attention_screen_history(patient)
                screen_scores = torch.where(
                    invalid,
                    fallback_scores.to(screen_scores.device),
                    screen_scores,
                )
                if float(screen_scores.abs().max().item()) <= 1e-12:
                    screen_scores = fallback_scores.to(screen_scores.device)

            selected_indices = self._select_history_indices_from_scores(
                screen_scores, history_len
            )
            return selected_indices, approximate_ce, screen_scores

        # An MLP proxy receives the terminal causal GRU state during exact
        # attribution. Its lightweight screen therefore uses the same state
        # space, but without pretending that it is already a counterfactual.
        selected_indices, attention_scores = self._attention_screen_history(patient)
        approximate_ce = reference.new_zeros(history_len, self.med_vocab_size)
        return selected_indices, approximate_ce, attention_scores

    def _exact_topk_counterfactual_effects(
        self,
        compact_patient,
        deterministic_causal_visit_embs=None,
    ):
        """Physically delete one selected history at a time and re-run only Top-K.

        ``deterministic_causal_visit_embs`` is the already-computed compact
        observational trajectory. Reusing it removes the redundant reference
        encoding that previously occurred before every strict attribution pass.
        Each counterfactual trajectory still must be re-encoded because removing
        a visit changes both temporal GRU states and previous-medication inputs.
        """
        history_len = max(len(compact_patient) - 1, 0)
        if history_len == 0:
            reference = self.embeddings[0].weight
            return reference.new_empty(0, self.med_vocab_size)

        with torch.no_grad(), self._deterministic_proxy():
            if self.proxy_model_type in ("gamenet", "safedrug", "armr"):
                y_obs = self._proxy_probability(compact_patient)
            else:
                if deterministic_causal_visit_embs is None:
                    _, deterministic_causal_visit_embs = self._encode_visit_sequences(
                        compact_patient, apply_dropout=False, include_raw=False
                    )
                y_obs = self._proxy_probability(
                    compact_patient,
                    causal_visit_embs=deterministic_causal_visit_embs,
                )

            effects = []
            for local_history_index in range(history_len):
                patient_do = (
                    list(compact_patient[:local_history_index])
                    + list(compact_patient[local_history_index + 1 :])
                )
                if self.proxy_model_type in ("gamenet", "safedrug", "armr"):
                    y_do = self._proxy_probability(patient_do)
                else:
                    _, do_causal_visit_embs = self._encode_visit_sequences(
                        patient_do, apply_dropout=False, include_raw=False
                    )
                    y_do = self._proxy_probability(
                        patient_do,
                        causal_visit_embs=do_causal_visit_embs,
                    )
                effects.append(y_obs - y_do)

        return torch.cat(effects, dim=0)

    def encode_patient_trajectory(self, patient):
        """Encode a trajectory with light screening followed by strict Top-K routing.

        The representation contract is deliberately simple:

        1. Screen the *full* history once with a lightweight mechanism.
        2. Build the compact sequence: selected historical visits + current visit.
        3. Encode the compact sequence once. Its terminal GRU states are
           ``h_raw`` and ``h_obs``; no second attention-pooled ``full_obs`` is
           created.
        4. Physically remove only selected histories to obtain strict effects.
        5. Use those effects to form ``h_causal``.

        The compact encoder is deterministic at the entity-input level. This is
        important because the observed compact representation and every strict
        counterfactual must live in the same representation space. Dropout is
        still present in the downstream prediction/routing layers.
        """
        if patient is None or len(patient) == 0:
            raise ValueError("patient trajectory must contain at least one admission.")

        if self.use_causal_routing:
            selected_indices, approximate_ce, screen_scores = self._approximate_history_screen(patient)
        else:
            with torch.no_grad():
                selected_indices, screen_scores = self._attention_screen_history(patient)
            approximate_ce = self.embeddings[0].weight.new_zeros(
                max(len(patient) - 1, 0), self.med_vocab_size
            )

        compact_patient, selected_indices = self._compact_patient(patient, selected_indices)

        # This is the only differentiable trajectory encoding on the main path.
        # The final state of each forward GRU already summarizes all retained
        # visits, so an additional cross-visit pooled patient vector is redundant.
        raw_visit_embs, causal_visit_embs = self._encode_visit_sequences(compact_patient, apply_dropout=False)

        h_raw = raw_visit_embs[-1:, :]
        h_obs = causal_visit_embs[-1:, :]
        selected_history_len = len(selected_indices)

        if not self.use_causal_routing:
            compact_attention = self._cross_visit_attention_weights(causal_visit_embs)
            if selected_history_len > 0:
                history_weights = compact_attention.squeeze(0)[:selected_history_len]
                history_weights = history_weights / history_weights.sum().clamp_min(1e-9)
            else:
                history_weights = causal_visit_embs.new_empty(0)

            a_hist = causal_visit_embs.new_tensor(
                [self.no_causal_routing_a_hist if selected_history_len > 0 else 0.0]
            )
            return {
                "h_raw": h_raw,
                "h_obs": h_obs,
                "h_causal": h_obs,
                "a_k": history_weights,
                "a_hist": a_hist,
                "CE_tensor": causal_visit_embs.new_zeros(
                    selected_history_len, self.med_vocab_size
                ),
                "approx_CE_tensor": approximate_ce,
                "approx_history_importance": screen_scores,
                "screen_history_scores": screen_scores,
                "visit_embs": causal_visit_embs,
                "selected_history_indices": selected_indices,
                "selected_history_visits": compact_patient[:-1],
                "compact_patient": compact_patient,
            }

        exact_ce = self._exact_topk_counterfactual_effects(
            compact_patient,
            deterministic_causal_visit_embs=causal_visit_embs,
        )

        if selected_history_len > 0:
            ce_magnitude = exact_ce.detach().abs()
            routing_scores = self.routing_mlp(ce_magnitude).squeeze(-1)
            a_k = F.softmax(routing_scores, dim=0)
            h_history = torch.sum(
                a_k.unsqueeze(-1) * causal_visit_embs[:selected_history_len],
                dim=0,
                keepdim=True,
            )

            # Mean prevents a_hist from becoming a proxy for trajectory length.
            total_ce = ce_magnitude.mean(dim=0)
            a_hist = self.hist_contribution_mlp(total_ce).view(1)
            h_causal = (1.0 - a_hist) * h_obs + a_hist * h_history
        else:
            a_k = causal_visit_embs.new_empty(0)
            a_hist = causal_visit_embs.new_zeros(1)
            h_causal = h_obs

        return {
            "h_raw": h_raw,  # raw_visit_embs[-1:]
            "h_obs": h_obs,  # causal_visit_embs[-1:]
            "h_causal": h_causal, # counterfactual + causal_visit_embs
            "a_k": a_k,
            "a_hist": a_hist,
            "CE_tensor": exact_ce,
            "approx_CE_tensor": approximate_ce,
            "approx_history_importance": screen_scores,
            "screen_history_scores": screen_scores,
            "visit_embs": causal_visit_embs,
            "selected_history_indices": selected_indices,
            "selected_history_visits": compact_patient[:-1],
            "compact_patient": compact_patient,
        }

    def aggregate_retrieved_meds(self, retrieved_meds, weights):
        """
        retrieved_meds: List[List[int]]
        weights: [K]
        return: [1, emb_dim]
        """
        med_vectors = []

        for med_list in retrieved_meds:
            if len(med_list) == 0:
                med_vec = torch.zeros(self.emb_dim).to(self.device)
            else:
                med_ids = torch.LongTensor(med_list).to(self.device)
                med_vec = self.embeddings[2](med_ids).mean(dim=0)

            med_vectors.append(med_vec)

        med_vectors = torch.stack(med_vectors, dim=0)  # [K, emb_dim]
        weights = weights.to(self.device).view(1, -1)  # [1, K]
        weights = torch.clamp(weights, min=0.0)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-9)

        sim_med = torch.mm(weights, med_vectors)       # [1, emb_dim]
        return sim_med

    def get_patient_representation(
        self,
        patient,
        memory_bank=None,
        patient_id=None,
        global_id=None,
        epoch=0,
        warmup_epochs=5,
        topk=10
    ):
        """
        阶段二 + 阶段三：
        1. 先得到 h_raw, h_causal, a_hist
        2. 再从 causal memory bank 中检索全局经验
        3. 最后得到 h_final
        """

        enc_info = self.encode_patient_trajectory(patient)

        h_raw = enc_info["h_raw"]          # [1, 3 * emb_dim]
        h_causal = enc_info["h_causal"]    # [1, 3 * emb_dim]
        a_hist = enc_info["a_hist"]        # [1]

        # epoch warm-up 系数
        epoch_gate = min(1.0, float(epoch) / float(max(1, warmup_epochs)))

        # raw -> causal
        rho = epoch_gate * float(a_hist.detach().cpu().item())

        if (not self.use_memory_bank) or memory_bank is None:
            sim_visit = torch.zeros_like(h_causal)
            sim_med = torch.zeros(1, self.emb_dim).to(self.device)
            retrieved_info = None
        else:
            retrieved_info = memory_bank.search(
                q_raw=h_raw,
                q_causal=h_causal,
                rho=rho,
                query_patient_id=patient_id,
                query_global_id=global_id,
                topk=topk
            )

            if retrieved_info is None:
                sim_visit = torch.zeros_like(h_causal)
                sim_med = torch.zeros(1, self.emb_dim).to(self.device)
            else:
                retrieved_values = retrieved_info["values"].unsqueeze(0) # [1, K, 3 * emb_dim]

                sim_visit = self.cross_att(
                    h_causal.unsqueeze(1),     # [1, 1, 3 * emb_dim]
                    retrieved_values,          # [1, K, 3 * emb_dim]
                    retrieved_values           # [1, K, 3 * emb_dim]
                ).squeeze(1)                   # [1, 3 * emb_dim]

                sim_med = self.aggregate_retrieved_meds(
                    retrieved_info["meds"],
                    retrieved_info["weights"]
                )                              # [1, emb_dim]

                # control strength
                sim_visit = sim_visit * (1.0 - a_hist)
                sim_med = sim_med * (1.0 - a_hist)

        # patient_emb = torch.cat([h_causal, sim_visit, sim_med], dim=-1)  # [1, 7 * emb_dim]

        # patient_final_repr = self.query(patient_emb)  # [1, emb_dim]

        h_obs = enc_info["h_obs"]                                      # [1, 3 * emb_dim]
        base_repr = self.base_query(h_obs)                                # [1, emb_dim]

        causal_input = torch.cat([h_causal, sim_visit, sim_med], dim=-1)
        causal_delta = self.causal_adapter(causal_input)

        epoch_gate = min(1.0, float(epoch) / float(max(1, warmup_epochs)))
        causal_gate = torch.sigmoid(self.raw_causal_gate) * epoch_gate

        patient_final_repr = base_repr + causal_gate * causal_delta

        return patient_final_repr, h_causal, {
            "a_hist": a_hist,
            "a_k": enc_info["a_k"],
            "retrieved": retrieved_info,
            "rho": rho,
            "h_raw": h_raw,
            "h_causal": h_causal,
            "selected_history_indices": enc_info["selected_history_indices"],
            "selected_history_visits": enc_info["selected_history_visits"],
            "compact_patient": enc_info["compact_patient"],
            "CE_tensor": enc_info["CE_tensor"],
            "approx_history_importance": enc_info["approx_history_importance"],
        }

    def forward(self, patient, memory_bank=None, patient_id=None, global_id=None, epoch=0, warmup_epochs=5, topk=10):

        query_final_repr, _, aux_info = self.get_patient_representation(
            patient,
            memory_bank=memory_bank,
            patient_id=patient_id,
            global_id=global_id,
            epoch=epoch,
            warmup_epochs=warmup_epochs,
            topk=topk
        )

        retrieved_info = aux_info["retrieved"]

        base_emb, ehr_emb, ddi_emb = self._compute_drug_views()

        # fusing molecular / EHR / negative-DDI views
        enhanced_emb, view_importance = self.fusion_layer(query_final_repr, base_emb, ehr_emb, ddi_emb)

        # MPNN embedding
        MPNN_match = torch.sigmoid(torch.mm(query_final_repr, enhanced_emb.t()))
        MPNN_att = self.MPNN_layernorm(MPNN_match + self.MPNN_output(MPNN_match))

        # local embedding
        bipartite_emb = self.bipartite_output(
            torch.sigmoid(self.bipartite_transform(query_final_repr)), self.tensor_ddi_mask_H.t()
        )

        result = torch.mul(bipartite_emb, MPNN_att)

        # CTCR: enabled only when use_causal_graph=True and use_ctcr=True.
        # If disabled, this becomes an identity mapping and review_reg is zero.
        if self.causal_review is not None:
            global_diags = None if retrieved_info is None else retrieved_info.get('diags')
            global_procs = None if retrieved_info is None else retrieved_info.get('procs')
            global_meds = None if retrieved_info is None else retrieved_info.get('meds')
            global_weights = None if retrieved_info is None else retrieved_info.get('weights')
            selected_history_visits = aux_info["selected_history_visits"]
            calibration_progress = min(
                1.0,
                max(
                    0.0,
                    float(epoch - self.ctcr_start_epoch)
                    / float(max(1, self.ctcr_ramp_epochs)),
                ),
            )
            reviewed_result, review_aux = self.causal_review(
                logits=result,
                cur_diags=patient[-1][0],
                cur_procs=patient[-1][1],
                history_diags=[adm[0] for adm in selected_history_visits],
                history_procs=[adm[1] for adm in selected_history_visits],
                route_weights=aux_info["a_k"],
                history_indices=aux_info["selected_history_indices"],
                a_hist=aux_info["a_hist"],
                global_diags=global_diags,
                global_procs=global_procs,
                global_meds=global_meds,
                global_weights=global_weights,
                ce_tensor=aux_info["CE_tensor"],
                calibration_progress=calibration_progress,
            )

            final_result = reviewed_result

            # raw_delta = reviewed_result - result.detach()
            # raw_delta = torch.nan_to_num(raw_delta, nan=0.0, posinf=0.0, neginf=0.0)
            # raw_delta = torch.clamp(
            #     raw_delta,
            #     min=-self.ctcr_delta_clip,
            #     max=self.ctcr_delta_clip
            # )

            # epoch_gate = min(
            #     1.0,
            #     max(0.0, float(epoch - self.ctcr_start_epoch) / float(self.ctcr_ramp_epochs))
            # )

            # ctcr_gate = self.ctcr_gate_cap * torch.sigmoid(self.raw_ctcr_gate) * epoch_gate

            # final_result = result + ctcr_gate * raw_delta

            # # 正则项必须约束“实际加入 final logits 的变化”，而不是约束未乘 gate 的 reviewed_logits
            # review_reg = torch.mean((final_result - result) ** 2)
        else:
            final_result = result
            # review_reg = result.new_tensor(0.0)

        neg_pred_prob = torch.sigmoid(final_result)
        neg_pred_pair_prob = neg_pred_prob.t() * neg_pred_prob  # [num_med, num_med]
        ddi_norm = self.tensor_ddi_adj.sum().clamp_min(1.0)
        batch_neg = self.ddi_loss_weight * neg_pred_pair_prob.mul(self.tensor_ddi_adj).sum() / ddi_norm

        if self.causal_review is not None:
            if "review_aux" in locals() and "g" in review_aux:
                effective_delta = final_result - result.detach()
                trust_g = review_aux["g"].to(final_result.device, final_result.dtype).view_as(effective_delta)
                review_reg = torch.mean((1.0 - trust_g) * effective_delta.pow(2))
            else:
                review_reg = torch.mean((final_result - result.detach()) ** 2)
        else:
            review_reg = result.new_tensor(0.0)

        return final_result, batch_neg, review_reg
