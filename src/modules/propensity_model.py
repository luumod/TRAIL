import torch
import torch.nn as nn

class TargetAwareAttention(nn.Module):
    """
    目标感知交叉注意力模块 (Target-aware Cross Attention)
    允许每个候选目标实体动态地从患者完整的历史序列中检索相关信息。
    """
    def __init__(self, emb_dim, num_heads=4, dropout=0.3):
        super(TargetAwareAttention, self).__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, target_query, patient_seq):
        """
        target_query: [1, num_targets, emb_dim] (作为 Query)
        patient_seq: [1, seq_len, emb_dim] (作为 Key 和 Value)
        """
        # attn_output: [1, num_targets, emb_dim] 包含了从病史中提取的专属上下文
        # attn_weights: [1, num_targets, seq_len] 记录了每个实体对每次就诊的注意力分数
        attn_output, attn_weights = self.cross_attn(
            query=target_query, 
            key=patient_seq, 
            value=patient_seq
        )
        
        enriched_targets = self.layer_norm(target_query + self.dropout(attn_output))
        
        return enriched_targets, attn_weights

class PropensityEstimator(nn.Module):
    def __init__(
        self,
        input_vocab_sizes,  # [diag_size, proc_size]
        target_vocab_size,  # diag_size, proc_size 或 med_size
        emb_dim=64,
        use_multimodal=False,
        device=torch.device("cuda:0"),
    ):
        super(PropensityEstimator, self).__init__()
        self.device = device
        self.emb_dim = emb_dim
        self.use_multimodal = use_multimodal
        
        # 1. 结构化特征编码器 (ID Embeddings)
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size, emb_dim) for vocab_size in input_vocab_sizes]
        )

        self.target_embedding = nn.Embedding(target_vocab_size, emb_dim)
        self.dropout = nn.Dropout(p=0.5)

        if self.use_multimodal:
            self.text_project = nn.Sequential(
                nn.Linear(768, emb_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ).to(device)

        # 3. 动态时间序列编码器 (增加了一个处理文本的 GRU)
        # index 0: Diag GRU, index 1: Proc GRU, index 2: Text GRU
        self.encoders = nn.ModuleList(
            [nn.GRU(emb_dim, emb_dim * 2, batch_first=True) for _ in range(4)] 
        )

        # 4. 多模态融合 Query 网络
        # 输入维度 = (Diag_emb*2) + (Proc_emb*2) + (Text_emb*2) = emb_dim * 6
        if self.use_multimodal:
            self.query = nn.Sequential(
                nn.Linear(emb_dim * 8, emb_dim * 2),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(emb_dim * 2, emb_dim),
            )
        else:
            self.query = nn.Sequential(
                nn.Linear(emb_dim * 6, emb_dim * 2),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(emb_dim * 2, emb_dim),
            )
            

        self.target_aware_attn = TargetAwareAttention(emb_dim=emb_dim, num_heads=4)

        # 5. 因果分数预测网络
        self.output = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(emb_dim, 1),
        )

        self.init_weights()

    def forward(self, seq_input, target_ids, es_type='d'):
        """
        seq_input: 患者截止到 t-1 的丰富历史就诊记录列表
        target_ids: tensor, 待预测的目标实体 ID 列表 (N,)
        """
        if len(seq_input) == 0:
            query = torch.zeros(1, self.emb_dim).to(self.device)
        else:
            i1_seq, i2_seq, i3_seq, text_seq = [], [], [], []

            def mean_embedding(embedding):
                return embedding.mean(dim=1).unsqueeze(dim=0)  # (1,1,dim)

            # 遍历历次就诊，分别提取 结构化 和 非结构化 序列
            for adm in seq_input:
                # 结构化
                i1 = mean_embedding(self.dropout(self.embeddings[0](
                    torch.LongTensor(adm[0]).unsqueeze(dim=0).to(self.device)
                ))) # (1, 1, dim)
                i2 = mean_embedding(self.dropout(self.embeddings[1](
                    torch.LongTensor(adm[1]).unsqueeze(dim=0).to(self.device)
                ))) # (1, 1, dim)
                i3 = mean_embedding(self.dropout(self.embeddings[2](
                    torch.LongTensor(adm[2]).unsqueeze(dim=0).to(self.device)
                ))) # (1, 1, dim)

                if adm == seq_input[-1]:  # 当前就诊跳过目标es_type对应实体
                    if es_type == 'd':
                        i1 = torch.zeros(1, 1, self.emb_dim).to(self.device)
                    elif es_type == 'p':
                        i2 = torch.zeros(1, 1, self.emb_dim).to(self.device)
                    elif es_type == 'm':
                        i3 = torch.zeros(1, 1, self.emb_dim).to(self.device)

                # 非结构化
                if self.use_multimodal:
                    raw_bert_emb = adm[-1].to(self.device).unsqueeze(0) # (1, 768)
                    i4 = self.text_project(raw_bert_emb).unsqueeze(0) # (1, 1, emb_dim)

                i1_seq.append(i1)
                i2_seq.append(i2)
                i3_seq.append(i3)
                if self.use_multimodal:
                    text_seq.append(i4)

            # 沿着序列维度拼接
            i1_seq = torch.cat(i1_seq, dim=1)  # (1, seq, dim)
            i2_seq = torch.cat(i2_seq, dim=1)  # (1, seq, dim)
            i3_seq = torch.cat(i3_seq, dim=1)  # (1, seq, dim)
            if self.use_multimodal:
                text_seq = torch.cat(text_seq, dim=1) # (1, seq, dim)

            # GRU 时序建模
            o1, h1 = self.encoders[0](i1_seq)  
            o2, h2 = self.encoders[1](i2_seq)
            o3, h3 = self.encoders[2](i3_seq)
            if self.use_multimodal:
                o4, h4 = self.encoders[3](text_seq)

            # 多模态表征拼接 (seq, dim*6)
            if self.use_multimodal:
                patient_representations = torch.cat([o1, o2, o3, o4], dim=-1).squeeze(dim=0) # (seq, dim * 8)
            else:
                patient_representations = torch.cat([o1, o2, o3], dim=-1).squeeze(dim=0) # (seq, dim * 6)
            
            # 降维与非线性映射
            queries = self.query(patient_representations)  # (seq, dim)
            patient_seq_embs = queries.unsqueeze(dim=0)  # (1, seq, dim)

        # 融入目标医疗实体 d 的特征
        target_embs = self.target_embedding(target_ids).unsqueeze(dim=0)  # (1, num_targets, dim)

        # enriched_targets: [1, num_targets, dim] 
        # attn_weights: [1, num_targets, seq_len]
        enriched_targets, attn_weights = self.target_aware_attn(
            target_query=target_embs, 
            patient_seq=patient_seq_embs
        )

        # 倾向分数预测
        logits = self.output(enriched_targets.squeeze(dim=0))  # (num_targets, 1)

        return logits.permute(1, 0) # (1, num_targets) logits
    
    def init_weights(self):
        """Initialize structured weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)
        self.target_embedding.weight.data.uniform_(-initrange, initrange)
