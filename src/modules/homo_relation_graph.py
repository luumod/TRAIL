import networkx as nx
import torch
import torch.nn as nn

class LearnableMaskLayer(nn.Module):
    def __init__(self, emb_dim):
        super(LearnableMaskLayer, self).__init__()
        # 创建一个与输入相同形状的权重，初始值设为1
        self.mask_weights = nn.Parameter(torch.ones(emb_dim))

    def forward(self, x):
        # 将输入与mask权重相乘
        return x * self.mask_weights


class CausalWeight(nn.Module):
    def __init__(self, emb_dim, device):
        super(CausalWeight, self).__init__()
        self.device = device
        self.list_weights = nn.ModuleList([LearnableMaskLayer(emb_dim) for _ in range(4)])

    def forward(self, x, causal_graph):
        # 遍历每个列表
        echelon = self.node_classify(causal_graph)
        x1 = torch.zeros_like(x)
        for i, node_list in enumerate(echelon):
            for node in node_list:
                x1[0, node, :] += self.list_weights[i](x[0, node, :])
        return x1

    def node_classify(self, causal_graph):
        """初始化四个类别的列表 从因到果
        0无入度有出度 因节点
        1无入度无出度 孤儿节点
        2有入度有出度 中间节点
        3有入度无出度 果节点"""

        # 根据node_type初始化列表
        echelon = [[], [], [], []]

        # 对每个节点进行分类
        for node in causal_graph.nodes():
            in_degree = causal_graph.in_degree(node)
            out_degree = causal_graph.out_degree(node)

            if in_degree == 0 and out_degree == 0:
                echelon[1].append(node)
            elif in_degree > 0 and out_degree == 0:
                echelon[3].append(node)
            elif in_degree == 0 and out_degree > 0:
                echelon[0].append(node)
            else:
                echelon[2].append(node)

        return echelon


class SignedDualChannelLayer(nn.Module):
    def __init__(self, emb_dim):
        super(SignedDualChannelLayer, self).__init__()
        self.self_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.pos_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.neg_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.act = nn.ReLU()

    def _aggregate(self, h, edge_index, alpha, mask):
        out = h.new_zeros(h.shape)
        if edge_index.numel() == 0 or not torch.any(mask):
            return out

        src = edge_index[0, mask]
        dst = edge_index[1, mask]
        msg = h[src] * alpha[mask].unsqueeze(-1)
        out.index_add_(0, dst, msg)
        return out

    def forward(self, h_pos, h_neg, edge_index, edge_weight, alpha):
        pos_mask = edge_weight > 0
        neg_mask = edge_weight < 0

        h_pos_next = self.self_proj(h_pos)
        h_pos_next = h_pos_next + self.pos_proj(self._aggregate(h_pos, edge_index, alpha, pos_mask))
        h_pos_next = h_pos_next + self.neg_proj(self._aggregate(h_neg, edge_index, alpha, neg_mask))

        h_neg_next = self.self_proj(h_neg)
        h_neg_next = h_neg_next + self.pos_proj(self._aggregate(h_neg, edge_index, alpha, pos_mask))
        h_neg_next = h_neg_next + self.neg_proj(self._aggregate(h_pos, edge_index, alpha, neg_mask))

        return self.act(h_pos_next), self.act(h_neg_next)


class homo_relation_graph(nn.Module):
    # 用因果推断来去偏，用图卷积来增强表达
    def __init__(self, emb_dim, device):
        super(homo_relation_graph, self).__init__()
        self.device = device
        self.causal_weight = CausalWeight(emb_dim, device)

        self.in_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.layers = nn.ModuleList([SignedDualChannelLayer(emb_dim) for _ in range(2)])
        self.res_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.norm = nn.LayerNorm(emb_dim)
        self.eps = 1e-8

    def forward(self, graph, node_features):
        x, edge_index, edge_weight = self.nx_to_pyg(graph, node_features)

        alpha = self.normalize_edge_weight(edge_index, edge_weight, x.size(0))
        h_pos = self.in_proj(x)
        h_neg = torch.zeros_like(h_pos)
        for layer in self.layers:
            h_pos, h_neg = layer(h_pos, h_neg, edge_index, edge_weight, alpha)

        x_gnn = self.norm(self.res_proj(x) + h_pos - h_neg)
        return x_gnn.unsqueeze(0) # [1, num_nodes, emb_dim]

    def normalize_edge_weight(self, edge_index, edge_weight, num_nodes):
        if edge_weight.numel() == 0:
            return edge_weight

        dst = edge_index[1]
        abs_weight = edge_weight.abs()
        denom = edge_weight.new_zeros(num_nodes)
        denom.index_add_(0, dst, abs_weight)
        return abs_weight / (denom[dst] + self.eps)

    def nx_to_pyg(self, graph, node_features):
        # 将字符串节点转换为整数节点
        mapping = {node: i for i, node in enumerate(graph.nodes())}
        graph = nx.relabel_nodes(graph, mapping)

        # 根据因果性重新分配各个节点权重
        node_features_weighted = self.causal_weight(node_features, graph)
        x = node_features_weighted.squeeze(0) # [num_nodes, emb_dim]

        # 边list
        edge_list = list(graph.edges(data=True))

        # 获得边
        if len(edge_list) > 0:
            edge_index = torch.tensor([[e[0] for e in edge_list], 
                                     [e[1] for e in edge_list]], 
                                    device=self.device, dtype=torch.long)
            # 提取建图时注入的 'weight' 属性
            edge_weight = torch.tensor([e[2]['weight'] for e in edge_list], 
                                     device=self.device, dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), device=self.device, dtype=torch.long)
            edge_weight = torch.empty((0,), device=self.device, dtype=torch.float)

        return x, edge_index, edge_weight


if __name__ == "__main__":
    # 创建一个简单的有向图
    G = nx.DiGraph()
    G.add_edge('A', 'B', weight=0.5)
    G.add_edge('B', 'C', weight=-0.3)
    G.add_edge('A', 'C', weight=0)

    # 创建节点特征 (假设每个节点的特征维度为64)
    node_features = torch.rand(1, 3, 64)  # (1, num_nodes, emb_dim)

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = homo_relation_graph(emb_dim=64, device=device).to(device)

    d = torch.arange(3, device=device)  # (1, 3, 64)
    emb = nn.Embedding(3, 64).to(device)

    node_features = emb(d)

    repr = model(graph=G, node_features=node_features)
