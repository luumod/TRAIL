import os
import torch
import dill
import copy
import networkx as nx
import pandas as pd
import statsmodels.api as sm
from cdt.causality.graph import GES
from dowhy import CausalModel
from tqdm import tqdm
from .propensity_model import PropensityEstimator
import os
import json
import time
import queue as py_queue
import traceback
import signal
import multiprocessing as mp
from datetime import datetime


PROJECT_ROOT = os.environ.get(
    "CSDG_PROJECT_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
)


def project_path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)


def _ges_predict_worker(visit_data, result_queue):
    """
    在独立子进程中执行 GES.predict。
    这样即使 GES 内部陷入长时间卡死，主进程也可以通过 terminate/kill 强制结束它。
    """
    try:
        # 让子进程及其可能启动的 R/外部进程进入独立进程组，便于超时时整体终止。
        if hasattr(os, "setsid"):
            os.setsid()

        cdt_algo = GES()
        causal_graph = cdt_algo.predict(visit_data)
        result_queue.put(("ok", causal_graph))
    except Exception as e:
        result_queue.put(("error", repr(e), traceback.format_exc()))


def _run_ges_predict_with_timeout(visit_data, timeout_seconds=300, mp_start_method=None):
    """
    带硬超时控制的 GES.predict。

    注意：这里不要使用 signal.alarm，因为 GES/CDT 底层可能进入 R/native code，
    Python 层信号未必能可靠打断。独立进程是更稳妥的做法。
    """
    if mp_start_method is None:
        # Linux 服务器上 fork 启动快；非 POSIX 系统自动回退到 spawn。
        mp_start_method = "fork" if hasattr(os, "fork") else "spawn"

    try:
        ctx = mp.get_context(mp_start_method)
    except ValueError:
        ctx = mp.get_context()

    result_queue = ctx.Queue(maxsize=1)
    process = ctx.Process(
        target=_ges_predict_worker,
        args=(visit_data.copy(), result_queue),
    )
    process.daemon = True

    start_time = time.time()
    process.start()
    process.join(timeout_seconds)
    elapsed = time.time() - start_time

    if process.is_alive():
        # 优先杀掉整个进程组，避免 CDT/GES 底层拉起的 R 进程残留。
        if hasattr(os, "killpg"):
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            except Exception:
                process.terminate()
        else:
            process.terminate()

        process.join(timeout=5)

        if process.is_alive():
            if hasattr(os, "killpg"):
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except Exception:
                    if hasattr(process, "kill"):
                        process.kill()
            elif hasattr(process, "kill"):
                process.kill()
            process.join(timeout=5)

        result_queue.cancel_join_thread()
        result_queue.close()
        raise TimeoutError(f"GES.predict 超过 {timeout_seconds} 秒仍未完成，已终止子进程。")

    try:
        status, *payload = result_queue.get(timeout=5)
    except py_queue.Empty as e:
        raise RuntimeError(
            f"GES.predict 子进程没有返回结果，exitcode={process.exitcode}, elapsed={elapsed:.2f}s"
        ) from e
    finally:
        result_queue.close()
        result_queue.join_thread()

    if status == "ok":
        return payload[0], elapsed

    error_repr, tb = payload
    raise RuntimeError(f"GES.predict 子进程异常：{error_repr}\n{tb}")


# Causal Graph Construction
class CausaltyGraph4Visit:
    def __init__(self, data_all, data_train, num_diagnosis, num_procedure, num_medication, data_type):
        """
        data_all: Global dataset
        data_train: Training set data
        The remaining three variables represent the counts of different types of entities.
        """
        self.data_type = data_type

        # The number of diag, proc, med
        self.num_d = num_diagnosis
        self.num_p = num_procedure
        self.num_m = num_medication

        self.data_train = data_train

        # The large df table generated from the training set marks the occurrence of each entity
        self.data = self.data_process(data_train)

        # The causal effect produced by self.data (graph from the training set)
        self.dd_effect = self.build_causal_effect(num_diagnosis, num_diagnosis, "Diag", "Diag")
        self.dp_effect = self.build_causal_effect(num_diagnosis, num_procedure, "Diag", "Proc")
        self.dm_effect = self.build_causal_effect(num_diagnosis, num_medication, "Diag", "Med")
        self.pd_effect = self.build_causal_effect(num_procedure, num_diagnosis, "Proc", "Diag")
        self.pp_effect = self.build_causal_effect(num_procedure, num_procedure, "Proc", "Proc")
        self.pm_effect = self.build_causal_effect(num_procedure, num_medication, "Proc", "Med")
        self.mm_effect = self.build_causal_effect(num_medication, num_medication, "Med", "Med")

        # The isomorphic graph generated with all data does not include the relationship between drugs and diseases.
        # It only contains three types of data: "d-d", "p-p" and "m-m". It does not participate in training,
        # but only participates in reasoning (because it includes test set samples)
        self.causal_graphs = self.build_graph(data_all)

    # Return a subgraph in the visit
    def get_graph(self, graph_id, graph_type):
        graph = self.causal_graphs[graph_id]

        if graph_type == "Diag":
            return graph[0]
        elif graph_type == "Proc":
            return graph[1]
        elif graph_type == "Med":
            return graph[2]

    # Returns the causal relationship between any two diseases-drugs
    def get_effect(self, a, b, A_type, B_type):
        a = A_type + '_' + str(int(a))
        b = B_type + '_' + str(int(b))

        if A_type == "Diag" and B_type == "Med":
            effect_df = self.dm_effect
        elif A_type == "Proc" and B_type == "Med":
            effect_df = self.pm_effect
        else:
            raise ValueError("Invalid A_type and B_type combination")

        effect = effect_df.loc[a, b]
        return effect

    def get_threshold_effect(self, threshold, A_type, B_type):
        if A_type == "Diag" and B_type == "Med":
            effect_df = self.dm_effect
        elif A_type == "Proc" and B_type == "Med":
            effect_df = self.pm_effect
        else:
            raise ValueError("Invalid A_type and B_type combination")

        # 将 DataFrame 转换为一维序列
        flattened = effect_df.stack()

        # 计算并返回对应的阈值
        threshold_value = flattened.quantile(threshold)
        return threshold_value

    def compute_ipw_ate_vectorized(self, T_matrix, E_matrix):
        """
        基于矩阵运算的高效 IPW 因果效应计算
        T_matrix: [N_visits, N_diags] 0/1 观测矩阵
        E_matrix: [N_visits, N_diags] (0, 1) 倾向分数概率矩阵
        返回: ATE_IPW 矩阵 [N_diags, N_diags]
        """
        # 截断极值，防止 1/0 出现除零异常（数值稳定性）
        eps = 1e-5
        E_matrix = torch.clamp(E_matrix, min=eps, max=1.0-eps)

        W1 = T_matrix / E_matrix  
        
        W0 = (1 - T_matrix) / (1 - E_matrix) 

        sum_W1 = W1.sum(dim=0)  # 处理组总权重，形状: [N_diags]
        sum_W0 = W0.sum(dim=0)  # 对照组总权重，形状: [N_diags]

        sum_W1_Y = torch.matmul(W1.T, T_matrix)  # 形状: [N_diags, N_diags]
        sum_W0_Y = torch.matmul(W0.T, T_matrix)  # 形状: [N_diags, N_diags]

        E_Y1 = sum_W1_Y / (sum_W1.unsqueeze(1) + eps)
        E_Y0 = sum_W0_Y / (sum_W0.unsqueeze(1) + eps)

        ATE_IPW = E_Y1 - E_Y0
        
        ATE_IPW.fill_diagonal_(0.0)
        return ATE_IPW

    def compute_hetero_ipw_ate_vectorized(self, T_matrix, E_matrix, Y_matrix):
        eps = 1e-5
        E_matrix = torch.clamp(E_matrix, min=eps, max=1.0-eps)

        W1 = T_matrix / E_matrix                # [N_visits, N_cause]
        W0 = (1 - T_matrix) / (1 - E_matrix)    # [N_visits, N_cause]

        sum_W1 = W1.sum(dim=0)  # [N_cause]
        sum_W0 = W0.sum(dim=0)  # [N_cause]

        sum_W1_Y = torch.matmul(W1.T, Y_matrix) # [N_cause, N_effect]
        sum_W0_Y = torch.matmul(W0.T, Y_matrix) # [N_cause, N_effect]

        E_Y1 = sum_W1_Y / (sum_W1.unsqueeze(1) + eps)
        E_Y0 = sum_W0_Y / (sum_W0.unsqueeze(1) + eps)

        ATE_IPW = E_Y1 - E_Y0
        return ATE_IPW

    def compute_cooccur_effect(self, num_a, num_b, a_type, b_type):
        """
        基于简单共现的边权：
            P(B=1 | A=1)

        返回 DataFrame:
            index:  A_type_i
            column: B_type_j
        """
        eps = 1e-8

        A_cols = [f"{a_type}_{i}" for i in range(num_a)]
        B_cols = [f"{b_type}_{j}" for j in range(num_b)]

        A = torch.tensor(self.data[A_cols].values, dtype=torch.float32)
        B = torch.tensor(self.data[B_cols].values, dtype=torch.float32)

        # numerator[a, b] = count(A_a=1 and B_b=1)
        numerator = A.T @ B  # [num_a, num_b]

        # denominator[a] = count(A_a=1)
        denominator = A.sum(dim=0).unsqueeze(1)  # [num_a, 1]

        cooccur = numerator / (denominator + eps)

        if a_type == b_type:
            cooccur.fill_diagonal_(0.0)

        return pd.DataFrame(
            cooccur.numpy(),
            index=A_cols,
            columns=B_cols
        )

    def build_causal_effect(self, num_a, num_b, a_type, b_type):
        file_path = project_path(
            "data", "graphs", self.data_type, "effect_matrix", "ipw",
            f"{a_type}_{b_type}_causal_effect.pkl",
        )
        
        try:
            effect_df = dill.load(open(file_path, "rb"))
        except FileNotFoundError:
            print(f"正在通过深度预训练模型构建 {a_type} -> {b_type} 全局无偏因果图，请稍候...")
            
            device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
            
            # 1. 实例化预训练的倾向分数模型
            model_es = PropensityEstimator(
                input_vocab_sizes=[self.num_d, self.num_p, self.num_m], 
                target_vocab_size=num_a, # 目标域是原因节点 (D、P、M)
                emb_dim=64, 
                device=device
            ).to(device)

            # 根据当前的源节点类型加载对应的权重
            if a_type == "Diag":
                # d 倾向分数估计器
                model_path = project_path(
                    "save", "propensity_model", self.data_type, "dd",
                    "best_pro_es_Brier0.0071_dd.pth",
                )
                es_type = 'd'
            elif a_type == "Proc":
                # p 倾向分数估计器
                model_path = project_path(
                    "save", "propensity_model", self.data_type, "pp",
                    "best_pro_es_Brier0.0024_pp.pth",
                )
                es_type = 'p'
            elif a_type == "Med":
                # m 倾向分数估计器
                model_path = project_path(
                    "save", "propensity_model", self.data_type, "mm",
                    "best_pro_es_Brier0.0833_mm.pth",
                )
                es_type = 'm'

            model_es.load_state_dict(torch.load(model_path, map_location=device), strict=False)
            model_es.eval()

            all_target_ids = torch.arange(num_a).to(device)

            T_list, E_list, Y_list = [], [], []

            # 2. 遍历整个训练集，提取所有就诊节点的局部概率矩阵
            with torch.no_grad():
                for input_seq in tqdm(self.data_train, desc=f"推断 {a_type} 倾向分数"):
                    for idx, adm in enumerate(input_seq):
                        seq_input = copy.deepcopy(input_seq[:idx+1])
                        if a_type == "Diag":
                            seq_input[-1][0] = [] 
                        elif a_type == "Proc":
                            seq_input[-1][1] = []
                        elif a_type == "Med":
                            seq_input[-1][2] = []
                        
                        D, P, M = adm[0], adm[1], adm[2]
                        if a_type == 'Diag':
                            pos_targets = D
                        elif a_type == 'Proc':
                            pos_targets = P
                        elif a_type == 'Med':
                            pos_targets = M

                        if len(pos_targets) == 0:
                            continue
                        # 构建原因、结果向量 (T)
                        t_vector = torch.zeros(num_a, device=device) # num_a == num_b
                        t_vector[pos_targets] = 1.0
                        
                        if a_type != b_type: # 异构
                            # 构建结果向量 (Y)
                            y_vector = torch.zeros(num_b, device=device)
                            if b_type == "Diag":
                                y_targets = D
                            elif b_type == "Proc":
                                y_targets = P
                            elif b_type == "Med":
                                y_targets = M
                            y_vector[y_targets] = 1.0
                            Y_list.append(y_vector)
                        
                        # 预测倾向得分 (E)
                        probs = model_es(seq_input, all_target_ids, es_type=es_type) 
                        probs = probs.squeeze(0) # (num, )
                        
                        T_list.append(t_vector)
                        E_list.append(probs)

            T_matrix = torch.stack(T_list)
            E_matrix = torch.stack(E_list)
            if a_type != b_type:
                Y_matrix = torch.stack(Y_list)

            # 3. 矩阵并行计算 IPW ATE
            if a_type != b_type:
                ate_matrix = self.compute_hetero_ipw_ate_vectorized(T_matrix, E_matrix, Y_matrix)
            else:
                ate_matrix = self.compute_ipw_ate_vectorized(T_matrix, E_matrix)
            
            # 4. 封装回 DataFrame 并保存
            ate_numpy = ate_matrix.cpu().numpy()
            effect_df = pd.DataFrame(
                ate_numpy, 
                index=[f"{a_type}_{i}" for i in range(num_a)],
                columns=[f"{b_type}_{j}" for j in range(num_b)]
            )

            # 确保保存目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "wb") as f:
                dill.dump(effect_df, f)

        return effect_df

    def get_effect_table(self, a_type, b_type, weight_mode="ipw"):
        """
        根据 weight_mode 返回对应的 effect matrix.
        """
        if weight_mode == "ipw":
            if a_type == "Diag" and b_type == "Diag":
                return self.dd_effect
            if a_type == "Diag" and b_type == "Proc":
                return self.dp_effect
            if a_type == "Diag" and b_type == "Med":
                return self.dm_effect
            if a_type == "Proc" and b_type == "Diag":
                return self.pd_effect
            if a_type == "Proc" and b_type == "Proc":
                return self.pp_effect
            if a_type == "Proc" and b_type == "Med":
                return self.pm_effect
            if a_type == "Med" and b_type == "Med":
                return self.mm_effect

        elif weight_mode == "cooccur":
            if a_type == "Diag" and b_type == "Diag":
                return self.compute_cooccur_effect(self.num_d, self.num_d, "Diag", "Diag")
            if a_type == "Diag" and b_type == "Proc":
                return self.compute_cooccur_effect(self.num_d, self.num_p, "Diag", "Proc")
            if a_type == "Diag" and b_type == "Med":
                return self.compute_cooccur_effect(self.num_d, self.num_m, "Diag", "Med")
            if a_type == "Proc" and b_type == "Diag":
                return self.compute_cooccur_effect(self.num_p, self.num_d, "Proc", "Diag")
            if a_type == "Proc" and b_type == "Proc":
                return self.compute_cooccur_effect(self.num_p, self.num_p, "Proc", "Proc")
            if a_type == "Proc" and b_type == "Med":
                return self.compute_cooccur_effect(self.num_p, self.num_m, "Proc", "Med")
            if a_type == "Med" and b_type == "Med":
                return self.compute_cooccur_effect(self.num_m, self.num_m, "Med", "Med")

        else:
            raise ValueError(f"Unknown weight_mode: {weight_mode}")

        raise ValueError(f"Unsupported relation: {a_type}->{b_type}")

    def assign_edge_weights(self, skeleton_graph, weight_mode="ipw"):
        """
        输入只含结构的 GES skeleton_graph。
        输出带权图。

        注意：边结构完全不变，只替换 weight。
        """
        weighted_graph = nx.DiGraph()
        weighted_graph.add_nodes_from(skeleton_graph.nodes())

        # 可选：缓存 cooccur matrix，避免每条边重复计算
        effect_cache = {}

        def parse_type(node):
            # node format: "Diag_12"
            return node.split("_")[0]

        for source, target in skeleton_graph.edges():
            s_type = parse_type(source)
            t_type = parse_type(target)

            if weight_mode == "binary":
                weight = 1.0
            else:
                key = (s_type, t_type, weight_mode)
                if key not in effect_cache:
                    effect_cache[key] = self.get_effect_table(
                        s_type,
                        t_type,
                        weight_mode=weight_mode
                    )

                effect_df = effect_cache[key]
                weight = float(effect_df.at[source, target])

            weighted_graph.add_edge(source, target, weight=weight)

        return weighted_graph

    def build_graph(self, data_all, save_interval=100, ges_timeout=300, weight_mode="ipw"):
        """
        weight_mode:
            "ipw"     : 使用 IPW ATE 作为边权
            "cooccur" : 使用简单共现 P(B|A) 作为边权
            "binary"  : 所有边权设为 1
        """
        save_dir = project_path("data", "graphs", self.data_type)
        os.makedirs(save_dir, exist_ok=True)

        skeleton_path = os.path.join(save_dir, "construct_skl/causal_graph_skl.pkl")
        weighted_path = os.path.join(save_dir, f"causal_graph_{weight_mode}.pkl")

        # 1. 如果对应权重版本已经存在，直接读取
        if os.path.exists(weighted_path):
            with open(weighted_path, "rb") as f:
                return dill.load(f)

        # 2. 若 skeleton 存在，直接读取结构
        if os.path.exists(skeleton_path):
            with open(skeleton_path, "rb") as f:
                skeleton_graphs = dill.load(f)
        else:
            skeleton_graphs = self.build_ges_skeleton(
                data_all=data_all,
                save_interval=save_interval,
                ges_timeout=ges_timeout
            )
            with open(skeleton_path, "wb") as f:
                dill.dump(skeleton_graphs, f)

        # 4. 在同一个 skeleton 上替换边权
        weighted_graphs = [
            self.assign_edge_weights(g, weight_mode=weight_mode)
            for g in tqdm(skeleton_graphs, desc=f"Assigning {weight_mode} weights")
        ]

        # 5. 拆成 D-D / P-P / M-M
        subgraph_list = self.split_homogeneous_subgraphs(weighted_graphs)

        with open(weighted_path, "wb") as f:
            dill.dump(subgraph_list, f)

        return subgraph_list

    def split_homogeneous_subgraphs(self, causal_graphs):
        """
        将完整异构图拆成 D-D, P-P, M-M 三类同构子图。
        """
        subgraph_list = []

        for graph in tqdm(causal_graphs, desc="拆分异构图为同构子图"):
            graph_type = []

            graph_d = graph.copy()
            graph_d.remove_nodes_from(
                [node for node in graph.nodes() if "Med" in node or "Proc" in node]
            )
            graph_type.append(graph_d)

            graph_p = graph.copy()
            graph_p.remove_nodes_from(
                [node for node in graph.nodes() if "Diag" in node or "Med" in node]
            )
            graph_type.append(graph_p)

            graph_m = graph.copy()
            graph_m.remove_nodes_from(
                [node for node in graph.nodes() if "Diag" in node or "Proc" in node]
            )
            graph_type.append(graph_m)

            subgraph_list.append(graph_type)

        return subgraph_list

    def _save_skipped_ges_sample(
        self,
        skipped_dir,
        idx,
        adm,
        D,
        P,
        M,
        visit,
        visit_data,
        reason,
        elapsed_seconds=None,
        traceback_text=None,
    ):
        """保存被跳过样本的原始信息和 visit_data，便于后续复现、排错或单独重试。"""
        os.makedirs(skipped_dir, exist_ok=True)

        csv_path = os.path.join(skipped_dir, f"visit_{idx}_data.csv")
        meta_path = os.path.join(skipped_dir, f"visit_{idx}_meta.pkl")
        jsonl_path = os.path.join(skipped_dir, "skipped_ges_samples.jsonl")

        visit_data.to_csv(csv_path, index=False)

        meta = {
            "idx": int(idx),
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "reason": str(reason),
            "elapsed_seconds": None if elapsed_seconds is None else float(elapsed_seconds),
            "num_diag": len(D),
            "num_proc": len(P),
            "num_med": len(M),
            "num_total_nodes": len(visit),
            "D": list(D),
            "P": list(P),
            "M": list(M),
            "visit_columns": list(visit),
            "visit_data_shape": tuple(visit_data.shape),
            "adm": adm,
            "visit_data_csv": csv_path,
            "meta_pkl": meta_path,
            "traceback": traceback_text,
        }

        with open(meta_path, "wb") as f:
            dill.dump(meta, f)

        # 额外保存一份 jsonl 摘要，方便不用 Python 也能快速查看有哪些样本被跳过。
        summary = {
            "idx": meta["idx"],
            "saved_at": meta["saved_at"],
            "reason": meta["reason"],
            "elapsed_seconds": meta["elapsed_seconds"],
            "num_diag": meta["num_diag"],
            "num_proc": meta["num_proc"],
            "num_med": meta["num_med"],
            "num_total_nodes": meta["num_total_nodes"],
            "D": meta["D"],
            "P": meta["P"],
            "M": meta["M"],
            "visit_data_shape": meta["visit_data_shape"],
            "visit_data_csv": csv_path,
            "meta_pkl": meta_path,
        }
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(self._to_jsonable(summary), ensure_ascii=False) + "\n")

        return meta

    def _build_empty_visit_graph(self, visit):
        """
        当某个样本的 GES 超时或异常时，用只含节点、不含边的图作为占位。
        这样可以保证 causal_graphs[i] 仍然对应 sessions[i]，不会破坏后续 get_graph(graph_id) 的索引。
        """
        graph = nx.DiGraph()
        graph.add_nodes_from(visit)
        return graph

    # Three graphs are built for each session"d-d","p-p","m-m"
    def build_ges_skeleton(self, data_all, save_interval=100, ges_timeout=300):
        
        """
        save_interval: 每处理多少个 visit 保存一次检查点，默认 100。
                       值太小会增加 I/O 开销，太大则中断时可能会丢失较多进度。
        ges_timeout: 单个 visit 的 GES.predict 最长运行时间，默认 300 秒，即 5 分钟。
                     超时后跳过该样本，并保存该样本信息供后续排查或重试。
        """
        save_dir = project_path("data", "graphs", self.data_type, "construct_skl")
        os.makedirs(save_dir, exist_ok=True)

        # 最终文件、检查点文件、跳过样本文件路径
        file_path = os.path.join(save_dir, "causal_graph_skl.pkl")
        checkpoint_path = os.path.join(save_dir, "causal_graph_skl_chkpt.pkl")
        skipped_path = os.path.join(save_dir, "skipped_ges_samples.pkl")
        skipped_dir = os.path.join(save_dir, "skipped_ges_samples")

        # 1. 如果最终文件已存在，直接读取并返回
        try:
            subgraph_list = dill.load(open(file_path, "rb"))
            return subgraph_list
        except FileNotFoundError:
            pass

        print("准备构建所有因果图...")
        sessions = self.sessions_process(data_all)
        causal_graphs = []
        skipped_samples = []

        if os.path.exists(skipped_path):
            try:
                with open(skipped_path, "rb") as f:
                    skipped_samples = dill.load(f)
                print(f"已读取历史跳过记录：{len(skipped_samples)} 个 visits。")
            except Exception as e:
                print(f"读取跳过记录失败，将重新创建。错误信息: {e}")
                skipped_samples = []

        # 2. 检查是否存在未完成的检查点文件
        if os.path.exists(checkpoint_path):
            print("发现检查点文件，正在恢复之前的进度...")
            try:
                with open(checkpoint_path, "rb") as f:
                    causal_graphs = dill.load(f)
                print(f"成功恢复进度：已完成 {len(causal_graphs)} / {len(sessions)} 个 visits。")
            except Exception as e:
                print(f"读取检查点失败，将重新开始。错误信息: {e}")
                causal_graphs = []

        start_idx = len(causal_graphs)

        # 3. 从上次中断的地方继续计算
        if start_idx < len(sessions):
            for i in tqdm(range(start_idx, len(sessions)), desc="构建 GES 因果图骨架", initial=start_idx, total=len(sessions)):
                adm = sessions[i]
                D = adm[0]
                P = adm[1]
                M = adm[2]

                # Convert data into DataFrame column names
                visit = [f"Diag_{d}" for d in D] + [f"Proc_{p}" for p in P] + [f"Med_{m}" for m in M]
                visit_data = self.data[visit]

                # Calculate the cause-and-effect diagram using the GES algorithm with timeout protection.
                try:
                    causal_graph, elapsed = _run_ges_predict_with_timeout(
                        visit_data,
                        timeout_seconds=ges_timeout,
                    )
                except TimeoutError as e:
                    print(f"\n[GES Timeout] visit idx={i} 超过 {ges_timeout} 秒，已跳过并保存样本。")
                    meta = self._save_skipped_ges_sample(
                        skipped_dir=skipped_dir,
                        idx=i,
                        adm=adm,
                        D=D,
                        P=P,
                        M=M,
                        visit=visit,
                        visit_data=visit_data,
                        reason=str(e),
                        elapsed_seconds=ges_timeout,
                        traceback_text=traceback.format_exc(),
                    )
                    skipped_samples.append(meta)
                    with open(skipped_path, "wb") as f:
                        dill.dump(skipped_samples, f)

                    # 用空边图占位，保证 graph_id 与原始 visit idx 一一对应。
                    causal_graphs.append(self._build_empty_visit_graph(visit))
                    with open(checkpoint_path, "wb") as f:
                        dill.dump(causal_graphs, f)
                    continue
                except Exception as e:
                    print(f"\n[GES Error] visit idx={i} 运行异常，已跳过并保存样本。错误信息: {e}")
                    meta = self._save_skipped_ges_sample(
                        skipped_dir=skipped_dir,
                        idx=i,
                        adm=adm,
                        D=D,
                        P=P,
                        M=M,
                        visit=visit,
                        visit_data=visit_data,
                        reason=repr(e),
                        elapsed_seconds=None,
                        traceback_text=traceback.format_exc(),
                    )
                    skipped_samples.append(meta)
                    with open(skipped_path, "wb") as f:
                        dill.dump(skipped_samples, f)

                    # 运行异常同样使用空边图占位，避免索引错位。
                    causal_graphs.append(self._build_empty_visit_graph(visit))
                    with open(checkpoint_path, "wb") as f:
                        dill.dump(causal_graphs, f)
                    continue

                # Remove Med-Diag, Med-Proc
                new_graph = nx.DiGraph()

                # First, add all nodes to the new graph
                for node in causal_graph.nodes():
                    new_graph.add_node(node)

                # Then, add the required edges to the new graph
                for edge in causal_graph.edges():
                    source, target = edge

                    # 根据类型从对应的 ATE 矩阵中提取 IPW 权重
                    if source.startswith("Diag") and target.startswith("Diag"):
                        new_graph.add_edge(source, target)
                    elif source.startswith("Diag") and target.startswith("Med"):
                        new_graph.add_edge(source, target)
                    elif source.startswith("Diag") and target.startswith("Proc"):
                        new_graph.add_edge(source, target)
                    elif source.startswith("Proc") and target.startswith("Proc"):
                        new_graph.add_edge(source, target)
                    elif source.startswith("Proc") and target.startswith("Diag"):
                        new_graph.add_edge(source, target)
                    elif source.startswith("Proc") and target.startswith("Med"):
                        new_graph.add_edge(source, target)
                    elif source.startswith("Med") and target.startswith("Med"):
                        new_graph.add_edge(source, target)

                causal_graph = new_graph

                # remove loop
                while not nx.is_directed_acyclic_graph(causal_graph):
                    cycle_nodes = nx.find_cycle(causal_graph, orientation="original")
                    for edge in cycle_nodes:
                        source, target, _ = edge
                        causal_graph.remove_edge(source, target)

                causal_graph = nx.DiGraph(causal_graph)
                causal_graphs.append(causal_graph)

                # 4. 定期保存检查点
                if (i + 1) % save_interval == 0:
                    with open(checkpoint_path, "wb") as f:
                        dill.dump(causal_graphs, f)

            # 循环结束后再保存一次，确保 100% 的进度被记录
            with open(checkpoint_path, "wb") as f:
                dill.dump(causal_graphs, f)

        # 6. 保存最终结果并清理临时文件
        with open(file_path, "wb") as f:
            dill.dump(causal_graphs, f)

        # if os.path.exists(checkpoint_path):
        #     os.remove(checkpoint_path)
        #     print("计算完成，已清理检查点文件。")

        if skipped_samples:
            print(f"本次构建中共有 {len(skipped_samples)} 个 GES 样本被跳过，记录保存在: {skipped_path}")
            print(f"对应 visit_data/meta 文件保存在: {skipped_dir}")

        return causal_graphs

    # Turn all medical visits into conversational format, one by one
    def sessions_process(self, raw_data):
        sessions = []
        for patient in raw_data:
            for adm in patient:
                sessions.append(adm)
        return sessions

    # Convert session data one by one into a large df table
    def data_process(self, data_train):
        # Get the directory where the current script file is located
        file_path = project_path("data", "graphs", self.data_type, "matrix4causalgraph.pkl")
        try:
            with open(file_path, "rb") as f:
                df = dill.load(f)
        except FileNotFoundError:

            print("整理数据集..")
            train_sessions = self.sessions_process(data_train)

            df = pd.DataFrame(0.0, index=range(len(train_sessions)), columns=
            [f'Diag_{i}' for i in range(self.num_d)] +
            [f'Proc_{i}' for i in range(self.num_p)] +
            [f'Med_{i}' for i in range(self.num_m)])

            for i, session in tqdm(enumerate(train_sessions)):
                D, P, M = session[:3]
                df.loc[i, [f'Diag_{d}' for d in D]] = 1
                df.loc[i, [f'Proc_{p}' for p in P]] = 1
                df.loc[i, [f'Med_{m}' for m in M]] = 1

            with open(file_path, "wb") as f:
                dill.dump(df, f)
        return df


if __name__ == '__main__':
    # test sample

    data_type = 'mimic-iii' # mimic-iv

    data_path = project_path("data", "output", data_type, "records_final.pkl")
    voc_path = project_path("data", "output", data_type, "voc_final.pkl")
    data = dill.load(open(data_path, "rb"))

    voc = dill.load(open(voc_path, "rb"))
    diag_voc, pro_voc, med_voc = voc["diag_voc"], voc["pro_voc"], voc["med_voc"]

    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    adm_id = 0
    for patient in data:
        for adm in patient:
            adm.append(adm_id)
            adm_id += 1

    split_point = int(len(data) * 3 / 5) # 训练 3/5
    data_train = data[:split_point] 
    causal_graph = CausaltyGraph4Visit(data, data_train, voc_size[0], voc_size[1], voc_size[2], data_type)
