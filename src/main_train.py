import dill
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
import time
from collections import defaultdict
import os
import json
from util import llprint, multi_label_metric, ddi_rate_score, get_n_params, buildMPNN, build_patient_memory_bank, build_block_graph, construct_patient_graph, tensorboard_write_pretrain, tensorboard_write_main,construct_patient_visit_graph
from models import CounterfactualCausalRouter
from torch.utils.tensorboard import SummaryWriter
import argparse
import random
from modules.causal_construction_skl import CausaltyGraph4Visit
from modules.causal_memory_bank import CausalVisitMemoryBank


PROJECT_ROOT = os.environ.get(
    "CSDG_PROJECT_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)


def project_path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Reproducible Top-K selection is important because a small score change can
    # alter the compact history set used by every downstream module.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1", "y"):
        return True
    if v.lower() in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

def get_args():
    parser = argparse.ArgumentParser(description="Train or evaluate CSDG-Rec.")
    parser.add_argument('--dataset', type=str, default='mimic-iii', choices=['mimic-iii', 'mimic-iv'])
    parser.add_argument('--data_dir', type=str, default=None, help='Directory containing processed dataset pickle files.')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=2048)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--emb_dim', type=int, default=64)
    parser.add_argument('--save_dir', type=str, default=project_path('save', 'CSDG-Rec'))
    parser.add_argument('--commit', type=str, default='default')
    parser.add_argument('--resume_path', type=str, default='', help='Checkpoint used when --test is enabled.')
    parser.add_argument('--resume_path_pretrained', type=str, default=project_path('save', 'pretrained_model', 'best_pretrained_model.pt'))
    parser.add_argument('--test', dest='Test', action='store_true', help='Run bootstrap evaluation from --resume_path.')
    parser.add_argument('--no_pretrain', dest='is_pretrain', action='store_false', help='Disable encoder/proxy warm-up.')
    parser.add_argument('--save_model_every', type=str2bool, default=False, help='Save one checkpoint per epoch.')
    parser.set_defaults(is_pretrain=True)
    args = parser.parse_args()

    data_dir = args.data_dir or project_path('data', 'output', args.dataset)
    args.data_path = os.path.join(data_dir, 'records_final.pkl')
    args.ddi_adj_path = os.path.join(data_dir, 'ddi_A_final.pkl')
    args.ehr_adj_path = os.path.join(data_dir, 'ehr_adj_final.pkl')
    args.ddi_mask_path = os.path.join(data_dir, 'ddi_mask_H.pkl')
    args.molecule_path = os.path.join(data_dir, 'atc3toSMILES.pkl')
    args.voc_path = os.path.join(data_dir, 'voc_final.pkl')

    # Fixed paper defaults. These are intentionally not exposed as CLI flags in
    # the release version to keep the public entry point compact and readable.
    defaults = {
        'weight_bce': 0.97,
        'weight_multi': 0.03,
        'lambda_review': 0.001,
        'weight_decay': 0.0,
        'grad_clip': 1.0,
        'dropout': 0.3,
        'att_tau': 20.0,
        'history_topk': 3,
        'eval_threshold': 0.5,
        'is_ddi_loss': False,
        'target_ddi': 0.05,
        'kp': 0.05,
        'ddi_T': 0.5,
        'ddi_decay_weight': 0.85,
        'pretrain_epochs': 6,
        'pretrain_lr': 1e-3,
        'pretrain_weight_decay': 1e-5,
        'pretrain_patience': 3,
        'warmup_detach_encoder': False,
        'freeze_proxy': True,
        'memory_topk': 10,
        'memory_warmup_epochs': 5,
        'memory_momentum': 0.7,
        'memory_refresh_every': 2,
        'use_causal_graph': True,
        'use_causal_routing': True,
        'use_ctcr': True,
        'use_memory_bank': True,
        'max_train_patients': 0,
        'max_eval_patients': 0,
        'early_stop_patience': 0,
        'select_metric': 'ja',
        'ddi_penalty': 0.0,
        'results_dir': '',
    }
    for key, value in defaults.items():
        setattr(args, key, value)
    return args



import copy
from torch.optim import Adam

def warmup_encoder_and_proxy(model, data_train, data_eval, vocab_size, device, args):
    """
    使用最终模型自身的 encoder 表示预训练：
    1. embeddings
    2. homo_graph
    3. causal GRU encoders
    4. cross-visit attention linear_layer
    5. proxy_predictor

    目标：让 proxy_predictor 在正式训练前已经能够根据 h_obs 预测当前药物。
    """

    if not args.is_pretrain:
        print("[Warmup] Disabled.")
        return None

    proxy_type = getattr(model, 'proxy_model_type', 'mlp')
    if proxy_type != 'mlp':
        print(f"[Warmup] Skipped because proxy_model_type={proxy_type}. The proxy is loaded from an external pretrained checkpoint instead of h_obs warmup.")
        return {"used": False, "reason": f"{proxy_type}_proxy_uses_external_pretraining"}

    if args.resume_path_pretrained and os.path.exists(args.resume_path_pretrained):
        model.load_state_dict(torch.load(args.resume_path_pretrained))
        print(f"[Warmup] Loaded pre-trained model from {args.resume_path_pretrained}")
        return None

    optimizer = Adam(
        [
            {"params": model.embeddings.parameters(), "lr": args.pretrain_lr},
            {"params": model.homo_graph.parameters(), "lr": args.pretrain_lr},
            {"params": model.causal_encoders.parameters(), "lr": args.pretrain_lr},
            {"params": model.linear_layer.parameters(), "lr": args.pretrain_lr},
            {"params": model.proxy_predictor.parameters(), "lr": args.pretrain_lr},
        ],
        weight_decay=args.pretrain_weight_decay
    )

    best_state = None
    best_ja = -1.0
    no_improve = 0

    for epoch in range(args.pretrain_epochs):
        model.train()
        total_loss = 0.0
        visit_cnt = 0

        for patient in tqdm(data_train, desc=f"Warmup Encoder+Proxy Epoch {epoch}"):
            for adm_idx, adm in enumerate(patient):
                seq_input = patient[: adm_idx + 1]
                target_meds = adm[2]

                target = torch.zeros(1, vocab_size[2], device=device)
                target[:, target_meds] = 1.0

                # 只取观察历史表示 h_obs，不执行反事实 do(v_k=0)
                h_obs = model.encode_patient_observational_representation(seq_input)
                h_obs = h_obs.detach() if args.warmup_detach_encoder else h_obs

                logits = model.proxy_predictor(h_obs)

                loss_bce = F.binary_cross_entropy_with_logits(logits, target)

                loss_multi_target = np.full((1, vocab_size[2]), -1)
                for i, med in enumerate(target_meds):
                    loss_multi_target[0][i] = med

                loss_multi = F.multilabel_margin_loss(
                    torch.sigmoid(logits),
                    torch.LongTensor(loss_multi_target).to(device)
                )

                loss = args.weight_bce * loss_bce + args.weight_multi * loss_multi

                optimizer.zero_grad()
                loss.backward()

                if args.grad_clip and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(model.embeddings.parameters())
                        + list(model.homo_graph.parameters())
                        + list(model.causal_encoders.parameters())
                        + list(model.linear_layer.parameters())
                        + list(model.proxy_predictor.parameters()),
                        args.grad_clip
                    )

                optimizer.step()

                total_loss += loss.item()
                visit_cnt += 1

        ja, prauc, avg_p, avg_r, avg_f1, avg_loss, avg_med = proxy_warmup_eval(
            model, data_eval, vocab_size, device, args
        )

        print(
            f"[Warmup] epoch={epoch}, loss={total_loss / max(1, visit_cnt):.6f}, "
            f"JA={ja:.4f}, PRAUC={prauc:.4f}, F1={avg_f1:.4f}, AVG_MED={avg_med:.4f}"
        )

        if ja > best_ja:
            best_ja = ja
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
            # 保存最佳模型
            torch.save(model.state_dict(), args.resume_path_pretrained)
        else:
            no_improve += 1

        if args.pretrain_patience > 0 and no_improve >= args.pretrain_patience:
            print(f"[Warmup] Early stop at epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "best_ja": float(best_ja),
        "used": True,
    }

@torch.no_grad()
def proxy_warmup_eval(model, data_eval, vocab_size, device, args):
    model.eval()

    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    losses = []
    med_cnt, visit_cnt = 0, 0

    for patient in tqdm(data_eval, desc="Warmup Eval"):
        y_gt, y_pred, y_prob = [], [], []

        for adm_idx, adm in enumerate(patient):
            seq_input = patient[: adm_idx + 1]
            target_meds = adm[2]

            target = torch.zeros(1, vocab_size[2], device=device)
            target[:, target_meds] = 1.0

            h_obs = model.encode_patient_observational_representation(seq_input)
            logits = model.proxy_predictor(h_obs)

            loss = F.binary_cross_entropy_with_logits(logits, target)
            losses.append(loss.item())

            prob = torch.sigmoid(logits).detach().cpu().numpy()[0]

            gt = np.zeros(vocab_size[2])
            gt[target_meds] = 1

            pred = prob.copy()
            pred[pred >= args.eval_threshold] = 1
            pred[pred < args.eval_threshold] = 0

            y_gt.append(gt)
            y_pred.append(pred)
            y_prob.append(prob)

            med_cnt += int(pred.sum())
            visit_cnt += 1

        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(
            np.array(y_gt),
            np.array(y_pred),
            np.array(y_prob)
        )

        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)

    return (
        np.mean(ja),
        np.mean(prauc),
        np.mean(avg_p),
        np.mean(avg_r),
        np.mean(avg_f1),
        np.mean(losses),
        med_cnt / max(1, visit_cnt),
    )

def eval(model, data_eval, vocab_size, memory_bank, epoch, args):
    model.eval()

    smm_record = []
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    med_cnt, visit_cnt = 0, 0

    with torch.no_grad():
        for step, patient in tqdm(enumerate(data_eval), total=len(data_eval), desc="eval"):
            y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []

            adm = patient[-1]
            target_meds = adm[2]
            seq_input = patient

            target_output, _, _ = model(
                seq_input,
                memory_bank=memory_bank,
                patient_id=None,
                global_id=None,
                epoch=epoch,
                warmup_epochs=args.memory_warmup_epochs,
                topk=args.memory_topk
            )

            y_gt_tmp = np.zeros(vocab_size[2])
            y_gt_tmp[target_meds] = 1
            y_gt.append(y_gt_tmp)

            target_output = torch.sigmoid(target_output).detach().cpu().numpy()[0]

            y_pred_prob.append(target_output)

            y_pred_tmp = target_output.copy()
            y_pred_tmp[y_pred_tmp >= args.eval_threshold] = 1
            y_pred_tmp[y_pred_tmp < args.eval_threshold] = 0
            y_pred.append(y_pred_tmp)

            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            y_pred_label.append(sorted(y_pred_label_tmp))

            visit_cnt += 1
            med_cnt += len(y_pred_label_tmp)

            smm_record.append(y_pred_label)

            adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(np.array(y_gt), np.array(y_pred), y_pred_prob)

            ja.append(adm_ja)
            prauc.append(adm_prauc)
            avg_p.append(adm_avg_p)
            avg_r.append(adm_avg_r)
            avg_f1.append(adm_avg_f1)

    ddi_rate = ddi_rate_score(smm_record, path=args.ddi_adj_path)

    llprint(
        "\nDDI Rate: {:.4}, Jaccard: {:.4},  PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, AVG_MED: {:.4}\n".format(
            ddi_rate,
            np.mean(ja),
            np.mean(prauc),
            np.mean(avg_p),
            np.mean(avg_r),
            np.mean(avg_f1),
            med_cnt / visit_cnt,
        )
    )

    return (
        ddi_rate,
        np.mean(ja),
        np.mean(prauc),
        np.mean(avg_p),
        np.mean(avg_r),
        np.mean(avg_f1),
        med_cnt / visit_cnt,
    )

if __name__ == "__main__":
    args = get_args()

    set_seed(args.seed)

    # 是否存在文件夹，没有就创建
    pretrained_model_path = f"{args.save_dir}/{args.commit}/pretrained_model/"
    tensorboard_logs_path = f"{args.save_dir}/{args.commit}/tensorboard_logs/"
    train_history_path = f"{args.save_dir}/{args.commit}/train_history/"

    os.makedirs(pretrained_model_path, exist_ok=True)
    os.makedirs(tensorboard_logs_path, exist_ok=True)
    os.makedirs(train_history_path, exist_ok=True)

    device = torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")

    # 使用 args 中的路径
    data_path = args.data_path
    ddi_adj_path = args.ddi_adj_path
    ehr_adj_path = args.ehr_adj_path
    ddi_mask_path = args.ddi_mask_path
    molecule_path = args.molecule_path
    voc_path = args.voc_path

    # 加载数据
    data = dill.load(open(data_path, "rb"))
    voc = dill.load(open(voc_path, "rb"))
    ehr_adj = dill.load(open(ehr_adj_path, "rb"))
    ddi_adj = dill.load(open(ddi_adj_path, "rb"))
    ddi_mask_H = dill.load(open(ddi_mask_path, "rb"))
    molecule = dill.load(open(molecule_path, "rb"))

    diag_voc, pro_voc, med_voc = voc["diag_voc"], voc["pro_voc"], voc["med_voc"]
    vocab_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    # 训练 评估 测试
    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_eval = data[split_point : split_point + eval_len]
    data_test = data[split_point + eval_len :]

    # 预训练 预评估
    pretrained_data = data_train
    pre_train_split_point = int(len(pretrained_data) * 4 / 5)
    pre_train_data_train = pretrained_data[:pre_train_split_point]
    pre_train_data_eval = pretrained_data[pre_train_split_point:]

    if args.max_train_patients and args.max_train_patients > 0:
        data_train = data_train[:args.max_train_patients]
        pre_train_data_train = pre_train_data_train[:args.max_train_patients]
    if args.max_eval_patients and args.max_eval_patients > 0:
        data_eval = data_eval[:args.max_eval_patients]
        pre_train_data_eval = pre_train_data_eval[:args.max_eval_patients]

    writer = SummaryWriter(log_dir=tensorboard_logs_path)

    diag_size, proc_size, med_size = vocab_size

    MPNNSet, N_fingerprint, average_projection = buildMPNN(molecule, med_voc.idx2word, 2, device)

    if args.use_causal_graph:
        causal_graph = CausaltyGraph4Visit(data, data_train, diag_size, proc_size, med_size, data_type=args.dataset)
    else:
        print('[Ablation] use_causal_graph=False: skip CausaltyGraph4Visit construction and all causal graph operations.')
        causal_graph = None

    if not args.use_causal_routing:
        print('[Ablation] use_causal_routing=False: remove Stage-2 counterfactual routing; use ordinary cross-visit attention aggregation.')
    print(
        f'[History] approximate screening + exact Top-K intervention: history_topk={args.history_topk}'
    )

    # 为每个 visit 追加全局 graph_id。Proxy 预训练、CCMB refresh 和正式训练都会使用 adm[3]
    # 从 causal_graph 中取对应 visit 的局部同构因果图。
    adm_id = 0
    for patient in data:
        for adm in patient:
            if len(adm) < 4:
                adm.append(adm_id)
            else:
                adm[3] = adm_id
            adm_id += 1


    counterfactualModel = CounterfactualCausalRouter(
        ddi_adj=ddi_adj,
        ehr_adj=ehr_adj,
        ddi_mask_H=ddi_mask_H,
        MPNNSet=MPNNSet,
        N_fingerprints=N_fingerprint,
        average_projection=average_projection,
        causal_graph=causal_graph,
        vocab_size=vocab_size,
        emb_dim=args.emb_dim,
        device=device,
    ).to(device)

    if args.Test and args.resume_path is not None:
        counterfactualModel.load_state_dict(torch.load(open(args.resume_path, "rb")))
        counterfactualModel.to(device=device)
        tic = time.time()

        memory_bank = CausalVisitMemoryBank(
            device=device,
            emb_dim=args.emb_dim,
            topk=args.memory_topk,
            momentum=args.memory_momentum,
            exclude_same_patient=True
        )

        # warmup memory bank with current model snapshot
        memory_bank.refresh(
            data_train=data_train,
            model=counterfactualModel,
            epoch=0,
            use_momentum=False
        )

        ddi_list, ja_list, prauc_list, f1_list, med_list = [], [], [], [], []

        result = []
        for _ in range(10):
            # test_sample = np.random.choice(
            #     data_test, round(len(data_test) * 0.8), replace=True
            # )
            n_samples = len(data_test)
            sample_size = round(n_samples * 0.8)
            # 1. 生成采样索引（一维）
            indices = np.random.choice(n_samples, size=sample_size, replace=True)
            # 2. 用索引从原数据中取样本，保持原结构不变
            test_sample = [data_test[i] for i in indices]  # 如果是 list
            ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(
                counterfactualModel, test_sample, vocab_size, memory_bank, 0, args
            )
            result.append([ddi_rate, ja, avg_f1, prauc, avg_med])

        result = np.array(result)
        mean = result.mean(axis=0)
        std = result.std(axis=0)

        outstring = ""
        for m, s in zip(mean, std):
            outstring += "{:.4f} $\\pm$ {:.4f} & ".format(m, s)

        print(outstring)

        print("test time: {}".format(time.time() - tic))
        exit()

    warmup_info = warmup_encoder_and_proxy(counterfactualModel, pre_train_data_train, pre_train_data_eval, vocab_size, device, args)

    # Freeze proxy before building the optimizer, otherwise the optimizer will still keep
    # unnecessary state for the pretrained GAMENet parameters.
    if args.freeze_proxy:
        if hasattr(counterfactualModel.proxy_predictor, 'set_frozen'):
            counterfactualModel.proxy_predictor.set_frozen(True)
        else:
            for param in counterfactualModel.proxy_predictor.parameters():
                param.requires_grad = False
            counterfactualModel.proxy_predictor.eval()

    trainable_params = [p for p in counterfactualModel.parameters() if p.requires_grad]
    optimizer = Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    if args.use_memory_bank:
        memory_bank = CausalVisitMemoryBank(
            device=device,
            emb_dim=args.emb_dim,
            topk=args.memory_topk,
            momentum=args.memory_momentum,
            exclude_same_patient=True
        )

        # warmup memory bank with current model snapshot
        memory_bank.refresh(
            data_train=data_train,
            model=counterfactualModel,
            epoch=0,
            use_momentum=False
        )
    else:
        memory_bank = None

    history = defaultdict(list)
    best_epoch, best_ja = 0, 0
    best_score = -float("inf")
    best_metrics = {}
    no_improve_epochs = 0

    if args.is_ddi_loss:
        T = args.ddi_T
        decay_weight = args.ddi_decay_weight

    counterfactualModel.train()
    for epoch in range(args.epochs):
        total_loss = 0
        tic = time.time()
        visit_emb = []
        
        global_id = 0
        for step, patient in tqdm(enumerate(data_train), total=len(data_train), desc="main train"):
            
            counterfactualModel.train()

            adm = patient[-1]
            target_meds = adm[2]
            seq_input = patient


            loss_bce_target = np.zeros((1, med_size))
            loss_bce_target[:, target_meds] = 1

            loss_multi_target = np.full((1, med_size), -1)
            for idx, item in enumerate(target_meds):
                loss_multi_target[0][idx] = item

            result, loss_ddi, review_reg = counterfactualModel(
                seq_input,
                memory_bank=memory_bank,
                patient_id=step,
                global_id=global_id,
                epoch=epoch,
                warmup_epochs=args.memory_warmup_epochs,
                topk=args.memory_topk
            )

            # result = model(seq_input)
            loss_bce = F.binary_cross_entropy_with_logits(
                result, torch.FloatTensor(loss_bce_target).to(device)
            )
            loss_multi = F.multilabel_margin_loss(
                F.sigmoid(result), torch.LongTensor(loss_multi_target).to(device)
            )

            result = F.sigmoid(result).detach().cpu().numpy()[0]
            result[result >= args.eval_threshold] = 1
            result[result < args.eval_threshold] = 0
            y_label = np.where(result == 1)[0]
            current_ddi_rate = ddi_rate_score([[y_label]], path=args.ddi_adj_path)

            if args.is_ddi_loss:
                if current_ddi_rate <= args.target_ddi:
                    loss = args.weight_bce * loss_bce + args.weight_multi * loss_multi + args.lambda_review * review_reg
                else:
                    beta = max(0, 1 + (args.target_ddi - current_ddi_rate) / args.kp)
                    loss = (
                        beta * (args.weight_bce * loss_bce + args.weight_multi * loss_multi) + (1 - beta) * loss_ddi
                    )
            else:
                loss =  args.weight_bce  * loss_bce + args.weight_multi * loss_multi + args.lambda_review * review_reg

            # ddi_loss = F.mse_loss(torch.FloatTensor([current_ddi_rate]),torch.FloatTensor([ddi_rate_score([[adm[2]]], path=args.ddi_adj_path)]))
            # loss = 0.97 * loss_bce + 0.03 * loss_multi + ddi_loss

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(counterfactualModel.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += loss.item()
            global_id += 1
        
        if args.is_ddi_loss:
            T *= decay_weight
        
        if memory_bank is not None and ((epoch + 1) % max(1, args.memory_refresh_every) == 0):
            memory_bank.refresh(
                data_train=data_train,
                model=counterfactualModel,
                epoch=epoch + 1,
                use_momentum=True
            )

        print()
        tic2 = time.time()
        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(
            counterfactualModel,
            data_eval,
            vocab_size,
            memory_bank,
            epoch=epoch,
            args=args
        )

        metrics = {
            'ddi_rate': float(ddi_rate),
            'ja': float(ja),
            'prauc': float(prauc),
            'avg_p': float(avg_p),
            'avg_r': float(avg_r),
            'avg_f1': float(avg_f1),
            'avg_med': float(avg_med),
            'train_loss': float(total_loss / max(1, len(data_train)))
        }
        metrics['score'] = float(metrics['ja'] + metrics['prauc'] + metrics['avg_f1'] - args.ddi_penalty * metrics['ddi_rate'])
        tensorboard_write_main(writer, total_loss / len(data_train), metrics, epoch)

        selected_value = metrics[args.select_metric]
        improved = selected_value > best_score
        if improved:
            best_score = selected_value
            best_epoch = epoch
            best_ja = ja
            best_metrics = dict(metrics)
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        print(
            "training time: {}, test time: {}".format(
                time.time() - tic, time.time() - tic2
            )
        )
        history["ja"].append(ja)
        history["ddi_rate"].append(ddi_rate)
        history["avg_p"].append(avg_p)
        history["avg_r"].append(avg_r)
        history["avg_f1"].append(avg_f1)
        history["prauc"].append(prauc)
        history["med"].append(avg_med)

        if epoch >= 5:
            print(
                "ddi: {}, Med: {}, Ja: {}, F1: {}, PRAUC: {}".format(
                    np.mean(history["ddi_rate"][-5:]),
                    np.mean(history["med"][-5:]),
                    np.mean(history["ja"][-5:]),
                    np.mean(history["avg_f1"][-5:]),
                    np.mean(history["prauc"][-5:]),
                )
            )


        '''
        > save
            > CSDG-Rec
                > commit
                    > pretrained_model
                         > similar_patient_pretrained.model
                    > tensorboard_logs
                        > events.out.tfevents.xxxx
                    > train_history
                        > Epoch_XX_JA_XX_DDI_XX_Loss.model ...
                        > history_CSDG-Rec.pkl
        '''          
        if args.save_model_every:
            torch.save(
                counterfactualModel.state_dict(),
                open(
                    os.path.join(
                        train_history_path,
                        "Epoch_{}_JA_{:.4}_DDI_{:.4}_Loss.model".format(
                            epoch, ja, ddi_rate,
                        ),
                    ),
                    "wb",
                ),
            )

        print("best_epoch: {}, best_{}: {:.6f}".format(best_epoch, args.select_metric, best_score))

        if args.early_stop_patience and args.early_stop_patience > 0 and no_improve_epochs >= args.early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch}; no improvement for {no_improve_epochs} epochs.")
            break

        dill.dump(
            history,
            open(
                os.path.join(
                    train_history_path,
                    "history_{}.pkl".format('CSDG-Rec')
                ),
                "wb",
            ),
        )

    # Write a compact machine-readable training summary.
    results_dir = args.results_dir if args.results_dir else train_history_path
    os.makedirs(results_dir, exist_ok=True)
    training_summary = {
        'commit': args.commit,
        'best_epoch': int(best_epoch),
        'best_score': float(best_score),
        'best_metrics': best_metrics,
        'last_metrics': {k: (float(v[-1]) if len(v) > 0 else None) for k, v in history.items()},
        'history': {k: [float(x) for x in v] for k, v in history.items()},
        'args': vars(args),
    }
    with open(os.path.join(results_dir, 'metrics_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(training_summary, f, indent=2, ensure_ascii=False)
    writer.close()
