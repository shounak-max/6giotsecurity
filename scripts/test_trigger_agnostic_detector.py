"""
Stage A: Trigger-Agnostic Gini Attribution Detector Benchmark.

Implements Claude's exact mathematical formulation:
1. Class-conditional margin attribution on clean server root data D_r:
   m_c(x) = z_c(x) - logsumexp_{c' != c} z_{c'}(x)
   g_{i,c,j} = mean_{x in D_r, y != c} | d m_c(x) / d x_j |
   p_{i,c,j} = g_{i,c,j} / sum_l g_{i,c,l}
2. Gini concentration per class:
   G_{i,c} = (1 / (2d)) sum_j sum_l | p_{i,c,j} - p_{i,c,l} |
3. Within-model class contrast anomaly score (protects honest atypical clients):
   a_i = max_c ( G_{i,c} - median_{c'} G_{i,c'} ) / ( 1.4826 * MAD_{c'}(G_{i,c'}) + eps )
4. Cross-client per-class ablation (flags honest atypical clients as false positives):
   a_{i,c}^{cross} = ( G_{i,c} - median_k G_{k,c} ) / ( 1.4826 * MAD_k(G_{k,c}) + eps )
5. Permutation invariance check:
   Permute features -> verify a_i is numerically identical.
6. Multi-seed test grid:
   Disjoint trigger positions (held-out features), varying sizes {1, 3, 5}, magnitudes {2, 5}.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fl_ids.data.nftoniot import load_nf_ton_iot_v2
from fl_ids.data.preprocessor import LeakageFreePreprocessor
from fl_ids.data.partitioner import DirichletPartitioner
from fl_ids.attacks import AdditiveBackdoorAttack
from fl_ids.federation import FederatedClient
from fl_ids.models import EdgeIDSMLP


def compute_gini(p):
    """
    Computes Gini coefficient for 1D non-negative normalized vector p.
    Formula: G = (1 / (2d)) * sum_j sum_l | p_j - p_l |
    """
    d = len(p)
    p_sorted = np.sort(p)
    # Gini = sum_{k=1}^d (2k - d - 1) * p_sorted[k-1] / d
    k = np.arange(1, d + 1)
    gini = float(np.sum((2 * k - d - 1) * p_sorted) / d)
    return max(0.0, gini)


def compute_within_model_attribution_scores(client_models, X_root_scaled, y_root, num_classes, input_dim):
    """
    Computes:
    - Within-model class contrast anomaly score a_i
    - Cross-client anomaly score a_i_cross
    - Detailed G_{i,c} matrix
    """
    N = len(client_models)
    G_matrix = np.zeros((N, num_classes), dtype=np.float32)
    
    tensor_x_root = torch.tensor(X_root_scaled, dtype=torch.float32)
    
    for cid in range(N):
        weights = client_models[cid]
        model = EdgeIDSMLP(input_dim=input_dim, num_classes=num_classes)
        model.set_weights_dict(weights)
        model.eval()
        
        for c in range(num_classes):
            mask = (y_root != c)
            if not np.any(mask):
                continue
            x_c = tensor_x_root[mask].clone().detach().requires_grad_(True)
            logits = model(x_c)
            
            # Margin m_c(x) = z_c(x) - logsumexp_{c' != c} z_{c'}(x)
            z_c = logits[:, c]
            other_indices = [idx for idx in range(num_classes) if idx != c]
            z_other = logits[:, other_indices]
            lse_other = torch.logsumexp(z_other, dim=1)
            margin = z_c - lse_other
            
            # Mean over samples of gradient
            margin_sum = torch.sum(margin)
            margin_sum.backward()
            
            grad_x = x_c.grad.detach().abs().mean(dim=0).numpy() # (d,)
            total_g = float(np.sum(grad_x)) + 1e-12
            p_c = grad_x / total_g
            
            G_matrix[cid, c] = compute_gini(p_c)

    # 1. Within-model class contrast:
    # a_i = max_c ( G_{i,c} - median_{c'} G_{i,c'} ) / ( 1.4826 * MAD_{c'} + eps )
    a_within = np.zeros(N, dtype=np.float32)
    for cid in range(N):
        row = G_matrix[cid]
        med = np.median(row)
        mad = np.median(np.abs(row - med))
        scale = 1.4826 * mad + 1e-6
        contrasts = (row - med) / scale
        a_within[cid] = float(np.max(contrasts))
        
    # 2. Cross-client contrast (ablation):
    # a_{i,c}^{cross} = ( G_{i,c} - median_k G_{k,c} ) / ( 1.4826 * MAD_k(G_{k,c}) + eps )
    a_cross = np.zeros(N, dtype=np.float32)
    cross_matrix = np.zeros((N, num_classes), dtype=np.float32)
    for c in range(num_classes):
        col = G_matrix[:, c]
        med = np.median(col)
        mad = np.median(np.abs(col - med))
        scale = 1.4826 * mad + 1e-6
        cross_matrix[:, c] = (col - med) / scale
    for cid in range(N):
        a_cross[cid] = float(np.max(cross_matrix[cid]))
        
    return a_within, a_cross, G_matrix


def run_stage_a_evaluation():
    print("=" * 70)
    print("STAGE A: TRIGGER-AGNOSTIC GINI ATTRIBUTION DETECTOR BENCHMARK")
    print("=" * 70)
    
    # 1. Load real NetFlow NF-ToN-IoT-V2 data
    seed = 42
    df, labels, meta = load_nf_ton_iot_v2(n_samples=4000, min_rare_samples=100, random_state=seed)
    X = df.to_numpy(dtype=np.float32)
    rare_classes = meta["rare_classes"]
    num_classes = meta["num_classes"]
    input_dim = X.shape[1]
    
    X_clients, y_clients, X_root, y_root = LeakageFreePreprocessor.extract_disjoint_root_set(
        X, labels, root_size=200, random_state=seed
    )
    preproc = LeakageFreePreprocessor(random_state=seed)
    X_clients_scaled = preproc.fit_transform(X_clients)
    X_root_scaled = preproc.transform(X_root)
    
    # Partition with k=3 co-located rare-class holders
    partitioner = DirichletPartitioner(num_clients=10, alpha=0.3, random_state=seed)
    client_splits, p_meta = partitioner.partition(y_clients, atypical_class_ids=rare_classes)
    atypical_clients = set(p_meta["atypical_clients"])
    
    print(f"Dataset: NF-ToN-IoT-V2, input_dim={input_dim}, num_classes={num_classes}")
    print(f"Atypical clients: {sorted(list(atypical_clients))}")
    
    # Disjoint test trigger grid (never using development trigger locations [0, 1, 2]):
    # Testing sizes {1, 3, 5}, magnitudes {2, 5}, and disjoint locations in [20..40]
    eval_configs = [
        {"name": "Size 1, Mag 5 (feat 25)", "feats": [25], "mag": 5.0, "boost": 2.0},
        {"name": "Size 3, Mag 5 (feats 22,23,24)", "feats": [22, 23, 24], "mag": 5.0, "boost": 2.0},
        {"name": "Size 5, Mag 5 (feats 30..34)", "feats": [30, 31, 32, 33, 34], "mag": 5.0, "boost": 2.0},
        {"name": "Size 3, Mag 2 (feats 26,27,28)", "feats": [26, 27, 28], "mag": 2.0, "boost": 2.0},
        {"name": "Size 3, Mag 5, Boost 5 (feats 35,36,37)", "feats": [35, 36, 37], "mag": 5.0, "boost": 5.0},
    ]
    
    # Seeds to evaluate
    eval_seeds = [101, 202, 303, 404, 505]
    
    results_within_all = []
    results_within_atyp = []
    results_cross_all = []
    results_cross_atyp = []
    
    # Set validation threshold tau = 2.5 (MAD multiplier)
    tau = 2.5
    atypical_rejections_within = []
    atypical_rejections_cross = []
    
    for cfg in eval_configs:
        print(f"\n--- Evaluating Config: {cfg['name']} ---")
        cfg_auc_within_all = []
        cfg_auc_within_atyp = []
        cfg_auc_cross_all = []
        cfg_auc_cross_atyp = []
        
        for s in eval_seeds:
            # Set random malicious clients from non-holders
            non_holders = [cid for cid in range(10) if cid not in atypical_clients]
            rng = np.random.RandomState(s)
            malicious_ids = list(rng.choice(non_holders, size=2, replace=False))
            
            attack_obj = AdditiveBackdoorAttack(
                trigger_features=cfg["feats"],
                trigger_magnitude=cfg["mag"],
                target_class=0,
                poison_rate=0.5,
                boost_factor=cfg["boost"],
            )
            
            global_model = EdgeIDSMLP(input_dim=input_dim, num_classes=num_classes)
            global_weights = global_model.get_weights_dict()
            
            # Train for 3 federated rounds so models converge
            for r in range(3):
                client_models = {}
                for cid in range(10):
                    c_atk = attack_obj if cid in malicious_ids else None
                    c = FederatedClient(
                        client_id=cid,
                        X_train=X_clients_scaled[client_splits[cid]],
                        y_train=y_clients[client_splits[cid]],
                        batch_size=32,
                        local_epochs=1,
                        learning_rate=0.01,
                        attack=c_atk,
                    )
                    w, _ = c.train(global_weights)
                    client_models[cid] = w
                
                # Server aggregates honest + malicious to update global_weights for next round
                # Simple average for federation
                new_weights = {}
                for k in global_weights.keys():
                    new_weights[k] = np.mean([client_models[cid][k] for cid in range(10)], axis=0)
                global_weights = new_weights
                
            a_within, a_cross, _ = compute_within_model_attribution_scores(
                client_models, X_root_scaled, y_root, num_classes, input_dim
            )
            
            # Ground truth: 1 for attacker, 0 for honest
            y_true_all = np.array([1 if cid in malicious_ids else 0 for cid in range(10)])
            auc_w_all = roc_auc_score(y_true_all, a_within)
            auc_c_all = roc_auc_score(y_true_all, a_cross)
            cfg_auc_within_all.append(auc_w_all)
            cfg_auc_cross_all.append(auc_c_all)
            
            # Attackers vs Honest Atypical ONLY
            sub_ids = malicious_ids + list(atypical_clients)
            y_true_atyp = np.array([1 if cid in malicious_ids else 0 for cid in sub_ids])
            auc_w_atyp = roc_auc_score(y_true_atyp, a_within[sub_ids])
            auc_c_atyp = roc_auc_score(y_true_atyp, a_cross[sub_ids])
            cfg_auc_within_atyp.append(auc_w_atyp)
            cfg_auc_cross_atyp.append(auc_c_atyp)
            
            # False rejection on honest atypical at tau
            atyp_list = list(atypical_clients)
            frr_w = np.mean([1 if a_within[c] > tau else 0 for c in atyp_list])
            frr_c = np.mean([1 if a_cross[c] > tau else 0 for c in atyp_list])
            atypical_rejections_within.append(frr_w)
            atypical_rejections_cross.append(frr_c)
            
        print(f"  Within-Model AUC (All):     {np.mean(cfg_auc_within_all):.4f} +/- {np.std(cfg_auc_within_all):.4f}")
        print(f"  Within-Model AUC (vs Atyp): {np.mean(cfg_auc_within_atyp):.4f} +/- {np.std(cfg_auc_within_atyp):.4f}")
        print(f"  Cross-Client AUC (vs Atyp): {np.mean(cfg_auc_cross_atyp):.4f} +/- {np.std(cfg_auc_cross_atyp):.4f}")
        
        results_within_all.extend(cfg_auc_within_all)
        results_within_atyp.extend(cfg_auc_within_atyp)
        results_cross_all.extend(cfg_auc_cross_all)
        results_cross_atyp.extend(cfg_auc_cross_atyp)
        
    print("\n" + "=" * 70)
    print("STAGE A SUMMARY & PRE-REGISTERED GO/NO-GO AUDIT")
    print("=" * 70)
    
    mean_w_all = np.mean(results_within_all)
    mean_w_atyp = np.mean(results_within_atyp)
    mean_c_atyp = np.mean(results_cross_atyp)
    frr_w_mean = np.mean(atypical_rejections_within)
    frr_c_mean = np.mean(atypical_rejections_cross)
    
    ci_w_atyp_lower = mean_w_atyp - 1.96 * (np.std(results_within_atyp) / np.sqrt(len(results_within_atyp)))
    
    print(f"Total Evaluations across seeds and trigger configs: {len(results_within_all)}")
    print(f"1. Within-Model Class Contrast AUC (All Clients):            {mean_w_all:.4f}")
    print(f"2. Within-Model Class Contrast AUC (vs Honest Atypical Only): {mean_w_atyp:.4f} (95% CI lower: {ci_w_atyp_lower:.4f})")
    print(f"3. Cross-Client Anomaly AUC (vs Honest Atypical Only):        {mean_c_atyp:.4f}")
    print(f"4. False Rejection Rate on Honest Atypical (Within-Model):    {frr_w_mean * 100:.1f}%")
    print(f"5. False Rejection Rate on Honest Atypical (Cross-Client):    {frr_c_mean * 100:.1f}%")
    
    # 7. Permutation Test
    print("\n--- PERMUTATION INVARIANCE TEST ---")
    p_rng = np.random.RandomState(999)
    perm = p_rng.permutation(input_dim)
    
    # Test on seed 101, config 0
    X_root_perm = X_root_scaled[:, perm]
    # In model, linear1 has shape (hidden, input_dim) -> permute columns
    perm_client_models = {}
    for cid, w in client_models.items():
        w_perm = {k: v.copy() for k, v in w.items()}
        w_perm["feature_extractor.0.weight"] = w["feature_extractor.0.weight"][:, perm]
        perm_client_models[cid] = w_perm
        
    a_w_perm, _, _ = compute_within_model_attribution_scores(
        perm_client_models, X_root_perm, y_root, num_classes, input_dim
    )
    max_diff = np.max(np.abs(a_within - a_w_perm))
    print(f"Max absolute score difference after arbitrary feature permutation: {max_diff:.8e}")
    perm_pass = max_diff < 1e-5
    print(f"Permutation Invariance Test Passed: {perm_pass}")
    
    passed_stage_a = (ci_w_atyp_lower >= 0.90) and (frr_w_mean <= 0.05) and perm_pass
    print(f"\nSTAGE A PRE-REGISTERED GATE STATUS: {'>>> PASSED <<<' if passed_stage_a else 'FAILED'}")
    
    return {
        "mean_w_all": float(mean_w_all),
        "mean_w_atyp": float(mean_w_atyp),
        "ci_w_atyp_lower": float(ci_w_atyp_lower),
        "mean_c_atyp": float(mean_c_atyp),
        "frr_w_mean": float(frr_w_mean),
        "frr_c_mean": float(frr_c_mean),
        "perm_pass": bool(perm_pass),
        "passed": bool(passed_stage_a),
    }


if __name__ == "__main__":
    run_stage_a_evaluation()
