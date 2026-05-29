# ============================================================
# Google Colab-ready implementation
# FL: NO_DP vs FIXED_HDP vs AS-HDP++ (Ablation A1..A4)
# Multi-seed + CI + clear DP accounting
# ============================================================

# In Google Colab, uncomment the next line if the packages are missing:
# !pip -q install numpy pandas scikit-learn matplotlib torch scipy

import os, math, copy, random, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy import stats

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_to_numeric_series(s: pd.Series) -> pd.Series:
    try:
        return pd.to_numeric(s)
    except Exception:
        return s


def clean_df_numeric_only(df: pd.DataFrame, keep_cols_as_is: Optional[List[str]] = None) -> pd.DataFrame:
    keep_cols_as_is = keep_cols_as_is or []
    df = df.dropna(axis=1, how="all").copy()
    df = df.replace(["?", "NA", "N/A", "na", "n/a", "null", "None", ""], np.nan)

    for c in df.columns:
        if c in keep_cols_as_is:
            continue
        if df[c].dtype == object:
            df[c] = safe_to_numeric_series(df[c])

    for c in df.columns:
        if c in keep_cols_as_is:
            continue
        if df[c].dtype == object:
            df[c], _ = pd.factorize(df[c].astype(str))

    for c in df.columns:
        if c in keep_cols_as_is:
            continue
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    return df


class TabDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class FCNN(nn.Module):
    def __init__(self, d_in: int, hidden=(128, 64), dropout=0.08):
        super().__init__()
        layers = []
        prev = d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def bce_logits_loss(logits, y, pos_weight: Optional[torch.Tensor] = None):
    return nn.functional.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)


@torch.no_grad()
def predict_proba(model: nn.Module, loader: DataLoader) -> np.ndarray:
    model.eval()
    out = []
    for xb, _ in loader:
        xb = xb.to(DEVICE)
        prob = torch.sigmoid(model(xb)).cpu().numpy().ravel()
        out.append(prob)
    return np.concatenate(out) if len(out) else np.array([])


def eval_metrics(model: nn.Module, X: np.ndarray, y: np.ndarray, batch_size=2048) -> Dict[str, float]:
    dl = DataLoader(TabDataset(X, y), batch_size=batch_size, shuffle=False)
    prob = predict_proba(model, dl)
    pred = (prob >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auprc": float(average_precision_score(y, prob)),
    }


def make_non_iid_partitions(X, y, n_clients=8, dirichlet=0.6, seed=42):
    rng = np.random.default_rng(seed)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    rng.shuffle(idx0)
    rng.shuffle(idx1)

    def split(idxs):
        p = rng.dirichlet([dirichlet] * n_clients)
        cuts = (p * len(idxs)).astype(int)
        cuts[-1] += len(idxs) - cuts.sum()
        out, s = [], 0
        for c in cuts:
            out.append(idxs[s:s + c])
            s += c
        return out

    s0 = split(idx0)
    s1 = split(idx1)
    clients = [np.concatenate([s0[i], s1[i]]) for i in range(n_clients)]
    for i in range(n_clients):
        rng.shuffle(clients[i])
    return clients


def l2_norm_update(update: List[torch.Tensor]) -> float:
    s = 0.0
    for t in update:
        s += float(torch.sum(t.detach() ** 2).cpu())
    return float(math.sqrt(s + 1e-12))


def clip_update(update: List[torch.Tensor], clip_norm: float):
    n = l2_norm_update(update)
    if n <= clip_norm:
        return update, n
    scale = clip_norm / (n + 1e-12)
    return [t * scale for t in update], n


def add_laplace_noise(update: List[torch.Tensor], scale_b: float):
    if scale_b <= 0:
        return update
    out = []
    for t in update:
        noise = torch.distributions.Laplace(0.0, scale_b).sample(t.shape).to(t.device)
        out.append(t + noise)
    return out


def add_gaussian_noise(update: List[torch.Tensor], sigma: float):
    if sigma <= 0:
        return update
    return [t + sigma * torch.randn_like(t) for t in update]


def apply_update(global_model, update: List[torch.Tensor], lr_global=1.0):
    with torch.no_grad():
        for p, du in zip(global_model.parameters(), update):
            p.add_(lr_global * du)


def compute_feature_sensitivity(Xc: np.ndarray, alpha=0.65, beta=0.35, eps=1e-12) -> float:
    std = Xc.std(axis=0)
    rng = Xc.max(axis=0) - Xc.min(axis=0)
    sens = alpha * (std + eps) + beta * (rng + eps)
    return float(np.mean(sens))


def squash_positive(x: float, cap: float = 3.0, eps: float = 1e-12) -> float:
    x = max(float(x), eps)
    return float(cap * (x / (x + cap)))


def server_sensitivity_proxy(client_sens_list: List[float], clipped_norms: List[float], clip_bound: float, cap: float = 3.0) -> float:
    s_med = float(np.median(np.array(client_sens_list, dtype=np.float64))) if client_sens_list else 1.0
    medn = float(np.median(np.array(clipped_norms, dtype=np.float64))) if clipped_norms else clip_bound
    closeness = float(np.clip(medn / (clip_bound + 1e-12), 0.0, 1.0))
    s_eff = s_med * (0.6 + 0.4 * closeness)
    return squash_positive(s_eff, cap=cap)


@dataclass
class ClipAdapt:
    init: float = 1.6
    q: float = 0.80
    ema: float = 0.95
    cmin: float = 0.25
    cmax: float = 6.0
    max_growth: float = 1.15
    min_growth: float = 0.90


def update_clip_bound(prev: float, raw_norms: List[float], cfg: ClipAdapt) -> float:
    if not raw_norms:
        return prev
    target = float(np.quantile(np.array(raw_norms, dtype=np.float64), cfg.q))
    new_c = cfg.ema * prev + (1.0 - cfg.ema) * target
    new_c = min(new_c, prev * cfg.max_growth)
    new_c = max(new_c, prev * cfg.min_growth)
    return float(np.clip(new_c, cfg.cmin, cfg.cmax))


class AdaptiveBudgetScheduler:
    def __init__(self, eps_total: float, rounds: int, warmup_rounds: int = 12,
                 warmup_mult: float = 2.6, late_gamma: float = 1.25,
                 stagnation_beta: float = 1.6, min_eps: float = 0.15,
                 max_eps: float = 2.5):
        self.eps_total = float(eps_total)
        self.rounds = int(rounds)
        self.warmup_rounds = int(warmup_rounds)
        self.warmup_mult = float(warmup_mult)
        self.late_gamma = float(late_gamma)
        self.stagnation_beta = float(stagnation_beta)
        self.min_eps = float(min_eps)
        self.max_eps = float(max_eps)
        self.used = 0.0
        r = np.arange(1, rounds + 1, dtype=np.float64)
        w = (r / rounds) ** self.late_gamma
        if self.warmup_rounds > 0:
            w[:self.warmup_rounds] *= self.warmup_mult
        self.base_w = w / (w.sum() + 1e-12)

    def alloc(self, r_idx: int, val_improve: float) -> float:
        rem = max(self.eps_total - self.used, 0.0)
        rem_rounds = self.rounds - r_idx + 1
        if r_idx == self.rounds:
            eps = rem
            self.used += eps
            return float(eps)
        if rem <= 0:
            return 0.0
        w_slice = self.base_w[r_idx - 1:]
        w_slice = w_slice / (w_slice.sum() + 1e-12)
        eps_base = rem * float(w_slice[0])
        stagnation = max(0.0, 0.0006 - float(val_improve))
        eps = eps_base * (1.0 + self.stagnation_beta * stagnation)
        avg_allow = rem / rem_rounds
        eps = min(eps, max(avg_allow * 2.3, self.min_eps))
        eps = float(np.clip(eps, self.min_eps, self.max_eps))
        eps = min(eps, rem)
        self.used += eps
        return float(eps)


def contribution_scores(updates, losses, tau=1.0, weight_cap=0.16):
    mags = np.array([l2_norm_update(u) for u in updates], dtype=np.float64)
    wperf = np.exp(-np.array(losses, dtype=np.float64) / max(tau, 1e-8))
    scores = np.maximum(mags * wperf, 1e-12)
    weights = scores / (scores.sum() + 1e-12)
    weights = np.minimum(weights, weight_cap)
    weights = weights / (weights.sum() + 1e-12)
    return weights


def gaussian_sigma_for_eps_delta(eps: float, delta: float, sensitivity: float) -> float:
    eps = max(float(eps), 1e-12)
    delta = max(float(delta), 1e-12)
    return float(sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / eps)


def eps_from_gaussian_sigma(delta: float, sensitivity: float, sigma: float) -> float:
    sigma = max(float(sigma), 1e-12)
    delta = max(float(delta), 1e-12)
    return float(sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / sigma)


def local_train_client(global_model, Xc, yc, cfg, pos_weight: Optional[torch.Tensor]):
    model = copy.deepcopy(global_model).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    dl = DataLoader(TabDataset(Xc, yc), batch_size=cfg.batch_size, shuffle=True)
    model.train()
    for _ in range(cfg.local_epochs):
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = bce_logits_loss(model(xb), yb, pos_weight=pos_weight)
            loss.backward()
            opt.step()
    update = []
    with torch.no_grad():
        for pn, po in zip(model.parameters(), global_model.parameters()):
            update.append((pn.data - po.data).detach().clone())
    m = eval_metrics(model, Xc, yc, batch_size=cfg.batch_size)
    loss_proxy = 1.0 - m["accuracy"]
    return update, loss_proxy


@dataclass
class FLConfig:
    n_clients: int = 8
    rounds: int = 30
    local_epochs: int = 3
    batch_size: int = 256
    lr: float = 9e-4
    weight_decay: float = 1e-4
    fixed_clip_norm: float = 1.0
    clip_adapt: ClipAdapt = field(default_factory=ClipAdapt)
    alpha: float = 0.65
    beta: float = 0.35
    sens_cap: float = 3.0
    eps_ldp_total: float = 12.0
    eps_cdp_total: float = 8.0
    delta_cdp: float = 1e-5
    warmup_rounds: int = 12
    warmup_mult_ldp: float = 2.4
    warmup_mult_cdp: float = 3.0
    late_gamma_ldp: float = 1.20
    late_gamma_cdp: float = 1.15
    stagnation_beta: float = 1.6
    min_eps_round: float = 0.15
    max_eps_round: float = 2.5
    tau: float = 1.0
    weight_cap: float = 0.16
    use_server_momentum: bool = True
    server_momentum: float = 0.6
    use_pos_weight: bool = True


@dataclass
class AblationFlags:
    adapt_clip: bool = True
    adapt_budget: bool = True
    adapt_client_sens: bool = True
    adapt_server_sens: bool = True


def run_fl_mode(mode: str, Xtr, ytr, Xva, yva, Xte, yte, d_in,
                cfg: FLConfig, flags: Optional[AblationFlags], seed: int):
    assert mode in {"NO_DP", "FIXED_HDP", "AS"}
    if flags is None:
        flags = AblationFlags()
    set_seed(seed)
    global_model = FCNN(d_in).to(DEVICE)
    client_idxs = make_non_iid_partitions(Xtr, ytr, cfg.n_clients, dirichlet=0.6, seed=seed)

    if cfg.use_pos_weight:
        pos = float(np.sum(ytr == 1))
        neg = float(np.sum(ytr == 0))
        pw = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=DEVICE)
    else:
        pw = None

    eps_ldp_fixed = cfg.eps_ldp_total / cfg.rounds
    eps_cdp_fixed = cfg.eps_cdp_total / cfg.rounds

    sched_ldp = sched_cdp = None
    if mode == "AS" and flags.adapt_budget:
        sched_ldp = AdaptiveBudgetScheduler(cfg.eps_ldp_total, cfg.rounds, cfg.warmup_rounds,
                                            cfg.warmup_mult_ldp, cfg.late_gamma_ldp,
                                            cfg.stagnation_beta, cfg.min_eps_round, cfg.max_eps_round)
        sched_cdp = AdaptiveBudgetScheduler(cfg.eps_cdp_total, cfg.rounds, cfg.warmup_rounds,
                                            cfg.warmup_mult_cdp, cfg.late_gamma_cdp,
                                            cfg.stagnation_beta, cfg.min_eps_round, cfg.max_eps_round)

    fixed_sens = squash_positive(compute_feature_sensitivity(Xtr, alpha=cfg.alpha, beta=cfg.beta), cap=cfg.sens_cap)
    C_r = cfg.clip_adapt.init
    v = None
    hist = {"val_auprc": [], "clip_bound": [], "epsL_r": [], "epsC_r": [],
            "lap_b_med": [], "sigma": [], "epsC_real": [], "epsL_real": []}
    best_state = None
    best_val = -1.0
    prev_val = None
    epsL_used = 0.0
    epsC_used = 0.0

    for r in range(1, cfg.rounds + 1):
        val_improve = 0.001 if prev_val is None else float(hist["val_auprc"][-1] - prev_val)
        if mode == "NO_DP":
            epsL_r, epsC_r = 0.0, 0.0
        elif mode == "FIXED_HDP":
            epsL_r, epsC_r = float(eps_ldp_fixed), float(eps_cdp_fixed)
        else:
            if flags.adapt_budget:
                epsL_r = float(sched_ldp.alloc(r, val_improve))
                epsC_r = float(sched_cdp.alloc(r, val_improve))
            else:
                epsL_r, epsC_r = float(eps_ldp_fixed), float(eps_cdp_fixed)
        epsL_used += epsL_r
        epsC_used += epsC_r

        updates, losses, raw_norms, clipped_norms, client_sens_list = [], [], [], [], []
        for k in range(cfg.n_clients):
            idx = client_idxs[k]
            Xc, yc = Xtr[idx], ytr[idx]
            up, loss_k = local_train_client(global_model, Xc, yc, cfg, pw)
            raw_norms.append(l2_norm_update(up))

            if mode == "NO_DP":
                clip_bound = 1e9
            elif mode == "FIXED_HDP":
                clip_bound = cfg.fixed_clip_norm
            else:
                clip_bound = C_r if flags.adapt_clip else cfg.fixed_clip_norm

            up, _ = clip_update(up, clip_bound)
            clipped_norms.append(l2_norm_update(up))

            if mode == "NO_DP":
                sens_c = 0.0
            elif mode == "FIXED_HDP":
                sens_c = float(fixed_sens)
            else:
                if flags.adapt_client_sens:
                    s_raw = compute_feature_sensitivity(Xc, alpha=cfg.alpha, beta=cfg.beta)
                    sens_c = squash_positive(s_raw, cap=cfg.sens_cap)
                else:
                    sens_c = float(fixed_sens)

            client_sens_list.append(float(sens_c))
            updates.append(up)
            losses.append(loss_k)

        if mode == "AS" and flags.adapt_clip:
            C_r = update_clip_bound(C_r, raw_norms, cfg.clip_adapt)

        lap_b_list, epsL_real_list = [], []
        if mode != "NO_DP":
            updates_noisy = []
            for i in range(cfg.n_clients):
                clip_bound = cfg.fixed_clip_norm if mode == "FIXED_HDP" else (C_r if flags.adapt_clip else cfg.fixed_clip_norm)
                S_client = float(client_sens_list[i]) * clip_bound
                b = S_client / max(epsL_r, 1e-12)
                epsL_real = S_client / max(b, 1e-12)
                lap_b_list.append(float(b))
                epsL_real_list.append(float(epsL_real))
                updates_noisy.append(add_laplace_noise([t.clone() for t in updates[i]], float(b)))
            updates = updates_noisy

        lap_b_med = float(np.median(np.array(lap_b_list, dtype=np.float64))) if lap_b_list else 0.0
        epsL_real_med = float(np.median(np.array(epsL_real_list, dtype=np.float64))) if epsL_real_list else 0.0

        weights = contribution_scores(updates, losses, tau=cfg.tau, weight_cap=cfg.weight_cap)
        agg = []
        for t in range(len(updates[0])):
            s = torch.zeros_like(updates[0][t])
            for i in range(len(updates)):
                s += float(weights[i]) * updates[i][t]
            agg.append(s)

        sigma = 0.0
        epsC_real = 0.0
        if mode == "NO_DP":
            agg_priv = agg
        else:
            if mode == "FIXED_HDP":
                srv_sens = float(fixed_sens)
                clip_bound = cfg.fixed_clip_norm
            else:
                clip_bound = C_r if flags.adapt_clip else cfg.fixed_clip_norm
                if flags.adapt_server_sens:
                    srv_sens = server_sensitivity_proxy(client_sens_list, clipped_norms, clip_bound=clip_bound, cap=cfg.sens_cap)
                else:
                    srv_sens = float(np.median(np.array(client_sens_list, dtype=np.float64)))
            S_server = srv_sens * clip_bound
            sigma = gaussian_sigma_for_eps_delta(epsC_r, cfg.delta_cdp, S_server)
            epsC_real = eps_from_gaussian_sigma(cfg.delta_cdp, S_server, sigma)
            agg_priv = add_gaussian_noise(agg, sigma)

        if cfg.use_server_momentum:
            if v is None:
                v = [torch.zeros_like(t) for t in agg_priv]
            v = [cfg.server_momentum * v_i + t_i for v_i, t_i in zip(v, agg_priv)]
            step = v
        else:
            step = agg_priv

        apply_update(global_model, step, lr_global=1.0)
        vm = eval_metrics(global_model, Xva, yva, batch_size=cfg.batch_size)

        hist["val_auprc"].append(vm["auprc"])
        hist["clip_bound"].append(float(C_r if (mode == "AS" and flags.adapt_clip) else (cfg.fixed_clip_norm if mode != "NO_DP" else 0.0)))
        hist["epsL_r"].append(float(epsL_r))
        hist["epsC_r"].append(float(epsC_r))
        hist["lap_b_med"].append(float(lap_b_med))
        hist["sigma"].append(float(sigma))
        hist["epsC_real"].append(float(epsC_real))
        hist["epsL_real"].append(float(epsL_real_med))

        if vm["auprc"] > best_val + 1e-4:
            best_val = vm["auprc"]
            best_state = copy.deepcopy(global_model.state_dict())
        prev_val = vm["auprc"]

        if r % 5 == 0 or r == 1:
            print(f"[seed={seed}][{mode}] r={r:02d} ValAUPRC={vm['auprc']:.4f} C={hist['clip_bound'][-1]:.3f} epsL={epsL_r:.3f} epsC={epsC_r:.3f} lap_b_med={lap_b_med:.4g} sigma={sigma:.4g} delta={cfg.delta_cdp:g}")

    if best_state is not None:
        global_model.load_state_dict(best_state)
    tm = eval_metrics(global_model, Xte, yte, batch_size=cfg.batch_size)
    privacy = {
        "epsL_total_target": float(cfg.eps_ldp_total),
        "epsC_total_target": float(cfg.eps_cdp_total),
        "epsL_total_used_sum_epsL_r": float(epsL_used),
        "epsC_total_used_sum_epsC_r": float(epsC_used),
        "delta_cdp": float(cfg.delta_cdp),
        "eps_hybrid_total_used": float(epsL_used + epsC_used),
        "delta_hybrid_total": float(cfg.delta_cdp),
    }
    return hist, tm, privacy


def mean_std_ci(x: np.ndarray, alpha=0.05):
    x = np.array(x, dtype=np.float64)
    n = len(x)
    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if n > 1 else 0.0
    if n <= 1:
        return mu, sd, (mu, mu)
    tcrit = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
    half = tcrit * sd / math.sqrt(n)
    return mu, sd, (mu - half, mu + half)


def run_experiments(dataset_name, X_train, y_train, X_val, y_val, X_test, y_test, d_in, output_prefix):
    cfg_tuned = FLConfig(
        n_clients=8, rounds=30, local_epochs=3, batch_size=256,
        lr=9e-4, weight_decay=1e-4, fixed_clip_norm=1.0,
        clip_adapt=ClipAdapt(init=1.6, q=0.80, ema=0.95, cmin=0.25, cmax=6.0, max_growth=1.15, min_growth=0.90),
        alpha=0.65, beta=0.35, sens_cap=3.0,
        eps_ldp_total=12.0, eps_cdp_total=8.0, delta_cdp=1e-5,
        warmup_rounds=12, warmup_mult_ldp=2.4, warmup_mult_cdp=3.0,
        late_gamma_ldp=1.20, late_gamma_cdp=1.15,
        stagnation_beta=1.6, min_eps_round=0.15, max_eps_round=2.5,
        use_pos_weight=True, tau=1.0, weight_cap=0.16,
        use_server_momentum=True, server_momentum=0.6
    )
    experiments = [
        ("NO_DP", "NO_DP", None),
        ("FIXED_HDP", "FIXED_HDP", None),
        ("A1_ClipOnly", "AS", AblationFlags(True, False, False, False)),
        ("A2_Clip+Budget", "AS", AblationFlags(True, True, False, False)),
        ("A3_Clip+Budget+ClientSens", "AS", AblationFlags(True, True, True, False)),
        ("A4_Full_AS_HDP_PP_Tuned", "AS", AblationFlags(True, True, True, True)),
    ]
    SEEDS = [11, 22, 33, 44, 55]
    all_rows, privacy_rows, histories = [], [], {}
    for seed in SEEDS:
        print("\n" + "#" * 80)
        print(f"SEED = {seed}")
        print("#" * 80)
        for exp_name, mode, flags in experiments:
            print("\n" + "=" * 70)
            print(f"RUN: {exp_name} | mode={mode} | seed={seed}")
            print("=" * 70)
            hist, tm, privacy = run_fl_mode(mode, X_train, y_train, X_val, y_val, X_test, y_test,
                                            d_in, copy.deepcopy(cfg_tuned), flags, seed)
            histories[(seed, exp_name)] = hist
            all_rows.append({"seed": seed, "Exp": exp_name,
                             "TestAcc": tm["accuracy"], "TestPrec": tm["precision"],
                             "TestRec": tm["recall"], "TestF1": tm["f1"], "TestAUPRC": tm["auprc"]})
            privacy_rows.append({"seed": seed, "Exp": exp_name, **privacy})
    results_df = pd.DataFrame(all_rows)
    privacy_df = pd.DataFrame(privacy_rows)
    print("\n=== PER-SEED RESULTS ===")
    display(results_df)
    print("\n=== PRIVACY LEDGER ===")
    display(privacy_df)

    def check_totals(privacy_df: pd.DataFrame, exp: str):
        sub = privacy_df[privacy_df["Exp"] == exp].copy()
        if len(sub) == 0:
            return
        max_ldp_err = float(np.max(np.abs(sub["epsL_total_used_sum_epsL_r"] - sub["epsL_total_target"])))
        max_cdp_err = float(np.max(np.abs(sub["epsC_total_used_sum_epsC_r"] - sub["epsC_total_target"])))
        print(f"[Fairness] {exp}: max |epsL_used - epsL_target| = {max_ldp_err:.6f}, max |epsC_used - epsC_target| = {max_cdp_err:.6f}")

    for exp in ["FIXED_HDP", "A4_Full_AS_HDP_PP_Tuned", "A3_Clip+Budget+ClientSens", "A2_Clip+Budget", "A1_ClipOnly"]:
        check_totals(privacy_df, exp)

    summary_rows = []
    metrics = ["TestAcc", "TestPrec", "TestRec", "TestF1", "TestAUPRC"]
    for exp_name, _, _ in experiments:
        sub = results_df[results_df["Exp"] == exp_name]
        row = {"Exp": exp_name, "n_seeds": len(sub)}
        for m in metrics:
            mu, sd, ci = mean_std_ci(sub[m].values, alpha=0.05)
            row[f"{m}_mean"] = mu
            row[f"{m}_std"] = sd
            row[f"{m}_ci95_low"] = ci[0]
            row[f"{m}_ci95_high"] = ci[1]
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows).sort_values("TestAUPRC_mean", ascending=False)
    print("\n=== SUMMARY: Mean ± Std with 95% CI ===")
    display(summary_df)

    results_df.to_csv(f"/content/{output_prefix}_results_per_seed.csv", index=False)
    privacy_df.to_csv(f"/content/{output_prefix}_privacy_ledger_per_seed.csv", index=False)
    summary_df.to_csv(f"/content/{output_prefix}_summary_mean_std_ci.csv", index=False)
    print("\nSaved outputs to /content")

    def plot_val_auprc_mean_ci(exp: str):
        series = []
        for seed in SEEDS:
            h = histories.get((seed, exp), None)
            if h is not None:
                series.append(np.array(h["val_auprc"], dtype=np.float64))
        if not series:
            return
        R = max(len(s) for s in series)
        arr = np.full((len(series), R), np.nan, dtype=np.float64)
        for i, s in enumerate(series):
            arr[i, :len(s)] = s
        mean = np.nanmean(arr, axis=0)
        n = np.sum(~np.isnan(arr), axis=0)
        sd = np.nanstd(arr, axis=0, ddof=1)
        se = sd / np.sqrt(np.maximum(n, 1))
        df_ci = max(int(np.nanmin(n) - 1), 1)
        tcrit = float(stats.t.ppf(0.975, df=df_ci))
        ci = tcrit * se
        x = np.arange(1, len(mean) + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(x, mean, label=f"{exp} mean")
        plt.fill_between(x, mean - ci, mean + ci, alpha=0.2, label="95% CI")
        plt.title(f"{dataset_name}: Validation AUPRC Across Rounds ({exp})")
        plt.xlabel("Round")
        plt.ylabel("Validation AUPRC")
        plt.legend()
        plt.grid(True)
        plt.show()

    for exp in ["NO_DP", "FIXED_HDP", "A4_Full_AS_HDP_PP_Tuned"]:
        plot_val_auprc_mean_ci(exp)

# ============================================================
# Load Diabetes dataset
# ============================================================

DATA_PATH = "/content/diabetes.csv"

def resolve_path(p: str) -> str:
    if os.path.exists(p):
        return p
    alt = os.path.join("/content", os.path.basename(p))
    if os.path.exists(alt):
        return alt
    raise FileNotFoundError(f"Could not find dataset at '{p}' or '{alt}'. Upload diabetes.csv to Colab.")

def infer_target_column(df: pd.DataFrame, preferred: List[str]) -> str:
    cols = list(df.columns)
    for c in preferred:
        if c in cols:
            return c
    last = cols[-1]
    if df[last].nunique() <= 10:
        return last
    small = [c for c in cols if df[c].nunique() <= 10]
    return small[-1] if small else last

def make_binary_labels(y: pd.Series) -> np.ndarray:
    if y.dtype == object:
        y2 = y.astype(str).str.strip().str.lower()
        if y2.nunique() == 2:
            codes, _ = pd.factorize(y2)
            return codes.astype(np.int64)
        pos = {"1", "true", "yes", "y", "positive", "pos"}
        return np.array([1 if v in pos else 0 for v in y2], dtype=np.int64)
    yv = y.values
    uniq = np.unique(yv[~pd.isna(yv)])
    if len(uniq) == 2:
        u0, u1 = np.sort(uniq)
        return (yv == u1).astype(np.int64)
    return (yv > np.median(yv)).astype(np.int64)

raw = pd.read_csv(resolve_path(DATA_PATH))
target = infer_target_column(raw, preferred=["Diabetes_binary", "Outcome", "diabetes", "target", "label"])
y = make_binary_labels(raw[target])
df = clean_df_numeric_only(raw.drop(columns=[target]))
X = df.values.astype(np.float32)
if len(np.unique(y)) < 2:
    raise ValueError("Need 2 classes for binary classification.")

print("Loaded Diabetes dataset | target =", target, "| N =", len(y), "| pos =", int(np.sum(y == 1)), "| neg =", int(np.sum(y == 0)))
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
d_in = X_train.shape[1]

run_experiments("Diabetes", X_train, y_train, X_val, y_val, X_test, y_test, d_in, "diabetes")
