from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from joblib import dump
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.features.build_features import (
    aggregate_events_to_users,
    enrich_features,
    split_Xy,
)

def tstamp() -> str:
    return time.strftime("%H:%M:%S")

def R(p: str | None) -> Path:
    return Path(p).expanduser().resolve() if p else Path("")

def _read_csv(p: Path) -> pd.DataFrame:
    print(f"[{tstamp()}] [READ] {p}", flush=True)
    df = pd.read_csv(p)
    print(f"[{tstamp()}]       shape={df.shape} cols={list(df.columns)}", flush=True)
    return df

def _read_many(files: List[str]) -> pd.DataFrame:
    parts = []
    for f in files:
        p = R(f)
        if not p.exists():
            raise FileNotFoundError(f"Файл не найден: {p}")
        parts.append(_read_csv(p))
    out = pd.concat(parts, axis=0, ignore_index=True, sort=False)
    print(f"[{tstamp()}] [READ] concatenated shape={out.shape}", flush=True)
    return out

def _is_event_level(df: pd.DataFrame) -> bool:
    return {"event_time", "event_type", "user_id"}.issubset(df.columns)

def _aggregate_and_label(
    df_hist_raw: pd.DataFrame,
    df_next_raw: pd.DataFrame,
    df_test_raw: pd.DataFrame,
    df_future_raw: pd.DataFrame | None,
    clip_q: float,
    label_noise_pct: float,
    use_last_window: bool,
    window_days: int,
    synth_cfg: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, bool]:
    assert _is_event_level(df_hist_raw) and _is_event_level(df_next_raw)
    assert _is_event_level(df_test_raw)

    hist_end = pd.to_datetime(df_hist_raw["event_time"], errors="coerce", utc=True).max()
    test_end = pd.to_datetime(df_test_raw["event_time"], errors="coerce", utc=True).max()

    print(f"[{tstamp()}] [PREP] aggregate TRAIN (hist) -> user-level…", flush=True)
    train_user = aggregate_events_to_users(df_hist_raw, period_end=hist_end)
    if use_last_window:
        before = len(train_user)
        train_user = train_user[train_user["days_since_last_visit"] <= float(window_days)].copy()
        print(f"[{tstamp()}] [COHORT] TRAIN active {window_days}d: {len(train_user)} / {before}", flush=True)
    next_users = set(df_next_raw["user_id"].astype("int64").unique())
    train_user["churn"] = (~train_user["user_id"].astype("int64").isin(next_users)).astype(int)
    train_user = enrich_features(train_user, clip_q=clip_q, label_noise_pct=label_noise_pct)

    print(f"[{tstamp()}] [PREP] aggregate TEST (Feb) -> user-level…", flush=True)
    test_user = aggregate_events_to_users(df_test_raw, period_end=test_end)
    if use_last_window:
        before = len(test_user)
        test_user = test_user[test_user["days_since_last_visit"] <= float(window_days)].copy()
        print(f"[{tstamp()}] [COHORT] TEST  active {window_days}d: {len(test_user)} / {before}", flush=True)

    synthetic = False
    if df_future_raw is not None and not df_future_raw.empty:
        future_users = set(df_future_raw["user_id"].astype("int64").unique())
        test_user["churn"] = (~test_user["user_id"].astype("int64").isin(future_users)).astype(int)
        print(f"[{tstamp()}] [LABEL] Test labels from FUTURE month (real)")
    else:
        thr = int(synth_cfg.get("inactive_days_threshold", 30))
        test_user["churn"] = (test_user["days_since_last_visit"] > thr).astype(int)
        synthetic = True
        print(f"[{tstamp()}] [LABEL] Test labels are SYNTHETIC: churn=1 if days_since_last_visit>{thr}d")

    test_user = enrich_features(test_user, clip_q=clip_q, label_noise_pct=0.0)
    return train_user, test_user, synthetic


def _get_candidate_models(cfg: Dict, y_train: pd.Series) -> Dict[str, Pipeline]:
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    spw = float(n_neg / max(n_pos, 1))
    print(f"[{tstamp()}] [MODEL] class_balance: pos={n_pos} neg={n_neg} spw={spw:.3f}", flush=True)

    pipes: Dict[str, Pipeline] = {}

    try:
        from lightgbm import LGBMClassifier
        params = cfg["candidates"].get("lightgbm", {})
        lgbm = LGBMClassifier(**params, scale_pos_weight=spw, random_state=42)
        pipes["lightgbm"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", lgbm),
        ])
    except Exception as e:
        print(f"[WARN] LightGBM недоступен: {e}")

    try:
        from xgboost import XGBClassifier
        xgbp = cfg["candidates"].get("xgboost", {})
        xgb = XGBClassifier(
            **xgbp,
            objective="binary:logistic",
            eval_metric="auc",
            scale_pos_weight=spw,
            random_state=42,
            use_label_encoder=False,
            n_jobs=-1,
        )
        pipes["xgboost"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", xgb),
        ])
    except Exception as e:
        print(f"[WARN] XGBoost недоступен: {e}")

    lrp = cfg["candidates"].get("logreg", {})
    lr = LogisticRegression(**lrp, solver="lbfgs")
    pipes["logreg"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", lr),
    ])

    return pipes


def train(cfg_path: str = "configs/config.yaml"):
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
    data_cfg = cfg["data"]
    feats_cfg = cfg["features"]
    cohort_cfg = data_cfg.get("cohort", {})
    model_cfg = cfg["model"]
    synth_cfg = cfg.get("synthetic_test_label", {"enabled": True, "inactive_days_threshold": 30})
    art = cfg["artifacts"]

    future_files = data_cfg.get("future_files") or []
    has_future = len(future_files) > 0

    print("[INFO] Time-based split: TRAIN hist→next; TEST test→(future or synthetic)", flush=True)

    df_hist_raw = _read_many(data_cfg["train_files"])
    df_next_raw = _read_many(data_cfg["test_files"])
    df_test_raw = _read_many(data_cfg["test_files"])
    df_future_raw = _read_many(future_files) if has_future else None

    train_user, test_user, is_synth = _aggregate_and_label(
        df_hist_raw, df_next_raw, df_test_raw, df_future_raw,
        clip_q=feats_cfg["clip_quantile"],
        label_noise_pct=feats_cfg.get("label_noise_pct", 0.0),
        use_last_window=bool(cohort_cfg.get("use_last_window", True)),
        window_days=int(cohort_cfg.get("window_days", 14)),
        synth_cfg=synth_cfg,
    )

    Path(data_cfg["processed_train_path"]).parent.mkdir(parents=True, exist_ok=True)
    Path(data_cfg["processed_test_path"]).parent.mkdir(parents=True, exist_ok=True)
    train_user.to_csv(R(data_cfg["processed_train_path"]), index=False)
    test_user.to_csv(R(data_cfg["processed_test_path"]), index=False)
    print(f"[{tstamp()}] [SAVE] processed train/test written", flush=True)

    X_train, y_train = split_Xy(train_user)
    X_test, y_test = split_Xy(test_user)
    print(f"[{tstamp()}] [SPLIT] X_train={X_train.shape} y_train={y_train.shape} | X_test={X_test.shape} y_test={y_test.shape}",
          flush=True)

    X_tr, X_va, y_tr, y_va = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    pipes = _get_candidate_models(model_cfg, y_train)

    rows = []
    best_name, best_pipe, best_auc = None, None, -1.0
    thresholds = np.linspace(0.05, 0.95, 19)

    for name, pipe in pipes.items():
        print(f"[{tstamp()}] [FIT] {name} …", flush=True)
        t0 = time.time()
        pipe.fit(X_tr, y_tr)
        oof = pipe.predict_proba(X_va)[:, 1]
        roc_auc = roc_auc_score(y_va, oof)
        pr_auc = average_precision_score(y_va, oof)
        f1s = [f1_score(y_va, (oof >= t).astype(int)) for t in thresholds]
        best_idx = int(np.argmax(f1s))
        rows.append({"model": name, "roc_auc": float(roc_auc), "pr_auc": float(pr_auc),
                     "f1": float(f1s[best_idx]), "best_thr": float(thresholds[best_idx]),
                     "fit_sec": float(time.time()-t0)})
        if roc_auc > best_auc:
            best_auc = roc_auc; best_name = name; best_pipe = pipe

    assert best_pipe is not None
    print(f"[{tstamp()}] [BEST] {best_name} (ROC-AUC={best_auc:.3f}) — дообучаем на всём train…", flush=True)
    best_pipe.fit(X_train, y_train)

    feat_cols = list(X_train.columns)
    Path(art["feature_columns_path"]).parent.mkdir(parents=True, exist_ok=True)
    with open(R(art["feature_columns_path"]), "w", encoding="utf-8") as f:
        json.dump(feat_cols, f, ensure_ascii=False, indent=2)

    proba_test = best_pipe.predict_proba(X_test)[:, 1]
    best_thr = float(max(rows, key=lambda r: r["roc_auc"])["best_thr"])
    y_pred_test = (proba_test >= best_thr).astype(int)

    test_metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba_test)),
        "pr_auc": float(average_precision_score(y_test, proba_test)),
        "f1": float(f1_score(y_test, y_pred_test)),
        "best_threshold_used": best_thr,
        "n_test": int(len(y_test)),
        "n_pos_test": int((y_test == 1).sum()),
        "n_neg_test": int((y_test == 0).sum()),
        "labels_source": "future" if (df_future_raw is not None and not df_future_raw.empty)
                         else f"synthetic_inactive_days>{int(synth_cfg.get('inactive_days_threshold',30))}",
    }
    print(f"[TEST] {test_metrics}", flush=True)

    Path(art["model_path"]).parent.mkdir(parents=True, exist_ok=True)
    dump(best_pipe, R(art["model_path"]))
    pd.DataFrame(rows).to_csv(R(art["candidates_report_path"]), index=False)

    preds = test_user[["user_id"]].copy()
    preds["churn_proba"] = proba_test
    preds["churn_true"] = y_test
    preds.to_csv(R(art["test_predictions_path"]), index=False)

    final_metrics = {
        "mode": "hist->next (train) + test->future_or_synthetic (test)",
        "chosen_model": best_name,
        "train_holdout": {"best_by": "roc_auc", "candidates": rows},
        "test_metrics": test_metrics,
    }
    R(art["metrics_path"]).write_text(json.dumps(final_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{tstamp()}] [SAVE] model + metrics + predictions written", flush=True)

    try:
        est = best_pipe.named_steps["model"]
        if hasattr(est, "feature_importances_"):
            imp = pd.Series(est.feature_importances_, index=X_train.columns).sort_values(ascending=False)
            imp.to_csv(R(art["feature_importances_path"]), header=["importance"])
            print(f"[{tstamp()}] [SAVE] feature importances", flush=True)
    except Exception as e:
        print(f"[WARN] importances: {e}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    train(args.config)
