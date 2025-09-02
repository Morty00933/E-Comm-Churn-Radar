import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as pdplt  # dummy alias to avoid style conflicts
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.metrics import auc, precision_recall_curve, roc_curve

st.set_page_config(page_title="Churn Dashboard — Improved", layout="wide")
st.title("📊 User Churn — Hist→Next (Train) & Test→Future (Test)")

MODEL_PATH = Path("models/model.pkl")
METRICS_PATH = Path("models/metrics.json")
PRED_TEST_PATH = Path("models/test_predictions.csv")
PROC_TRAIN = Path("data/processed_train.csv")
PROC_TEST = Path("data/processed_test.csv")
FEATURE_COLS_PATH = Path("models/feature_columns.json")
CANDS_PATH = Path("models/candidates_report.csv")

# -------- helpers --------
def downcast(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], downcast="float")
        elif pd.api.types.is_integer_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], downcast="integer")
    return out

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

@st.cache_data
def load_df(p: Path):
    if not p.exists():
        return pd.DataFrame()
    return downcast(pd.read_csv(p))

def load_feature_columns(model):
    if FEATURE_COLS_PATH.exists():
        return json.loads(FEATURE_COLS_PATH.read_text(encoding="utf-8"))
    try:
        return list(model.named_steps["model"].feature_name_)
    except Exception:
        return None

def align_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    X = df.copy()
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0
    extra = [c for c in X.columns if c not in feature_cols]
    if extra:
        X = X.drop(columns=extra)
    return X[feature_cols]

# -------- load --------
model = load_model()
metrics = load_json(METRICS_PATH)
cands = load_df(CANDS_PATH)
df_train = load_df(PROC_TRAIN)
df_test = load_df(PROC_TEST)
df_pred_test = load_df(PRED_TEST_PATH)
feature_cols = load_feature_columns(model)

if feature_cols is None:
    st.error("Нет списка признаков обучения. Переобучи модель.")
    st.stop()

# -------- sidebar controls --------
st.sidebar.header("⚙️ Настройки")
sample_n = st.sidebar.number_input("Размер сэмпла для графиков/SHAP", 1000, 100000, 20000, step=1000)
random_state = st.sidebar.number_input("random_state", 0, 10000, 42)
topn = st.sidebar.number_input("Top-N at-risk (Test)", 10, 10000, 200, step=10)

# -------- summary --------
st.subheader("Summary")
c1, c2, c3 = st.columns(3)
test_m = metrics.get("test_metrics", {})
c1.metric("Chosen model", metrics.get("chosen_model", "n/a"))
c2.metric("Test ROC-AUC", f"{test_m.get('roc_auc', float('nan')):.3f}")
c3.metric("Test PR-AUC", f"{test_m.get('pr_auc', float('nan')):.3f}")
st.caption("Train выбирался по holdout ROC-AUC среди кандидатов (lightgbm/xgboost/logreg). "
           "Test-метрики считаем по Feb с разметкой из Mar.")

with st.expander("Сравнение кандидатов (holdout из train)"):
    if not cands.empty:
        st.dataframe(cands.sort_values("roc_auc", ascending=False), use_container_width=True)
    else:
        st.info("Нет models/candidates_report.csv")

# -------- tabs --------
tab_metrics, tab_importances, tab_shap, tab_users = st.tabs(
    ["📈 Метрики (Train/Test)", "⭐ Важности", "🧠 SHAP", "👤 Пользователи (Test)"]
)

with tab_metrics:
    if not df_train.empty:
        df_s = df_train if len(df_train) <= sample_n else df_train.sample(sample_n, random_state=random_state)
        y_true = df_s["churn"].astype(int).values
        X_raw = df_s.drop(columns=["user_id", "churn"], errors="ignore")
        X = align_features(X_raw, feature_cols)
        y_pred = model.predict_proba(X)[:, 1]

        fpr, tpr, _ = roc_curve(y_true, y_pred)
        prec, rec, _ = precision_recall_curve(y_true, y_pred)

        st.write("**Train (sample)**")
        fig1 = plt.figure(figsize=(6,5)); plt.plot(fpr, tpr, label=f"AUC={auc(fpr,tpr):.3f}")
        plt.plot([0,1],[0,1], "--"); plt.title("ROC (Train)"); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend()
        st.pyplot(fig1)

        fig2 = plt.figure(figsize=(6,5)); plt.plot(rec, prec, label=f"AUC={auc(rec,prec):.3f}")
        plt.title("PR (Train)"); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend()
        st.pyplot(fig2)

    if not df_pred_test.empty and "churn_true" in df_pred_test.columns:
        st.write("**Test (полный месяц)**")
        y_true_t = df_pred_test["churn_true"].astype(int).values
        y_pred_t = df_pred_test["churn_proba"].values
        fpr, tpr, _ = roc_curve(y_true_t, y_pred_t)
        prec, rec, _ = precision_recall_curve(y_true_t, y_pred_t)

        fig3 = plt.figure(figsize=(6,5)); plt.plot(fpr, tpr, label=f"AUC={auc(fpr,tpr):.3f}")
        plt.plot([0,1],[0,1], "--"); plt.title("ROC (Test)"); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend()
        st.pyplot(fig3)

        fig4 = plt.figure(figsize=(6,5)); plt.plot(rec, prec, label=f"AUC={auc(rec,prec):.3f}")
        plt.title("PR (Test)"); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend()
        st.pyplot(fig4)

with tab_importances:
    fi_path = Path("models/feature_importances.csv")
    if fi_path.exists():
        fi = pd.read_csv(fi_path, index_col=0).reset_index().rename(columns={"index":"feature"})
        st.dataframe(fi, use_container_width=True, height=500)
        fig = plt.figure(figsize=(7,6)); plt.barh(fi["feature"], fi["importance"]); plt.title("Feature Importances")
        st.pyplot(fig)
    else:
        st.info("Важности недоступны (например, для логрег).")

with tab_shap:
    if not df_train.empty:
        df_s = df_train if len(df_train) <= sample_n else df_train.sample(sample_n, random_state=random_state)
        X_raw = df_s.drop(columns=["user_id", "churn"], errors="ignore")
        X = align_features(X_raw, feature_cols)
        try:
            explainer = shap.TreeExplainer(model.named_steps["model"])
            shap_vals = explainer.shap_values(X)
            fig = plt.figure(figsize=(10,6))
            shap.summary_plot(shap_vals, X, show=False, plot_type="bar")
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"SHAP недоступен: {e}")

with tab_users:
    if not df_test.empty:
        st.write("**Top-N at-risk (по вероятности churn на Test)**")
        if not df_pred_test.empty:
            df_top = df_pred_test.sort_values("churn_proba", ascending=False).head(topn)
            st.dataframe(df_top, use_container_width=True, height=500)
        uid = st.number_input("user_id из Test", min_value=int(df_test["user_id"].min()),
                              max_value=int(df_test["user_id"].max()),
                              value=int(df_test["user_id"].iloc[0]), step=1)
        row = df_test[df_test["user_id"] == uid]
        if not row.empty:
            Xr = align_features(row.drop(columns=["user_id","churn"], errors="ignore"), feature_cols)
            proba = float(model.predict_proba(Xr)[:,1])
            st.metric("Churn probability", f"{proba:.2%}")
            try:
                explainer = shap.TreeExplainer(model.named_steps["model"])
                sv = explainer.shap_values(Xr)
                st.write("Локальный SHAP")
                fig = plt.figure(figsize=(8,4))
                shap.force_plot(explainer.expected_value, sv, Xr, matplotlib=True, show=False)
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"SHAP недоступен: {e}")
    else:
        st.info("Нет данных Test.")
