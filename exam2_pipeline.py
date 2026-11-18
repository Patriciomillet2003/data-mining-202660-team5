"""
Midterm Test 2, Part 2 - Team 5 Smart Grid (IIND4417, period 202660)
Dataset: exam2_team5_smartgrid.csv | target: power_demand_mw

Run from the repo root:
  py exam2_pipeline.py 1   -> Task 1 (3 regression models + 4-panel residual dashboard)
  py exam2_pipeline.py 2   -> Task 2 (MLPRegressor (64, 32) relu early stopping + loss curve + joblib export)
  py exam2_pipeline.py     -> both
"""
import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed" / "exam2_team5_smartgrid.csv"
OUT = ROOT / "reports" / "exam2"
MODELS = ROOT / "models"
TARGET = "power_demand_mw"
SEED = 42


def load():
    df = pd.read_csv(DATA)
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=SEED)
    scaler = StandardScaler().fit(X_tr)  # fit on TRAIN only: no test leakage
    return df, X, X_tr, X_te, y_tr, y_te, scaler


def scores(y_true, y_pred, p):
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    return {
        "R2": r2,
        "Adj_R2": 1 - (1 - r2) * (n - 1) / (n - p - 1),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
    }


def regression_models():
    return {
        "1 Baseline OLS": LinearRegression(),
        # log1p on the target, predictions come back in MW through expm1
        "2 Log-target OLS": TransformedTargetRegressor(
            regressor=LinearRegression(), func=np.log1p, inverse_func=np.expm1
        ),
        "3 Polynomial OLS (deg 2)": make_pipeline(PolynomialFeatures(degree=2, include_bias=False), LinearRegression()),
    }


def task1():
    df, X, X_tr, X_te, y_tr, y_te, scaler = load()
    Xs_tr, Xs_te = scaler.transform(X_tr), scaler.transform(X_te)
    p = X.shape[1]
    print("=" * 74)
    print("TASK 1 - REGRESSION MODEL COMPARISON & RESIDUAL DIAGNOSTICS")
    print("=" * 74)
    print(f"Data: {DATA.name} | rows={len(df)} | features={p} | nulls={int(df.isna().sum().sum())}")
    print(f"Split 80/20 random_state={SEED}: train={len(X_tr)} test={len(X_te)}")
    print(f"Target {TARGET}: mean {df[TARGET].mean():.2f} MW, std {df[TARGET].std():.2f} MW\n")

    rows, fitted = [], {}
    for name, model in regression_models().items():
        model.fit(Xs_tr, y_tr)
        s = scores(y_te, model.predict(Xs_te), p)
        s["Train_R2"] = r2_score(y_tr, model.predict(Xs_tr))
        rows.append({"Model": name, **s})
        fitted[name] = model
    bench = pd.DataFrame(rows)[["Model", "Train_R2", "R2", "Adj_R2", "MAE", "RMSE"]]
    print("Holdout test metrics (n = %d):" % len(y_te))
    print(bench.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    best = bench.loc[bench["RMSE"].idxmin(), "Model"]
    print(f"\nBest regression model by test RMSE: {best}")

    # 4-panel residual dashboard (on the TRAIN residuals of the best model, the classical OLS checks)
    model = fitted[best]
    fit_tr = model.predict(Xs_tr)
    res = y_tr.values - fit_tr
    std_res = (res - res.mean()) / res.std()
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    ax[0, 0].scatter(fit_tr, res, s=10, alpha=0.6)
    ax[0, 0].axhline(0, color="red", lw=1)
    ax[0, 0].set(title="1. Linearity: residuals vs fitted", xlabel="Fitted (MW)", ylabel="Residual (MW)")
    ax[0, 1].scatter(fit_tr, np.sqrt(np.abs(std_res)), s=10, alpha=0.6, color="tab:orange")
    ax[0, 1].set(title="2. Homoscedasticity: scale-location", xlabel="Fitted (MW)", ylabel="sqrt(|standardized residual|)")
    ax[1, 0].plot(res, lw=0.8, color="tab:green")
    ax[1, 0].axhline(0, color="red", lw=1)
    ax[1, 0].set(title="3. Independence: residuals vs observation order", xlabel="Observation index", ylabel="Residual (MW)")
    stats.probplot(res, dist="norm", plot=ax[1, 1])
    ax[1, 1].set_title("4. Normality: Q-Q plot")
    fig.suptitle(f"Residual diagnostics - {best} (train residuals)", fontsize=13)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "task1_residual_dashboard.png", dpi=130)
    plt.close(fig)

    shapiro_p = stats.shapiro(res).pvalue
    dw = np.sum(np.diff(res) ** 2) / np.sum(res ** 2)
    lo, hi = fit_tr < np.median(fit_tr), fit_tr >= np.median(fit_tr)
    print("\nResidual health checks (best model, train residuals):")
    print(f"  Mean residual: {res.mean():.4f} MW")
    print(f"  Residual std, low vs high fitted half: {res[lo].std():.2f} vs {res[hi].std():.2f} MW")
    print(f"  Durbin-Watson: {dw:.3f} (about 2 = independent)")
    print(f"  Shapiro-Wilk p-value: {shapiro_p:.4f} (> 0.05 = consistent with normal)")
    print(f"Figure: {(OUT / 'task1_residual_dashboard.png').relative_to(ROOT)}")
    bench.to_csv(OUT / "task1_benchmark.csv", index=False)
    return bench, fitted, best


def task2():
    df, X, X_tr, X_te, y_tr, y_te, scaler = load()
    Xs_tr, Xs_te = scaler.transform(X_tr), scaler.transform(X_te)
    p = X.shape[1]
    print("=" * 74)
    print("TASK 2 - ANN TRAINING & LOSS CONVERGENCE")
    print("=" * 74)
    ann = MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
                       early_stopping=True, max_iter=3000, random_state=SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ann.fit(Xs_tr, y_tr)
    s = scores(y_te, ann.predict(Xs_te), p)
    print(f"MLPRegressor(64, 32), relu, adam, early_stopping=True, random_state={SEED}")
    print(f"Epochs run: {ann.n_iter_} | best validation R2 (10% of train): {ann.best_validation_score_:.4f}")
    print(f"Test: R2 {s['R2']:.4f} | Adj R2 {s['Adj_R2']:.4f} | MAE {s['MAE']:.2f} MW | RMSE {s['RMSE']:.2f} MW")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ann.loss_curve_, lw=2)
    ax.set_yscale("log")
    ax.set(title="ANN (64, 32) relu - training loss convergence (loss_curve_)", xlabel="Epoch",
           ylabel="Training loss (squared error / 2, log scale)")
    ax2 = ax.twinx()
    ax2.plot(ann.validation_scores_, color="tab:orange", lw=1.5, label="Validation R2")
    ax2.set_ylabel("Validation R2 (early-stopping split)")
    ax2.set_ylim(min(0, min(ann.validation_scores_)), 1.02)
    ax2.legend(loc="center right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "task2_loss_curve.png", dpi=130)
    plt.close(fig)

    # Winner across the 3 OLS models + the ANN, same split, by test RMSE
    cands = {}
    for name, model in regression_models().items():
        model.fit(Xs_tr, y_tr)
        cands[name] = (model, scores(y_te, model.predict(Xs_te), p))
    cands["4 ANN (64, 32)"] = (ann, s)
    table = pd.DataFrame([{"Model": k, **v[1]} for k, v in cands.items()])
    print("\nAll 4 models on the same holdout:")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    win = table.loc[table["RMSE"].idxmin(), "Model"]
    print(f"Winning model (lowest test RMSE): {win}")

    MODELS.mkdir(exist_ok=True)
    joblib.dump(cands[win][0], MODELS / "best_model.joblib")
    joblib.dump(scaler, MODELS / "scaler.joblib")
    joblib.dump(ann, MODELS / "ann_model.joblib")
    m, sc = joblib.load(MODELS / "best_model.joblib"), joblib.load(MODELS / "scaler.joblib")
    diff = float(np.abs(m.predict(sc.transform(X_te)) - cands[win][0].predict(Xs_te)).max())
    print(f"Saved models/best_model.joblib ({win}), models/scaler.joblib, models/ann_model.joblib")
    print(f"Reload check: max |reloaded - original| = {diff:.2e} MW")
    print(f"Figure: {(OUT / 'task2_loss_curve.png').relative_to(ROOT)}")

    meta = {
        "target": TARGET, "features": list(X.columns), "winner": win,
        "metrics": table.to_dict(orient="records"),
        "ann_loss_curve": list(map(float, ann.loss_curve_)),
        "ann_validation_r2": list(map(float, ann.validation_scores_)),
        "train_demand_p75": float(np.percentile(y_tr, 75)),
        "train_demand_p90": float(np.percentile(y_tr, 90)),
    }
    (MODELS / "exam2_metrics.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    if arg in ("1", "all"):
        task1()
    if arg in ("2", "all"):
        if arg == "all":
            print()
        task2()
