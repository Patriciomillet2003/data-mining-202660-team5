"""
IIND4417 — Final Capstone, Team 5 (Big99 Inventory Intelligence)
capstone_streamlit_app/explain_models.py

Builds models/explainability.json: the model facts the dashboard's LLM assistant
is allowed to quote.

  * Permutation importance of every input, measured on the hold-out lots.
  * Average effect of each input level on every model's score.
  * A short card per model and the printed rules of the decision tree.

Nothing is retrained. The script loads the serialized artifacts, rebuilds the
lot-grouped hold-out of train_models.py and refuses to write anything unless it
reproduces the hold-out ROC-AUC stored in metrics.json.

From capstone_streamlit_app/:
    python explain_models.py
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.tree import export_text

from train_models import DATA, GROUP, MODELS, RANDOM_STATE, TARGET

OUT = MODELS / "explainability.json"
N_REPEATS = 30
QUANTILES = (10, 50, 90)


def make_scorer(prep, model, kind):
    """Raw score for a frame of units: predicted log1p(days) or stall probability."""
    def score(frame):
        Z = prep.transform(frame)
        return model.predict(Z) if kind == "regression" else model.predict_proba(Z)[:, 1]
    return score


def typical(kind, scores):
    """One readable number for a group of units: median predicted days or mean probability."""
    return float(np.expm1(np.median(scores))) if kind == "regression" else float(np.mean(scores))


def permutation_importance(score, X, y, rng):
    """Drop in hold-out ROC-AUC when one input is shuffled, so its link to the label breaks."""
    base = roc_auc_score(y, score(X))
    rows = []
    for col in X.columns:
        drops = []
        for _ in range(N_REPEATS):
            shuffled = X.copy()
            shuffled[col] = rng.permutation(X[col].to_numpy())
            drops.append(base - roc_auc_score(y, score(shuffled)))
        rows.append({"feature": col, "auc_drop_mean": round(float(np.mean(drops)), 4),
                     "auc_drop_std": round(float(np.std(drops, ddof=1)), 4)})
    return base, sorted(rows, key=lambda r: -r["auc_drop_mean"])


def partial_effects(score, kind, X, meta):
    """Typical score over all units when one input is forced to a single level."""
    effects = {}
    for col in meta["categorical"]:
        effects[col] = {str(level): round(typical(kind, score(X.assign(**{col: level}))), 4)
                        for level in X[col].value_counts().index}
    for col in meta["numeric"]:
        effects[col] = {}
        for q in QUANTILES:
            value = float(X[col].quantile(q / 100))
            effects[col][f"P{q}"] = {"input": round(value, 2),
                                     "score": round(typical(kind, score(X.assign(**{col: value}))), 4)}
    for col in meta["accessories"]:
        effects[col] = {str(flag): round(typical(kind, score(X.assign(**{col: flag}))), 4)
                        for flag in (0, 1)}
    return effects


def model_card(name, model):
    if name == "Decision Tree":
        return {"estimator": "DecisionTreeClassifier", "max_depth": model.max_depth,
                "min_samples_leaf": model.min_samples_leaf, "class_weight": model.class_weight,
                "leaves": int(model.get_n_leaves())}
    if name == "Random Forest":
        return {"estimator": "RandomForestClassifier", "n_estimators": model.n_estimators,
                "max_depth": model.max_depth, "min_samples_leaf": model.min_samples_leaf,
                "class_weight": model.class_weight}
    if name == "MLR":
        return {"estimator": "LinearRegression", "target": "log1p(days_in_stock)",
                "coefficients": int(model.coef_.size)}
    return {"estimator": "MLPClassifier", "hidden_layers": list(model.hidden_layer_sizes),
            "activation": model.activation, "alpha": model.alpha,
            "learning_rate_init": model.learning_rate_init, "batch_size": model.batch_size,
            "trainable_parameters": int(sum(w.size for w in model.coefs_) +
                                        sum(b.size for b in model.intercepts_))}


def main():
    meta = json.loads((MODELS / "metrics.json").read_text())
    df = pd.read_csv(DATA)
    X, y, groups = df[meta["features"]], df[TARGET], df[GROUP]

    # Same first fold as train_models.py, so "hold-out" means the same 25 lots everywhere.
    outer = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
    _, test_idx = next(outer.split(X, y, groups))
    X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]
    if len(test_idx) != meta["test_units"]:
        raise SystemExit(f"Hold-out has {len(test_idx)} units, metrics.json says {meta['test_units']}.")

    prep = joblib.load(MODELS / "preprocessor.joblib")
    rng = np.random.default_rng(RANDOM_STATE)
    out = {"basis": {"holdout_units": int(len(test_idx)), "holdout_lots": int(groups.iloc[test_idx].nunique()),
                     "repeats": N_REPEATS, "metric": "drop in hold-out ROC-AUC",
                     "partial_effect_units": int(len(X))},
           "models": {}}
    for name, file in meta["files"].items():
        model = joblib.load(MODELS / file)
        kind = meta["policy"][name]["kind"]
        score = make_scorer(prep, model, kind)
        base, ranking = permutation_importance(score, X_te, y_te, rng)
        stored = meta["results"][name]["roc_auc"]
        if abs(base - stored) > 1e-9:
            raise SystemExit(f"{name}: hold-out ROC-AUC {base:.6f} does not match metrics.json {stored:.6f}. "
                             "Re-run train_models.py before explaining the models.")
        out["models"][name] = {
            "kind": kind,
            "score_unit": "median predicted days in stock" if kind == "regression" else "mean stall probability",
            "holdout_roc_auc": round(base, 4),
            "card": model_card(name, model),
            "permutation_importance": ranking,
            "partial_effects": partial_effects(score, kind, X, meta),
        }
        if name == "Decision Tree":
            out["models"][name]["rules"] = export_text(model, feature_names=meta["encoded_columns"], decimals=2)
        top = ", ".join(f"{r['feature']} {r['auc_drop_mean']:+.3f}" for r in ranking[:4])
        print(f"{name:<14} hold-out AUC {base:.3f} (matches metrics.json) | top inputs: {top}")

    OUT.write_text(json.dumps(out, indent=1))
    print(f"Wrote {OUT.relative_to(MODELS.parent)}")


if __name__ == "__main__":
    main()
