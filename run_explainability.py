"""SHAP + LIME explanations for the adaptive River ensemble NIDS.

Run from the project directory (where app.py is located):
    python run_explainability.py

This script trains ARF, HAT and Hoeffding Tree online on NSL-KDD training data,
updates model weights from a rolling accuracy window, then explains a small set of
held-out records. Explanations are model-agnostic approximations of the ensemble
wrapper's attack probability.
"""
from __future__ import annotations

import os
import sys
import warnings
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ------------------------- Config -------------------------
TRAIN_LIMIT = 20_000       # online training records; increase after first successful run
ROLLING_WINDOW = 500
WEIGHT_TEMPERATURE = 5.0
WEIGHT_UPDATE_EVERY = 100
EXPLAIN_ROWS = 6          # small on purpose: model-agnostic explainers are expensive
LIME_SAMPLES = 1_000
SHAP_MAX_EVALS = 2_000    # capped below based on feature count
BACKGROUND_ROWS = 40
RANDOM_SEED = 42
OUTPUT_DIR = Path("results/explainability")

# ------------------------- Project imports -------------------------
try:
    from app import load
except Exception as exc:
    raise SystemExit(
        "Could not import load() from app.py. Run this script from your project folder.\n"
        f"Import error: {exc}"
    )

try:
    from river import forest, tree
except ImportError as exc:
    raise SystemExit(
        "Could not import River. River may be installed, but an import path or dependency failed.\n"
        f"Details: {exc}"
    )

try:
    import shap
    from lime.lime_tabular import LimeTabularExplainer
except ImportError:
    raise SystemExit("Install explainability packages first: pip install shap lime")


# ------------------------- Utilities -------------------------
def to_dict(row, feature_names):
    return {name: float(value) if isinstance(value, (np.number, int, float)) else value
            for name, value in zip(feature_names, row)}


def class_probability(model, x, class_id=1):
    """River predict_proba_one with a safe fallback to hard prediction."""
    try:
        probs = model.predict_proba_one(x) or {}
        p = float(probs.get(class_id, 0.0))
        if np.isfinite(p):
            return float(np.clip(p, 0.0, 1.0))
    except Exception:
        pass
    try:
        return float(model.predict_one(x) == class_id)
    except Exception:
        return 0.5


def softmax_weights(accuracies, temperature=WEIGHT_TEMPERATURE):
    a = np.asarray(accuracies, dtype=float)
    if len(a) == 0 or not np.all(np.isfinite(a)):
        return np.ones(len(a), dtype=float) / max(1, len(a))
    # Shift for numerical stability. Temperature controls how strongly better models dominate.
    z = (a - np.max(a)) * temperature
    e = np.exp(np.clip(z, -50, 50))
    return e / e.sum() if e.sum() else np.ones(len(a)) / len(a)


class EnsembleProbability:
    def __init__(self, models, weights, feature_names):
        self.models = models
        self.weights = np.asarray(weights, dtype=float)
        self.feature_names = list(feature_names)

    def predict_proba(self, matrix):
        matrix = np.asarray(matrix)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        out = []
        for row in matrix:
            x = to_dict(row, self.feature_names)
            probs = [class_probability(m, x, 1) for m in self.models]
            p_attack = float(np.dot(self.weights, probs))
            p_attack = np.clip(p_attack, 0.0, 1.0)
            out.append([1.0 - p_attack, p_attack])
        return np.asarray(out, dtype=float)


# ------------------------- Main -------------------------
def main():
    np.random.seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading NSL-KDD through app.load()...")
    loaded = load(shuffle_train=False)
    # Expected project API: train_df, test_df. Support dict and tuple too.
    if isinstance(loaded, dict):
        train_df = loaded.get("train", loaded.get("train_df"))
        test_df = loaded.get("test", loaded.get("test_df"))
    elif isinstance(loaded, (tuple, list)) and len(loaded) >= 2:
        train_df, test_df = loaded[0], loaded[1]
    else:
        raise RuntimeError("Unexpected app.load() return value. Expected (train_df, test_df) or dict.")

    # app.load in the project prepares target 'y' and one-hot encoded features.
    if "y" not in train_df.columns or "y" not in test_df.columns:
        raise RuntimeError("Expected y column in train/test data returned by app.load().")
    feature_names = [c for c in train_df.columns if c not in {"y", "label", "attack_type", "difficulty"}]
    test_features = [c for c in test_df.columns if c not in {"y", "label", "attack_type", "difficulty"}]
    if feature_names != test_features:
        test_df = test_df.reindex(columns=feature_names + [c for c in test_df.columns if c in {"y", "label", "attack_type", "difficulty"}], fill_value=0)
    X_train = train_df[feature_names].astype(float).to_numpy()
    y_train = train_df["y"].astype(int).to_numpy()
    X_test = test_df[feature_names].astype(float).to_numpy()
    y_test = test_df["y"].astype(int).to_numpy()

    # Models correspond to the project's adaptive ensemble family.
    models = [
        forest.ARFClassifier(n_models=10, seed=RANDOM_SEED),
        tree.HoeffdingAdaptiveTreeClassifier(seed=RANDOM_SEED),
        tree.HoeffdingTreeClassifier(),
    ]
    model_names = ["ARF", "HAT", "Hoeffding Tree"]
    recent_correct = [deque(maxlen=ROLLING_WINDOW) for _ in models]
    weights = np.ones(len(models), dtype=float) / len(models)

    n_train = min(TRAIN_LIMIT, len(X_train))
    print(f"Online training {n_train:,} records...")
    for i in range(n_train):
        x = to_dict(X_train[i], feature_names)
        y = int(y_train[i])
        # Prequential: predict before learning, then update model.
        for j, model in enumerate(models):
            pred = model.predict_one(x)
            recent_correct[j].append(int(pred == y) if pred is not None else 0)
            model.learn_one(x, y)
        if (i + 1) % WEIGHT_UPDATE_EVERY == 0:
            accs = [np.mean(q) if q else 0.5 for q in recent_correct]
            weights = softmax_weights(accs)
        if (i + 1) % 5_000 == 0 or i + 1 == n_train:
            print(f"  trained {i+1:,}/{n_train:,}; weights={dict(zip(model_names, np.round(weights, 3)))}")

    # Final weights from recent prequential correctness.
    accuracies = [np.mean(q) if q else 0.5 for q in recent_correct]
    weights = softmax_weights(accuracies)
    ensemble = EnsembleProbability(models, weights, feature_names)
    print("Final rolling accuracies:", dict(zip(model_names, np.round(accuracies, 4))))
    print("Final ensemble weights:", dict(zip(model_names, np.round(weights, 4))))

    # Select a balanced set of normal/attack test rows.
    normal_idx = np.flatnonzero(y_test == 0)[: EXPLAIN_ROWS // 2]
    attack_idx = np.flatnonzero(y_test == 1)[: EXPLAIN_ROWS - len(normal_idx)]
    chosen = np.concatenate([normal_idx, attack_idx])
    if len(chosen) == 0:
        chosen = np.arange(min(EXPLAIN_ROWS, len(X_test)))
    X_explain = X_test[chosen]
    y_explain = y_test[chosen]
    probs = ensemble.predict_proba(X_explain)
    preds = (probs[:, 1] >= 0.5).astype(int)

    result_rows = []
    for k, idx in enumerate(chosen):
        result_rows.append({
            "test_row_index": int(idx),
            "actual": int(y_test[idx]),
            "predicted": int(preds[k]),
            "p_normal": float(probs[k, 0]),
            "p_attack": float(probs[k, 1]),
            "correct": bool(preds[k] == y_test[idx]),
        })
    prediction_df = pd.DataFrame(result_rows)
    prediction_df.to_csv(OUTPUT_DIR / "explained_predictions.csv", index=False)
    print("Explained records:")
    print(prediction_df.to_string(index=False))

    # ---------------- SHAP ----------------
    print("\nComputing SHAP permutation explanations (may take a few minutes)...")
    background = X_train[np.random.choice(n_train, min(BACKGROUND_ROWS, n_train), replace=False)]
    shap_values = None
    try:
        masker = shap.maskers.Independent(background, max_samples=min(BACKGROUND_ROWS, len(background)))
        explainer = shap.Explainer(ensemble.predict_proba, masker, algorithm="permutation", output_names=["Normal", "Attack"])
        # SHAP permutation requires >= 2*features+1 evaluations; cap selected records.
        max_evals = min(SHAP_MAX_EVALS, 2 * len(feature_names) + 1)
        shap_result = explainer(X_explain, max_evals=max_evals)
        vals = np.asarray(shap_result.values)
        # Standardize shape: samples x features x outputs; extract attack output.
        if vals.ndim == 3:
            attack_vals = vals[:, :, 1] if vals.shape[2] > 1 else vals[:, :, 0]
        elif vals.ndim == 2:
            attack_vals = vals
        else:
            raise ValueError(f"Unexpected SHAP output shape {vals.shape}")
        shap_values = attack_vals

        # Global summary of these selected samples.
        plt.figure(figsize=(11, 8))
        shap.summary_plot(attack_vals, X_explain, feature_names=feature_names, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "shap_summary.png", dpi=180, bbox_inches="tight")
        plt.close("all")

        # One individual SHAP bar chart for the first explained sample.
        first_vals = attack_vals[0]
        top = np.argsort(np.abs(first_vals))[-15:]
        plt.figure(figsize=(10, 7))
        plt.barh(np.array(feature_names)[top], first_vals[top])
        plt.xlabel("SHAP value (effect on attack probability)")
        plt.title(f"SHAP — test row {chosen[0]} (actual={y_explain[0]}, predicted={preds[0]})")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "shap_sample.png", dpi=180, bbox_inches="tight")
        plt.close()

        shap_long = []
        for sample_i, row_idx in enumerate(chosen):
            for feature_i, name in enumerate(feature_names):
                shap_long.append({"test_row_index": int(row_idx), "feature": name,
                                 "feature_value": X_explain[sample_i, feature_i],
                                 "shap_attack_probability": float(attack_vals[sample_i, feature_i])})
        pd.DataFrame(shap_long).to_csv(OUTPUT_DIR / "shap_values.csv", index=False)
        print("SHAP saved.")
    except Exception as exc:
        print(f"SHAP step failed: {exc}")
        print("The rest of the LIME run will continue. Check installed SHAP version if needed.")

    # ---------------- LIME ----------------
    print("\nComputing LIME explanation for one normal and one attack example...")
    lime_explainer = LimeTabularExplainer(
        training_data=background,
        feature_names=feature_names,
        class_names=["Normal", "Attack"],
        mode="classification",
        discretize_continuous=True,
        random_state=RANDOM_SEED,
    )
    lime_rows = []
    lime_html_paths = []
    # Explicitly select one Normal and one Attack record for LIME.
    lime_indices = []
    normal_candidates = np.flatnonzero(y_test == 0)
    attack_candidates = np.flatnonzero(y_test == 1)
    if len(normal_candidates):
        lime_indices.append(int(normal_candidates[0]))
    if len(attack_candidates):
        lime_indices.append(int(attack_candidates[0]))

    for sample_i, row_idx in enumerate(lime_indices):
        lime_x = X_test[row_idx]
        lime_y = int(y_test[row_idx])
        lime_pred = int(ensemble.predict_proba(lime_x.reshape(1, -1))[0, 1] >= 0.5)
        exp = lime_explainer.explain_instance(
            data_row=lime_x,
            predict_fn=ensemble.predict_proba,
            num_features=min(15, len(feature_names)),
            num_samples=LIME_SAMPLES,
            labels=(1,),
        )
        html_path = OUTPUT_DIR / f"lime_sample_{sample_i+1}_{'attack' if lime_y else 'normal'}.html"
        exp.save_to_file(str(html_path))
        lime_html_paths.append(html_path)
        for feature, contribution in exp.as_list(label=1):
            lime_rows.append({"test_row_index": int(row_idx), "actual": lime_y,
                             "predicted": lime_pred, "feature_rule": feature,
                             "lime_contribution_to_attack": float(contribution)})
    pd.DataFrame(lime_rows).to_csv(OUTPUT_DIR / "lime_values.csv", index=False)

    # Summary of prediction/explanation artifacts.
    print("\nDONE. Files created in:", OUTPUT_DIR.resolve())
    for file in sorted(OUTPUT_DIR.iterdir()):
        print(" -", file.name)
    print("\nOpen lime_sample_*.html in your browser for interactive LIME explanations.")
    print("SHAP charts are PNG files; CSVs contain the underlying values.")


if __name__ == "__main__":
    main()
