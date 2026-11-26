import os
import random
import warnings
from collections import deque

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from river import drift, ensemble, tree
from app import load

warnings.filterwarnings("ignore")

SEEDS = [42, 123, 456, 789, 2026]
STREAM_SIZE = 30000
DRIFT_POINT = 15000
WINDOW_SIZE = 500
WEIGHT_UPDATE_INTERVAL = 100
MIN_WEIGHT = 0.05
CANDIDATE_WARMUP = 500
CANDIDATE_EVALUATION = 1000
REPLACEMENT_THRESHOLD = 0.01
ADWIN_DELTA = 0.002

RESULTS_DIR = "results/experiment_1B_corrected"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

train, test = load(shuffle_train=False)
df = train.reset_index(drop=True)
labels = df["y"]

X = df.drop(columns=["y", "attack_type"], errors="ignore")
X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

normal_data = X[labels == 0].copy()
attack_df = df.loc[labels == 1].copy()
attack_df["attack_type"] = attack_df["attack_type"].astype(str)

attack_groups = {
    "dos": ["neptune", "smurf", "teardrop", "back", "pod", "land"],
    "probe": ["satan", "ipsweep", "portsweep", "nmap"],
    "r2l": ["warezclient", "guess_passwd", "ftp_write", "imap", "multihop", "phf", "spy"],
    "u2r": ["buffer_overflow", "rootkit", "loadmodule", "perl", "sqlattack", "xterm"]
}

attack_df["attack_group"] = attack_df["attack_type"].apply(
    lambda t: next((g for g, attacks in attack_groups.items() if t.lower() in attacks), "other")
)

attack_group_data = {
    g: X.loc[attack_df[attack_df["attack_group"] == g].index].copy()
    for g in attack_groups
}
attack_group_data = {g: d for g, d in attack_group_data.items() if len(d) > 0}

MODEL_NAMES = ["SRP", "HAT", "HT"]

def make_model(name, seed=42):
    if name == "SRP":
        return ensemble.SRPClassifier(n_models=3, seed=seed)
    if name == "HAT":
        return tree.HoeffdingAdaptiveTreeClassifier(
            grace_period=100, delta=0.01, leaf_prediction="nba"
        )
    if name == "HT":
        return tree.HoeffdingTreeClassifier(
            grace_period=100, delta=0.01, leaf_prediction="nba"
        )
    raise ValueError(name)

def create_models(seed):
    return {n: make_model(n, seed) for n in MODEL_NAMES}

def init_weights():
    return {n: 1 / len(MODEL_NAMES) for n in MODEL_NAMES}

def generate_stream(seed):
    rng = np.random.default_rng(seed)
    phases = []

    configs = [
        (DRIFT_POINT, {"dos": 0.70, "probe": 0.25, "r2l": 0.03, "u2r": 0.02}),
        (STREAM_SIZE - DRIFT_POINT, {"dos": 0.15, "probe": 0.40, "r2l": 0.30, "u2r": 0.15})
    ]

    for phase_size, attack_distribution in configs:
        normal_count = phase_size // 2
        attack_count = phase_size - normal_count

        phase = [
            (normal_data.iloc[rng.integers(len(normal_data))].to_dict(), 0)
            for _ in range(normal_count)
        ]

        groups = list(attack_distribution)
        probabilities = np.array([attack_distribution[g] for g in groups])
        probabilities /= probabilities.sum()

        sampled_groups = rng.choice(groups, size=attack_count, p=probabilities)

        for group in sampled_groups:
            data = attack_group_data.get(group)
            if data is None or len(data) == 0:
                available = list(attack_group_data)
                group = rng.choice(available)
                data = attack_group_data[group]

            phase.append((data.iloc[rng.integers(len(data))].to_dict(), 1))

        rng.shuffle(phase)
        phases.append(phase)

    return phases[0] + phases[1]

def predict_proba(model, x):
    try:
        p = model.predict_proba_one(x)
        if p:
            return p
    except Exception:
        pass

    pred = model.predict_one(x)
    return {0: 1.0, 1: 0.0} if pred == 0 else {0: 0.0, 1: 1.0}

def predict_model(model, x):
    p = predict_proba(model, x)
    return int(p.get(1, 0.0) >= p.get(0, 0.0))

def ensemble_predict(models, weights, x):
    p0, p1 = 0.0, 0.0

    for name, model in models.items():
        p = predict_proba(model, x)
        p0 += weights[name] * p.get(0, 0.0)
        p1 += weights[name] * p.get(1, 0.0)

    return int(p1 >= p0)

class RollingStats:
    def __init__(self, size=WINDOW_SIZE):
        self.data = deque(maxlen=size)

    def update(self, y, pred):
        self.data.append((y, pred))

    def f1(self):
        tp = sum(y == 1 and p == 1 for y, p in self.data)
        fp = sum(y == 0 and p == 1 for y, p in self.data)
        fn = sum(y == 1 and p == 0 for y, p in self.data)

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0

        return 2 * precision * recall / (precision + recall) if precision + recall else 0

def update_weights(trackers):
    scores = {n: max(trackers[n].f1(), 0.001) for n in MODEL_NAMES}
    total = sum(scores.values())

    weights = {n: max(scores[n] / total, MIN_WEIGHT) for n in MODEL_NAMES}
    total = sum(weights.values())

    return {n: weights[n] / total for n in MODEL_NAMES}

def calculate_metrics(y_true, predictions):
    tp = sum(y == 1 and p == 1 for y, p in zip(y_true, predictions))
    tn = sum(y == 0 and p == 0 for y, p in zip(y_true, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(y_true, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(y_true, predictions))

    accuracy = (tp + tn) / len(y_true) if y_true else 0
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def run_experiment(seed):
    stream = generate_stream(seed)

    static_models = create_models(seed)
    adaptive_models = create_models(seed + 10)
    drift_models = create_models(seed + 20)

    static_weights = init_weights()
    adaptive_weights = init_weights()
    drift_weights = init_weights()

    adaptive_trackers = {n: RollingStats() for n in MODEL_NAMES}
    drift_trackers = {n: RollingStats() for n in MODEL_NAMES}

    predictions = {"static": [], "adaptive": [], "drift": []}
    weight_history = []
    drift_events = []
    replacement_events = []

    detector = drift.ADWIN(delta=ADWIN_DELTA)

    candidate = None
    candidate_name = None
    candidate_start = None
    candidate_eval_start = None
    candidate_eval_end = None

    candidate_eval_y = []
    candidate_eval_incumbent_preds = []
    candidate_eval_candidate_preds = []

    replacement_attempts = 0
    replacements = 0
    false_alarms = 0
    drift_detected = False

    for i, (x, y) in enumerate(stream):

        # All predictions happen BEFORE learning the current sample
        ps = ensemble_predict(static_models, static_weights, x)
        pa = ensemble_predict(adaptive_models, adaptive_weights, x)
        pd_ = ensemble_predict(drift_models, drift_weights, x)

        predictions["static"].append((y, ps))
        predictions["adaptive"].append((y, pa))
        predictions["drift"].append((y, pd_))

        adaptive_preds = {
            n: predict_model(m, x) for n, m in adaptive_models.items()
        }
        drift_preds = {
            n: predict_model(m, x) for n, m in drift_models.items()
        }

        for n in MODEL_NAMES:
            adaptive_trackers[n].update(y, adaptive_preds[n])
            drift_trackers[n].update(y, drift_preds[n])

        # Candidate predictions are also pre-learning
        candidate_pred = None
        incumbent_pred = None

        if candidate is not None and candidate_eval_start <= i < candidate_eval_end:
            incumbent_pred = predict_model(drift_models[candidate_name], x)
            candidate_pred = predict_model(candidate, x)

            candidate_eval_y.append(y)
            candidate_eval_incumbent_preds.append(incumbent_pred)
            candidate_eval_candidate_preds.append(candidate_pred)

        # Drift detection
        detector.update(int(pd_ != y))

        if detector.drift_detected:
            if i < DRIFT_POINT:
                false_alarms += 1

            elif not drift_detected:
                drift_detected = True

                candidate_name = min(
                    drift_trackers,
                    key=lambda n: drift_trackers[n].f1()
                )

                candidate = make_model(candidate_name, seed + 1000 + i)
                candidate_start = i + 1
                candidate_eval_start = candidate_start + CANDIDATE_WARMUP
                candidate_eval_end = candidate_eval_start + CANDIDATE_EVALUATION

                replacement_attempts += 1

                drift_events.append({
                    "seed": seed,
                    "actual_drift": DRIFT_POINT,
                    "detected": i,
                    "delay": i - DRIFT_POINT,
                    "candidate_model": candidate_name,
                    "false_alarms_before_drift": false_alarms
                })

        # Learn all main models once
        for model in static_models.values():
            model.learn_one(x, y)

        for model in adaptive_models.values():
            model.learn_one(x, y)

        for model in drift_models.values():
            model.learn_one(x, y)

        # Candidate warmup
        if candidate is not None and candidate_start <= i < candidate_eval_start:
            candidate.learn_one(x, y)

        # Candidate evaluation learning
        if candidate is not None and candidate_eval_start <= i < candidate_eval_end:
            candidate.learn_one(x, y)

        # Replacement decision
        if candidate is not None and i == candidate_eval_end - 1:

            incumbent_metrics = calculate_metrics(
                candidate_eval_y,
                candidate_eval_incumbent_preds
            )

            candidate_metrics = calculate_metrics(
                candidate_eval_y,
                candidate_eval_candidate_preds
            )

            improvement = candidate_metrics["f1"] - incumbent_metrics["f1"]
            accepted = improvement >= REPLACEMENT_THRESHOLD

            replacement_events.append({
                "seed": seed,
                "model": candidate_name,
                "incumbent_f1": incumbent_metrics["f1"],
                "candidate_f1": candidate_metrics["f1"],
                "improvement": improvement,
                "accepted": accepted,
                "evaluation_start": candidate_eval_start,
                "evaluation_end": candidate_eval_end
            })

            if accepted:
                drift_models[candidate_name] = candidate
                replacements += 1

            candidate = None

        # Update ensemble weights
        if i > 0 and i % WEIGHT_UPDATE_INTERVAL == 0:
            adaptive_weights = update_weights(adaptive_trackers)
            drift_weights = update_weights(drift_trackers)

        if i % 100 == 0:
            weight_history.append({
                "seed": seed,
                "index": i,
                **{f"static_{n}": static_weights[n] for n in MODEL_NAMES},
                **{f"adaptive_{n}": adaptive_weights[n] for n in MODEL_NAMES},
                **{f"drift_{n}": drift_weights[n] for n in MODEL_NAMES}
            })

        if (i + 1) % 5000 == 0:
            print(
                f"Seed {seed} | {i+1}/{STREAM_SIZE} | "
                f"Static {calculate_metrics(*zip(*predictions['static']))['accuracy']:.4f} | "
                f"Adaptive {calculate_metrics(*zip(*predictions['adaptive']))['accuracy']:.4f} | "
                f"Drift-aware {calculate_metrics(*zip(*predictions['drift']))['accuracy']:.4f}"
            )

    result = {"seed": seed}

    for name, data in predictions.items():
        ys, preds = zip(*data)

        for metric, value in calculate_metrics(ys, preds).items():
            result[f"{name}_{metric}"] = value

    result["detection_index"] = drift_events[0]["detected"] if drift_events else None
    result["detection_delay"] = drift_events[0]["delay"] if drift_events else None
    result["false_alarms"] = false_alarms
    result["replacement_attempts"] = replacement_attempts
    result["successful_replacements"] = replacements

    return result, predictions, weight_history, drift_events, replacement_events

all_results = []
all_predictions = []
all_weights = []
all_drift_events = []
all_replacements = []

for seed in SEEDS:
    print(f"\nRunning seed {seed}")

    result, predictions, weights, drift_events, replacements = run_experiment(seed)

    all_results.append(result)

    for name, data in predictions.items():
        all_predictions.extend([
            {
                "seed": seed,
                "model": name,
                "index": i,
                "y": y,
                "prediction": p
            }
            for i, (y, p) in enumerate(data)
        ])

    all_weights.extend(weights)
    all_drift_events.extend(drift_events)
    all_replacements.extend(replacements)

results_df = pd.DataFrame(all_results)

results_df.to_csv(
    os.path.join(RESULTS_DIR, "individual_runs.csv"),
    index=False
)

results_df.agg(["mean", "std"]).to_csv(
    os.path.join(RESULTS_DIR, "statistical_summary.csv")
)

pd.DataFrame(all_predictions).to_csv(
    os.path.join(RESULTS_DIR, "predictions.csv"),
    index=False
)

pd.DataFrame(all_weights).to_csv(
    os.path.join(RESULTS_DIR, "weight_history.csv"),
    index=False
)

pd.DataFrame(all_drift_events).to_csv(
    os.path.join(RESULTS_DIR, "drift_detection.csv"),
    index=False
)

pd.DataFrame(all_replacements).to_csv(
    os.path.join(RESULTS_DIR, "replacement_results.csv"),
    index=False
)

print("\nSUMMARY")

for metric in [
    "static_accuracy",
    "adaptive_accuracy",
    "drift_accuracy",
    "static_f1",
    "adaptive_f1",
    "drift_f1",
    "detection_delay"
]:
    values = results_df[metric].dropna()

    if len(values):
        print(f"{metric}: {values.mean():.4f} ± {values.std():.4f}")

print("False alarms:", results_df["false_alarms"].sum())
print("Replacement attempts:", results_df["replacement_attempts"].sum())
print("Successful replacements:", results_df["successful_replacements"].sum())

pred_df = pd.DataFrame(all_predictions)

for model_name in ["static", "adaptive", "drift"]:
    plt.figure(figsize=(12, 6))

    for seed in SEEDS:
        sub = pred_df[
            (pred_df["seed"] == seed) &
            (pred_df["model"] == model_name)
        ].copy()

        sub["correct"] = sub["y"] == sub["prediction"]
        sub["rolling_accuracy"] = sub["correct"].rolling(
            WINDOW_SIZE, min_periods=1
        ).mean()

        plt.plot(
            sub["index"],
            sub["rolling_accuracy"],
            alpha=0.6,
            label=f"Seed {seed}"
        )

    plt.axvline(DRIFT_POINT, linestyle="--", label="Actual Drift")
    plt.xlabel("Stream Instance")
    plt.ylabel("Rolling Accuracy")
    plt.title(f"{model_name.title()} Ensemble")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(PLOTS_DIR, f"{model_name}_rolling_accuracy.png"),
        dpi=300
    )
    plt.close()

print(f"\nCompleted. Results saved in {RESULTS_DIR}")