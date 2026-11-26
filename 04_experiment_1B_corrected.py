import os
import time
import random
import warnings
from collections import deque

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from river import drift, ensemble, tree, metrics
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

RESULTS_DIR = "results/experiment_1B"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

train, test = load(shuffle_train=False)
df = train.reset_index(drop=True)
labels = df["y"]
X = df.drop(columns=["y", "attack_type"], errors="ignore").apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
normal_data = X[labels == 0].copy()
attack_df = df.loc[labels == 1].copy()
attack_df["attack_type"] = attack_df["attack_type"].astype(str)

attack_groups = {"dos": ["neptune", "smurf", "teardrop", "back", "pod", "land"], "probe": ["satan", "ipsweep", "portsweep", "nmap"], "r2l": ["warezclient", "guess_passwd", "ftp_write", "imap", "multihop", "phf", "spy"], "u2r": ["buffer_overflow", "rootkit", "loadmodule", "perl", "sqlattack", "xterm"]}
attack_df["attack_group"] = attack_df["attack_type"].apply(lambda t: next((g for g, attacks in attack_groups.items() if t.lower() in attacks), "other"))
attack_group_data = {g: X.loc[attack_df[attack_df["attack_group"] == g].index].copy() for g in attack_groups}
attack_group_data = {g: d for g, d in attack_group_data.items() if len(d) > 0}

MODEL_NAMES = ["SRP", "HAT", "HT"]

def make_model(name, seed=42):
    if name == "SRP":
        return ensemble.SRPClassifier(n_models=3, seed=seed)
    if name == "HAT":
        return tree.HoeffdingAdaptiveTreeClassifier(grace_period=100, delta=0.01, leaf_prediction="nba")
    if name == "HT":
        return tree.HoeffdingTreeClassifier(grace_period=100, delta=0.01, leaf_prediction="nba")
    raise ValueError(name)

def create_models(seed):
    return {n: make_model(n, seed) for n in MODEL_NAMES}

def init_weights():
    return {n: 1 / len(MODEL_NAMES) for n in MODEL_NAMES}

def generate_stream(seed):
    rng = np.random.default_rng(seed)
    stream = []
    phase_configs = [
        (DRIFT_POINT, 0.50, {"dos": 0.70, "probe": 0.25, "r2l": 0.03, "u2r": 0.02}),
        (STREAM_SIZE - DRIFT_POINT, 0.30, {"dos": 0.15, "probe": 0.40, "r2l": 0.30, "u2r": 0.15})
    ]
    for phase_size, normal_ratio, attack_distribution in phase_configs:
        n_normal = int(phase_size * normal_ratio)
        n_attack = phase_size - n_normal
        for _ in range(n_normal):
            row = normal_data.iloc[rng.integers(len(normal_data))]
            stream.append((row.to_dict(), 0))
        groups = list(attack_distribution)
        probabilities = np.array([attack_distribution[g] for g in groups], dtype=float)
        probabilities /= probabilities.sum()
        sampled_groups = rng.choice(groups, size=n_attack, p=probabilities)
        for group in sampled_groups:
            data = attack_group_data.get(group)
            if data is None or len(data) == 0:
                available = list(attack_group_data)
                group = rng.choice(available)
                data = attack_group_data[group]
            row = data.iloc[rng.integers(len(data))]
            stream.append((row.to_dict(), 1))
    rng.shuffle(stream[:DRIFT_POINT])
    phase1 = stream[:DRIFT_POINT]
    phase2 = stream[DRIFT_POINT:]
    rng.shuffle(phase1)
    rng.shuffle(phase2)
    return phase1 + phase2

def predict(model, x):
    try:
        p = model.predict_proba_one(x)
        if p:
            return p
    except Exception:
        pass
    return {0: 0.5, 1: 0.5}

def ensemble_predict(models, weights, x):
    probabilities = {0: 0.0, 1: 0.0}
    for name, model in models.items():
        p = predict(model, x)
        probabilities[0] += weights[name] * p.get(0, 0.0)
        probabilities[1] += weights[name] * p.get(1, 0.0)
    return int(probabilities[1] >= probabilities[0])

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

    def accuracy(self):
        return sum(y == p for y, p in self.data) / len(self.data) if self.data else 0

def update_weights(trackers):
    scores = {n: max(trackers[n].f1(), 0.001) for n in MODEL_NAMES}
    total = sum(scores.values())
    weights = {n: max(scores[n] / total, MIN_WEIGHT) for n in MODEL_NAMES}
    total = sum(weights.values())
    return {n: weights[n] / total for n in MODEL_NAMES}

def f1_score(y_true, predictions):
    tp = sum(y == 1 and p == 1 for y, p in zip(y_true, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(y_true, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(y_true, predictions))
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0

def evaluate_replacement(incumbent, candidate, stream, start, end):
    incumbent_predictions = []
    candidate_predictions = []
    for j in range(start, end):
        x, y = stream[j]
        incumbent_predictions.append(int(incumbent.predict_one(x) or 0))
        candidate_predictions.append(int(candidate.predict_one(x) or 0))
        incumbent.learn_one(x, y)
        candidate.learn_one(x, y)
    ys = [stream[j][1] for j in range(start, end)]
    incumbent_f1 = f1_score(ys, incumbent_predictions)
    candidate_f1 = f1_score(ys, candidate_predictions)
    return incumbent_f1, candidate_f1, candidate_f1 - incumbent_f1

def run_experiment(seed):
    random.seed(seed)
    np.random.seed(seed)
    stream = generate_stream(seed)
    static_models = create_models(seed)
    adaptive_models = create_models(seed)
    drift_models = create_models(seed)
    static_weights = init_weights()
    adaptive_weights = init_weights()
    drift_weights = init_weights()
    adaptive_trackers = {n: RollingStats() for n in MODEL_NAMES}
    drift_trackers = {n: RollingStats() for n in MODEL_NAMES}
    static_metric = metrics.ConfusionMatrix()
    adaptive_metric = metrics.ConfusionMatrix()
    drift_metric = metrics.ConfusionMatrix()
    histories = {"static": [], "adaptive": [], "drift": []}
    weight_history = []
    drift_events = []
    replacement_events = []
    detector = drift.ADWIN(delta=ADWIN_DELTA)
    drift_detected = False
    detection_index = None
    candidate = None
    candidate_name = None
    candidate_warmup_start = None
    candidate_eval_start = None
    candidate_eval_end = None
    replacement_attempts = 0
    successful_replacements = 0
    start_time = time.time()

    for i, (x, y) in enumerate(stream):
        ps = ensemble_predict(static_models, static_weights, x)
        pa = ensemble_predict(adaptive_models, adaptive_weights, x)
        pd_ = ensemble_predict(drift_models, drift_weights, x)
        static_metric.update(y, ps)
        adaptive_metric.update(y, pa)
        drift_metric.update(y, pd_)
        histories["static"].append((i, y, ps))
        histories["adaptive"].append((i, y, pa))
        histories["drift"].append((i, y, pd_))

        adaptive_preds = {n: int(m.predict_one(x) or 0) for n, m in adaptive_models.items()}
        drift_preds = {n: int(m.predict_one(x) or 0) for n, m in drift_models.items()}
        for n in MODEL_NAMES:
            adaptive_trackers[n].update(y, adaptive_preds[n])
            drift_trackers[n].update(y, drift_preds[n])

        detector.update(int(pd_ != y))
        if detector.drift_detected and not drift_detected:
            drift_detected = True
            detection_index = i
            drift_events.append({"seed": seed, "actual_drift": DRIFT_POINT, "detected_drift": i, "delay": i - DRIFT_POINT})
            candidate_name = min(drift_trackers, key=lambda n: drift_trackers[n].f1())
            candidate = make_model(candidate_name, seed + 1000)
            candidate_warmup_start = i + 1
            candidate_eval_start = candidate_warmup_start + CANDIDATE_WARMUP
            candidate_eval_end = candidate_eval_start + CANDIDATE_EVALUATION
            replacement_attempts += 1

        for m in static_models.values():
            m.learn_one(x, y)
        for m in adaptive_models.values():
            m.learn_one(x, y)
        for m in drift_models.values():
            m.learn_one(x, y)

        if candidate is not None and candidate_warmup_start <= i < candidate_eval_start:
            candidate.learn_one(x, y)

        if i > 0 and i % WEIGHT_UPDATE_INTERVAL == 0:
            adaptive_weights = update_weights(adaptive_trackers)
            drift_weights = update_weights(drift_trackers)

        if candidate is not None and i == candidate_eval_start:
            candidate_warmup_start = None

        if candidate is not None and i == candidate_eval_end:
            incumbent = drift_models[candidate_name]
            incumbent_f1, candidate_f1, improvement = evaluate_replacement(incumbent, candidate, stream, candidate_eval_start, candidate_eval_end)
            replacement = improvement >= REPLACEMENT_THRESHOLD
            replacement_events.append({"seed": seed, "model": candidate_name, "incumbent_f1": incumbent_f1, "candidate_f1": candidate_f1, "improvement": improvement, "replacement": replacement})
            if replacement:
                drift_models[candidate_name] = candidate
                successful_replacements += 1
            candidate = None

        if i % 100 == 0:
            weight_history.append({"seed": seed, "index": i, **{f"static_{n}": static_weights[n] for n in MODEL_NAMES}, **{f"adaptive_{n}": adaptive_weights[n] for n in MODEL_NAMES}, **{f"drift_{n}": drift_weights[n] for n in MODEL_NAMES}})

        if (i + 1) % 5000 == 0:
            print(f"Seed {seed} | {i+1}/{STREAM_SIZE} | Static {static_metric.accuracy():.4f} | Adaptive {adaptive_metric.accuracy():.4f} | Drift-aware {drift_metric.accuracy():.4f}")

    result = {"seed": seed, "static_accuracy": static_metric.accuracy(), "adaptive_accuracy": adaptive_metric.accuracy(), "drift_accuracy": drift_metric.accuracy(), "static_precision": static_metric.precision(), "adaptive_precision": adaptive_metric.precision(), "drift_precision": drift_metric.precision(), "static_recall": static_metric.recall(), "adaptive_recall": adaptive_metric.recall(), "drift_recall": drift_metric.recall(), "static_f1": static_metric.f1(), "adaptive_f1": adaptive_metric.f1(), "drift_f1": drift_metric.f1(), "detection_index": detection_index, "detection_delay": detection_index - DRIFT_POINT if detection_index is not None else None, "replacement_attempts": replacement_attempts, "successful_replacements": successful_replacements, "runtime_seconds": time.time() - start_time}
    return result, histories, weight_history, drift_events, replacement_events

all_results = []
all_histories = {"static": [], "adaptive": [], "drift": []}
all_weights = []
all_drift_events = []
all_replacements = []

for seed in SEEDS:
    result, histories, weights, drift_events, replacements = run_experiment(seed)
    all_results.append(result)
    for key in all_histories:
        all_histories[key].extend([{"seed": seed, "index": i, "y": y, "prediction": p} for i, y, p in histories[key]])
    all_weights.extend(weights)
    all_drift_events.extend(drift_events)
    all_replacements.extend(replacements)

results_df = pd.DataFrame(all_results)
results_df.to_csv(os.path.join(RESULTS_DIR, "individual_runs.csv"), index=False)
results_df.agg(["mean", "std"]).to_csv(os.path.join(RESULTS_DIR, "statistical_summary.csv"))
for key, data in all_histories.items():
    pd.DataFrame(data).to_csv(os.path.join(RESULTS_DIR, f"{key}_predictions.csv"), index=False)
pd.DataFrame(all_weights).to_csv(os.path.join(RESULTS_DIR, "weight_history.csv"), index=False)
pd.DataFrame(all_drift_events).to_csv(os.path.join(RESULTS_DIR, "drift_detection.csv"), index=False)
pd.DataFrame(all_replacements).to_csv(os.path.join(RESULTS_DIR, "replacement_results.csv"), index=False)

print("\nSTATISTICAL SUMMARY")
for metric in ["static_accuracy", "adaptive_accuracy", "drift_accuracy", "static_f1", "adaptive_f1", "drift_f1", "detection_delay"]:
    print(f"{metric}: {results_df[metric].mean():.4f} ± {results_df[metric].std():.4f}")
print("Replacement attempts:", results_df["replacement_attempts"].sum())
print("Successful replacements:", results_df["successful_replacements"].sum())

for key, data in all_histories.items():
    df_plot = pd.DataFrame(data)
    df_plot["correct"] = df_plot["y"] == df_plot["prediction"]
    df_plot["rolling_accuracy"] = df_plot.groupby("seed")["correct"].transform(lambda s: s.rolling(WINDOW_SIZE, min_periods=1).mean())
    plt.figure(figsize=(12, 6))
    for seed in SEEDS:
        subset = df_plot[df_plot["seed"] == seed]
        plt.plot(subset["index"], subset["rolling_accuracy"], alpha=0.5, label=str(seed))
    plt.axvline(DRIFT_POINT, linestyle="--", label="Drift")
    plt.xlabel("Stream Instance")
    plt.ylabel("Rolling Accuracy")
    plt.title(f"{key.title()} Ensemble")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{key}_rolling_accuracy.png"), dpi=300)
    plt.close()

print("\nCompleted. Results saved to:", RESULTS_DIR)