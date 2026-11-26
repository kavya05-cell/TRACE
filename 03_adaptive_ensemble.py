import time
import csv
import os
from collections import deque

import pandas as pd
from river import tree, forest, metrics


# ============================================================
# CONFIGURATION
# ============================================================

STREAM_SIZE = 30000
WINDOW_SIZE = 500
WEIGHT_TEMPERATURE = 5.0

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

from app import load, stream

train_df, test_df = load(shuffle_train=False)

# Preserve the natural order of the training stream.
data = train_df.iloc[:STREAM_SIZE].copy()

print(f"Stream size: {len(data)}")
print(f"Attack ratio: {data['y'].mean():.2%}")


# ============================================================
# BASE MODELS
# ============================================================

models = {
    "ARF": forest.ARFClassifier(
        n_models=10,
        seed=42
    ),
    "HAT": tree.HoeffdingAdaptiveTreeClassifier(
        seed=42
    ),
    "HT": tree.HoeffdingTreeClassifier()
}


# ============================================================
# PERFORMANCE TRACKING
# ============================================================

recent_correct = {
    name: deque(maxlen=WINDOW_SIZE)
    for name in models
}

weights = {
    name: 1.0 / len(models)
    for name in models
}


# ============================================================
# METRICS
# ============================================================

ensemble_accuracy = metrics.Accuracy()
ensemble_precision = metrics.Precision()
ensemble_recall = metrics.Recall()
ensemble_f1 = metrics.F1()

individual_metrics = {
    name: {
        "accuracy": metrics.Accuracy(),
        "f1": metrics.F1()
    }
    for name in models
}


# ============================================================
# DYNAMIC WEIGHT CALCULATION
# ============================================================

def calculate_weights(performance_dict, temperature=WEIGHT_TEMPERATURE):
    """
    Converts recent learner performance into normalized weights.

    Learners with better recent accuracy receive larger weights.
    """

    scores = {}

    for name, history in performance_dict.items():

        if len(history) == 0:
            scores[name] = 0.5
        else:
            scores[name] = sum(history) / len(history)

    # Numerically stable softmax
    max_score = max(scores.values())

    exponentials = {
        name: pow(2.718281828, temperature * (score - max_score))
        for name, score in scores.items()
    }

    total = sum(exponentials.values())

    calculated_weights = {
        name: value / total
        for name, value in exponentials.items()
    }

    return calculated_weights


# ============================================================
# ENSEMBLE PREDICTION
# ============================================================

def weighted_vote(predictions, current_weights):
    """
    Weighted hard voting for binary classification.
    """

    vote_0 = 0.0
    vote_1 = 0.0

    for name, prediction in predictions.items():

        if prediction == 0:
            vote_0 += current_weights[name]

        else:
            vote_1 += current_weights[name]

    return 1 if vote_1 > vote_0 else 0


# ============================================================
# PREQUENTIAL EVALUATION
# ============================================================

results = []

start_time = time.time()

for i, (x, y) in enumerate(stream(data), start=1):

    # --------------------------------------------------------
    # 1. PREDICT BEFORE LEARNING
    # --------------------------------------------------------

    predictions = {}

    for name, model in models.items():

        predictions[name] = model.predict_one(x)

        # Cold-start handling
        if predictions[name] is None:
            predictions[name] = 0


    # --------------------------------------------------------
    # 2. ENSEMBLE PREDICTION
    # --------------------------------------------------------

    ensemble_prediction = weighted_vote(
        predictions,
        weights
    )


    # --------------------------------------------------------
    # 3. EVALUATE
    # --------------------------------------------------------

    ensemble_accuracy.update(y, ensemble_prediction)
    ensemble_precision.update(y, ensemble_prediction)
    ensemble_recall.update(y, ensemble_prediction)
    ensemble_f1.update(y, ensemble_prediction)

    for name in models:

        individual_metrics[name]["accuracy"].update(
            y,
            predictions[name]
        )

        individual_metrics[name]["f1"].update(
            y,
            predictions[name]
        )


    # --------------------------------------------------------
    # 4. UPDATE RECENT PERFORMANCE
    # --------------------------------------------------------

    for name in models:

        correct = int(predictions[name] == y)

        recent_correct[name].append(correct)


    # --------------------------------------------------------
    # 5. UPDATE DYNAMIC WEIGHTS
    # --------------------------------------------------------

    weights = calculate_weights(
        recent_correct
    )


    # --------------------------------------------------------
    # 6. LEARN AFTER PREDICTION
    # --------------------------------------------------------

    for name, model in models.items():

        model.learn_one(x, y)


    # --------------------------------------------------------
    # 7. LOG RESULTS
    # --------------------------------------------------------

    if i % 5000 == 0:

        elapsed = time.time() - start_time

        print(
            f"\n{i} instances | "
            f"Time: {elapsed:.2f}s | "
            f"Ensemble Accuracy: "
            f"{ensemble_accuracy.get():.4f} | "
            f"Ensemble F1: "
            f"{ensemble_f1.get():.4f}"
        )

        print("Current weights:")

        for name, weight in weights.items():

            print(
                f"  {name}: {weight:.4f} "
                f"(recent accuracy: "
                f"{sum(recent_correct[name]) / max(len(recent_correct[name]), 1):.4f})"
            )


        results.append({
            "instances": i,
            "accuracy": ensemble_accuracy.get(),
            "precision": ensemble_precision.get(),
            "recall": ensemble_recall.get(),
            "f1": ensemble_f1.get(),
            "ARF_weight": weights["ARF"],
            "HAT_weight": weights["HAT"],
            "HT_weight": weights["HT"]
        })


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 60)
print("ADAPTIVE ENSEMBLE FINAL RESULTS")
print("=" * 60)

print(f"Accuracy:  {ensemble_accuracy.get():.4f}")
print(f"Precision: {ensemble_precision.get():.4f}")
print(f"Recall:    {ensemble_recall.get():.4f}")
print(f"F1:        {ensemble_f1.get():.4f}")

print("\nFinal weights:")

for name, weight in weights.items():

    print(f"{name}: {weight:.4f}")


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

output_path = os.path.join(
    RESULTS_DIR,
    "adaptive_ensemble_results.csv"
)

results_df.to_csv(output_path, index=False)

print(f"\nResults saved to: {output_path}")