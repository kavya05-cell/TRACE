import os, random, numpy as np, pandas as pd
from collections import deque
from river import forest, tree
from app import load
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

SEEDS = [42, 123, 2025, 7, 99]
PRE_N, POST_N, EVAL_SIZE = 10000, 10000, 200
WINDOW, UPDATE, TEMP = 500, 100, 5
RESULT_DIR = "results/explainability_drift_multiseed_corrected"

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)

def create_models(seed):
    return {"ARF": forest.ARFClassifier(n_models=10, seed=seed), "HAT": tree.HoeffdingAdaptiveTreeClassifier(seed=seed), "HT": tree.HoeffdingTreeClassifier()}

def softmax(values):
    values = np.array(values) * TEMP; values -= values.max()
    exp = np.exp(values); return exp / exp.sum()

def predict(model, x):
    return int(model.predict_one(x) or 0)

def ensemble_predict(models, weights, x):
    preds = np.array([predict(m, x) for m in models.values()])
    return int(np.dot(preds, weights) >= 0.5)

def train(models, X, y, histories, weights, seed):
    set_seed(seed)
    for i, (x, target) in enumerate(zip(X.to_dict("records"), y)):
        target = int(target); preds = {n: predict(m, x) for n, m in models.items()}
        if i and i % UPDATE == 0:
            weights = softmax([np.mean(histories[n]) if histories[n] else 0.5 for n in models])
        for n, m in models.items():
            histories[n].append(int(preds[n] == target)); m.learn_one(x, target)
    return models, histories, weights

def evaluate(models, weights, X, y, phase, seed):
    preds = [ensemble_predict(models, weights, x) for x in X.to_dict("records")]
    return {"phase": phase, "seed": seed, "accuracy": accuracy_score(y, preds), "precision": precision_score(y, preds, zero_division=0), "recall": recall_score(y, preds, zero_division=0), "f1": f1_score(y, preds, zero_division=0)}

def make_tests(test, seed):
    normal = test[test.y == 0]; neptune = test[(test.y == 1) & (test.attack_type == "neptune")]; satan = test[(test.y == 1) & (test.attack_type == "satan")]
    if min(len(normal), len(neptune), len(satan)) == 0: raise ValueError("Required test classes/subtypes missing.")
    n = EVAL_SIZE // 2
    normal = normal.sample(n, random_state=seed, replace=len(normal) < n)
    def build(attacks, rs):
        attack = attacks.sample(n, random_state=rs, replace=len(attacks) < n); df = pd.concat([normal, attack]).sample(frac=1, random_state=rs).reset_index(drop=True)
        return df.drop(columns=["y", "attack_type"]), df.y
    return build(neptune, seed), build(satan, seed + 1)

def run(seed, train_df, test_df):
    print(f"\nSeed {seed}: Neptune -> Satan")
    features = train_df.drop(columns=["y", "attack_type"]); y = train_df.y.reset_index(drop=True); subtype = train_df.attack_type.reset_index(drop=True)
    normal_idx = np.where(y == 0)[0]; neptune_idx = np.where((y == 1) & (subtype == "neptune"))[0]; satan_idx = np.where((y == 1) & (subtype == "satan"))[0]
    rng = np.random.default_rng(seed)
    def stream(size, attack_pool):
        idx = [rng.choice(normal_idx if rng.random() < .5 else attack_pool) for _ in range(size)]; rng.shuffle(idx)
        return features.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)
    (X_pre_test, y_pre_test), (X_post_test, y_post_test) = make_tests(test_df, seed)
    X_pre, y_pre = stream(PRE_N, neptune_idx); X_post, y_post = stream(POST_N, satan_idx)
    models = create_models(seed); histories = {n: deque(maxlen=WINDOW) for n in models}; weights = np.ones(len(models)) / len(models)
    models, histories, weights = train(models, X_pre, y_pre, histories, weights, seed)
    results = [evaluate(models, weights, X_pre_test, y_pre_test, "pre_drift", seed), evaluate(models, weights, X_post_test, y_post_test, "post_drift_before_adaptation", seed)]
    models, histories, weights = train(models, X_post, y_post, histories, weights, seed + 1000)
    results.append(evaluate(models, weights, X_post_test, y_post_test, "post_drift_after_adaptation", seed))
    for r in results: print(r)
    return results

def main():
    print("Loading NSL-KDD..."); train_df, test_df = load()
    print("Train:", train_df.shape, "Test:", test_df.shape)
    results = []
    for seed in SEEDS: results.extend(run(seed, train_df, test_df))
    os.makedirs(RESULT_DIR, exist_ok=True); df = pd.DataFrame(results); df.to_csv(f"{RESULT_DIR}/seed_results.csv", index=False)
    summary = df.groupby("phase")[["accuracy", "precision", "recall", "f1"]].agg(["mean", "std"]); summary.to_csv(f"{RESULT_DIR}/summary.csv")
    print("\nMEAN ± STD\n", summary.round(4)); print("\nSaved:", os.path.abspath(RESULT_DIR))

if __name__ == "__main__": main()