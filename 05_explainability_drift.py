"""Corrected explainability-aware subtype-shift experiment for adaptive NIDS.
Run: python 05_explainability_drift.py
NOTE: This is a synthetic subtype distribution shift, not real temporal drift.
"""
from __future__ import annotations
import copy, warnings
from collections import deque
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

SEED = 42
PRE_N = 10_000
POST_N = 10_000
ROLLING = 500
UPDATE_EVERY = 100
TEMP = 5.0
TEST_PER_CLASS = 500
SHAP_NORMAL = 4
SHAP_ATTACK = 4
BACKGROUND_N = 30
SHAP_MAX_EVALS = 300
OUT = Path('results/explainability_drift')

from app import load
from river import forest, tree
try:
    import shap
except ImportError:
    raise SystemExit('Install SHAP: python -m pip install shap')


def rowdict(row, names):
    return {n: float(v) for n, v in zip(names, row)}

def p_attack(model, x):
    try:
        d = model.predict_proba_one(x) or {}
        p = float(d.get(1, 0.0))
        if np.isfinite(p): return float(np.clip(p, 0, 1))
    except Exception: pass
    pred = model.predict_one(x)
    return float(pred == 1) if pred is not None else 0.5

def weights_from_acc(acc):
    z = (np.asarray(acc, dtype=float) - max(acc)) * TEMP
    ex = np.exp(np.clip(z, -50, 50))
    return ex / ex.sum()

class Ensemble:
    def __init__(self, models, weights, features):
        self.models = models
        self.weights = np.asarray(weights, dtype=float)
        self.features = list(features)
    def predict_proba(self, X):
        X = np.asarray(X)
        if X.ndim == 1: X = X.reshape(1, -1)
        probs = []
        for row in X:
            x = rowdict(row, self.features)
            p = np.clip(sum(w * p_attack(m, x) for w, m in zip(self.weights, self.models)), 0, 1)
            probs.append([1-p, p])
        return np.asarray(probs)

def get_subtypes(df, subtype_col):
    attack_rows = df[df.y.astype(int) == 1]
    counts = attack_rows[subtype_col].astype(str).value_counts()
    counts = counts[counts.index.str.lower() != 'normal']
    if len(counts) < 2: raise RuntimeError('Need at least two attack subtypes.')
    return list(counts.index[:2])

def sample_shift(df, subtype_col, n, dominant, minority, rng):
    normals = np.flatnonzero(df.y.astype(int).to_numpy() == 0)
    attacks = np.flatnonzero(df.y.astype(int).to_numpy() == 1)
    sub = df[subtype_col].astype(str).to_numpy()
    dom_idx = attacks[sub[attacks] == dominant]
    min_idx = attacks[sub[attacks] == minority]
    if not len(normals) or not len(dom_idx) or not len(min_idx):
        raise RuntimeError(f'Missing rows for normal/{dominant}/{minority} in this split.')
    n0, n1 = n // 2, n - n // 2
    attack_dom = int(n1 * .8)
    picked = np.r_[rng.choice(normals, n0, replace=True),
                   rng.choice(dom_idx, attack_dom, replace=True),
                   rng.choice(min_idx, n1-attack_dom, replace=True)]
    rng.shuffle(picked)
    return picked

def train_phase(models, X, y, features, phase):
    # Reset the rolling score histories for each phase so post-phase scores
    # cannot be dominated by pre-phase performance.
    histories = [deque(maxlen=ROLLING) for _ in models]
    w = np.ones(len(models)) / len(models)
    prequential_correct = [[] for _ in models]
    for i, (row, yi) in enumerate(zip(X, y)):
        x = rowdict(row, features); yi = int(yi)
        for j, model in enumerate(models):
            pred = model.predict_one(x)
            ok = int(pred == yi) if pred is not None else 0
            histories[j].append(ok); prequential_correct[j].append(ok)
            model.learn_one(x, yi)
        if (i+1) % UPDATE_EVERY == 0:
            w = weights_from_acc([np.mean(h) if h else .5 for h in histories])
    acc = [float(np.mean(q)) for q in prequential_correct]
    rolling = [float(np.mean(h)) for h in histories]
    w = weights_from_acc(rolling)
    return w, acc, rolling

def evaluate(tag, ensemble, X, y):
    probs = ensemble.predict_proba(X)[:, 1]
    pred = (probs >= .5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp/(tp+fp) if tp+fp else 0.; recall = tp/(tp+fn) if tp+fn else 0.
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.
    return {'evaluation': tag, 'n': len(y), 'accuracy': float((pred == y).mean()), 'precision': precision, 'recall': recall, 'f1': f1}, probs, pred

def shap_snapshot(ensemble, background, X, features, phase):
    masker = shap.maskers.Independent(background, max_samples=len(background))
    explainer = shap.Explainer(lambda z: ensemble.predict_proba(z)[:, 1], masker,
                               algorithm='permutation', output_names=['Attack probability'])
    result = explainer(X, max_evals=max(SHAP_MAX_EVALS, 2*len(features)+1))
    vals = np.asarray(result.values)
    if vals.ndim == 3: vals = vals[:, :, 0]
    if vals.ndim != 2: raise RuntimeError(f'Unexpected SHAP shape: {vals.shape}')
    pd.DataFrame([{'sample': i, 'feature': f, 'shap_value': float(vals[i,j])}
                  for i in range(len(vals)) for j,f in enumerate(features)]).to_csv(OUT/f'shap_{phase}.csv', index=False)
    return vals, np.mean(np.abs(vals), axis=0)

def plot_importance(imp, features, title, path):
    top = np.argsort(imp)[-20:]
    plt.figure(figsize=(10, 8)); plt.barh(np.asarray(features)[top], imp[top])
    plt.xlabel('Mean absolute SHAP value (attack probability)'); plt.title(title)
    plt.tight_layout(); plt.savefig(path, dpi=180, bbox_inches='tight'); plt.close()

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng_train = np.random.default_rng(SEED); rng_test = np.random.default_rng(SEED+1); rng_shap = np.random.default_rng(SEED+2)
    print('Loading NSL-KDD via app.load()...')
    loaded = load(shuffle_train=False)
    if isinstance(loaded, dict):
        train_df = loaded.get('train', loaded.get('train_df')); test_df = loaded.get('test', loaded.get('test_df'))
    else: train_df, test_df = loaded[0], loaded[1]
    subtype_col = 'attack_type' if 'attack_type' in train_df.columns else 'label' if 'label' in train_df.columns else None
    if subtype_col is None or subtype_col not in test_df.columns: raise RuntimeError('Need attack_type/label subtype in both train and test.')
    features = [c for c in train_df.columns if c not in {'y','label','attack_type','difficulty'}]
    test_df = test_df.reindex(columns=features + [c for c in test_df.columns if c in {'y','label','attack_type','difficulty'}], fill_value=0)
    X = train_df[features].astype(float).to_numpy(); y = train_df.y.astype(int).to_numpy()
    Xt = test_df[features].astype(float).to_numpy(); yt = test_df.y.astype(int).to_numpy()
    subtype_a, subtype_b = get_subtypes(train_df, subtype_col)
    print(f'Simulated shift: {subtype_a} dominant -> {subtype_b} dominant; binary class ratio ~50/50.')

    # Construct fixed, independently sampled test streams with matching shift logic.
    pre_test_idx = sample_shift(test_df, subtype_col, TEST_PER_CLASS*2, subtype_a, subtype_b, rng_test)
    post_test_idx = sample_shift(test_df, subtype_col, TEST_PER_CLASS*2, subtype_b, subtype_a, rng_test)
    Xpre, ypre = Xt[pre_test_idx], yt[pre_test_idx]
    Xpost, ypost = Xt[post_test_idx], yt[post_test_idx]

    pre_idx = sample_shift(train_df, subtype_col, PRE_N, subtype_a, subtype_b, rng_train)
    post_idx = sample_shift(train_df, subtype_col, POST_N, subtype_b, subtype_a, rng_train)
    models = [forest.ARFClassifier(n_models=10, seed=SEED),
              tree.HoeffdingAdaptiveTreeClassifier(seed=SEED),
              tree.HoeffdingTreeClassifier()]
    names = ['ARF','HAT','Hoeffding Tree']

    print(f'Training pre-drift phase ({PRE_N:,} examples)...')
    w_pre, pre_prequential, pre_rolling = train_phase(models, X[pre_idx], y[pre_idx], features, 'pre')
    before = copy.deepcopy(Ensemble(models, w_pre, features))
    print('Pre-phase prequential accuracy:', dict(zip(names, np.round(pre_prequential,4))))
    print('Pre-phase rolling accuracy:', dict(zip(names, np.round(pre_rolling,4))))
    print('Pre-phase weights:', dict(zip(names, np.round(w_pre,4))))

    print('Evaluating pre-drift model on both test distributions...')
    metrics=[]; pred_data={}
    for tag, xx, yy in [('pre_model_on_pre_distribution',Xpre,ypre),('pre_model_on_post_distribution',Xpost,ypost)]:
        m, probs, preds = evaluate(tag, before, xx, yy); metrics.append(m); pred_data[tag]=(probs,preds)

    print(f'Training shifted phase ({POST_N:,} examples)...')
    w_post, post_prequential, post_rolling = train_phase(models, X[post_idx], y[post_idx], features, 'post')
    after = copy.deepcopy(Ensemble(models, w_post, features))
    print('Post-phase prequential accuracy:', dict(zip(names, np.round(post_prequential,4))))
    print('Post-phase rolling accuracy:', dict(zip(names, np.round(post_rolling,4))))
    print('Post-phase weights:', dict(zip(names, np.round(w_post,4))))
    m, probs, preds = evaluate('adapted_model_on_post_distribution', after, Xpost, ypost)
    metrics.append(m); pred_data[m['evaluation']] = (probs,preds)
    pd.DataFrame(metrics).to_csv(OUT/'drift_experiment_metrics.csv', index=False)
    print('\nEvaluation results:'); print(pd.DataFrame(metrics).to_string(index=False))

    pd.DataFrame({'test_row_index':post_test_idx,'actual':ypost,
        'p_attack_before_adaptation':pred_data['pre_model_on_post_distribution'][0],
        'pred_before_adaptation':pred_data['pre_model_on_post_distribution'][1],
        'p_attack_after_adaptation':pred_data['adapted_model_on_post_distribution'][0],
        'pred_after_adaptation':pred_data['adapted_model_on_post_distribution'][1]}).to_csv(OUT/'post_shift_prediction_comparison.csv', index=False)
    pd.DataFrame([{'model':n,'pre_phase_prequential_accuracy':a,'pre_phase_rolling_accuracy':r,'pre_phase_weight':w,
                   'post_phase_prequential_accuracy':b,'post_phase_rolling_accuracy':s,'post_phase_weight':v}
                  for n,a,r,w,b,s,v in zip(names,pre_prequential,pre_rolling,w_pre,post_prequential,post_rolling,w_post)]).to_csv(OUT/'ensemble_diagnostics.csv', index=False)

    # Balanced SHAP set: same exact rows for both snapshots, including normal and attack.
    n_idx = np.flatnonzero(ypost == 0); a_idx = np.flatnonzero(ypost == 1)
    if len(n_idx) < SHAP_NORMAL or len(a_idx) < SHAP_ATTACK: raise RuntimeError('Not enough examples for balanced SHAP sample.')
    shap_ids = np.r_[rng_shap.choice(n_idx, SHAP_NORMAL, replace=False), rng_shap.choice(a_idx, SHAP_ATTACK, replace=False)]
    Xshap = Xpost[shap_ids]
    bg = X[rng_shap.choice(len(X), min(BACKGROUND_N,len(X)), replace=False)]
    pd.DataFrame({'post_test_index':post_test_idx[shap_ids], 'actual':ypost[shap_ids]}).to_csv(OUT/'shap_sample_manifest.csv',index=False)
    print('\nComputing SHAP for the SAME balanced post-shift samples before adaptation...')
    _, imp_before = shap_snapshot(before,bg,Xshap,features,'before_adaptation_on_post_shift')
    print('Computing SHAP after adaptation...')
    _, imp_after = shap_snapshot(after,bg,Xshap,features,'after_adaptation_on_post_shift')
    plot_importance(imp_before,features,'SHAP — pre-adaptation model on post-shift samples',OUT/'shap_importance_before.png')
    plot_importance(imp_after,features,'SHAP — adapted model on same post-shift samples',OUT/'shap_importance_after.png')
    delta=imp_after-imp_before; ix=np.argsort(np.abs(delta))[-20:]
    plt.figure(figsize=(10,8)); plt.barh(np.asarray(features)[ix],delta[ix]); plt.axvline(0)
    plt.xlabel('Change in mean absolute SHAP (adapted − pre-adaptation)'); plt.title('Attribution change after simulated subtype shift')
    plt.tight_layout(); plt.savefig(OUT/'shap_importance_change.png',dpi=180,bbox_inches='tight'); plt.close()
    pd.DataFrame({'feature':features,'importance_before':imp_before,'importance_after':imp_after,
                  'importance_change':delta,'absolute_change':np.abs(delta)}).sort_values('absolute_change',ascending=False).to_csv(OUT/'shap_importance_delta.csv',index=False)
    print('\nCompleted. Results:', OUT.resolve())
    for f in sorted(OUT.iterdir()): print(' -',f.name)
    print('\nInterpretation caveat: simulated subtype shift only; this is not real timestamped concept drift.')

if __name__ == '__main__': main()
