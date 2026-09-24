"""Use case 4: train a grade-prediction model with DP-SGD
and report the (epsilon, delta) guarantee."""
import numpy as np, pandas as pd
from dp_accounting import dp_event, rdp
from sklearn.model_selection import train_test_split
rng = np.random.default_rng(3)
df = pd.read_csv("students.csv")
FEATS = ["attendance_pct","sem1_cgpa","internal_marks",
         "video_completion","forum_posts","late_submissions"]
X = ((df[FEATS]-df[FEATS].mean())/df[FEATS].std()).values
X = np.hstack([X, np.ones((len(X),1))]); y = df.passed.values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
sig = lambda z: 1/(1+np.exp(-z))
def evaluate(w):
    p = sig(Xte@w) > .5
    return ((p[yte==1]).mean() + (~p[yte==0]).mean())/2, (~p[yte==0]).mean()

# ---- DP-SGD hyper-parameters (Abadi et al., 2016)
N, D = Xtr.shape; BATCH = 64; EPOCHS = 30; LR = 0.5
CLIP = 1.0                      # per-example gradient L2 norm bound
STEPS = EPOCHS * N // BATCH; q = BATCH / N; DELTA = 1/(10*N)

def compute_eps(sigma):
    acct = rdp.RdpAccountant()
    step = dp_event.PoissonSampledDpEvent(q, dp_event.GaussianDpEvent(sigma))
    acct.compose(dp_event.SelfComposedDpEvent(step, STEPS))
    return acct.get_epsilon(DELTA)

def train(sigma):
    w = np.zeros(D)
    for _ in range(STEPS):
        idx = rng.random(N) < q                                   # Poisson sampling
        Xb, yb = Xtr[idx], ytr[idx]
        g = Xb * (sig(Xb@w) - yb)[:, None]                        # per-example gradients
        g = g / np.maximum(1, np.linalg.norm(g, axis=1, keepdims=True)/CLIP)   # clip
        noise = rng.normal(0, sigma*CLIP, D)                      # Gaussian noise
        w -= LR * (g.sum(0) + noise) / BATCH
    return w

print(f"train rows={N}  test rows={len(yte)}  batch={BATCH}  epochs={EPOCHS}  "
      f"steps={STEPS}  clip={CLIP}  delta={DELTA:.1e}")
print(f"{'noise sigma':>12s} {'epsilon':>9s} {'bal_acc':>9s} {'fail_recall':>12s}")
w0 = np.zeros(D)
for _ in range(STEPS):
    idx = rng.integers(0, N, BATCH)
    w0 -= LR * Xtr[idx].T @ (sig(Xtr[idx]@w0) - ytr[idx]) / BATCH
b, r = evaluate(w0); print(f"{'none (non-DP)':>12s} {'inf':>9s} {b:9.3f} {r:12.3f}")
for sigma in [0.5, 0.8, 1.2, 2.0]:
    w = train(sigma); b, r = evaluate(w)
    print(f"{sigma:12.1f} {compute_eps(sigma):9.2f} {b:9.3f} {r:12.3f}")
