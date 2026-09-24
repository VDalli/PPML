"""Use case 3: federated averaging across four departments
with secure aggregation (pairwise masks)."""
import numpy as np, pandas as pd
rng = np.random.default_rng(1)
df = pd.read_csv("students_MIC.csv")            # one college, four departmental silos
FEATS = ["attendance_pct","sem1_cgpa","internal_marks",
         "video_completion","forum_posts","late_submissions"]
mu, sd = df[FEATS].mean(), df[FEATS].std()      # agreed schema + normalisation
def xy(frame):
    X = ((frame[FEATS]-mu)/sd).values
    return np.hstack([X, np.ones((len(X),1))]), frame.passed.values
silos = {d: xy(g) for d, g in df.groupby("dept")}
# each department keeps a private 20% validation split
split = {d: (X[:int(.8*len(X))], y[:int(.8*len(y))],
             X[int(.8*len(X)):], y[int(.8*len(y)):]) for d, (X, y) in silos.items()}

sig = lambda z: 1/(1+np.exp(-z))
def local_train(w, X, y, epochs=5, lr=0.5):
    w = w.copy()
    for _ in range(epochs):
        w -= lr * X.T @ (sig(X@w) - y) / len(y)
    return w
def acc(w, X, y):
    p = sig(X@w) > .5; tpr = (p[y==1]).mean(); tnr = (~p[y==0]).mean()
    return (tpr+tnr)/2, tnr        # balanced accuracy, recall on the FAIL class
def val_acc(w):
    r = np.array([acc(w, *split[d][2:]) for d in split]).mean(0)
    return f"bal_acc={r[0]:.3f} fail_recall={r[1]:.3f}"

D = len(FEATS)+1; w_global = np.zeros(D); ROUNDS = 30
print(f"Departments: {list(split)}   rows: {[len(split[d][1]) for d in split]}")
print(f"{'round':>5s} {'validation (avg of 4 private dept splits)':>42s}"
      "   coordinator sees (first 3 coords of ONE dept update)")
for r in range(1, ROUNDS+1):
    updates, n = {}, {}
    for d, (Xtr, ytr, _, _) in split.items():
        updates[d] = local_train(w_global, Xtr, ytr) * len(ytr); n[d] = len(ytr)
    # ---- secure aggregation: pairwise masks that cancel in the sum
    depts = list(updates); masked = {d: updates[d].copy() for d in depts}
    for i, a in enumerate(depts):
        for b in depts[i+1:]:
            m = rng.normal(0, 1000, D)      # secret shared by a and b (key agreement)
            masked[a] += m; masked[b] -= m
    total = sum(masked.values())            # coordinator only ever sums masked vectors
    assert np.allclose(total, sum(updates.values()))
    w_global = total / sum(n.values())
    if r in (1, 5, 10, 20, 30):
        seen, true = np.round(masked[depts[0]][:3],1), np.round(updates[depts[0]][:3],1)
        print(f"{r:5d} {val_acc(w_global):>42s}   {seen}  (true update {true})")

# ---- baselines
Xall = np.vstack([split[d][0] for d in split])
yall = np.concatenate([split[d][1] for d in split])
w_central = np.zeros(D)
for _ in range(ROUNDS*5): w_central = local_train(w_central, Xall, yall, epochs=1)
print(f"\nCentralised (pooled data)   {val_acc(w_central)}")
print(f"Federated + secure agg      {val_acc(w_global)}")
for d in split:
    w_loc = np.zeros(D)
    for _ in range(ROUNDS*5):
        w_loc = local_train(w_loc, split[d][0], split[d][1], epochs=1)
    print(f"Local-only {d:6s}            {val_acc(w_loc)}")
np.save("federated_model.npy", w_global)
