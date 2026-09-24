"""Use case 2: differentially private release of grade statistics
with a privacy-budget ledger."""
import numpy as np, pandas as pd, json, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rng = np.random.default_rng(7)
df = pd.read_csv("students.csv")

LEDGER = "privacy_ledger.json"
ledger = (json.load(open(LEDGER)) if os.path.exists(LEDGER)
          else {"budget_total": 3.0, "spent": 0.0, "queries": []})

def spend(eps, desc):
    if ledger["spent"] + eps > ledger["budget_total"]:
        raise RuntimeError(f"REFUSED: '{desc}' needs eps={eps} but only "
                           f"{ledger['budget_total']-ledger['spent']:.2f} remains")
    ledger["spent"] += eps; ledger["queries"].append({"query": desc, "eps": eps})
    json.dump(ledger, open(LEDGER, "w"), indent=1)

def dp_count(series, eps):                 # sensitivity of a count = 1
    noisy = series + rng.laplace(0, 1/eps, len(series))
    return np.maximum(0, np.round(noisy)).astype(int)

def dp_mean(values, eps, lo, hi):          # sensitivity of a bounded mean = (hi-lo)/n
    n = len(values); sens = (hi - lo) / n
    return float(np.clip(np.mean(values) + rng.laplace(0, sens/eps), lo, hi))

# ---- Query 1: first-class (CGPA >= 7.5) counts by department
eps1 = 0.5
true_counts = (df[df.sem1_cgpa >= 7.5].groupby("dept").size()
               .reindex(["CSE","ECE","MECH","CIVIL"]))
spend(eps1, "first-class counts by dept")
pub_counts = dp_count(true_counts.values, eps1)
print(f"Query 1 (eps={eps1}): first-class counts by department")
print(pd.DataFrame({"true": true_counts.values, "published": pub_counts},
                   index=true_counts.index).to_string())

# ---- Query 2: mean CGPA by department and category (small cells get large noise)
eps2 = 0.5
spend(eps2, "mean CGPA by dept x category")
print(f"\nQuery 2 (eps={eps2}): mean CGPA by dept x category  (noise scale = 10/(n*eps))")
print(f"{'dept':6s}{'category':10s}{'n':>5s}{'true':>8s}"
      f"{'published':>11s}{'noise scale':>13s}")
for (d, c), g in df.groupby(["dept","category"]):
    n = len(g); scale = 10/(n*eps2)
    flag = "  <- unpublishable (scale>0.5)" if scale > 0.5 else ""
    pub = dp_mean(g.sem1_cgpa.values, eps2, 0, 10)
    print(f"{d:6s}{c:10s}{n:5d}{g.sem1_cgpa.mean():8.2f}{pub:11.2f}{scale:13.3f}{flag}")

# ---- Query 3: attempt to overspend the budget
try:
    spend(2.5, "ad hoc request from placement cell")
except RuntimeError as e:
    print("\n" + str(e))
print(f"\nLedger: spent {ledger['spent']:.2f} of {ledger['budget_total']:.2f} eps "
      f"across {len(ledger['queries'])} queries (see {LEDGER})")

# ---- Plot true vs published counts
fig, ax = plt.subplots(figsize=(5,3), dpi=200)
x = np.arange(4)
ax.bar(x-0.18, true_counts.values, 0.36, label="true")
ax.bar(x+0.18, pub_counts, 0.36, label=f"published (eps={eps1})")
ax.set_xticks(x); ax.set_xticklabels(true_counts.index)
ax.set_ylabel("first-class students"); ax.legend(frameon=False)
plt.tight_layout(); plt.savefig("uc2_counts.png"); print("Saved uc2_counts.png")
