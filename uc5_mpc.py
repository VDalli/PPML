"""Use case 5: three colleges compute pooled statistics
with additive secret sharing."""
import numpy as np, pandas as pd
P = 2**61 - 1                      # Mersenne prime field
SCALE = 10**6                      # fixed-point scaling for real numbers
rng = np.random.default_rng(11)
PARTIES = ["MIC", "SVEC", "PVPSIT"]
data = {c: pd.read_csv(f"students_{c}.csv") for c in PARTIES}

def to_field(x):  return int(round(x*SCALE)) % P
def from_field(v):
    v = int(v);  v = v - P if v > P//2 else v;  return v / SCALE
def share(v, n=3):
    s = [int(rng.integers(0, P)) for _ in range(n-1)]
    s.append((v - sum(s)) % P); return s

# ---- Step 1: each college computes LOCAL sufficient statistics on its own data ...
FEATS = ["attendance_pct", "sem1_cgpa", "video_completion"]
def local_stats(df):
    return {"n": len(df), "sum_pass": df.passed.sum(),
            "sum_x": df[FEATS].sum().values, "sum_x2": (df[FEATS]**2).sum().values,
            "sum_xy": (df[FEATS].mul(df.passed, axis=0)).sum().values}
# ---- Step 2: ... and splits every statistic into 3 shares, one per party
inbox = {p: [] for p in PARTIES}       # what each party RECEIVES
for owner, df in data.items():
    st = local_stats(df)
    for k, v in st.items():
        vals = np.atleast_1d(v)
        for j, val in enumerate(vals):
            for p, s in zip(PARTIES, share(to_field(float(val)))):
                inbox[p].append((k, j, s))
print("What party SVEC receives (first 6 shares) - indistinguishable from random:")
for k, j, s in inbox["SVEC"][:6]: print(f"   {k:8s}[{j}]  {s}")

# ---- Step 3: each party sums the shares it holds (locally), then publishes the partial sum
partials = {p: {} for p in PARTIES}
for p in PARTIES:
    for k, j, s in inbox[p]: partials[p][(k, j)] = (partials[p].get((k, j), 0) + s) % P
keys = sorted(partials["MIC"])
recon = {key: from_field(sum(partials[p][key] for p in PARTIES) % P) for key in keys}

n = recon[("n", 0)]; pass_rate = recon[("sum_pass", 0)] / n
print(f"\nPooled rows across 3 colleges: {int(n)}   pooled pass rate = {pass_rate:.3f}")
print(f"{'feature':18s}{'pooled mean':>12s}{'pooled std':>12s}{'corr with pass':>16s}")
for j, f in enumerate(FEATS):
    m = recon[("sum_x", j)]/n; var = recon[("sum_x2", j)]/n - m**2
    cov = recon[("sum_xy", j)]/n - m*pass_rate
    corr = cov/np.sqrt(var*pass_rate*(1-pass_rate))
    print(f"{f:18s}{m:12.3f}{np.sqrt(var):12.3f}{corr:16.3f}")

# ---- Step 4 (verification only): compare with pooled raw data, which no party ever saw
pooled = pd.concat(data.values())
print(f"\nCheck against pooled raw data: pass rate {pooled.passed.mean():.3f}, "
      f"corr(attendance, pass) = {pooled.attendance_pct.corr(pooled.passed):.3f}")
print("No party saw any other college's student-level rows.")
