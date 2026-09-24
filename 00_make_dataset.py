"""Step 0: generate a synthetic multi-college student dataset (no real students)."""
import numpy as np, pandas as pd
rng = np.random.default_rng(42)
COLLEGES = ["MIC", "SVEC", "PVPSIT"]
DEPTS = ["CSE", "ECE", "MECH", "CIVIL"]
DISTRICTS = ["Krishna", "NTR", "Guntur", "West Godavari", "Eluru", "Prakasam"]
rows = []
n = 0
for c in COLLEGES:
    for d in DEPTS:
        size = int(rng.integers(150, 260))
        att = np.clip(rng.normal(78, 12, size), 35, 100)
        cg  = np.clip(rng.normal(7.2, 1.1, size), 4.0, 10.0)
        im  = np.clip(rng.normal(18, 5, size) + 0.15*(att-78), 0, 30)  # internal /30
        vid = np.clip(rng.beta(4, 2, size), 0, 1)                      # video completion
        forum = rng.poisson(3, size)
        late  = rng.poisson(1.5, size)
        # probability of passing depends on the features (department offsets differ)
        z = (-15.6 + 0.09*att + 0.9*cg + 0.12*im + 1.5*vid + 0.12*forum - 0.5*late
             + {"CSE":0.3,"ECE":0.1,"MECH":-0.2,"CIVIL":-0.1}[d])
        p = 1/(1+np.exp(-z))
        passed = rng.random(size) < p
        for i in range(size):
            n += 1
            rows.append(dict(
                roll_no=f"{c}{d}{2024}{i+1:03d}", name=f"Student {n}",
                dob=pd.Timestamp("2006-01-01")
                    + pd.Timedelta(days=int(rng.integers(0, 730))),
                gender=rng.choice(["M","F"], p=[0.62,0.38]),
                category=rng.choice(["OC","BC","SC","ST"], p=[0.35,0.42,0.16,0.07]),
                district=rng.choice(DISTRICTS), college=c, dept=d,
                attendance_pct=round(float(att[i]),1), sem1_cgpa=round(float(cg[i]),2),
                internal_marks=round(float(im[i]),1),
                video_completion=round(float(vid[i]),2),
                forum_posts=int(forum[i]), late_submissions=int(late[i]),
                passed=int(passed[i])))
df = pd.DataFrame(rows)
df.to_csv("students.csv", index=False)
for c in COLLEGES:
    df[df.college==c].to_csv(f"students_{c}.csv", index=False)
print(f"Wrote students.csv with {len(df)} rows and {df.shape[1]} columns")
print(df.head(5).to_string(index=False))
print("\nRows per college/department:")
print(df.groupby(["college","dept"]).size().unstack())
