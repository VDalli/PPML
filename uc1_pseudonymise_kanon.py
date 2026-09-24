"""Use case 1: pseudonymise identifiers and measure k-anonymity of quasi-identifiers."""
import hmac, hashlib, os, pandas as pd
df = pd.read_csv("students.csv", parse_dates=["dob"])
QI = ["gender", "category", "district", "dept", "birth_year"]

# ---- Step 1: keyed pseudonyms (key stays in the examination cell, never with analysts)
key = os.environ.get("PSEUDO_KEY", "change-me-in-production").encode()
df["student_id"] = [hmac.new(key, r.encode(), hashlib.sha256).hexdigest()[:12]
                    for r in df.roll_no]

# ---- Step 2: drop direct identifiers, generalise dates
df["birth_year"] = df.dob.dt.year
released = df.drop(columns=["roll_no", "name", "dob"])

def k_anonymity(frame, qi):
    sizes = frame.groupby(qi).size()
    return int(sizes.min()), int((sizes == 1).sum()), len(sizes)

k, singletons, groups = k_anonymity(released, QI)
print(f"Quasi-identifiers: {QI}")
print(f"BEFORE generalisation : k = {k:2d}  unique (k=1) = {singletons:4d}  groups = {groups}")

# ---- Step 3: generalise district -> region and coarsen category until k >= 5
REGION = {"Krishna":"Central", "NTR":"Central", "Guntur":"South", "Prakasam":"South",
          "West Godavari":"North", "Eluru":"North"}
released["region"] = released.district.map(REGION)
released = released.drop(columns=["district"])
QI2 = ["gender", "category", "region", "dept", "birth_year"]
k, singletons, groups = k_anonymity(released, QI2)
print(f"AFTER  district->region: k = {k:2d}  unique (k=1) = {singletons:4d}  groups = {groups}")

released["category"] = released.category.replace({"SC":"SC/ST", "ST":"SC/ST"})
k, singletons, groups = k_anonymity(released, QI2)
print(f"AFTER  SC/ST merge     : k = {k:2d}  unique (k=1) = {singletons:4d}  groups = {groups}")

# ---- Step 4: suppress any remaining groups smaller than 5 and write the analyst file
sizes = released.groupby(QI2)["student_id"].transform("size")
safe = released[sizes >= 5]
print(f"Suppressed {len(released)-len(safe)} records in groups with k<5; "
      f"releasing {len(safe)} rows")
safe.to_csv("students_pseudonymised.csv", index=False)
print("\nFirst rows of the analyst-facing file:")
COLS = ["student_id","gender","category","region","dept","birth_year","sem1_cgpa","passed"]
print(safe[COLS].head(5).to_string(index=False))
