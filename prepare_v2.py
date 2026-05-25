# prepare_v2.py
# ---------------------------------------------------------------
# THIS FILE IS FROZEN. THE AGENT MUST NEVER MODIFY IT.
#
# Run this ONCE to set up the expanded feature dataset.
# Everything comes from the raw BRFSS 2015 XPT file — target
# and features are aligned by construction because they come
# from the same rows of the same file.
#
# Target: DIABETE3 — binary (1 = diabetes/prediabetes, 0 = no)
# Features: 37 clinically selected variables from the codebook
# Rows: completed interviews only (DISPCODE == 1100)
#
# Usage (run once only):
#   python prepare_v2.py
#
# Requirements:
#   pip install pyreadstat scikit-learn numpy pandas
# ---------------------------------------------------------------

import os
import urllib.request
import zipfile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

OUTPUT_DIR   = "data_v2"
BRFSS_URL    = "https://www.cdc.gov/brfss/annual_data/2015/files/LLCP2015XPT.ZIP"
BRFSS_ZIP    = "data_v2/LLCP2015.zip"
BRFSS_XPT    = "data_v2/LLCP2015.XPT"
RANDOM_STATE = 42

# ── Clinically selected features (domain knowledge filter) ───────────────
# These are NEW variables not in the original 21-feature UCI dataset
# plus the original 21 mapped to their raw BRFSS names.
# Every variable here has a documented clinical or socioeconomic
# relationship with diabetes risk.

FEATURES = [
    # ── Health status ──────────────────────────────────────────
    "GENHLTH",    # General health (1=Excellent to 5=Poor)
    "PHYSHLTH",   # Days physical health not good (0-30)
    "MENTHLTH",   # Days mental health not good (0-30)
    "POORHLTH",   # Days poor health limited activities (0-30)

    # ── Healthcare access ───────────────────────────────────────
    "HLTHPLN1",   # Has health care coverage (1=Yes, 2=No)
    "PERSDOC2",   # Has personal doctor (1=Yes one, 2=More than one, 3=No)
    "MEDCOST",    # Couldn't see doctor due to cost (1=Yes, 2=No)
    "CHECKUP1",   # Time since last routine checkup (1=<1yr ... 4=5+yrs)

    # ── Cardiovascular / chronic conditions ─────────────────────
    "BPHIGH4",    # Ever told high blood pressure (1=Yes, 3=No)
    "BPMEDS",     # Currently taking BP medication (1=Yes, 2=No)
    "BLOODCHO",   # Ever had cholesterol checked (1=Yes, 2=No)
    "TOLDHI2",    # Ever told high cholesterol (1=Yes, 2=No)
    "CVDINFR4",   # Ever had heart attack (1=Yes, 2=No)
    "CVDCRHD4",   # Ever had coronary heart disease (1=Yes, 2=No)
    "CVDSTRK3",   # Ever had stroke (1=Yes, 2=No)
    "CHCKIDNY",   # Ever told have kidney disease (1=Yes, 2=No)  ← NEW
    "HAVARTH3",   # Ever told have arthritis (1=Yes, 2=No)        ← NEW
    "ADDEPEV2",   # Ever told have depression (1=Yes, 2=No)       ← NEW
    "CHCCOPD1",   # Ever told have COPD (1=Yes, 2=No)             ← NEW
    "ASTHMA3",    # Ever told have asthma (1=Yes, 2=No)           ← NEW

    # ── Lifestyle ───────────────────────────────────────────────
    "EXERANY2",   # Exercise in past 30 days (1=Yes, 2=No)
    "SMOKE100",   # Smoked 100+ cigarettes lifetime (1=Yes, 2=No)

    # ── Demographics ────────────────────────────────────────────
    "SEX",        # Sex (1=Male, 2=Female)
    "MARITAL",    # Marital status (1=Married ... 6=Unmarried couple) ← NEW
    "EDUCA",      # Education level (1-6 scale)
    "RENTHOM1",   # Own/rent home (1=Own, 2=Rent, 3=Other)            ← NEW
    "EMPLOY1",    # Employment status (1=Employed ... 8=Unable to work)← NEW
    "INCOME2",    # Annual household income (1=<$10k ... 8=$75k+)
    "INTERNET",   # Internet use past 30 days (1=Yes, 2=No)           ← NEW
    "VETERAN3",   # Veteran status (1=Yes, 2=No)                      ← NEW

    # ── Functional status ────────────────────────────────────────
    "QLACTLM2",  # Activity limitation due to health (1=Yes, 2=No)   ← NEW
    "USEEQUIP",  # Requires special equipment (1=Yes, 2=No)           ← NEW

    # ── CDC calculated variables ─────────────────────────────────
    "_BMI5",      # BMI x100 (divide by 100 for actual BMI)
    "_RFHYPE5",   # High blood pressure calculated (1=No, 2=Yes)
    "_RFCHOL",    # High cholesterol calculated (1=No, 2=Yes)
    "_SMOKER3",   # Smoking status (1=Every day ... 4=Never)
    "_RFDRHV5",   # Heavy drinker (1=No, 2=Yes)
    "_AGEG5YR",   # Age group (1=18-24 ... 13=80+)

    # ── Extended features (v3 additions) ─────────────────────────
    "_FRUTSUM",   # Fruit consumption: total daily servings x100 (CDC computed)
    "_VEGESUM",   # Vegetable consumption: total daily servings x100 (CDC computed)
    "ALCDAY5",    # Days drank alcohol past 30 days (101-199=weekly, 201-299=monthly, 888=none)
    "AVEDRNK2",   # Average drinks per drinking occasion
    "PA1MIN_",    # Total physical activity minutes per week (CDC computed)
    "_PACAT1",    # Physical activity category (1=Highly active ... 4=Inactive)
    "WTKG3",      # Weight in kg x100 (CDC computed)
    "CIMEMLOS",   # Confusion/memory loss past 12 months (1=Yes, 2=No)
    "CHCSCNCR",   # Ever told have skin cancer (1=Yes, 2=No)
    "CHCOCNCR",   # Ever told have other cancer (1=Yes, 2=No)
    "_PA150R2",   # Met aerobic PA guideline (1=No, 2=Yes)

    # ── Extended features (v4 additions) ─────────────────────────
    "SLEPTIM1",   # Hours of sleep per 24-hour period (U-shaped T2D risk: <6 and >9 hrs both elevated)
    "DIFFWALK",   # Difficulty walking or climbing stairs (1=Yes, 2=No)
    "DECIDE",     # Difficulty concentrating/remembering/making decisions (1=Yes, 2=No)
    "DIFFALON",   # Difficulty doing errands alone due to health (1=Yes, 2=No)
    "_DRNKWEK",   # CDC-computed total drinks per week (continuous float)
    "_RFBING5",   # Binge drinker: 5+ drinks men / 4+ women on one occasion (1=No, 2=Yes)
    "FLUSHOT6",   # Flu shot in past 12 months (1=Yes, 2=No) — healthcare engagement proxy

    # ── Extended features (v5 additions) ─────────────────────────
    "HIVTST6",    # Ever tested for HIV (1=Yes, 2=No) — healthcare engagement proxy (90% coverage)
    "CHILDREN",   # Number of children in household (01-87=count, 88=none; 100% coverage)
    "NUMADULT",   # Number of adults in household (01-76=count; 58% coverage)
    "LMTJOIN3",   # Limited activities due to joint symptoms (1=Yes, 2=No; 31% coverage)

    # ── Extended features (v6 additions) ─────────────────────────
    "SMOKDAY2",   # Current smoking frequency (1=Every day, 2=Some days, 3=Not at all) — granular smoking
    "PNEUVAC3",   # Ever had pneumococcal vaccine (1=Yes, 2=No) — healthcare engagement proxy
    "_PASTRNG",   # Met muscle-strengthening guidelines (1=No, 2=Yes; CDC computed)
    "DRNK3GE5",   # Days had 5+ drinks on one occasion in past 30 days (88=none→0)

    # ── Extended features (v7 additions / Phase_L) ───────────────
    "LASTDEN4",   # Time since last dental visit (1=<1yr … 5=Never) — oral-metabolic link
    "RMVTETH4",   # Permanent teeth removed due to decay (1=None, 2=1-5, 3=6+) — chronic disease marker
    "DIFFDRES",   # Difficulty dressing or bathing (1=Yes, 2=No) — severe functional limitation
    "STRENGTH",   # Days of muscle-strengthening activities in past week (0-7) — continuous complement to _PASTRNG
    "BLIND",      # Blind or serious difficulty seeing (1=Yes, 2=No) — diabetic retinopathy marker
]

TARGET_COL = "DIABETE3"


def download_brfss():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.exists(BRFSS_XPT):
        print("  Raw BRFSS XPT already available.")
        return
    print(f"  Downloading BRFSS 2015 from CDC (~60MB)...")
    urllib.request.urlretrieve(BRFSS_URL, BRFSS_ZIP)
    print("  Extracting...")
    with zipfile.ZipFile(BRFSS_ZIP, 'r') as z:
        for name in z.namelist():
            if '.XPT' in name.upper():
                data = z.read(name)
                with open(BRFSS_XPT, 'wb') as f:
                    f.write(data)
                print(f"  Extracted: {BRFSS_XPT}")
                return


def load_and_prepare():
    try:
        import pyreadstat
    except ImportError:
        raise ImportError("Run: pip install pyreadstat")

    print("  Loading BRFSS XPT (1-2 minutes)...")
    df, _ = pyreadstat.read_xport(BRFSS_XPT)
    print(f"  Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

    # Keep only completed interviews
    df = df[df['DISPCODE'] == 1100].copy()
    print(f"  Completed interviews: {df.shape[0]:,} rows")

    # Build target from DIABETE3
    # 1 = diabetes, 4 = prediabetes → positive (1)
    # 3 = no diabetes               → negative (0)
    # 2 = gestational only, 7/9 = unknown → drop
    df = df[df[TARGET_COL].isin([1, 3, 4])].copy()
    y = ((df[TARGET_COL] == 1) | (df[TARGET_COL] == 4)).astype(int).values
    print(f"  After target cleaning: {len(y):,} rows")
    print(f"  Positive rate: {y.mean():.1%}")

    # Select feature columns
    available = [f for f in FEATURES if f in df.columns]
    missing   = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"  Warning — not found in dataset: {missing}")
    print(f"  Features available: {len(available)}")

    X = df[available].copy()

    # ALCDAY5: special frequency encoding before generic cleaning
    # 101-199 = days per week (subtract 100), 201-299 = days per month (subtract 200)
    # 888 = no drinks → 0; anything else → NaN
    if 'ALCDAY5' in X.columns:
        alc = X['ALCDAY5'].copy()
        weekly   = (alc >= 101) & (alc <= 199)
        monthly  = (alc >= 201) & (alc <= 299)
        none     = (alc == 888)
        alc_out  = alc.copy() * np.nan
        alc_out[weekly]  = ((alc[weekly] - 100) * 4.3).round(1)   # days/week → days/month
        alc_out[monthly] = alc[monthly] - 200
        alc_out[none]    = 0.0
        X['ALCDAY5'] = alc_out

    # Large missing-value codes used by CDC computed vars
    for col in X.columns:
        X[col] = X[col].replace([99900, 999900], np.nan)

    # Clean BRFSS coding
    # Don't know / refused → NaN, then fill with median
    for col in X.columns:
        X[col] = X[col].replace([7, 9, 77, 99, 7777, 9999], np.nan)
        X[col] = X[col].replace([8, 88], 0)
        X[col] = X[col].fillna(X[col].median())

    # Binary recoding: 1=Yes/2=No → 1=Yes/0=No (fixes sign of LR-poly interaction terms)
    binary_1yes_2no = [
        'HLTHPLN1', 'MEDCOST', 'BPMEDS', 'BLOODCHO', 'TOLDHI2',
        'CVDINFR4', 'CVDCRHD4', 'CVDSTRK3', 'CHCKIDNY', 'HAVARTH3',
        'ADDEPEV2', 'CHCCOPD1', 'ASTHMA3', 'EXERANY2', 'SMOKE100',
        'INTERNET', 'VETERAN3', 'QLACTLM2', 'USEEQUIP', 'CIMEMLOS',
        'CHCSCNCR', 'CHCOCNCR', 'DIFFWALK', 'DECIDE', 'DIFFALON',
        'FLUSHOT6', 'HIVTST6', 'LMTJOIN3', 'PNEUVAC3', 'DIFFDRES', 'BLIND', 'SEX',
    ]
    for col in binary_1yes_2no:
        if col in X.columns:
            X[col] = X[col].replace(2, 0)

    # BPHIGH4: 1=Yes, 2=pregnancy-only, 3=No, 4=borderline → binary 1=elevated/0=normal
    if 'BPHIGH4' in X.columns:
        X['BPHIGH4'] = X['BPHIGH4'].replace({2: 1, 3: 0, 4: 1})

    # CDC computed vars: 1=No, 2=Yes → 0=No, 1=Yes
    binary_cdc_1no_2yes = ['_RFHYPE5', '_RFCHOL', '_RFDRHV5', '_RFBING5', '_PA150R2', '_PASTRNG']
    for col in binary_cdc_1no_2yes:
        if col in X.columns:
            X[col] = X[col] - 1

    # Fix BMI — stored as BMI*100 in raw BRFSS
    if '_BMI5' in X.columns:
        X['_BMI5'] = X['_BMI5'] / 100.0

    # Fix fruit/veg/weight — stored as value*100 in CDC computed vars
    for col in ['_FRUTSUM', '_VEGESUM', 'WTKG3']:
        if col in X.columns:
            X[col] = X[col] / 100.0

    return X.values.astype(np.float32), y, available


def prepare_data():
    print("=" * 60)
    print("  prepare_v2.py — Expanded BRFSS Feature Dataset")
    print("=" * 60)

    required = ["X_train.npy", "y_train.npy", "X_val.npy",
                "y_val.npy",   "X_test.npy",  "y_test.npy",
                "feature_names.npy"]
    if all(os.path.exists(f"{OUTPUT_DIR}/{f}") for f in required):
        print("\n  data_v2/ already exists.")
        names = np.load(f"{OUTPUT_DIR}/feature_names.npy", allow_pickle=True)
        print(f"  {len(names)} features: {list(names)}\n")
        return

    download_brfss()
    X, y, feature_names = load_and_prepare()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )

    n = len(y)
    print(f"\n  Split summary:")
    print(f"    Train: {X_train.shape[0]:>6,} rows ({X_train.shape[0]/n:.0%})")
    print(f"    Val:   {X_val.shape[0]:>6,} rows ({X_val.shape[0]/n:.0%})")
    print(f"    Test:  {X_test.shape[0]:>6,} rows ({X_test.shape[0]/n:.0%})")
    print(f"    Features: {X_train.shape[1]}")
    print(f"\n  Class balance:")
    print(f"    Train: {y_train.mean():.3%} positive")
    print(f"    Val:   {y_val.mean():.3%} positive")
    print(f"    Test:  {y_test.mean():.3%} positive")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(f"{OUTPUT_DIR}/X_train.npy", X_train)
    np.save(f"{OUTPUT_DIR}/y_train.npy", y_train)
    np.save(f"{OUTPUT_DIR}/X_val.npy",   X_val)
    np.save(f"{OUTPUT_DIR}/y_val.npy",   y_val)
    np.save(f"{OUTPUT_DIR}/X_test.npy",  X_test)
    np.save(f"{OUTPUT_DIR}/y_test.npy",  y_test)
    np.save(f"{OUTPUT_DIR}/feature_names.npy", np.array(feature_names))

    print(f"\n  All splits saved to {OUTPUT_DIR}/")
    print(f"  TEST SET IS NOW LOCKED.")
    print(f"\n  Now run:")
    print(f'    python run_v2.py "baseline: LogisticRegression 37 expanded features"')
    print("=" * 60)


if __name__ == "__main__":
    prepare_data()