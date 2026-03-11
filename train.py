
# ============================================================
# 🎯 InternPath — Learning Path Recommendation System
# Author   : MAhmadMakhdoom
# Dataset  : Coursera Course Reviews
# Model    : SVD Matrix Factorization (Surprise Library)
# ============================================================

# ── STEP 1 : IMPORT LIBRARIES ────────────────────
import pandas as pd
import numpy as np
import pickle
import os
import kagglehub
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate

print("✅ Libraries Loaded!")

# ── STEP 2 : DOWNLOAD DATASET ────────────────────
path  = kagglehub.dataset_download("imuhammad/course-reviews-on-coursera")
files = os.listdir(path)
print(f"✅ Dataset Downloaded → {path}")

# ── STEP 3 : LOAD & MERGE ────────────────────────
reviews  = pd.read_csv(f"{path}/Coursera_reviews.csv")
courses  = pd.read_csv(f"{path}/Coursera_courses.csv")
df       = reviews.merge(courses, on="course_id", how="left")
print(f"✅ Merged → {df.shape[0]:,} rows")

# ── STEP 4 : SAMPLE & CLEAN ──────────────────────
df_sample = df.sample(n=100000, random_state=42).copy()
df_clean  = df_sample[["reviewers", "course_id", "name", "rating"]].copy()
df_clean.columns = ["User_ID", "Course_ID", "Course_Name", "Rating"]
df_clean  = df_clean.dropna().reset_index(drop=True)
print(f"✅ Cleaned → {df_clean.shape[0]:,} rows")
print(f"Unique Users   : {df_clean['User_ID'].nunique():,}")
print(f"Unique Courses : {df_clean['Course_ID'].nunique():,}")

# ── STEP 5 : PREPARE SURPRISE DATA ───────────────
reader = Reader(rating_scale=(1, 5))
data   = Dataset.load_from_df(
    df_clean[["User_ID", "Course_ID", "Rating"]],
    reader
)
print("✅ Surprise Data Ready!")

# ── STEP 6 : TRAIN SVD MODEL ─────────────────────
model_svd = SVD(n_factors=50, random_state=42)

print("\n📊 Cross Validation Results:")
results = cross_validate(
    model_svd, data,
    measures=["RMSE", "MAE"],
    cv=3,
    verbose=True
)
print(f"\n✅ Avg RMSE : {results['test_rmse'].mean():.4f}")
print(f"✅ Avg MAE  : {results['test_mae'].mean():.4f}")

# ── STEP 7 : TRAIN ON FULL DATA ──────────────────
trainset_full = data.build_full_trainset()
model_svd.fit(trainset_full)
print("✅ Model Trained on Full Dataset!")

# ── STEP 8 : SAVE MODEL & DATA ───────────────────
with open("model_svd.pkl", "wb") as f:
    pickle.dump(model_svd, f)

df_clean.to_csv("df_clean.csv", index=False)

print("✅ Model & Data Saved!")
print("🎯 Run app.py to launch Gradio UI!")
