# 🎯 InternPath — Personalized Learning Path Recommender

> *"Every intern learns differently. This system learns that too."*

---

## 📌 Project Overview

| Item | Detail |
|------|--------|
| Dataset | Coursera Course Reviews (1.45M reviews) |
| Sample Size | 100,000 reviews |
| Model | SVD Matrix Factorization |
| Library | Scikit-Surprise |
| RMSE | 0.6651 ✅ |
| MAE | 0.4317 ✅ |
| UI | Gradio |

---

## 🗂️ Project Structure
```
├── train.py           → Full training pipeline
├── app.py             → Gradio UI
├── requirements.txt   → Dependencies
└── README.md          → Documentation
```

---

## 🔄 Complete Project Journey

### Step 1 — Understanding The Problem
The goal was to build a system that recommends personalized
learning paths for interns based on their past learning patterns.
The key insight was that interns with similar learning histories
tend to benefit from similar courses — this is the foundation
of Collaborative Filtering.

### Step 2 — Dataset Selection
Used the Coursera Course Reviews dataset with 1.45 million
real user reviews. Merged two files — reviews and course
metadata — to create a rich dataset with user IDs, course
names, ratings and institution information.

### Step 3 — Data Preparation
Sampled 100,000 rows for Colab efficiency. Cleaned and
structured data into the User-Item-Rating Matrix format
required by the Surprise library.

### Step 4 — Model: SVD Matrix Factorization
SVD (Singular Value Decomposition) is a Matrix Factorization
technique that finds hidden patterns in user-course rating
data. It predicts what rating a user would give to courses
they haven't taken yet — and recommends the highest predicted ones.

### Step 5 — Model Results
| Metric | Score |
|--------|-------|
| RMSE | 0.6651 ✅ |
| MAE | 0.4317 ✅ |

> RMSE of 0.66 on a 1-5 scale means predictions
> are off by less than 1 star on average — excellent! 🎯

---

## 💡 Key Challenges & Learnings

### Challenge 1 — Understanding Collaborative Filtering
The hardest conceptual challenge was understanding that
Collaborative Filtering doesn't care WHAT a course is about —
it cares WHO else liked similar things. This shift in thinking
from content-based to behavior-based recommendations was
the most important learning of this project.

### Challenge 2 — Cold Start Problem
When a user doesn't exist in the system, the model has
nothing to base recommendations on. Solved this by showing
the most popular courses as fallback — a real world solution
used by Netflix and Spotify.

### Challenge 3 — Scale
1.45 million rows is too large for Colab. Sampling 100,000
rows while maintaining data quality and diversity required
careful consideration of random state and distribution.

---

## ✅ Best Practices Applied
- Used real world dataset with 1.45M reviews
- Applied SVD for efficient Matrix Factorization
- Built Cold Start handler for unknown users
- Separated training code from UI code
- Added institution filter for targeted recommendations
- Showed rating history for user transparency

---

## 🚀 How To Run
```bash
# Install dependencies
pip install -r requirements.txt

# Train model
python train.py

# Launch app
python app.py
```

---

## 🛠️ Tech Stack
- Python
- Pandas & NumPy
- Scikit-Surprise (SVD)
- Gradio
- Matplotlib
- KaggleHub

---

## 👤 Author
**MAhmadMakhdoom**
> Future Data Scientist | ML & AI Enthusiast
> Building models today. Solving real problems tomorrow.

---

## 📦 Dataset

This project uses the **Coursera Course Reviews** dataset.

| Detail | Info |
|--------|------|
| Source | Kaggle |
| Author | imuhammad |
| Rows | 1,454,711 reviews |
| Files | Coursera_reviews.csv + Coursera_courses.csv |

🔗 **Download Dataset →** https://www.kaggle.com/datasets/imuhammad/course-reviews-on-coursera

> Dataset is NOT included in this repo.
> Run `python train.py` to auto-download via kagglehub. ✅
