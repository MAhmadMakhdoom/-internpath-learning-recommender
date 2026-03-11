
# ============================================================
# 🎯 InternPath — Gradio UI
# ============================================================
import gradio as gr
import pandas as pd
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import kagglehub
import os

# ── Load Model & Data ────────────────────────────
with open("model_svd.pkl", "rb") as f:
    model_svd = pickle.load(f)

df_clean = pd.read_csv("df_clean.csv")
path     = kagglehub.dataset_download("imuhammad/course-reviews-on-coursera")

# ── Recommendation Function ──────────────────────
def recommend_courses(user_id, n=5):
    all_courses   = df_clean["Course_ID"].unique()
    taken_courses = df_clean[df_clean["User_ID"] == user_id]["Course_ID"].values
    unseen        = [c for c in all_courses if c not in taken_courses]
    predictions   = [(c, model_svd.predict(user_id, c).est) for c in unseen]
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_n = predictions[:n]
    results = []
    for course_id, score in top_n:
        name = df_clean[df_clean["Course_ID"] == course_id]["Course_Name"].values
        results.append({
            "Course"          : name[0] if len(name) > 0 else course_id,
            "Course_ID"       : course_id,
            "Predicted_Rating": round(score, 2)
        })
    return pd.DataFrame(results)

# ── Popular Courses ──────────────────────────────
popular_courses = (
    df_clean.groupby(["Course_Name", "Course_ID"])["Rating"]
    .agg(["mean", "count"]).reset_index()
)
popular_courses.columns = ["Course Name", "Course ID", "Avg Rating", "Total Reviews"]
popular_courses = popular_courses[popular_courses["Total Reviews"] >= 10]
popular_courses = popular_courses.sort_values("Avg Rating", ascending=False).head(10)
popular_courses["Avg Rating"] = popular_courses["Avg Rating"].round(2)

# ── Institution List ─────────────────────────────
courses_df   = pd.read_csv(f"{path}/Coursera_courses.csv")
inst_list    = courses_df["institution"].dropna().unique().tolist()
institutions = ["All"] + sorted(inst_list)

# ── Sample Users ─────────────────────────────────
sample_users = df_clean["User_ID"].value_counts().head(5).index.tolist()
sample_text  = "📋 Sample Users:\n" + "\n".join(sample_users)

def get_recommendations(user_id, institution_filter):
    user_id = user_id.strip()

    if user_id not in df_clean["User_ID"].values:
        msg = f"⚠️ User not found! Showing Popular Courses."
        pop = popular_courses[["Course Name", "Avg Rating", "Total Reviews"]].reset_index(drop=True)
        return msg, None, pop, pop

    taken = df_clean[df_clean["User_ID"] == user_id][
        ["Course_Name", "Rating"]
    ].drop_duplicates().reset_index(drop=True)
    taken.columns = ["Course Name", "Your Rating"]
    taken = taken.sort_values("Your Rating", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, max(3, len(taken) * 0.5)))
    ax.barh(taken["Course Name"].str[:35], taken["Your Rating"], color="mediumpurple")
    ax.set_xlim(0, 5)
    ax.set_xlabel("Rating")
    ax.set_title(f"Rating History — {user_id}")
    plt.tight_layout()

    reco        = recommend_courses(user_id, n=20)
    courses_df2 = pd.read_csv(f"{path}/Coursera_courses.csv")

    if institution_filter != "All":
        filtered = courses_df2[
            courses_df2["institution"] == institution_filter
        ]["course_id"].values
        reco = reco[reco["Course_ID"].isin(filtered)]

    reco = reco.head(5).reset_index(drop=True)
    reco = reco.merge(
        courses_df2[["course_id", "course_url", "institution"]],
        left_on="Course_ID", right_on="course_id", how="left"
    )
    reco = reco[["Course", "Predicted_Rating", "institution", "course_url"]].reset_index(drop=True)
    reco.columns = ["Recommended Course", "Predicted Rating", "Institution", "Course URL"]

    msg = (
        f"✅ User: {user_id}\n"
        f"📚 Courses Taken  : {len(taken)}\n"
        f"⭐ Avg Rating     : {taken['Your Rating'].mean():.2f}\n"
        f"🎯 Top Recommendations Ready!"
    )

    return msg, fig, taken, reco

theme = gr.themes.Soft(
    primary_hue="violet",
    secondary_hue="blue",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter")
)

with gr.Blocks(theme=theme, title="InternPath Recommender") as app:
    gr.Markdown("# 🎯 InternPath — Personalized Learning Recommender")
    gr.Markdown("Enter your User ID to get a personalized learning path!")

    with gr.Row():
        with gr.Column(scale=2):
            user_input = gr.Textbox(label="Enter User ID", placeholder="e.g. By James F", lines=1)
            gr.Markdown(sample_text)
        with gr.Column(scale=1):
            inst_filter = gr.Dropdown(choices=institutions, value="All", label="Filter by Institution 🏫")

    recommend_btn = gr.Button("🚀 Get My Learning Path", variant="primary")
    status_out    = gr.Text(label="📊 User Summary")
    chart_out     = gr.Plot(label="📈 Your Rating History")

    with gr.Row():
        taken_out = gr.Dataframe(label="📚 Courses You Already Took", wrap=True, interactive=False)
        reco_out  = gr.Dataframe(label="🎯 Your Top 5 Recommended Courses", wrap=True, interactive=False)

    recommend_btn.click(
        fn=get_recommendations,
        inputs=[user_input, inst_filter],
        outputs=[status_out, chart_out, taken_out, reco_out]
    )

app.launch()
