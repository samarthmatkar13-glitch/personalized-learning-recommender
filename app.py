import streamlit as st
import pandas as pd
from data_loader import load_data
from preprocessing import prepare_data
from recommender import recommend_courses
from clustering import group_students
from evaluation import dataset_summary

st.set_page_config(page_title="Personalized Learning Recommender", page_icon="🎓", layout="wide")
st.title("🎓 Personalized Learning and Course Recommendation")
st.caption("A demonstration ML system that recommends courses using student activity and course information.")

try:
    courses, interactions = load_data()
    courses, interactions = prepare_data(courses, interactions)
except Exception as exc:
    st.error(f"Could not load or prepare the datasets: {exc}")
    st.stop()

with st.sidebar:
    st.header("Student profile")
    student_ids = sorted(interactions["student_id"].astype(str).unique())
    student_id = st.selectbox("Choose student", student_ids)
    top_n = st.slider("Number of recommendations", 3, 10, 5)

tab1, tab2, tab3 = st.tabs(["Recommendations", "Student groups", "Dataset overview"])

with tab1:
    st.subheader(f"Recommended courses for {student_id}")
    recs = recommend_courses(student_id, courses, interactions, top_n=top_n)
    if recs.empty:
        st.info("No recommendations available for this student yet.")
    else:
        display_cols = [c for c in ["course_id", "course_title", "category", "difficulty", "recommendation_score"] if c in recs]
        st.dataframe(recs[display_cols], use_container_width=True, hide_index=True)
        st.download_button("Download recommendations (CSV)", recs.to_csv(index=False), "recommendations.csv", "text/csv")

with tab2:
    st.subheader("K-Means student grouping")
    grouped = group_students(interactions)
    st.dataframe(grouped, use_container_width=True, hide_index=True)
    if "cluster" in grouped:
        st.bar_chart(grouped["cluster"].value_counts().sort_index())

with tab3:
    st.subheader("Dataset statistics")
    summary = dataset_summary(courses, interactions)
    c1, c2, c3 = st.columns(3)
    c1.metric("Courses", summary["course_count"])
    c2.metric("Students", summary["student_count"])
    c3.metric("Interactions", summary["interaction_count"])
    st.write("Course categories")
    st.bar_chart(courses["category"].value_counts())
    st.write("Average rating by course")
    if "rating" in interactions:
        st.dataframe(interactions.groupby("course_id")["rating"].mean().reset_index().rename(columns={"rating":"average_rating"}), use_container_width=True)

st.divider()
st.caption("Demo note: the included CSV data is synthetic and intended for learning/testing.")
