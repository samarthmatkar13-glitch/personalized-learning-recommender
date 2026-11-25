import pandas as pd

def prepare_data(courses, interactions):
    """Clean columns, remove unusable rows, and standardize basic fields."""
    courses = courses.copy()
    interactions = interactions.copy()
    courses.columns = courses.columns.str.strip().str.lower()
    interactions.columns = interactions.columns.str.strip().str.lower()

    required_courses = {"course_id", "course_title", "category"}
    required_interactions = {"student_id", "course_id"}
    if not required_courses.issubset(courses.columns):
        raise ValueError(f"courses.csv must contain: {sorted(required_courses)}")
    if not required_interactions.issubset(interactions.columns):
        raise ValueError(f"interactions.csv must contain: {sorted(required_interactions)}")

    courses = courses.dropna(subset=["course_id", "course_title", "category"])
    interactions = interactions.dropna(subset=["student_id", "course_id"])
    courses["course_id"] = courses["course_id"].astype(str).str.strip()
    interactions["course_id"] = interactions["course_id"].astype(str).str.strip()
    interactions["student_id"] = interactions["student_id"].astype(str).str.strip()
    courses["course_title"] = courses["course_title"].astype(str).str.strip()
    courses["category"] = courses["category"].astype(str).str.strip()

    if "rating" in interactions.columns:
        interactions["rating"] = pd.to_numeric(interactions["rating"], errors="coerce")
    if "completion" in interactions.columns:
        interactions["completion"] = pd.to_numeric(interactions["completion"], errors="coerce").fillna(0).clip(0, 1)
    interactions = interactions[interactions["course_id"].isin(courses["course_id"])]
    return courses.reset_index(drop=True), interactions.reset_index(drop=True)
