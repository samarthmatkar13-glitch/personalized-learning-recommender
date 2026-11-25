import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def _content_scores(courses, interactions, student_id):
    """Score unseen courses by similarity to categories the student engaged with."""
    if courses.empty:
        return pd.Series(dtype=float)
    if "category" not in courses:
        return pd.Series(0.0, index=courses.index)
    student_rows = interactions[interactions["student_id"].astype(str) == str(student_id)]
    seen = set(student_rows["course_id"].astype(str))
    liked_ids = set()
    if "rating" in student_rows:
        liked_ids = set(student_rows.loc[student_rows["rating"] >= 4, "course_id"].astype(str))
    if not liked_ids:
        liked_ids = seen
    liked_categories = courses[courses["course_id"].astype(str).isin(liked_ids)]["category"].astype(str).tolist()
    if not liked_categories:
        return pd.Series(0.0, index=courses.index)
    corpus = courses["category"].fillna("").astype(str).tolist()
    vectorizer = TfidfVectorizer()
    try:
        matrix = vectorizer.fit_transform(corpus + [" ".join(liked_categories)])
        sims = cosine_similarity(matrix[:-1], matrix[-1]).ravel()
        return pd.Series(sims, index=courses.index)
    except ValueError:
        return pd.Series(0.0, index=courses.index)

def recommend_courses(student_id, courses, interactions, top_n=5):
    """Hybrid ranking: content similarity plus collaborative rating signal."""
    data = courses.copy()
    student_rows = interactions[interactions["student_id"].astype(str) == str(student_id)]
    seen = set(student_rows["course_id"].astype(str))
    data["content_score"] = _content_scores(data, interactions, student_id)
    if "rating" in interactions.columns:
        global_avg = interactions.groupby("course_id")["rating"].mean()
        data["collaborative_score"] = data["course_id"].astype(str).map(global_avg).fillna(interactions["rating"].mean() if interactions["rating"].notna().any() else 0)
    else:
        data["collaborative_score"] = 0.0
    max_collab = data["collaborative_score"].max()
    if pd.notna(max_collab) and max_collab > 0:
        data["collaborative_score"] = data["collaborative_score"] / max_collab
    data["recommendation_score"] = 0.65 * data["content_score"] + 0.35 * data["collaborative_score"]
    data = data[~data["course_id"].astype(str).isin(seen)]
    return data.sort_values("recommendation_score", ascending=False).head(top_n).reset_index(drop=True)
