import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def group_students(interactions, n_clusters=3):
    """Group students by activity volume, mean rating, and completion rate."""
    data = interactions.copy()
    if data.empty:
        return pd.DataFrame(columns=["student_id", "activity_count", "average_rating", "completion_rate", "cluster"])
    aggregations = {"course_id": "count"}
    if "rating" in data.columns:
        aggregations["rating"] = "mean"
    if "completion" in data.columns:
        aggregations["completion"] = "mean"
    features = data.groupby("student_id").agg(aggregations).reset_index()
    features = features.rename(columns={"course_id":"activity_count", "rating":"average_rating", "completion":"completion_rate"})
    for col in ["average_rating", "completion_rate"]:
        if col not in features:
            features[col] = 0.0
    cols = ["activity_count", "average_rating", "completion_rate"]
    X = features[cols].fillna(0)
    if len(features) < 2:
        features["cluster"] = 0
        return features
    k = min(max(1, n_clusters), len(features))
    scaled = StandardScaler().fit_transform(X)
    features["cluster"] = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(scaled)
    return features.sort_values("cluster").reset_index(drop=True)
