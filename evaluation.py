def dataset_summary(courses, interactions):
    """Return basic dataset-size statistics for the dashboard."""
    return {
        "course_count": int(courses["course_id"].nunique()) if "course_id" in courses else len(courses),
        "student_count": int(interactions["student_id"].nunique()) if "student_id" in interactions else 0,
        "interaction_count": int(len(interactions)),
        "missing_course_values": int(courses.isna().sum().sum()),
        "missing_interaction_values": int(interactions.isna().sum().sum()),
    }
