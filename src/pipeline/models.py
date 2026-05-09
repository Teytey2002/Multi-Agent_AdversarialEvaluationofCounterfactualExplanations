from sklearn.linear_model import LogisticRegression


def get_model(random_state=42):
    """
    Return the Logistic Regression classifier used throughout the pipeline.

    Args:
        random_state: seed for reproducibility.

    Returns:
        Tuple (name, estimator).
    """
    return "logistic_regression", LogisticRegression(
        max_iter=1000,
        random_state=random_state
    )