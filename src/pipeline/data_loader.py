from sklearn.datasets import fetch_openml


def load_adult_dataset():
    """
    Load the Adult Income dataset from OpenML.

    Returns:
        X: pandas DataFrame of features
        y: pandas Series of binary labels
    """
    adult = fetch_openml(name="adult", version=2, as_frame=True)

    X = adult.data.copy()
    y = adult.target.copy()

    # Convert target to binary
    y = y.map({"<=50K": 0, ">50K": 1})

    return X, y