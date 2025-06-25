import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Task 2a(i-iv) implementation
def load_and_filter_data(path=Path.cwd() / "p2-texts" / "hansard40000.csv"):
    # 1. Load
    df = pd.read_csv(path)
    # 2. Rename
    df["party"] = df["party"].replace("Labour (Co-op)", "Labour")
    # 3. Filter for top-4 parties
    top4 = df["party"].value_counts().nlargest(4).index
    df = df[df["party"].isin(top4)]
    # also drop any rows where party == 'Speaker'
    df = df[df["party"] != "Speaker"]
    # 4. Keep only real speeches
    df = df[df["speech_class"] == "Speech"]
    # 5. Minimum length
    df = df[df["speech"].str.len() >= 1000]
    return df

# Task 2b implementation
def vectorize_and_split(df, random_state=26, test_size=0.25):
    # 1) fit & transform
    vects = TfidfVectorizer(stop_words='english', max_features=3000)
    X = vects.fit_transform(df['speech'])
    y = df['party']
    # 2) stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    return X_train, X_test, y_train, y_test, vects


if __name__ == "__main__":
    df = load_and_filter_data()
    # Print the shape - Task 2a
    print(df.shape)
    # Task 2b
    X_train, X_test, y_train, y_test, vect = vectorize_and_split(df)