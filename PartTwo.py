import pandas as pd
from pathlib import Path

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

if __name__ == "__main__":
    df = load_and_filter_data()
    # Task asks only to print the shape
    print(df.shape)