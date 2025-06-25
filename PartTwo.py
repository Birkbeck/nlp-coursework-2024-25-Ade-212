import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, classification_report


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
    y = df['party']  # Label to predict
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
    
    # Task 2c - Train and evaluate classifiers
    def train_and_evaluate(X_train, X_test, y_train, y_test):
        # Random Forest Classifier
        rf = RandomForestClassifier(n_estimators=300, random_state=26)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        macro_f1_rf = f1_score(y_test, y_pred_rf, average="macro")
        print(f"Random Forest macro-average F1 score: {macro_f1_rf:.4f}")
        print(classification_report(y_test, y_pred_rf))
        
        # Linear SVM Classifier
        svm = SVC(kernel='linear', random_state=26)
        svm.fit(X_train, y_train)
        y_pred_svm = svm.predict(X_test)
        macro_f1_svm = f1_score(y_test, y_pred_svm, average="macro")
        print(f"Linear SVM macro-average F1 score: {macro_f1_svm:.4f}")
        print(classification_report(y_test, y_pred_svm))
    train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # Task 2d: unigrams, bi-grams & tri-grams
    vect_ngram = TfidfVectorizer(
        stop_words='english',
        max_features=3000,
        ngram_range=(1, 3)
    )
    X_ngram = vect_ngram.fit_transform(df["speech"])
    y = df["party"]
    Xn_train, Xn_test, yn_train, yn_test = train_test_split(
        X_ngram, y,
        test_size=0.25,
        stratify=y,
        random_state=26
    )

    # Print macro-F1 and classification report again, now with n-grams range (1, 3)
    train_and_evaluate(Xn_train, Xn_test, yn_train, yn_test)


