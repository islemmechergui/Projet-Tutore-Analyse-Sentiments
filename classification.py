from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def run_classification(
    df,
    text_col,
    label_col="label",
    test_size=0.2,
    models_choice=("Logistic Regression", "Naive Bayes")
):
    df = df[[text_col, label_col]].dropna()
    df[text_col] = df[text_col].astype(str)

    if df[label_col].nunique() < 2:
        raise ValueError("Le dataset doit contenir au moins 2 classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col],
        df[label_col],
        test_size=test_size,
        random_state=42,
        stratify=df[label_col]
    )

    tfidf = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        norm=None,
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False,
        stop_words="english"
    )

    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    results = {}

    if "Logistic Regression" in models_choice:
        lr = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            n_jobs=-1
        )

        lr.fit(X_train_vec, y_train)
        y_pred = lr.predict(X_test_vec)

        results["Logistic Regression"] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "report": classification_report(
                y_test,
                y_pred,
                labels=[-1, 0, 1],
                target_names=["Négatif", "Neutre", "Positif"],
                output_dict=True,
                zero_division=0
            ),
            "confusion_matrix": confusion_matrix(
                y_test,
                y_pred,
                labels=[-1, 0, 1]
            )
        }

    if "Naive Bayes" in models_choice:
        try:
            nb = MultinomialNB(alpha=1.0)
            nb.fit(X_train_vec, y_train)
            y_pred = nb.predict(X_test_vec)

            results["Naive Bayes"] = {
                "accuracy": accuracy_score(y_test, y_pred),
                "report": classification_report(
                    y_test,
                    y_pred,
                    labels=[-1, 0, 1],
                    target_names=["Négatif", "Neutre", "Positif"],
                    output_dict=True,
                    zero_division=0
                ),
                "confusion_matrix": confusion_matrix(
                    y_test,
                    y_pred,
                    labels=[-1, 0, 1]
                )
            }

        except Exception as e:
            results["Naive Bayes"] = {"error": str(e)}

    return results