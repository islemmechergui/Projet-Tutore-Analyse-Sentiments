from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import time


def run_classification(
    df,
    text_col,
    label_col="label",
    test_size=0.2,
    models_choice=("Logistic Regression", "Naive Bayes"),
    save_model=True
):
    # ===============================
    # Préparation des données
    # ===============================
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

    # ===============================
    # Vectorisation TF-IDF
    # ===============================
    tfidf = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        stop_words="english"
    )

    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    results = {}
    best_model = None
    best_accuracy = 0
    best_model_name = ""

    # ===============================
    # Logistic Regression
    # ===============================
    if "Logistic Regression" in models_choice:
        start = time.time()

        lr = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            n_jobs=-1
        )

        lr.fit(X_train_vec, y_train)
        y_pred = lr.predict(X_test_vec)

        acc = accuracy_score(y_test, y_pred)

        results["Logistic Regression"] = {
            "accuracy": acc,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "train_percent": round((1 - test_size) * 100, 2),
            "test_percent": round(test_size * 100, 2),
            "training_time": round(time.time() - start, 2),
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

        if acc > best_accuracy:
            best_accuracy = acc
            best_model = lr
            best_model_name = "Logistic Regression"

    # ===============================
    # Naive Bayes
    # ===============================
    if "Naive Bayes" in models_choice:
        try:
            start = time.time()

            nb = MultinomialNB(alpha=1.0)
            nb.fit(X_train_vec, y_train)
            y_pred = nb.predict(X_test_vec)

            acc = accuracy_score(y_test, y_pred)

            results["Naive Bayes"] = {
                "accuracy": acc,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "train_percent": round((1 - test_size) * 100, 2),
                "test_percent": round(test_size * 100, 2),
                "training_time": round(time.time() - start, 2),
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

            if acc > best_accuracy:
                best_accuracy = acc
                best_model = nb
                best_model_name = "Naive Bayes"

        except Exception as e:
            results["Naive Bayes"] = {"error": str(e)}

    # ===============================
    # Sauvegarde du meilleur modèle
    # ===============================
    if save_model and best_model is not None:
        with open("sentiment_model.pkl", "wb") as f:
            pickle.dump(
                {
                    "model": best_model,
                    "vectorizer": tfidf,
                    "model_name": best_model_name,
                    "accuracy": best_accuracy
                },
                f
            )

    return results
