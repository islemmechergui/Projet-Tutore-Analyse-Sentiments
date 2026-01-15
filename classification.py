from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import time

def run_classification(
    df,
    text_col="Review Text",
    label_col="label",
    test_size=0.2,
    models_choice=("Logistic Regression", "Naive Bayes"),
    save_model=True
):
    # Séparer textes et labels
    X = df[text_col].astype(str)
    y = df[label_col].astype(int)

    # Vérifier au moins 2 classes
    if y.nunique() < 2:
        raise ValueError("Le dataset doit contenir au moins 2 classes.")

    # Split train/test stratifié
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    # Pourcentage train/test
    train_percent = round(len(X_train) / (len(X_train) + len(X_test)) * 100, 2)
    test_percent = round(len(X_test) / (len(X_train) + len(X_test)) * 100, 2)

    # Vectorisation TF-IDF
    tfidf = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1,2),
        min_df=3,
        max_df=0.9,
        stop_words="english"
    )
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    # Labels uniques pour classification report
    unique_labels = sorted(y.unique())
    target_names = [str(l) for l in unique_labels]

    results = {}
    best_model = None
    best_accuracy = 0
    best_model_name = ""

    # Logistic Regression
    if "Logistic Regression" in models_choice:
        start = time.time()
        lr = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)
        lr.fit(X_train_vec, y_train)
        y_pred = lr.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)

        results["Logistic Regression"] = {
            "accuracy": acc,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "train_percent": train_percent,
            "test_percent": test_percent,
            "training_time": round(time.time() - start, 2),
            "report": classification_report(y_test, y_pred,
                                            labels=unique_labels,
                                            target_names=target_names,
                                            output_dict=True,
                                            zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred, labels=unique_labels),
            "model_object": lr,
            "vectorizer": tfidf
        }

        if acc > best_accuracy:
            best_accuracy = acc
            best_model = lr
            best_model_name = "Logistic Regression"

    # Naive Bayes
    if "Naive Bayes" in models_choice:
        start = time.time()
        nb = MultinomialNB(alpha=1.0)
        nb.fit(X_train_vec, y_train)
        y_pred = nb.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)

        results["Naive Bayes"] = {
            "accuracy": acc,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "train_percent": train_percent,
            "test_percent": test_percent,
            "training_time": round(time.time() - start, 2),
            "report": classification_report(y_test, y_pred,
                                            labels=unique_labels,
                                            target_names=target_names,
                                            output_dict=True,
                                            zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred, labels=unique_labels),
            "model_object": nb,
            "vectorizer": tfidf
        }

        if acc > best_accuracy:
            best_accuracy = acc
            best_model = nb
            best_model_name = "Naive Bayes"

    # Sauvegarde du meilleur modèle
    if save_model and best_model is not None:
        filename = f"sentiment_model_{best_model_name.replace(' ', '_')}.pkl"
        with open(filename, "wb") as f:
            pickle.dump({
                "model": best_model,
                "vectorizer": tfidf,
                "model_name": best_model_name,
                "accuracy": best_accuracy
            }, f)

    print(f"Meilleur modèle : {best_model_name} avec accuracy = {best_accuracy:.4f}")
    return results
