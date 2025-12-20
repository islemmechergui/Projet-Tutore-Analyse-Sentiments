import pandas as pd
import re
import nltk
import unicodedata
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns


nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemm = WordNetLemmatizer()


def preprocess_text(text):
    if pd.isna(text):
        return ""

    text = text.lower()
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('utf-8')

    text = re.sub(r"http\S+|www\S+", "", text)  # URLs
    text = re.sub(r"@\w+", "", text)            # mentions
    text = re.sub(r"#\w+", "", text)            # hashtags
    text = re.sub(r"[:;=8][\-o\*']?[\)\]\(\[dDpP/:\}\{@\|\\]", " ", text)  # smileys
    text = re.sub(r"[^a-z\s]", " ", text)      # lettres uniquement
    text = re.sub(r"\s+", " ", text).strip()   # espaces multiples

    tokens = text.split()
    tokens = [lemm.lemmatize(w) for w in tokens if w not in stop_words]

    return " ".join(tokens)


df = pd.read_csv("data/Amazon_Reviews.csv")
print("Colonnes disponibles :", df.columns.tolist())


review_text_col = None
review_title_col = None

for col in df.columns:
    name = col.strip().lower()
    if "review" in name and "text" in name:
        review_text_col = col
    if "title" in name or "summary" in name:
        review_title_col = col

if review_text_col is None:
    raise ValueError(f"❌ Aucune colonne de review trouvée. Colonnes : {df.columns.tolist()}")

print("✔ Colonne review texte :", review_text_col)
print("✔ Colonne review titre :", review_title_col)


def extract_rating(value):
    if pd.isna(value):
        return None
    match = re.search(r'Rated\s+(\d+)', str(value))
    return int(match.group(1)) if match else None

df['numeric_rating'] = df['Rating'].apply(extract_rating)
df = df.dropna(subset=['numeric_rating'])


df[review_text_col] = df[review_text_col].apply(preprocess_text)
if review_title_col:
    df[review_title_col] = df[review_title_col].apply(preprocess_text)

df = df[df[review_text_col].str.split().str.len() > 2]

print("\nTEST APRÈS NETTOYAGE :")
cols_show = [review_text_col]
if review_title_col:
    cols_show.append(review_title_col)
print(df[cols_show].head())


def rating_to_sentiment(r):
    r = int(r)
    if r <= 2:
        return -1
    elif r == 3:
        return 0
    else:
        return 1

df['label'] = df['numeric_rating'].apply(rating_to_sentiment)


print("\nNombre total de reviews :", len(df))
print("Répartition des sentiments :\n", df['label'].value_counts())


sns.set(style="whitegrid")

plt.figure(figsize=(6,4))
sns.countplot(x='label', data=df, palette='coolwarm')
plt.title("Répartition des sentiments")
plt.xlabel("Sentiment (-1, 0, 1)")
plt.ylabel("Nombre de reviews")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df['numeric_rating'], bins=5)
plt.title("Distribution des ratings")
plt.xlabel("Rating")
plt.ylabel("Nombre de reviews")
plt.show()


df.to_csv("data/amazon_reviews_cleaned.csv", index=False)
print("\n✅ Dataset FINAL nettoyé et sauvegardé : data/amazon_reviews_cleaned.csv")
