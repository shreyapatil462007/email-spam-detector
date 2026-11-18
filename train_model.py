import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------- File locations ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "spam.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "spam_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.pkl")

# ---------- Step 1: Load the dataset ----------
print("Step 1: Loading the dataset...")

if not os.path.exists(DATA_PATH):
    print("ERROR: Could not find data/spam.csv")
    print("Please check that spam.csv is inside the data folder.")
    raise SystemExit(1)

df = pd.read_csv(
    DATA_PATH,
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="latin-1",
    quoting=3,
)
df = df.dropna()
print("Total messages loaded:", len(df))

# ---------- Step 2: Convert labels to numbers ----------
# ham (NOT SPAM) = 0, spam = 1
df["label_num"] = df["label"].str.strip().map({"ham": 0, "spam": 1})
df = df.dropna(subset=["label_num"])
df["label_num"] = df["label_num"].astype(int)

X = df["message"]
y = df["label_num"]

# ---------- Step 3: Split into training and testing data ----------
print("Step 2: Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Training messages:", len(X_train))
print("Testing messages:", len(X_test))

# ---------- Step 4: TF-IDF text processing ----------
print("Step 3: Applying TF-IDF...")
vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ---------- Step 5: Train the Multinomial Naive Bayes model ----------
print("Step 4: Training the Multinomial Naive Bayes model...")
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# ---------- Step 6: Test the model ----------
print("Step 5: Testing the model...")
y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["NOT SPAM", "SPAM"])

print("")
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("")
print("Confusion Matrix:")
print(cm)
print("")
print("Classification Report:")
print(report)

# ---------- Step 7: Save everything ----------
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(model, MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)

metrics = {
    "accuracy": float(accuracy),
    "confusion_matrix": cm.tolist(),
    "report": report,
}
joblib.dump(metrics, METRICS_PATH)

print("Done! The model was saved in the model folder.")