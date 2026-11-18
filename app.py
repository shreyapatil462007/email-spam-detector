import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ---------- File locations ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "spam_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.pkl")

# ---------- Page settings ----------
st.set_page_config(page_title="Email Spam Detector", page_icon="📧", layout="centered")

# ---------- Check that the model files exist ----------
files_exist = (
    os.path.exists(MODEL_PATH)
    and os.path.exists(VECTORIZER_PATH)
    and os.path.exists(METRICS_PATH)
)

if not files_exist:
    st.error("Model files not found. Please run this command first: python train_model.py")
    st.stop()


# ---------- Load the saved model ----------
@st.cache_resource
def load_files():
    loaded_model = joblib.load(MODEL_PATH)
    loaded_vectorizer = joblib.load(VECTORIZER_PATH)
    loaded_metrics = joblib.load(METRICS_PATH)
    return loaded_model, loaded_vectorizer, loaded_metrics


model, vectorizer, metrics = load_files()

# ---------- Sidebar ----------
with st.sidebar:
    st.header("About this project")
    st.write("This app checks if an email or text message is SPAM or NOT SPAM.")
    st.write("**Dataset:** SMS Spam Collection")
    st.write("**Text processing:** TF-IDF")
    st.write("**Algorithm:** Multinomial Naive Bayes")
    st.write("**Library:** Scikit-learn")

# ---------- Page title ----------
st.title("📧 Email Spam Detector")
st.caption("A Machine Learning mini-project using Python, Scikit-learn and Streamlit")

# ---------- Tabs ----------
tab1, tab2 = st.tabs(["🔍 Check Message", "📊 Model Performance"])

# ---------- Tab 1: Check a message ----------
with tab1:
    st.subheader("Enter a message")

    user_message = st.text_area(
        "Type or paste an email or SMS message below:",
        height=150,
        placeholder="Example: Congratulations! You won a free prize. Click here to claim now.",
    )

    check_clicked = st.button("Check Message", type="primary")

    if check_clicked:
        if user_message.strip() == "":
            st.warning("Please enter a message first.")
        else:
            message_tfidf = vectorizer.transform([user_message])
            prediction = int(model.predict(message_tfidf)[0])
            probabilities = model.predict_proba(message_tfidf)[0]
            confidence = float(probabilities[prediction])

            if prediction == 1:
                st.error("### 🚨 SPAM")
            else:
                st.success("### ✅ NOT SPAM")

            st.write("Confidence: {:.2f}%".format(confidence * 100))
            st.progress(confidence)

    with st.expander("Need example messages to try?"):
        st.write("Spam example:")
        st.code("WINNER!! You have won a free cash prize. Call now to claim your reward. Limited time offer!")
        st.write("Not spam example:")
        st.code("Hey, are we still meeting for lunch tomorrow at 1 PM?")

# ---------- Tab 2: Model performance ----------
with tab2:
    st.subheader("Model Performance")

    accuracy = metrics["accuracy"]
    cm = np.array(metrics["confusion_matrix"])

    st.metric(label="Model Accuracy", value="{:.2f}%".format(accuracy * 100))

    st.write("**Confusion Matrix**")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")

    labels = ["NOT SPAM", "SPAM"]
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    threshold = cm.max() / 2
    for i in range(2):
        for j in range(2):
            text_color = "white" if cm[i, j] > threshold else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=text_color, fontsize=16)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.write("**Classification Report**")
    st.code(metrics["report"])