import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
import joblib

# 1. Load dataset
df = pd.read_csv("emails.csv")  # Your CSV with 'Category' and 'Message' columns

# 2. Rename columns to standard format
df = df.rename(columns={"Category": "label", "Message": "text"})

# 3. Clean labels
df["label"] = df["label"].str.strip().str.lower()

# 4. Balance dataset
spam_df = df[df['label'] == 'spam']
ham_df = df[df['label'] == 'ham']
spam_df_upsampled = resample(spam_df, replace=True, n_samples=len(ham_df), random_state=42)
df_balanced = pd.concat([ham_df, spam_df_upsampled])

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df_balanced["text"], df_balanced["label"], test_size=0.2, random_state=42
)

# 6. Create pipeline with TF-IDF + Naive Bayes
pipeline = Pipeline([
    ("vectorizer", TfidfVectorizer(lowercase=True, stop_words="english")),
    ("classifier", MultinomialNB())
])

# 7. Train model
pipeline.fit(X_train, y_train)

# 8. Evaluate model
accuracy = pipeline.score(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# 9. Save model
joblib.dump(pipeline, "spam_model.pkl")
print("Model saved as spam_model.pkl")


