import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load the dataset
data = pd.read_csv("data/items.csv")

print("=" * 50)
print("       AI RECOMMENDATION SYSTEM")
print("=" * 50)

print("\nAvailable Items:")
print(data["item"].to_string(index=False))


# Take user preferences
user_input = input(
    "\nEnter your interests (example: python machine learning): "
).lower()


# Combine category and tags for matching
data["features"] = (
    data["category"].fillna("")
    + " "
    + data["tags"].fillna("")
)


# Convert item features into numerical vectors
vectorizer = TfidfVectorizer()

item_vectors = vectorizer.fit_transform(data["features"])

user_vector = vectorizer.transform([user_input])


# Calculate similarity between user preferences and items
similarity_scores = cosine_similarity(
    user_vector, item_vectors
).flatten()


# Add similarity score to the dataset
data["similarity"] = similarity_scores


# Sort items by similarity
recommendations = data.sort_values(
    by="similarity",
    ascending=False
)


# Display top recommendations
print("\n" + "=" * 50)
print("          RECOMMENDED ITEMS")
print("=" * 50)

recommended = recommendations.head(5)

for index, row in recommended.iterrows():
    score = row["similarity"] * 100

    print(
        f"{row['item']} "
        f"- Similarity: {score:.2f}%"
    )


print("\nRecommendation completed successfully!")