from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 🔥 Training Data (Expanded)
products = [
    # Food
    "milk", "bread", "rice", "apple", "vegetables", "fruits",

    # Electronics
    "laptop", "phone", "charger", "tv", "earphones", "earpods", "headphones", "speaker",

    # Fragile
    "glass", "ceramic", "mirror", "cup", "plate",

    # Liquid
    "oil", "juice", "water", "bottle", "liquid soap",

    # Heavy
    "machine", "metal parts", "equipment",

    # Medical
    "medicine", "tablets", "syrup", "medical kit",

    # Cosmetics
    "cream", "perfume", "cosmetics", "lotion"
]

categories = [
    # Food
    "food", "food", "food", "food", "food", "food",

    # Electronics
    "electronics", "electronics", "electronics", "electronics",
    "electronics", "electronics", "electronics", "electronics",

    # Fragile
    "fragile", "fragile", "fragile", "fragile", "fragile",

    # Liquid
    "liquid", "liquid", "liquid", "liquid", "liquid",

    # Heavy
    "heavy", "heavy", "heavy",

    # Medical
    "medical", "medical", "medical", "medical",

    # Cosmetics
    "cosmetics", "cosmetics", "cosmetics", "cosmetics"
]

# 🔥 Vectorizer (converts text → numbers)
vectorizer = TfidfVectorizer()

# Train on existing product data
X = vectorizer.fit_transform(products)


# 🚀 MAIN FUNCTION (used in app.py)
def predict_category(product_name):
    product_name = product_name.lower().strip()

    # Convert input to vector
    input_vec = vectorizer.transform([product_name])

    # Compute similarity
    similarity = cosine_similarity(input_vec, X)

    # Get best match index
    index = similarity.argmax()

    # Return corresponding category
    return categories[index]