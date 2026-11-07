from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "Connection timeout when sending HTTP request",
    "SSL certificate verification failed",
    "HTTP request fails due to connection timeout",
    "Update the project documentation",
]

query = "certificate error"

vectorizer = TfidfVectorizer()

document_vectors = vectorizer.fit_transform(documents)
query_vector = vectorizer.transform([query])

scores = cosine_similarity(
    query_vector,
    document_vectors
).flatten()

ranked_indices = scores.argsort()[::-1]

for rank, idx in enumerate(ranked_indices, start=1):
    print(f"{rank}. Score={scores[idx]:.4f} | {documents[idx]}")