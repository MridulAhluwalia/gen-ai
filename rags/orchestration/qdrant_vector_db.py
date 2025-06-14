from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
import uuid

# 1. Connect to Qdrant
client = QdrantClient(host="localhost", port=6333)

# 2. Define Collection
collection_name = "my_collection"
embedding_size = 384  # Depends on the model used

client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=embedding_size, distance=Distance.COSINE),
)

# 3. Generate Embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [
    "Qdrant is a vector database",
    "It supports similarity search",
    "Run locally or in cloud",
]
embeddings = model.encode(texts).tolist()

# 4. Upload to Qdrant
points = [
    PointStruct(id=str(uuid.uuid4()), vector=vector, payload={"text": text})
    for text, vector in zip(texts, embeddings)
]
client.upsert(collection_name=collection_name, points=points)

print("Embeddings saved to Qdrant.")

# 5. Search (Load and Query)
query = "What is Qdrant?"
query_vector = model.encode(query).tolist()

search_result = client.search(
    collection_name=collection_name,
    query_vector=query_vector,
    limit=3,
)

print("Search Results:")
for hit in search_result:
    print(f"- {hit.payload['text']} (score: {hit.score:.3f})")
