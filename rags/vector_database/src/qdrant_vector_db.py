from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
import uuid
from typing import List, Dict, Optional


class QdrantVectorDB:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        """Initialize QdrantVectorDB with connection parameters."""
        self.host = host
        self.port = port
        self.client = None
        self.model = SentenceTransformer(model_name)
        self.embedding_size = 384  # Default for all-MiniLM-L6-v2

    def connect(self) -> None:
        """Establish connection to Qdrant."""
        try:
            self.client = QdrantClient(host=self.host, port=self.port)
        except Exception as e:
            print(f"Connection failed: {str(e)}")
            raise ConnectionError(f"Failed to connect to Qdrant: {str(e)}")

    def stop(self) -> None:
        """Close the connection to Qdrant."""
        if self.client:
            self.client.close()
            self.client = None

    def create_collection(self, collection_name: str) -> None:
        """Create a new collection."""
        existing = self.client.get_collections().collections
        collection_names = [collection.name for collection in existing]

        if collection_name in collection_names:
            print(f"✅ Collection '{collection_name}' already exists.")
        else:
            try:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_size, distance=Distance.COSINE
                    ),
                )
                print(f"🎉 Created new collection '{collection_name}'.")
            except Exception as e:
                print(f"Failed to create collection: {str(e)}")
                raise RuntimeError(f"Failed to create collection: {str(e)}")

    def recreate_collection(self, collection_name: str) -> None:
        """Create a new collection."""

        try:
            self.client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_size, distance=Distance.COSINE
                ),
            )
            print(f"🎉 Recreated new collection '{collection_name}'.")
        except Exception as e:
            print(f"Failed to create collection: {str(e)}")
            raise RuntimeError(f"Failed to create collection: {str(e)}")

    def delete_collection(self, collection_name: str) -> None:
        """Delete an existing collection."""
        try:
            self.client.delete_collection(collection_name=collection_name)
        except Exception as e:
            print(f"Failed to delete collection: {str(e)}")
            raise RuntimeError(f"Failed to delete collection: {str(e)}")

    def add_texts(
        self,
        collection_name: str,
        texts: List[str],
        metadata: Optional[List[Dict]] = None,
    ) -> None:
        """Add texts to the collection."""
        try:
            embeddings = self.model.encode(texts).tolist()

            points = []
            for i, (text, vector) in enumerate(zip(texts, embeddings)):
                payload = {"text": text}
                if metadata and i < len(metadata):
                    payload.update(metadata[i])

                points.append(
                    PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload)
                )

            self.client.upsert(collection_name=collection_name, points=points)
        except Exception as e:
            print(f"Failed to add texts: {str(e)}")
            raise RuntimeError(f"Failed to add texts: {str(e)}")

    def search(self, collection_name: str, query: str, limit: int = 3) -> List[Dict]:
        """Search for similar texts."""
        results: List[Dict] = []

        try:
            query_vector = self.model.encode(query).tolist()
            search_result = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
            )

            for hit in search_result:
                results.append(
                    {
                        "text": hit.payload["text"],
                        "score": hit.score,
                        "metadata": {
                            k: v for k, v in hit.payload.items() if k != "text"
                        },
                    }
                )
            return results
        except Exception as e:
            print(f"Search failed: {str(e)}")
            raise RuntimeError(f"Search operation failed: {str(e)}")


# Example usage:
if __name__ == "__main__":
    # Initialize and connect
    db = QdrantVectorDB()
    db.connect()

    # Create collection
    collection_name = "my_collection"
    db.create_collection(collection_name)

    # Add some texts
    texts = [
        "Qdrant is a vector database",
        "It supports similarity search",
        "Run locally or in cloud",
    ]
    db.add_texts(collection_name, texts)

    # Search
    results = db.search(collection_name, "What is Qdrant?")
    print("\nSearch Results:")
    for result in results:
        print(f"- {result['text']} (score: {result['score']:.3f})")

    # Clean up
    db.stop()
