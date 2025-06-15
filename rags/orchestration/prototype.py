# --- Imports ---
import os
from src.qdrant_vector_db import QdrantVectorDB
from typing import Tuple, List

# --- Main Function ---


def load_data(data_folder_path: str) -> Tuple[List, List]:
    """Load book descriptions and metadata from text files."""

    book_description_files = [
        f for f in os.listdir(data_folder_path) if f.endswith(".txt")
    ]

    texts = []
    metadata = []

    for book_description_file in book_description_files:
        with open(os.path.join(data_folder_path, book_description_file), "r") as f:
            book_descriptions = f.readlines()

        titles = [
            book_description.split(":::")[1].strip()
            for book_description in book_descriptions
        ]
        authors = [
            book_description.split(":::")[2].strip()
            for book_description in book_descriptions
        ]
        book_description_text = [
            book_description.split(":::")[3].strip()
            for book_description in book_descriptions
        ]
        # create metadata
        payload = [
            {"title": title, "author": author} for title, author in zip(titles, authors)
        ]

        texts.extend(book_description_text)
        metadata.extend(payload)

    return texts, metadata


def main(collection_name: str, query: str, data_folder_path: str) -> None:
    """Main function to initialize the Qdrant vector database, load data, and perform a search."""

    # Initialize and connect to the Qdrant vector database
    db = QdrantVectorDB()
    db.connect()
    # Create or recreate the collection
    db.recreate_collection(collection_name)
    # Load book descriptions and metadata
    texts, metadata = load_data(data_folder_path)
    # Add texts to the collection
    db.add_texts(collection_name, texts, metadata)
    # Search for similar texts
    results = db.search(collection_name, query, limit=5)
    # Print search results
    for index, result in enumerate(results):
        print("*" * 10)
        print(
            f"Index: {index}, Description: {result['text']}\n Score: {result['score']}, Metadata: {result['metadata']}"
        )
    # Stop the database connection
    db.stop()


if __name__ == "__main__":
    COLLECTION_NAME = "Books"
    BOOK_DESCRIPTION_FOLDER = "data/books"
    QUERY = "A metarealism world of fantasy and adventure"

    main(COLLECTION_NAME, QUERY, BOOK_DESCRIPTION_FOLDER)
