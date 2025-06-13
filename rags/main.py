# --- Imports ---
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# --- 0. Initialize OpenAI Client ---
# The API key is read automatically from the OPENAI_API_KEY environment variable
try:
    client = OpenAI()
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    print("Please ensure your OPENAI_API_KEY environment variable is set correctly.")
    exit()

# --- 1. Knowledge Base ---
# Our simple knowledge base: a list of text documents
documents = [
    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
    "Paris is the capital and most populous city of France.",
    "The Louvre is the world's largest art museum and a historic monument in Paris, France.",
    "MacBooks are a line of Mac laptops designed, manufactured by Apple Inc.",
    "Python is a high-level, general-purpose programming language.",
]
document_embeddings = []  # To store embeddings of our documents


# --- 2. Embedding Function ---
def get_embedding(text: str, model="text-embedding-3-small"):
    """Generates an embedding for the given text using OpenAI API."""
    try:
        text = text.replace("\n", " ")  # API recommendation
        response = client.embeddings.create(input=[text], model=model)
        return np.array(response.data[0].embedding)
    except Exception as e:
        print(f"Error getting embedding for '{text[:30]}...': {e}")
        return None


# --- 3. Indexing our Knowledge Base ---
print("Indexing knowledge base...")
for doc in documents:
    embedding = get_embedding(doc)
    if embedding is not None:
        document_embeddings.append(embedding)
    else:
        print(f"Skipping document due to embedding error: {doc[:50]}...")

# Convert to NumPy array for easier calculations
document_embeddings = np.array(document_embeddings)
# Filter out original documents for which embedding failed
valid_documents = [
    doc for i, doc in enumerate(documents) if i < len(document_embeddings)
]


# --- 4. Retrieval Function ---
def retrieve_relevant_document(query_embedding, top_k=1):
    """Retrieves the top_k most relevant document(s) based on cosine similarity."""
    if document_embeddings.shape[0] == 0:
        print("No documents were successfully embedded. Cannot retrieve.")
        return []

    # Calculate cosine similarities
    similarities = cosine_similarity(
        query_embedding.reshape(1, -1), document_embeddings
    )

    # Get the indices of the top_k most similar documents
    # argsort returns indices that would sort the array. We take the last 'top_k' for highest similarity.
    top_k_indices = np.argsort(similarities[0])[-top_k:][
        ::-1
    ]  # [::-1] to get descending order

    retrieved_docs = [valid_documents[i] for i in top_k_indices]
    return retrieved_docs


# --- 5. Augmentation & Generation Function ---
def generate_answer_with_rag(
    query, retrieved_docs, model="gpt-4o-mini"
):  # or "gpt-3.5-turbo"
    """Generates an answer using GPT, augmented with retrieved documents."""
    if not retrieved_docs:
        context_str = "No relevant context found."
    else:
        context_str = "\n\n".join(retrieved_docs)

    prompt = f"""
    Based on the following context, please answer the question.
    If the context does not provide enough information, say so.

    Context:
    {context_str}

    Question: {query}

    Answer:
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on provided context.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,  # Lower temperature for more factual answers
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "Sorry, I encountered an error while trying to generate an answer."


# --- Main RAG Workflow ---
def ask_rag(user_query):
    print(f"\nUser Query: {user_query}")

    # 1. Embed the user query
    query_embedding = get_embedding(user_query)
    if query_embedding is None:
        print("Could not generate embedding for the query.")
        return "Sorry, I couldn't process your query."

    # 2. Retrieve relevant document(s)
    relevant_docs = retrieve_relevant_document(
        query_embedding, top_k=1
    )  # Get the top 1 relevant doc
    if relevant_docs:
        print(f'Retrieved context: "{relevant_docs[0][:100]}..."')
    else:
        print("No relevant context found in the knowledge base.")

    # 3. Generate answer using LLM with retrieved context
    answer = generate_answer_with_rag(user_query, relevant_docs)
    print(f"RAG Answer: {answer}")
    return answer


# --- Example Usage ---
if __name__ == "__main__":
    if not document_embeddings.any():
        print("Knowledge base could not be indexed. Exiting.")
    else:
        ask_rag("Tell me about the Eiffel Tower.")
        ask_rag("What is the capital of France?")
        ask_rag("What programming language is known for being high-level?")
        ask_rag("What kind of computer is a MacBook?")
        ask_rag(
            "What is the weather like today?"
        )  # Example where context might not be helpful
