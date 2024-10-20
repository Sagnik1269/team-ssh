# semantic_search.py

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModel
import torch

def load_embeddings(language, dataset_type):
    """
    Load embeddings for a specific language and dataset type.
    """
    embeddings_path = f'embeddings/{language}_{dataset_type}_embeddings.npy'
    return np.load(embeddings_path)

def load_dataset(language, dataset_type):
    """
    Load the original dataset for a specific language and dataset type.
    """
    dataset_path = f'processed_datasets/{language}/{dataset_type}'
    return load_from_disk(dataset_path)

def get_query_embedding(query, model, tokenizer):
    """
    Convert a text query into an embedding using the model and tokenizer.
    """
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True).to('cuda')
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

def semantic_search(query_embedding, embeddings, top_k=5):
    """
    Perform semantic search using cosine similarity.
    """
    similarities = cosine_similarity(query_embedding, embeddings)
    top_k_indices = similarities.argsort()[0][-top_k:][::-1]
    return top_k_indices, similarities[0][top_k_indices]

def main():
    # Load model and tokenizer
    model_name = "microsoft/codebert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to('cuda')

    # Example: Load embeddings and dataset for a specific language and dataset type
    language = 'javascript'
    dataset_type = 'train'
    embeddings = load_embeddings(language, dataset_type)
    dataset = load_dataset(language, dataset_type)

    while True:
        query = input("Enter your search query (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break

        # Generate query embedding
        query_embedding = get_query_embedding(query, model, tokenizer)

        # Perform semantic search
        top_k_indices, top_k_similarities = semantic_search(query_embedding, embeddings)

        # Print results
        print("Top K similar items:")
        for idx, similarity in zip(top_k_indices, top_k_similarities):
            code_snippet = dataset[int(idx)]['code']  # Convert numpy.int64 to int
            print(f"Index: {idx}, Similarity: {similarity}")
            print(f"Code Snippet:\n{code_snippet}\n")
            print("-" * 80)

if __name__ == "__main__":
    main()