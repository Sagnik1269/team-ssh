import sys
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_from_disk

def embed_query(query, model, tokenizer):
    """
    Generate an embedding for a query using a pre-trained model.
    """
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

def semantic_search(query, code_embeddings, model, tokenizer, dataset, top_k=5):
    """
    Perform semantic search using cosine similarity.
    """
    query_embedding = embed_query(query, model, tokenizer)
    similarities = cosine_similarity(query_embedding, code_embeddings)
    top_indices = np.argsort(similarities[0])[::-1][:top_k]
    return top_indices, similarities[0][top_indices]

def main(query, lang):
    # Load the pre-trained model and tokenizer  
    model_name = "microsoft/codebert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Load the code embeddings
    code_embeddings = np.load('code_embeddings.npy')

    # Load the processed dataset
    dataset = load_from_disk('processed_dataset')

    # Perform semantic search
    top_indices, scores = semantic_search(query, code_embeddings, model, tokenizer, dataset)
    
    result = []
    for idx, score in zip(top_indices, scores):
        idx = int(idx)  # Convert numpy.int64 to Python int
        result.append({
            "index": idx,
            "score": float(score),
            "code": dataset[idx]['code'],
            "description": dataset[idx]['description']
        })
    
    return result

if __name__ == "__main__":
    query = sys.argv[1]
    lang = sys.argv[2]
    results = main(query, lang)
    print(results)