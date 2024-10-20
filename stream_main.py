from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_from_disk
# import streamlit as st
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_from_disk
import nltk
from nltk.tokenize import word_tokenize
import os
import streamlit as st

def load_resources():
    # nltk.download('punkt')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    # Load tokenizer and model
    model_name = "microsoft/codebert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Load code embeddings and dataset
    code_embeddings = np.load('code_embeddings.npy')
    dataset = load_from_disk('processed_dataset')

    # Prepare BM25
    
    # tokenized_corpus = [word_tokenize(doc['code'].lower()) for doc in dataset]
    tokenized_corpus = [doc['code'].lower().split() for doc in dataset]
    
    bm25 = BM25Okapi(tokenized_corpus)

    return tokenizer, model, code_embeddings, dataset, bm25

def embed_query(query, model, tokenizer):
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

def semantic_search(query, code_embeddings, model, tokenizer, dataset, top_k=5):
    query_embedding = embed_query(query, model, tokenizer)
    similarities = cosine_similarity(query_embedding, code_embeddings)
    top_indices = np.argsort(similarities[0])[::-1][:top_k]
    return top_indices, similarities[0][top_indices]

def bm25_search(query, bm25, top_k=5):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return top_indices, scores[top_indices]


def hybrid_search(query, bm25, code_embeddings, model, tokenizer, dataset, top_k=5, alpha=0.5):
    # BM25
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # Semantic
    query_embedding = embed_query(query, model, tokenizer)
    semantic_scores = cosine_similarity(query_embedding, code_embeddings).flatten()
    
    # Normalize scores
    bm25_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-10)
    semantic_norm = (semantic_scores - semantic_scores.min()) / (semantic_scores.max() - semantic_scores.min() + 1e-10)
    
    # Combine
    combined_scores = alpha * bm25_norm + (1 - alpha) * semantic_norm
    top_indices = np.argsort(combined_scores)[::-1][:top_k]
    return top_indices, combined_scores[top_indices]


def main():
    st.title("🔍 Semantic and BM25 Code Search")

    # Load resources
    tokenizer, model, code_embeddings, dataset, bm25 = load_resources()

    # User input
    query = st.text_input("Enter your search query:", "")

    if st.button("Search"):
        if query:
            st.markdown("### 🔹 Semantic Search Results:")
            top_indices_sem, scores_sem = semantic_search(query, code_embeddings, model, tokenizer, dataset)
            for idx, score in zip(top_indices_sem, scores_sem):
                idx = int(idx)
                st.write(f"**Index:** {idx} | **Similarity Score:** {score:.4f}")
                st.code(dataset[idx]['code'], language='python')  # Adjust language as needed
                st.write(f"**Description:** {dataset[idx]['description']}")
                st.markdown("---")

            st.markdown("### 🔸 BM25 Search Results:")
            top_indices_bm25, scores_bm25 = bm25_search(query, bm25)
            for idx, score in zip(top_indices_bm25, scores_bm25):
                idx = int(idx)
                st.write(f"**Index:** {idx} | **BM25 Score:** {score:.4f}")
                st.code(dataset[idx]['code'], language='python')  # Adjust language as needed
                st.write(f"**Description:** {dataset[idx]['description']}")
                st.markdown("---")
        else:
            st.warning("Please enter a search query.")

if __name__ == "__main__":
    main()