# generate_embeddings.py

from transformers import AutoTokenizer, AutoModel
import torch
from datasets import load_from_disk
import numpy as np

def embed_text(texts, model, tokenizer):
    """
    Generate embeddings for a list of texts using a pre-trained model.
    """
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

def main():
    # Load the pre-trained model and tokenizer
    model_name = "microsoft/codebert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Load the processed dataset
    dataset = load_from_disk('processed_dataset')

    # Generate embeddings for code snippets
    code_embeddings = []
    for sample in dataset:
        code = sample['code']
        embedding = embed_text([code], model, tokenizer)
        code_embeddings.append(embedding)

    # Convert to numpy array for easier handling
    code_embeddings = np.vstack(code_embeddings)

    # Save embeddings to disk
    np.save('code_embeddings.npy', code_embeddings)
    print("Embeddings generated and saved to 'code_embeddings.npy'.")

if __name__ == "__main__":
    main()