import os
import torch
from transformers import AutoTokenizer, AutoModel
from datasets import load_from_disk
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def embed_text(texts, model, tokenizer):
    """
    Generate embeddings for a list of texts using a pre-trained model.
    """
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to('cuda')
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

def process_dataset(language, dataset_type):
    """
    Generate embeddings for a specific language and dataset type.
    """
    model_name = "microsoft/codebert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to('cuda')

    dataset_path = f'processed_datasets/{language}/{dataset_type}'
    dataset = load_from_disk(dataset_path)

    code_embeddings = []
    for sample in dataset:
        code = sample['code']
        embedding = embed_text([code], model, tokenizer)
        code_embeddings.append(embedding)

    code_embeddings = np.vstack(code_embeddings)
    np.save(f'embeddings/{language}_{dataset_type}_embeddings.npy', code_embeddings)
    print(f"Embeddings generated for {language} - {dataset_type}")

def main():
    languages = ['python', 'java', 'javascript', 'ruby', 'go', 'php']
    dataset_types = ['train', 'valid', 'test']

    os.makedirs('embeddings', exist_ok=True)

    with ThreadPoolExecutor(max_workers=4) as executor:
        for language in languages:
            for dataset_type in dataset_types:
                print(f"Submitting embedding task for {language} - {dataset_type} dataset...")
                executor.submit(process_dataset, language, dataset_type)

if __name__ == "__main__":
    main()