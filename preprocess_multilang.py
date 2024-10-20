# preprocess_multilang.py

import os
import json
from datasets import Dataset, DatasetDict, load_dataset

def preprocess_data(language, dataset_type):
    """
    Preprocess the CodeSearchNet dataset for a specific language and dataset type.
    """
    data_dir = f'path/to/codesearchnet/{language}/{dataset_type}'
    files = [f for f in os.listdir(data_dir) if f.endswith('.jsonl.gz')]

    all_samples = []
    for file in files:
        file_path = os.path.join(data_dir, file)
        with open(file_path, 'r') as f:
            for line in f:
                sample = json.loads(line)
                code = sample.get('code', '')
                description = sample.get('docstring', '')
                if code and description:
                    all_samples.append({'code': code, 'description': description})

    return Dataset.from_list(all_samples)

def main():
    languages = ['python', 'java', 'javascript', 'ruby', 'go', 'php']
    dataset_types = ['train', 'valid', 'test']

    for language in languages:
        for dataset_type in dataset_types:
            print(f"Processing {language} - {dataset_type} dataset...")
            dataset = preprocess_data(language, dataset_type)
            output_dir = f'processed_datasets/{language}/{dataset_type}'
            os.makedirs(output_dir, exist_ok=True)
            dataset.save_to_disk(output_dir)
            print(f"Saved {language} - {dataset_type} dataset to {output_dir}")

if __name__ == "__main__":
    main()