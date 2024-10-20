# preprocess_multilang.py

import os
import json
import gzip
from datasets import Dataset

def preprocess_data(language, dataset_type):
    """
    Preprocess the CodeSearchNet dataset for a specific language and dataset type.
    """
    # Navigate from the scripts directory to the datasets directory
    base_path = '../../../../..'
    data_dir = f'{base_path}/{language}_data/{language}/final/jsonl/{dataset_type}'
    
    # Debug: Print the data directory being processed
    print(f"Processing directory: {data_dir}")
    
    files = [f for f in os.listdir(data_dir) if f.endswith('.jsonl.gz')]
    
    # Debug: Print the files found
    print(f"Found files: {files}")

    all_samples = []
    for file in files:
        file_path = os.path.join(data_dir, file)
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line)
                code = sample.get('code', '')
                description = sample.get('docstring', '')
                if code and description:
                    all_samples.append({'code': code, 'description': description})
    
    # Debug: Print the number of samples collected
    print(f"Collected {len(all_samples)} samples for {language} - {dataset_type}")

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