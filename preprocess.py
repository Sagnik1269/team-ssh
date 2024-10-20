import os
import gzip
import json
from datasets import Dataset

def load_jsonl_gz(file_path):
    """
    Loads a .jsonl.gz file and returns a list of JSON objects.
    """
    data = []
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_all_data(directory_path):
    """
    Loads all .jsonl.gz files in the given directory and combines them into a single list.
    """
    all_data = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.jsonl.gz'):
            file_path = os.path.join(directory_path, filename)
            print(f"Loading {file_path}...")
            data = load_jsonl_gz(file_path)
            all_data.extend(data)
    return all_data

def main():
    # Path to the 'jsonl' directory
    train_dir = '../train'  # Adjust the path if necessary

    # Load all training data
    train_data = load_all_data(train_dir)
    print(f"Total training samples loaded: {len(train_data)}")

    # Extract code snippets and docstrings
    code_snippets = [entry['code'] for entry in train_data if 'code' in entry]
    descriptions = [entry['docstring'] for entry in train_data if 'docstring' in entry]

    print(f"Total code snippets: {len(code_snippets)}")
    print(f"Total descriptions: {len(descriptions)}")

    # Optional: Sample a subset for initial testing
    sample_size = 1000  # Adjust the sample size as needed
    if len(code_snippets) > sample_size:
        code_snippets = code_snippets[:sample_size]
        descriptions = descriptions[:sample_size]
        print(f"Sampled {sample_size} code snippets and descriptions for testing.")

    # Save the samples to a new dataset or proceed with further preprocessing
    # For demonstration, we'll print the first sample
    if code_snippets and descriptions:
        print("\nSample Code Snippet:\n", code_snippets[0])
        print("\nSample Description:\n", descriptions[0])
    else:
        print("No code snippets or descriptions found.")

    # (Optional) Create a Hugging Face Dataset for easier handling
    if code_snippets and descriptions:
        dataset = Dataset.from_dict({
            'code': code_snippets,
            'description': descriptions
        })

        # Save the dataset to disk for later use
        dataset.save_to_disk('processed_dataset')
        print("Processed dataset saved to 'processed_dataset' directory.")

if __name__ == "__main__":
    main()