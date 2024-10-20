# index_datasets.py

import os
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, ID
from whoosh.analysis import StemmingAnalyzer
from datasets import load_from_disk
from concurrent.futures import ThreadPoolExecutor

def create_index(language, dataset_type):
    """
    Create a Whoosh index for a specific language and dataset type.
    """
    schema = Schema(
        id=ID(stored=True, unique=True),
        code=TEXT(stored=True, analyzer=StemmingAnalyzer()),
        description=TEXT(stored=True, analyzer=StemmingAnalyzer())
    )

    index_dir = f'indices/{language}/{dataset_type}'
    os.makedirs(index_dir, exist_ok=True)
    ix = create_in(index_dir, schema)

    writer = ix.writer()

    dataset_path = f'processed_datasets/{language}/{dataset_type}'
    dataset = load_from_disk(dataset_path)

    for i, sample in enumerate(dataset):
        writer.add_document(
            id=str(i),
            code=sample['code'],
            description=sample['description']
        )

    writer.commit()
    print(f"Index created for {language} - {dataset_type}")

def main():
    languages = ['python', 'java', 'javascript', 'ruby', 'go', 'php']
    dataset_types = ['train', 'valid', 'test']

    with ThreadPoolExecutor(max_workers=4) as executor:
        for language in languages:
            for dataset_type in dataset_types:
                print(f"Submitting indexing task for {language} - {dataset_type} dataset...")
                executor.submit(create_index, language, dataset_type)

if __name__ == "__main__":
    main()