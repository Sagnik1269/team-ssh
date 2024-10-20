# index_search.py

from whoosh import index
from whoosh.fields import Schema, TEXT, ID
from datasets import load_from_disk
import os

def create_schema():
    """
    Defines the schema for the Whoosh index.
    """
    return Schema(
        id=ID(stored=True, unique=True),
        code=TEXT(stored=True),
        description=TEXT(stored=True)
    )

def create_search_index(dataset, index_dir='code_index'):
    """
    Creates a Whoosh index from the dataset.
    """
    if not os.path.exists(index_dir):
        os.mkdir(index_dir)
        ix = index.create_in(index_dir, create_schema())
    else:
        ix = index.open_dir(index_dir)
    
    writer = ix.writer()
    
    for idx, sample in enumerate(dataset):
        writer.add_document(
            id=str(idx),
            code=sample['code'],
            description=sample['description']
        )
        if idx % 100 == 0:
            print(f"Indexed {idx} documents...")
    
    writer.commit()
    print("Indexing completed.")

def main():
    # Load the processed dataset
    dataset = load_from_disk('processed_dataset')
    print(f"Total samples in dataset: {len(dataset)}")
    
    # Create the search index
    create_search_index(dataset)

if __name__ == "__main__":
    main()