# search_index.py

from whoosh.index import open_dir
from whoosh.qparser import QueryParser
import os

def search_index(language, dataset_type, query_str):
    """
    Search the Whoosh index for a specific language and dataset type.
    """
    index_dir = f'indices/{language}/{dataset_type}'
    ix = open_dir(index_dir)

    with ix.searcher() as searcher:
        query = QueryParser("description", ix.schema).parse(query_str)
        results = searcher.search(query, limit=2)
        for result in results:
            print(f"ID: {result['id']}")
            print(f"Code: {result['code']}")
            print(f"Description: {result['description']}\n")
            print("-" * 80)

def main():
    languages = ['python', 'java', 'javascript', 'ruby', 'go', 'php']
    dataset_types = ['train', 'valid', 'test']

    while True:
        query_str = input("Enter your search query (or type 'exit' to quit): ")
        if query_str.lower() == 'exit':
            break

        language = input(f"Enter language ({', '.join(languages)}): ")
        if language not in languages:
            print("Invalid language. Please try again.")
            continue

        dataset_type = input(f"Enter dataset type ({', '.join(dataset_types)}): ")
        if dataset_type not in dataset_types:
            print("Invalid dataset type. Please try again.")
            continue

        print(f"Searching {language} - {dataset_type} dataset for query: '{query_str}'")
        search_index(language, dataset_type, query_str)

if __name__ == "__main__":
    main()