# search.py

from whoosh import index
from whoosh.qparser import MultifieldParser

def search_query(query_str, index_dir='code_index', top_k=5):
    """
    Searches the index for the given query string and returns the top_k results.
    """
    ix = index.open_dir(index_dir)
    parser = MultifieldParser(["code", "description"], schema=ix.schema)
    query = parser.parse(query_str)
    
    with ix.searcher() as searcher:
        results = searcher.search(query, limit=top_k)
        for hit in results:
            print(f"ID: {hit['id']}")
            print(f"Code Snippet:\n{hit['code']}\n")
            print(f"Description:\n{hit['description']}\n")
            print("-" * 80)

def main():
    while True:
        query_str = input("Enter your search query (or type 'exit' to quit): ")
        if query_str.lower() == 'exit':
            break
        print(f"\nResults for query: '{query_str}'\n")
        search_query(query_str)
        print("\n")

if __name__ == "__main__":
    main()