from pathlib import Path
from project.processor import DocumentProcessor
from project.milvus import MilvusVectorStore
from project.pydantic_models import ProcessingConfig

def main():
    print("DOCUMENT PROCESSING")
    print("=" * 30)
    file_paths = [
        r"D:\genai\RAG\uploaded.pdf",
        r"D:\genai\RAG\test.json",
        r"D:\genai\RAG\test.csv",
        r"D:\genai\RAG\test.tsv",
        r"D:\genai\RAG\test.txt"
    ]
    # Check DB state
    stats = MilvusVectorStore.get_stats()
    existing_chunks = stats.get('total_chunks', 0)
    if existing_chunks > 0:
        print(f"Database has {existing_chunks} chunks")
        choice = input("Replace? (y/n): ").lower().strip()
        if choice == 'y':
            MilvusVectorStore.clear_all_data()
        else:
            print("Cancelled")
            return
    total_chunks = 0
    # Use config defaults from pydantic_models.py
    config = ProcessingConfig()
    for file_path in file_paths:
        if not Path(file_path).exists():
            print(f"File not found: {file_path}")
            continue
        print(f"\n--- Processing: {file_path} ---")
        processor = DocumentProcessor(config)
        try:
            document, chunks = processor.process_document(file_path)
            print(f"Processed: {document.title}")
            print(f"Chunks created: {len(chunks)} (method auto-selected based on filetype!)")
            total_chunks += len(chunks)
        except Exception as e:
            print(f"Error for {file_path}: {e}")
    print(f"\nAll done! Total new chunks: {total_chunks}")
    print("Run query_document.py to search.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled")
    except Exception as e:
        print(f"Fatal error: {e}")
