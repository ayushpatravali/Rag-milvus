from typing import Tuple, List
from project.pydantic_models import Document, Chunk, ProcessingConfig, ChunkingMethod
from project.doc_reader import DocumentLoader
from project.chunker import ChunkingService
from project.embedder import EmbeddingService
from project.milvus import MilvusVectorStore

class DocumentProcessor:
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.embedding_service = EmbeddingService(config.embedding_model)
        MilvusVectorStore.setup_schema()

    def process_document(self, file_path: str) -> Tuple[Document, List[Chunk]]:
        print(f"Processing: {file_path}")
        
        # Load document
        document = DocumentLoader.load_document(file_path)
        print(f"Loaded: {len(document.content)} characters")
        
        # Create chunks export file
        #self._export_document_content(document.content, document.title)
        
        # Chunk document
        chunks = ChunkingService.chunk_document(document, self.config)
        print(f"Created {len(chunks)} chunks")
        
        # Export chunks for inspection
        #self._export_chunks(chunks, document.title)
        
        # Generate embeddings
        chunks_with_embeddings = self.embedding_service.embed_chunks(chunks)
        
        # Store
        MilvusVectorStore.store_chunks(chunks_with_embeddings, document)
        
        # Verify storage
        stats = MilvusVectorStore.get_stats()
        print(f"Stored: {stats.get('total_chunks', 0)} chunks in database")
        
        return document, chunks_with_embeddings

    def _export_document_content(self, content: str, title: str):
        """Export original document content to file"""
        filename = f"original_{title}.txt"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"ORIGINAL DOCUMENT CONTENT: {title}\n")
                f.write("=" * 50 + "\n\n")
                f.write(content)
            print(f"Original content saved to: {filename}")
        except Exception as e:
            print(f"Export error: {e}")

    def _export_chunks(self, chunks: List[Chunk], title: str):
        """Export chunks to file for inspection"""
        filename = f"chunks_{title}.txt"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"CHUNKS FOR: {title}\n")
                f.write(f"Total chunks: {len(chunks)}\n")
                f.write("=" * 50 + "\n\n")
                
                for i, chunk in enumerate(chunks, 1):
                    f.write(f"CHUNK {i} (ID: {chunk.id})\n")
                    f.write(f"Length: {len(chunk.content)} chars\n")
                    f.write(f"Method: {chunk.chunking_method.value}\n")
                    f.write("-" * 30 + "\n")
                    f.write(chunk.content)
                    f.write("\n\n" + "=" * 50 + "\n\n")
            
            print(f"Chunks saved to: {filename}")
        except Exception as e:
            print(f"Chunk export error: {e}")
