import time
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Dict, Any
from project.pydantic_models import Chunk, Document, SearchResult

class MilvusVectorStore:
    _vectorstore = None
    _embeddings = None
    _connected = False

    @classmethod
    def _wait_for_milvus(cls, max_retries=10, delay=3):
        print("Connecting to Milvus...")
        for attempt in range(max_retries):
            try:
                from pymilvus import connections
                connections.connect("default", host="localhost", port="19530", timeout=10)
                if connections.get_connection_addr("default"):
                    print("Milvus ready")
                    return True
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(delay)
        raise ConnectionError("Milvus connection failed")

    @classmethod
    def get_client(cls):
        if cls._embeddings is None:
            print("Loading embeddings...")
            cls._embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        if not cls._connected:
            cls._wait_for_milvus()
            cls._connected = True
        return True

    @classmethod
    def setup_schema(cls, class_name: str = "rag_chunks"):
        cls.get_client()
        connection_args = {"uri": "http://localhost:19530"}
        try:
            cls._vectorstore = Milvus(
                embedding_function=cls._embeddings,
                collection_name=class_name,
                connection_args=connection_args,
                consistency_level="Strong",
                drop_old=False,
                vector_field="embedding_vector",  # Match schema field name
                text_field="chunk_text",          # Match schema field name
                # CRITICAL: Enable dynamic fields to allow LangChain's automatic fields
                enable_dynamic_field=True,
                index_params={
                    "metric_type": "COSINE",
                    "index_type": "HNSW",
                    "params": {"M": 16, "efConstruction": 200}
                }
            )
            print(f"Collection '{class_name}' ready")
        except Exception as e:
            print(f"Error: {e}")
            raise e

    @classmethod
    def insert_chunks(cls, chunk_dicts: List[Dict], class_name: str = "rag_chunks"):
        """NEW: Store with dicts for batch/bulk mode (agentic and efficient)"""
        if cls._vectorstore is None:
            cls.setup_schema(class_name)
        from langchain_core.documents import Document as LangChainDoc
        
        docs = []
        ids = []
        for chunk in chunk_dicts:
            # Clean metadata - remove any problematic fields
            metadata = {
                "chunk_id": chunk.get("chunk_id"),
                "doc_id": chunk.get("doc_id"),
                "chunk_index": chunk.get("chunk_index"),
                "chunking_method": chunk.get("chunk_method"),
                "file_type": chunk.get("content_type"),
                "word_count": chunk.get("chunk_tokens", 0),
                "domain": chunk.get("domain"),
                "embedding_model": chunk.get("embedding_model")
            }
            
            doc = LangChainDoc(
                page_content=chunk['chunk_text'],
                metadata=metadata
            )
            docs.append(doc)
            ids.append(chunk.get("chunk_id"))

        batch_size = 50  # Reduced batch size for stability
        for i in range(0, len(docs), batch_size):
            batch_docs = docs[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            try:
                cls._vectorstore.add_documents(documents=batch_docs, ids=batch_ids)
                print(f"Inserted batch {i//batch_size + 1}: {len(batch_docs)} chunks")
            except Exception as e:
                print(f"Error inserting batch {i//batch_size + 1}: {e}")
                continue

        print(f"Completed insertion of {len(docs)} chunk dicts to Milvus.")

    # ... rest of methods remain the same ...
    @classmethod
    def store_chunks(cls, chunks: List[Chunk], document: Document, class_name: str = "rag_chunks"):
        """Store list of Chunk objects (old method)"""
        if cls._vectorstore is None:
            cls.setup_schema(class_name)
        from langchain_core.documents import Document as LangChainDoc
        print(f"Storing {len(chunks)} chunks...")

        langchain_docs = []
        ids = []
        for chunk in chunks:
            doc = LangChainDoc(
                page_content=chunk.content,
                metadata={
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                    "chunking_method": chunk.chunking_method.value,
                    "file_type": document.file_type.value,
                    "word_count": len(chunk.content.split())
                }
            )
            langchain_docs.append(doc)
            ids.append(chunk.id)

        batch_size = 50
        for i in range(0, len(langchain_docs), batch_size):
            batch_docs = langchain_docs[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            cls._vectorstore.add_documents(documents=batch_docs, ids=batch_ids)
        print("Storage complete")

    @classmethod
    def search_by_text(cls, query_text: str, limit: int = 5) -> List[SearchResult]:
        if cls._vectorstore is None:
            cls.setup_schema()
        try:
            results_with_scores = cls._vectorstore.similarity_search_with_relevance_scores(
                query=query_text,
                k=limit
            )

            search_results = []
            for rank, (doc, score) in enumerate(results_with_scores, 1):
                similarity_score = float(score)
                distance = 1.0 - similarity_score
                chunk = Chunk(
                    id=doc.metadata.get("chunk_id", f"chunk_{rank}"),
                    document_id=doc.metadata.get("document_id", ""),
                    content=doc.page_content,
                    chunk_index=doc.metadata.get("chunk_index", 0),
                    chunking_method=doc.metadata.get("chunking_method", "unknown"),
                    metadata=doc.metadata
                )
                search_result = SearchResult(
                    chunk=chunk,
                    similarity_score=similarity_score,
                    distance=distance,
                    rank=rank
                )
                search_results.append(search_result)
            return search_results
        except Exception as e:
            print(f"Search error: {e}")
            return []

    @classmethod
    def get_stats(cls, class_name: str = "rag_chunks") -> Dict[str, Any]:
        try:
            cls.get_client()
            from pymilvus import Collection, utility
            if utility.has_collection(class_name):
                collection = Collection(class_name)
                collection.flush()
                collection.load()
                return {
                    "total_chunks": collection.num_entities,
                    "status": "ready"
                }
            else:
                return {"total_chunks": 0, "status": "no_collection"}
        except Exception as e:
            return {"error": str(e), "status": "error", "total_chunks": 0}

    @classmethod
    def clear_all_data(cls, class_name: str = "rag_chunks"):
        try:
            print("Clearing database...")
            from pymilvus import utility
            cls.get_client()
            if utility.has_collection(class_name):
                utility.drop_collection(class_name)
            cls._vectorstore = None
            cls.setup_schema(class_name)
            print("Database cleared")
        except Exception as e:
            print(f"Clear error: {e}")
