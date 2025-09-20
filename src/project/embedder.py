from typing import List
from project.pydantic_models import Chunk, EmbeddingModel

# LangChain embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

class EmbeddingService:
    """LangChain-based embedding service"""
    
    def __init__(self, model_type: EmbeddingModel = EmbeddingModel.HUGGINGFACE):
        self.model_type = model_type
        self.embeddings = self._load_model()
    
    def _load_model(self):
        """Load embedding model"""
        print(f"Loading embedding model: {self.model_type.value}")
        
        if self.model_type == EmbeddingModel.HUGGINGFACE:
            # LangChain HuggingFace embeddings
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        
        elif self.model_type == EmbeddingModel.SENTENCE_TRANSFORMER:
            # Direct sentence-transformers (faster)
            return SentenceTransformer('all-MiniLM-L6-v2')
        
        else:
            raise ValueError(f"Unknown embedding model: {self.model_type}")
    
    def embed_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Add embeddings to chunks"""
        if not chunks:
            return chunks
        
        print(f"Generating embeddings for {len(chunks)} chunks...")
        
        # Extract text content
        texts = [chunk.content for chunk in chunks]
        
        # Generate embeddings
        if self.model_type == EmbeddingModel.HUGGINGFACE:
            # LangChain embeddings
            embeddings = self.embeddings.embed_documents(texts)
        else:
            # Direct sentence-transformers
            embeddings = self.embeddings.encode(texts, convert_to_numpy=True).tolist()
        
        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        
        print("Embeddings generated successfully")
        return chunks
    
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for query"""
        if self.model_type == EmbeddingModel.HUGGINGFACE:
            return self.embeddings.embed_query(query)
        else:
            return self.embeddings.encode(query, convert_to_numpy=True).tolist()
