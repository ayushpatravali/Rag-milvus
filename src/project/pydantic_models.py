from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

class FileType(str, Enum):
    PDF = "pdf"
    TXT = "txt"
    JSON = "json"
    CSV = "csv"
    TSV = "tsv" or "tsv#"
    

class ChunkingMethod(str, Enum):
    RECURSIVE = "recursive"
    CHARACTER = "character"
    TOKEN = "token"
    SENTENCE = "sentence"
    JSON = "json"

class EmbeddingModel(str, Enum):
    SENTENCE_TRANSFORMER = "sentence_transformer"
    HUGGINGFACE = "huggingface"

class ProcessingConfig(BaseModel):
    chunking_method: ChunkingMethod = ChunkingMethod.RECURSIVE
    chunk_size: int = 1024  
    chunk_overlap: int = 254  
    embedding_model: EmbeddingModel = EmbeddingModel.SENTENCE_TRANSFORMER

class Document(BaseModel):
    id: str
    title: str
    content: str = Field(..., min_length=50)
    file_type: FileType
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Chunk(BaseModel):
    id: str
    doc_id: str
    content: str = Field(..., min_length=20)
    chunk_index: int
    chunking_method: ChunkingMethod
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SearchResult(BaseModel):
    chunk: Chunk
    similarity_score: float
    distance: float
    rank: int
