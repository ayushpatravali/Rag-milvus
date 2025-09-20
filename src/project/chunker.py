from typing import List
from project.pydantic_models import Chunk, ChunkingMethod, ProcessingConfig, Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    SentenceTransformersTokenTextSplitter,
    RecursiveJsonSplitter,
)
import pandas as pd
import io
import json

class ChunkingService:
    """LangChain-based chunking service with method toggle"""

    @staticmethod
    def chunk_document(document: Document, config: ProcessingConfig) -> List[Chunk]:
        print(f"Chunking with method: {config.chunking_method.value}")
        if document.file_type.value in ("csv", "tsv", "tsv#"):
            return ChunkingService._csv_tsv_chunking(document, config)
        if document.file_type.value == "json":
            chunks = ChunkingService._json_chunking(document, config)
        elif config.chunking_method == ChunkingMethod.RECURSIVE:
            chunks = ChunkingService._recursive_chunking(document, config)
        elif config.chunking_method == ChunkingMethod.CHARACTER:
            chunks = ChunkingService._character_chunking(document, config)
        elif config.chunking_method == ChunkingMethod.TOKEN:
            chunks = ChunkingService._token_chunking(document, config)
        elif config.chunking_method == ChunkingMethod.SENTENCE:
            chunks = ChunkingService._sentence_chunking(document, config)
        else:
            raise ValueError(f"Unknown chunking method: {config.chunking_method}")
        print(f"Created {len(chunks)} chunks")
        return chunks

    @staticmethod
    def _csv_tsv_chunking(document: Document, config: ProcessingConfig) -> List[Chunk]:
        sep = "," if document.file_type.value == "csv" else "\t"
        has_header = False
        if has_header:
            df = pd.read_csv(io.StringIO(document.content), sep=sep)
        else:
            df = pd.read_csv(io.StringIO(document.content), sep=sep, header=None)
            df.columns = [f"Column{i+1}" for i in range(df.shape[1])]
        max_kb = 2
        max_bytes = (max_kb * 1024) if max_kb else 4096
        chunks = []
        current_rows = []
        running_len = 0
        start_idx = 0
        for i, row in df.iterrows():
            row_dict = {str(col): str(row[col]) for col in df.columns}
            row_text = json.dumps(row_dict, ensure_ascii=False)
            row_byte_len = len(row_text.encode("utf-8"))
            if running_len + row_byte_len > max_bytes and current_rows:
                chunk_text = json.dumps(current_rows, ensure_ascii=False)
                chunk = Chunk(
                    id=f"{document.id}_chunk_{start_idx}",
                    doc_id=document.id,     # <-- FIX
                    content=chunk_text,
                    chunk_index=start_idx,
                    chunking_method=ChunkingMethod.RECURSIVE,
                    metadata={
                        "document_title": document.title,
                        "file_type": document.file_type.value,
                        "columns": list(df.columns),
                        "row_start": start_idx,
                        "row_end": i - 1,
                    }
                )
                chunks.append(chunk)
                current_rows = []
                running_len = 0
                start_idx = i
            current_rows.append(row_dict)
            running_len += row_byte_len
        if current_rows:
            chunk_text = json.dumps(current_rows, ensure_ascii=False)
            chunk = Chunk(
                id=f"{document.id}_chunk_{start_idx}",
                doc_id=document.id,   # <-- FIX
                content=chunk_text,
                chunk_index=start_idx,
                chunking_method=ChunkingMethod.RECURSIVE,
                metadata={
                    "document_title": document.title,
                    "file_type": document.file_type.value,
                    "columns": list(df.columns),
                    "row_start": start_idx,
                    "row_end": start_idx + len(current_rows) - 1,
                }
            )
            chunks.append(chunk)
        return chunks

    @staticmethod
    def _json_chunking(document: Document, config: ProcessingConfig) -> List[Chunk]:
        try:
            # No chunk_overlap here (not supported), manual overlap below
            splitter = RecursiveJsonSplitter(
                max_chunk_size=config.chunk_size
            )
            splits = splitter.split_text(document.content)
            texts = [split['text'] for split in splits]
            metas = [split.get('metadata', {}) for split in splits]

            # Manual overlap for JSON
            window = config.chunk_overlap or 0
            if window > 0 and len(texts) > 1:
                overlapped_texts = []
                overlapped_metas = []
                for idx in range(len(texts)):
                    start = max(0, idx - window)
                    overlap_text = " ".join(texts[start:idx+1])
                    overlapped_texts.append(overlap_text)
                    overlapped_metas.append(metas[idx])
                texts = overlapped_texts
                metas = overlapped_metas

            return ChunkingService._create_chunks_json(texts, metas, document, config)
        except Exception as e:
            print(f"JSON splitter failed: {e}, using fallback.")
            return ChunkingService._recursive_chunking(document, config)

    @staticmethod
    def _recursive_chunking(document: Document, config: ProcessingConfig) -> List[Chunk]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        texts = splitter.split_text(document.content)
        return ChunkingService._create_chunks(texts, document, config)

    @staticmethod
    def _character_chunking(document: Document, config: ProcessingConfig) -> List[Chunk]:
        splitter = CharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separator="\n\n"
        )
        texts = splitter.split_text(document.content)
        return ChunkingService._create_chunks(texts, document, config)

    @staticmethod
    def _token_chunking(document: Document, config: ProcessingConfig) -> List[Chunk]:
        splitter = TokenTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        texts = splitter.split_text(document.content)
        return ChunkingService._create_chunks(texts, document, config)

    @staticmethod
    def _sentence_chunking(document: Document, config: ProcessingConfig) -> List[Chunk]:
        splitter = SentenceTransformersTokenTextSplitter(
            chunk_overlap=config.chunk_overlap,
            tokens_per_chunk=config.chunk_size
        )
        texts = splitter.split_text(document.content)
        return ChunkingService._create_chunks(texts, document, config)

    @staticmethod
    def _create_chunks(texts: List[str], document: Document, config: ProcessingConfig) -> List[Chunk]:
        chunks = []
        for i, text in enumerate(texts):
            if len(text.strip()) >= 20:
                chunk = Chunk(
                    id=f"{document.id}_chunk_{i}",
                    doc_id=document.id,    # <-- FIX
                    content=text.strip(),
                    chunk_index=i,
                    chunking_method=config.chunking_method,
                    metadata={
                        "document_title": document.title,
                        "file_type": document.file_type.value,
                        "chunk_size": len(text),
                        "word_count": len(text.split())
                    }
                )
                chunks.append(chunk)
        return chunks

    @staticmethod
    def _create_chunks_json(texts: List[str], metas: List[dict], document: Document, config: ProcessingConfig) -> List[Chunk]:
        chunks = []
        for i, (text, meta) in enumerate(zip(texts, metas)):
            if len(text.strip()) >= 20:
                chunk_meta = {
                    "document_title": document.title,
                    "file_type": document.file_type.value,
                    "chunk_size": len(text),
                    "word_count": len(text.split()),
                    "json_path": meta.get('path', []),
                    **meta
                }
                chunk = Chunk(
                    id=f"{document.id}_chunk_{i}",
                    doc_id=document.id,    # <-- FIX
                    content=text.strip(),
                    chunk_index=i,
                    chunking_method=config.chunking_method,
                    metadata=chunk_meta
                )
                chunks.append(chunk)
        return chunks
