import uuid
from pathlib import Path
from project.pydantic_models import Document, FileType
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import pandas as pd

class DocumentLoader:
    """LangChain-based document loader"""
    
    @staticmethod
    def load_document(file_path: str) -> Document:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # CRITICAL FIX: Proper file type detection
        ext = path.suffix.lower().replace(".", "")
        
        # Map extensions to FileType enum
        if ext == "pdf":
            file_type = FileType.PDF
        elif ext == "txt":
            file_type = FileType.TXT  
        elif ext == "json":
            file_type = FileType.JSON
        elif ext == "csv":
            file_type = FileType.CSV
        elif ext == "tsv" or "tsv#":
            file_type = FileType.TSV
        else:
            raise ValueError(f"Unsupported file type: {ext}")
            
        doc_id = str(uuid.uuid4())[:8]
        print(f"Loading {file_type.value.upper()}: {path.name}")
        
        # Route to correct loader based on actual file type
        if file_type == FileType.PDF:
            return DocumentLoader.load_pdf(str(path), doc_id)
        elif file_type == FileType.TXT:
            return DocumentLoader.load_txt(str(path), doc_id)
        elif file_type == FileType.JSON:
            return DocumentLoader.load_json(str(path), doc_id)
        elif file_type in [FileType.CSV, FileType.TSV]:
            return DocumentLoader.load_csv_tsv(str(path), doc_id, file_type)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    @staticmethod
    def load_pdf(file_path: str, doc_id: str) -> Document:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        content = "\n".join([page.page_content for page in pages])
        metadata = pages[0].metadata if pages else {}
        
        return Document(
            id=doc_id,
            title=Path(file_path).stem,
            content=content,
            file_type=FileType.PDF,
            metadata=metadata
        )

    @staticmethod
    def load_txt(file_path: str, doc_id: str) -> Document:
        """Load TXT file properly"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        return Document(
            id=doc_id,
            title=Path(file_path).stem,
            content=content,
            file_type=FileType.TXT,
            metadata={"source": file_path}
        )

    @staticmethod
    def load_json(file_path: str, doc_id: str) -> Document:
        """Load JSON file as raw text"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        return Document(
            id=doc_id,
            title=Path(file_path).stem,
            content=content,
            file_type=FileType.JSON,
            metadata={"source": file_path}
        )

    @staticmethod  
    def load_csv_tsv(file_path: str, doc_id: str, file_type: FileType) -> Document:
        """Load CSV/TSV file"""
        sep = "," if file_type == FileType.CSV else "\t"
        try:
            df = pd.read_csv(file_path, sep=sep)
            content = df.to_csv(index=False, sep=sep)
        except Exception as e:
            raise ValueError(f"Error loading {file_type.value}: {e}")
            
        return Document(
            id=doc_id,
            title=Path(file_path).stem,
            content=content,
            file_type=file_type,
            metadata={
                "source": file_path,
                "columns": ",".join(df.columns)
            }
        )
