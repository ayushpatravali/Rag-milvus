import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from project.pydantic_models import Document, Chunk
from project.weaviate_ut import VectorStore

class StorageManager:
    """Handle JSON storage and backup operations"""
    
    def __init__(self, storage_dir: str = "data/storage"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    def save_chunks_to_json(self, chunks: List[Chunk], filename: str = None) -> str:
        """Save chunks to JSON file"""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chunks_{timestamp}.json"
        
        filepath = self.storage_dir / filename
        
        # Convert to JSON format
        chunks_data = []
        for chunk in chunks:
            chunk_dict = chunk.dict()
            chunks_data.append(chunk_dict)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "total_chunks": len(chunks),
                    "chunking_method": chunks[0].chunking_method.value if chunks else "unknown"
                },
                "chunks": chunks_data
            }, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(chunks)} chunks to: {filepath}")
        return str(filepath)
    
    def load_chunks_from_json(self, filepath: str) -> List[Chunk]:
        """Load chunks from JSON file"""
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        chunks = []
        for chunk_data in data.get("chunks", []):
            chunk = Chunk(**chunk_data)
            chunks.append(chunk)
        
        print(f"Loaded {len(chunks)} chunks from: {filepath}")
        return chunks
    
    def backup_weaviate_data(self) -> str:
        """Backup all Weaviate data to JSON"""
        
        print("Backing up Weaviate data...")
        
        try:
            client = VectorStore.get_client()
            
            # Get all objects
            result = client.query.get("DocumentChunk").with_additional(["vector"]).do()
            
            if "data" not in result or "Get" not in result["data"]:
                print("No data found in Weaviate")
                return ""
            
            objects = result["data"]["Get"].get("DocumentChunk", [])
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.storage_dir / f"weaviate_backup_{timestamp}.json"
            
            backup_data = {
                "metadata": {
                    "backup_date": datetime.now().isoformat(),
                    "total_objects": len(objects)
                },
                "objects": objects
            }
            
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False)
            
            print(f"Backup saved: {backup_file}")
            return str(backup_file)
            
        except Exception as e:
            print(f"Backup failed: {e}")
            return ""
    
    def restore_weaviate_data(self, backup_file: str):
        """Restore Weaviate data from JSON backup"""
        
        print(f"Restoring from: {backup_file}")
        
        try:
            with open(backup_file, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)
            
            objects = backup_data.get("objects", [])
            client = VectorStore.get_client()
            
            # Clear existing data
            VectorStore.clear_all_data()
            
            # Restore objects
            with client.batch as batch:
                batch.batch_size = 50
                
                for obj in objects:
                    vector = obj.get("_additional", {}).get("vector", [])
                    properties = {k: v for k, v in obj.items() if not k.startswith("_")}
                    
                    batch.add_data_object(
                        class_name="DocumentChunk",
                        data_object=properties,
                        vector=vector
                    )
            
            print(f"Restored {len(objects)} objects")
            
        except Exception as e:
            print(f"Restore failed: {e}")
