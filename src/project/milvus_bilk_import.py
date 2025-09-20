from pymilvus import utility
import time

COLLECTION = "rag_chunks"
NDJSON_PATH = "/var/lib/milvus/import/chunks_bulk.ndjson"  # Or your configured path

task_id = utility.do_bulk_insert(
    collection_name=COLLECTION,
    files=[NDJSON_PATH]
)
print(f"Submitted bulk import task: {task_id}")

while True:
    status = utility.get_bulk_insert_state(task_id)
    print("Bulk import status:", status)
    if status.get('state') == 'BulkImportCompleted':
        print("Bulk import completed successfully.")
        break
    if status.get('state') == 'BulkImportFailed':
        print("Bulk import failed.", status)
        break
    time.sleep(30)
