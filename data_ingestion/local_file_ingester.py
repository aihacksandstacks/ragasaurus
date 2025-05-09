from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document as LlamaDocument # Or your LlamaIndex Node type
from typing import List
import os
import hashlib
import time
import json
from persistence.source_metadata_db import get_source_metadata, update_source_metadata

def load_local_directory(directory_path: str, force_refresh: bool = False) -> List[LlamaDocument]:
    processed_files = []
    new_documents = []
    reader = SimpleDirectoryReader(input_dir=directory_path, recursive=True)
    
    # SimpleDirectoryReader.load_data() returns LlamaIndex Documents directly
    # We need to iterate through files to check for modifications before loading all.
    for filepath in reader.input_files:
        source_identifier = str(filepath)
        file_meta = get_source_metadata(source_identifier)
        
        current_mtime = os.path.getmtime(filepath)
        current_hash = "" # Calculate hash if needed for more robust change detection

        if not force_refresh and file_meta:
            if file_meta.get('last_modified_at_source') == current_mtime:
                # Could add hash check here if mtime isn't reliable enough
                print(f"Skipping {filepath}, no modification detected.")
                processed_files.append(source_identifier)
                continue
        
        # File is new or modified, or force_refresh is True
        print(f"Loading/re-loading {filepath}")
        # Load just this one file
        single_file_reader = SimpleDirectoryReader(input_files=[filepath])
        docs_from_file = single_file_reader.load_data() # Returns List[LlamaDocument]

        # Update metadata for each document from this file
        doc_ids_for_this_file = []
        for doc in docs_from_file:
            doc.doc_id = hashlib.md5(f"{source_identifier}_{doc.text[:100]}".encode()).hexdigest() # Example stable ID
            doc_ids_for_this_file.append(doc.doc_id)
            # Ensure metadata from SimpleDirectoryReader (like file_path) is preserved
            if 'file_path' not in doc.metadata:
                 doc.metadata['file_path'] = source_identifier # Add it if missing

        new_documents.extend(docs_from_file)

        with open(filepath, 'rb') as f: # Calculate hash for change detection
            current_hash = hashlib.md5(f.read()).hexdigest()

        update_source_metadata(source_identifier, {
            "source_type": "local_file",
            "last_ingested": time.time(),
            "last_modified_at_source": current_mtime,
            "content_hash": current_hash,
            "llama_index_doc_ids": json.dumps(doc_ids_for_this_file)
        })
        processed_files.append(source_identifier)
        
    return new_documents