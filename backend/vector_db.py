"""
Vector Database for Insurance Document Retrieval
Uses BAAI/bge-large-en-v1.5 embeddings with LanceDB for similarity search.
"""

import os
import ast
from pathlib import Path
from typing import List, Dict, Any
import lancedb
from sentence_transformers import SentenceTransformer
import pyarrow as pa


class VectorDB:
    def __init__(self, db_path: str = "vector_db", model_name: str = "BAAI/bge-large-en-v1.5"):
        """Initialize vector database with embedding model."""
        self.db_path = db_path
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.db = lancedb.connect(db_path)
        self.table_name = "insurance_chunks"
        
    def load_chunks(self, chunks_dir: str = "chunks") -> List[Dict[str, Any]]:
        """Load all text chunks from the chunks directory."""
        chunks = []
        chunks_path = Path(chunks_dir)
        
        for category_dir in chunks_path.iterdir():
            if not category_dir.is_dir():
                continue
                
            category = category_dir.name
            
            for doc_dir in category_dir.iterdir():
                if not doc_dir.is_dir():
                    continue
                    
                doc_name = doc_dir.name
                
                for chunk_file in doc_dir.glob("*.txt"):
                    try:
                        with open(chunk_file, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                            
                        # Parse the chunk data (it's stored as a Python dict string)
                        chunk_data = ast.literal_eval(content)
                        
                        # Add category and clean document name
                        chunk_data['category'] = category
                        chunk_data['document_name'] = doc_name.replace('.pdf', '')
                        
                        chunks.append(chunk_data)
                        
                    except Exception as e:
                        print(f"Error loading {chunk_file}: {e}")
                        continue
                        
        return chunks
    
    def create_database(self, chunks_dir: str = "chunks"):
        """Create vector database from chunks."""
        print("Loading chunks...")
        chunks = self.load_chunks(chunks_dir)
        print(f"Loaded {len(chunks)} chunks")
        
        if not chunks:
            raise ValueError("No chunks found to process")
        
        print("Generating embeddings...")
        texts = [chunk['chunk_embedding_text'] for chunk in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Prepare data for LanceDB - only embed the text
        data = []
        for i, chunk in enumerate(chunks):
            data.append({
                'chunk_id': chunk['chunk_id'],
                'text': chunk['chunk_embedding_text'],
                'category': chunk['category'],
                'document_name': chunk['document_name'],
                'page': chunk['metadata'].get('page', ''),
                'embedding': embeddings[i]
            })
        
        print("Creating LanceDB table...")
        # Drop existing table if it exists
        try:
            self.db.drop_table(self.table_name)
        except:
            pass
        
        # Define schema with proper vector type
        schema = pa.schema([
            pa.field("chunk_id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("category", pa.string()),
            pa.field("document_name", pa.string()),
            pa.field("page", pa.string()),
            pa.field("embedding", pa.list_(pa.float32(), list_size=1024))  # BGE-large has 1024 dimensions
        ])
            
        # Create new table with schema
        table = self.db.create_table(self.table_name, data, schema=schema)
        print(f"Created vector database with {len(data)} chunks")
        
        return table
    
    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar chunks using vector similarity."""
        try:
            table = self.db.open_table(self.table_name)
        except:
            raise ValueError("Vector database not found. Please create it first.")
        
        # Generate query embedding
        query_embedding = self.model.encode([query])[0]
        
        # Search for similar vectors
        results = table.search(query_embedding, vector_column_name="embedding").limit(limit).to_list()
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                'chunk_id': result['chunk_id'],
                'text': result['text'],
                'category': result['category'],
                'document_name': result['document_name'],
                'page': result['page'],
                'score': 1.0 - float(result['_distance']) if '_distance' in result else 0.0  # Convert distance to similarity
            })
        
        return formatted_results


def initialize_vector_db(chunks_dir: str = "chunks") -> VectorDB:
    """Initialize and create vector database if it doesn't exist."""
    vdb = VectorDB()
    
    # Check if database exists
    try:
        table = vdb.db.open_table(vdb.table_name)
        print("Vector database already exists")
    except:
        print("Creating new vector database...")
        vdb.create_database(chunks_dir)
    
    return vdb


if __name__ == "__main__":
    # Create the vector database
    vdb = initialize_vector_db()
    
    # Test search
    results = vdb.search("comprehensive coverage for car insurance", limit=3)
    print(f"\nTest search results ({len(results)} chunks):")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['document_name']}")
        print(f"   Category: {result['category']}")
        print(f"   Score: {result['score']:.3f}")
        print(f"   Text: {result['text'][:100]}...")
        print()
