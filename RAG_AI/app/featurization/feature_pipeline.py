# app/featurization/feature_pipeline.py
import logging
import time
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeaturePipeline:
    def __init__(
        self,
        mongodb_uri: str,
        qdrant_host: str,
        model_name: str = 'all-MiniLM-L6-v2',
        chunk_size: int = 512
    ):
        """Initialize feature pipeline"""
        self.mongodb_client = MongoClient(mongodb_uri)
        self.db = self.mongodb_client.rag_ros2
        self.qdrant = QdrantClient(host=qdrant_host)
        self.model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.collection_name = "ros2_vectors"
        
        # Initialize Qdrant collection
        self._init_qdrant()
    
    def _init_qdrant(self):
        """Initialize Qdrant collection"""
        collections = self.qdrant.get_collections().collections
        if not any(col.name == self.collection_name for col in collections):
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.model.get_sentence_embedding_dimension(),
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"Created Qdrant collection: {self.collection_name}")
    
    def chunk_document(self, text: str) -> List[str]:
        """Split document into chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size):
            chunk = ' '.join(words[i:i + self.chunk_size])
            chunks.append(chunk)
        
        return chunks
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for texts"""
        return self.model.encode(texts, normalize_embeddings=True)
    
    def process_document(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            chunks = self.chunk_document(doc['content'])
            embeddings = self.create_embeddings(chunks)
            
            points = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # Use a pure integer ID
                point_id = int(time.time() * 1000) + i  # Millisecond timestamp + index
                
                point = {
                    'id': point_id,  # Integer ID instead of string
                    'vector': embedding.tolist(),
                    'payload': {
                        'text': chunk,
                        'source_url': doc['original_url'],
                        'chunk_index': i,
                        'metadata': doc['metadata']
                    }
                }
                points.append(point)
            
            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=[models.PointStruct(**point) for point in points]
            )
            
            return points
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            logger.error(f"Raw response content: {e}")
            raise
    
    def process_all_documents(self):
        """Process all documents in MongoDB"""
        processed_docs = self.db.processed_documents.find({})
        
        for doc in processed_docs:
            try:
                self.process_document(doc)
                logger.info(f"Processed document: {doc['original_url']}")
            except Exception as e:
                logger.error(f"Error processing document {doc['original_url']}: {str(e)}")
                continue
    
    def search_similar(
        self,
        query: str,
        limit: int = 5,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Search for similar content"""
        query_vector = self.create_embeddings([query])[0]
        
        results = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            limit=limit,
            score_threshold=threshold
        )
        
        return [
            {
                'text': result.payload['text'],
                'source': result.payload['source_url'],
                'score': result.score,
                'metadata': result.payload['metadata']
            }
            for result in results
        ]

    def __del__(self):
        """Cleanup connections"""
        self.mongodb_client.close()