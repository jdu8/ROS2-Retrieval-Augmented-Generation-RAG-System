# featurization/embeddings.py
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import Distance, VectorParams, PointStruct
import numpy as np
from pymongo import MongoClient
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    def __init__(
        self, 
        model_name: str = 'all-MiniLM-L6-v2',
        mongodb_uri: str = "mongodb://localhost:27017",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "ros2_embeddings"
    ):
        """
        Initialize the EmbeddingGenerator with model and database connections.
        
        Args:
            model_name: Name of the sentence-transformer model to use
            mongodb_uri: URI for MongoDB connection
            qdrant_host: Hostname for Qdrant vector database
            qdrant_port: Port for Qdrant connection
            collection_name: Name of the Qdrant collection to use
        """
        self.model = SentenceTransformer(model_name)
        self.collection_name = collection_name
        self.vector_size = self.model.get_sentence_embedding_dimension()
        
        # Initialize database connections
        self.mongo_client = MongoClient(mongodb_uri)
        self.db = self.mongo_client.rag_ros2
        
        # Initialize Qdrant client and create collection if it doesn't exist
        self.qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)
        self._initialize_collection()

    def _initialize_collection(self) -> None:
        """Initialize Qdrant collection with proper configuration."""
        try:
            collections = self.qdrant.get_collections().collections
            exists = any(col.name == self.collection_name for col in collections)
            
            if not exists:
                self.qdrant.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created new collection: {self.collection_name}")
                
                # Create payload index for filtering
                self.qdrant.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="metadata.domain",
                    field_schema="keyword"
                )
        except Exception as e:
            logger.error(f"Error initializing Qdrant collection: {e}")
            raise

    def create_embedding(self, text: str) -> np.ndarray:
        """
        Create embedding for a single text input.
        
        Args:
            text: Input text to embed
            
        Returns:
            numpy.ndarray: Embedding vector
        """
        try:
            return self.model.encode(text, normalize_embeddings=True)
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            raise

    def create_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """
        Create embeddings for a batch of texts.
        
        Args:
            texts: List of input texts to embed
            
        Returns:
            numpy.ndarray: Array of embedding vectors
        """
        try:
            return self.model.encode(texts, normalize_embeddings=True, batch_size=32)
        except Exception as e:
            logger.error(f"Error creating batch embeddings: {e}")
            raise

    def store_embeddings(
        self, 
        texts: List[str], 
        metadata: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> None:
        """
        Store embeddings and metadata in Qdrant and MongoDB.
        
        Args:
            texts: List of texts to embed
            metadata: List of metadata dictionaries for each text
            batch_size: Size of batches for processing
        """
        try:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_metadata = metadata[i:i + batch_size]
                
                # Create embeddings for batch
                embeddings = self.create_embeddings_batch(batch_texts)
                
                # Prepare points for Qdrant
                points = [
                    PointStruct(
                        id=i + idx,
                        vector=embedding.tolist(),
                        payload={
                            "text": text,
                            "metadata": meta,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    )
                    for idx, (embedding, text, meta) in enumerate(zip(embeddings, batch_texts, batch_metadata))
                ]
                
                # Store in Qdrant
                self.qdrant.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                
                # Store in MongoDB for backup and additional querying
                mongo_docs = [
                    {
                        "text": text,
                        "metadata": meta,
                        "embedding_id": i + idx,
                        "timestamp": datetime.utcnow()
                    }
                    for idx, (text, meta) in enumerate(zip(batch_texts, batch_metadata))
                ]
                self.db.embeddings.insert_many(mongo_docs)
                
                logger.info(f"Stored {len(batch_texts)} embeddings")
                
        except Exception as e:
            logger.error(f"Error storing embeddings: {e}")
            raise

    def search(
        self, 
        query: str, 
        limit: int = 5, 
        domain_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar texts using embedding similarity.
        
        Args:
            query: Query text to search for
            limit: Maximum number of results to return
            domain_filter: Optional filter for specific domain (ros2, nav2, etc.)
            
        Returns:
            List of dictionaries containing matched texts and metadata
        """
        try:
            # Create query embedding
            query_vector = self.create_embedding(query)
            
            # Prepare search filters
            search_params = {}
            if domain_filter:
                search_params["query_filter"] = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.domain",
                            match=models.MatchValue(value=domain_filter)
                        )
                    ]
                )
            
            # Search in Qdrant
            search_results = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=query_vector.tolist(),
                limit=limit,
                **search_params
            )
            
            # Format results
            results = [
                {
                    "text": result.payload["text"],
                    "metadata": result.payload["metadata"],
                    "score": result.score
                }
                for result in search_results
            ]
            
            return results
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            raise

    def delete_embeddings(self, ids: List[int]) -> None:
        """
        Delete embeddings by their IDs.
        
        Args:
            ids: List of embedding IDs to delete
        """
        try:
            # Delete from Qdrant
            self.qdrant.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=ids
                )
            )
            
            # Delete from MongoDB
            self.db.embeddings.delete_many({"embedding_id": {"$in": ids}})
            
            logger.info(f"Deleted {len(ids)} embeddings")
            
        except Exception as e:
            logger.error(f"Error deleting embeddings: {e}")
            raise

    def update_metadata(self, embedding_id: int, new_metadata: Dict[str, Any]) -> None:
        """
        Update metadata for a specific embedding.
        
        Args:
            embedding_id: ID of the embedding to update
            new_metadata: New metadata dictionary
        """
        try:
            # Update in Qdrant
            self.qdrant.set_payload(
                collection_name=self.collection_name,
                payload={"metadata": new_metadata},
                points=[embedding_id]
            )
            
            # Update in MongoDB
            self.db.embeddings.update_one(
                {"embedding_id": embedding_id},
                {"$set": {"metadata": new_metadata}}
            )
            
            logger.info(f"Updated metadata for embedding {embedding_id}")
            
        except Exception as e:
            logger.error(f"Error updating metadata: {e}")
            raise

    def close(self) -> None:
        """Clean up database connections."""
        self.mongo_client.close()