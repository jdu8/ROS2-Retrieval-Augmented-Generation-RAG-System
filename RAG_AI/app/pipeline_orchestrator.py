# app/pipeline_orchestrator.py

# app/pipeline_orchestrator.py
import logging
import time
from typing import Dict, Any, List
from etl.youtube_crawler import YouTubeCrawler
from etl.github_crawler import GitHubCrawler
from etl.document_processor import DocumentProcessor
from featurization.feature_pipeline import FeaturePipeline
from clearml import Task, Logger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PipelineOrchestrator:
    def __init__(self, mongodb_uri: str, qdrant_host: str):
        """Initialize pipeline orchestrator"""
        self.mongodb_uri = mongodb_uri
        self.qdrant_host = qdrant_host
        
        # Use existing ClearML task instead of creating a new one
        self.task = Task.current_task()
        if not self.task:
            # Only create a new task if one doesn't exist
            timestamp = int(time.time())
            self.task = Task.create(
                project_name="ROS2-RAG",
                task_name=f"Pipeline-Execution-{timestamp}",
                task_type="data_processing"
            )
        
        # Initialize logger with default if none exists
        self.logger = Logger.current_logger()
        if not self.logger:
           self.logger = self.task.get_logger()
        
        # Initialize components
        self.github_crawler = GitHubCrawler(mongodb_uri)
        self.document_processor = DocumentProcessor(mongodb_uri)
        self.feature_pipeline = FeaturePipeline(mongodb_uri, qdrant_host)
        self.youtube_crawler = YouTubeCrawler(mongodb_uri)  # Add YouTube crawler


    def run_pipeline(self) -> Dict[str, Any]:
        """Run the complete pipeline"""
        start_time = time.time()
        iteration = 0
        stats = {
            'crawled_docs': 0,
            'processed_docs': 0,
            'vectorized_docs': 0,
            'errors': []
        }
        
        try:
            # Crawl repositories
            logger.info("Starting GitHub crawl...")
            self.logger.report_text("Starting GitHub crawl...")
            documents = self.github_crawler.crawl_all_repositories()
            stats['crawled_docs'] = len(documents)
            self.logger.report_scalar("stats", "crawled_docs", value=stats['crawled_docs'], iteration=iteration)
            
             # Process YouTube videos
            logger.info("Processing YouTube videos...")
            self.logger.report_text("Processing YouTube videos...")
            video_ids = [
                "Gg25GfA456o&t=23s"
            ]
            youtube_docs = self.youtube_crawler.crawl_videos(video_ids)
            stats['youtube_docs'] = len(youtube_docs)
            self.logger.report_scalar("stats", "youtube_docs", value=stats['youtube_docs'], iteration=iteration)
            
            # Process documents
            logger.info("Processing documents...")
            self.logger.report_text("Processing documents...")
            processed_docs = self.document_processor.process_all_documents()
            stats['processed_docs'] = len(processed_docs)
            self.logger.report_scalar("stats", "processed_docs", value=stats['processed_docs'], iteration=iteration)
            
            # Create embeddings
            logger.info("Creating embeddings...")
            self.logger.report_text("Creating embeddings...")
            self.feature_pipeline.process_all_documents()
            stats['vectorized_docs'] = len(processed_docs)
            self.logger.report_scalar("stats", "vectorized_docs", value=stats['vectorized_docs'], iteration=iteration)
            
            # Log execution time
            execution_time = time.time() - start_time
            self.logger.report_scalar("timing", "execution_time", value=execution_time, iteration=iteration)
            
            return stats
            
        except Exception as e:
            error_msg = f"Pipeline failed: {str(e)}"
            logger.error(error_msg)
            self.logger.report_text(error_msg)
            stats['errors'].append(str(e))
            return stats

    def search_navigation_content(
        self,
        query: str,
        limit: int = 5,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Search for navigation-related content"""
        try:
            logger.info(f"Searching for: {query}")
            self.logger.report_text(f"Search query: {query}")
            
            # Search in vector store
            results = self.feature_pipeline.search_similar(
                query=query,
                limit=limit,
                threshold=threshold
            )
            
            # Filter for navigation-related content
            nav_results = []
            for result in results:
                if any(term in result['text'].lower() 
                      for term in ['navigation', 'path', 'planning', 'motion', 'nav2']):
                    nav_results.append(result)
                    
            self.logger.report_text(f"Found {len(nav_results)} relevant results")
            return nav_results
            
        except Exception as e:
            error_msg = f"Search failed: {str(e)}"
            logger.error(error_msg)
            self.logger.report_text(error_msg)
            raise

def main():
    import os
    
    # Get configuration
    mongodb_uri = os.getenv('MONGODB_URI', 'mongodb://mongodb:27017')
    qdrant_host = os.getenv('QDRANT_HOST', 'qdrant')
    
    # Run pipeline
    orchestrator = PipelineOrchestrator(mongodb_uri, qdrant_host)
    logger.info("Starting pipeline execution...")
    stats = orchestrator.run_pipeline()
    logger.info(f"Pipeline stats: {stats}")

if __name__ == "__main__":
    main()
    
    
# import logging
# from etl.github_crawler import GitHubCrawler
# from etl.document_processor import DocumentProcessor
# from featurization.feature_pipeline import FeaturePipeline
# import time
# from typing import Dict, Any, List

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class PipelineOrchestrator:
#     def __init__(
#         self,
#         mongodb_uri: str,
#         qdrant_host: str
#     ):
#         """Initialize pipeline orchestrator"""
#         self.mongodb_uri = mongodb_uri
#         self.qdrant_host = qdrant_host
        
#         # Initialize components
#         self.github_crawler = GitHubCrawler(mongodb_uri)
#         self.document_processor = DocumentProcessor(mongodb_uri)
#         self.feature_pipeline = FeaturePipeline(mongodb_uri, qdrant_host)
    
#     def run_full_pipeline(self) -> Dict[str, Any]:
#         """Run the complete pipeline"""
#         start_time = time.time()
#         stats = {
#             'crawled_docs': 0,
#             'processed_docs': 0,
#             'vectorized_docs': 0,
#             'errors': []
#         }
        
#         try:
#             # 1. Crawl GitHub repositories
#             logger.info("Starting GitHub crawl...")
#             documents = self.github_crawler.crawl_all_repositories()
#             stats['crawled_docs'] = len(documents)
#             logger.info(f"Crawled {len(documents)} documents")
            
#             # 2. Process documents
#             logger.info("Processing documents...")
#             processed_docs = self.document_processor.process_all_documents()
#             stats['processed_docs'] = len(processed_docs)
#             logger.info(f"Processed {len(processed_docs)} documents")
            
#             # 3. Create embeddings and store in vector DB
#             logger.info("Creating embeddings and storing in vector DB...")
#             self.feature_pipeline.process_all_documents()
#             stats['vectorized_docs'] = len(processed_docs)
            
#             # Calculate execution time
#             execution_time = time.time() - start_time
#             stats['execution_time'] = execution_time
#             logger.info(f"Pipeline completed successfully in {execution_time:.2f} seconds")
            
#             return stats
            
#         except Exception as e:
#             logger.error(f"Pipeline failed: {str(e)}")
#             stats['errors'].append(str(e))
#             raise

#     def run_navigation_pipeline(self) -> Dict[str, Any]:
#         """Run pipeline specifically for navigation-related content"""
#         start_time = time.time()
#         stats = {
#             'nav_docs': 0,
#             'processed_nav_docs': 0,
#             'vectorized_nav_docs': 0,
#             'errors': []
#         }
        
#         try:
#             # 1. Get navigation-specific documentation
#             logger.info("Fetching navigation-related documents...")
#             nav_docs = self.github_crawler.get_navigation_docs()
#             stats['nav_docs'] = len(nav_docs)
#             logger.info(f"Found {len(nav_docs)} navigation-related documents")
            
#             # 2. Process navigation documents
#             logger.info("Processing navigation documents...")
#             processed_nav_docs = []
#             for doc in nav_docs:
#                 processed_doc = self.document_processor.process_document(doc)
#                 processed_nav_docs.append(processed_doc)
#             stats['processed_nav_docs'] = len(processed_nav_docs)
            
#             # 3. Create embeddings for navigation content
#             logger.info("Creating embeddings for navigation content...")
#             for doc in processed_nav_docs:
#                 self.feature_pipeline.process_document(doc)
#             stats['vectorized_nav_docs'] = len(processed_nav_docs)
            
#             # Calculate execution time
#             execution_time = time.time() - start_time
#             stats['execution_time'] = execution_time
#             logger.info(f"Navigation pipeline completed in {execution_time:.2f} seconds")
            
#             return stats
            
#         except Exception as e:
#             logger.error(f"Navigation pipeline failed: {str(e)}")
#             stats['errors'].append(str(e))
#             raise

#     def search_navigation_content(
#         self,
#         query: str,
#         limit: int = 5,
#         threshold: float = 0.7
#     ) -> List[Dict[str, Any]]:
#         """Search for navigation-related content"""
#         try:
#             results = self.feature_pipeline.search_similar(
#                 query=query,
#                 limit=limit,
#                 threshold=threshold
#             )
            
#             # Filter for navigation-related content
#             nav_results = [
#                 result for result in results
#                 if any(
#                     term in result['text'].lower()
#                     for term in ['navigation', 'path', 'planning', 'motion', 'nav2']
#                 )
#             ]
            
#             return nav_results
            
#         except Exception as e:
#             logger.error(f"Search failed: {str(e)}")
#             raise

#     def get_pipeline_status(self) -> Dict[str, Any]:
#         """Get current status of the pipeline"""
#         try:
#             status = {
#                 'total_documents': self.github_crawler.db.raw_documents.count_documents({}),
#                 'processed_documents': self.github_crawler.db.processed_documents.count_documents({}),
#                 'navigation_documents': len(self.github_crawler.get_navigation_docs()),
#                 'vector_count': self.feature_pipeline.qdrant.get_collection(
#                     self.feature_pipeline.collection_name
#                 ).vectors_count
#             }
#             return status
#         except Exception as e:
#             logger.error(f"Failed to get pipeline status: {str(e)}")
#             raise

# def main():
#     """Run the pipeline"""
#     # Get configuration from environment variables
#     mongodb_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017')
#     qdrant_host = os.getenv('QDRANT_HOST', 'localhost')
    
#     # Initialize and run pipeline
#     orchestrator = PipelineOrchestrator(mongodb_uri, qdrant_host)
    
#     # Run navigation pipeline
#     logger.info("Starting navigation pipeline...")
#     stats = orchestrator.run_navigation_pipeline()
#     logger.info(f"Pipeline stats: {stats}")

# if __name__ == "__main__":
#     main()