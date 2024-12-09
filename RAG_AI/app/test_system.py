# app/test_system.py
import sys
import logging
import traceback
from etl.github_crawler import GitHubCrawler
from featurization.embeddings import EmbeddingGenerator
from models.llm import LLMHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_github_crawler():
    """Test GitHub crawler functionality"""
    try:
        logger.info("Testing GitHub Crawler...")
        # Initialize crawler without ClearML
        github_crawler = GitHubCrawler(
            mongodb_uri="mongodb://mongodb:27017",
            use_clearml=False  # Disable ClearML for now
        )
        content = github_crawler.crawl_ros2_docs("https://github.com/ros2/ros2_documentation")
        logger.info("GitHub Crawler test completed successfully")
        return content
    except Exception as e:
        logger.error(f"GitHub Crawler test failed: {str(e)}")
        logger.error(traceback.format_exc())
        return None


def test_embedding_generator():
    """Test embedding generator functionality"""
    try:
        logger.info("Testing Embedding Generator...")
        embedder = EmbeddingGenerator(
            mongodb_uri="mongodb://mongodb:27017",
            qdrant_host="qdrant"
        )
        
        # Test with sample text
        sample_texts = [
            "ROS2 is a flexible framework for writing robot software.",
            "Navigation2 provides path planning capabilities for robots."
        ]
        sample_metadata = [
            {"domain": "ros2", "source": "test"},
            {"domain": "nav2", "source": "test"}
        ]
        
        embedder.store_embeddings(sample_texts, sample_metadata)
        search_results = embedder.search("How does ROS2 work?")
        logger.info(f"Search results: {search_results}")
        logger.info("Embedding Generator test completed successfully")
        return search_results
    except Exception as e:
        logger.error(f"Embedding Generator test failed: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def test_llm_handler():
    """Test LLM handler functionality"""
    try:
        logger.info("Testing LLM Handler...")
        llm = LLMHandler()
        
        # Test simple query
        test_prompt = "What is ROS2?"
        logger.info(f"Testing with prompt: {test_prompt}")
        
        response = llm.generate_response(test_prompt)
        
        if "error" in response.lower():
            logger.error(f"LLM generation failed: {response}")
            return False
            
        logger.info(f"LLM Response: {response}")
        
        # Test with context
        test_context = "ROS2 is a robotics middleware that provides tools and libraries for robot software development."
        test_prompt_with_context = "What are the main features of ROS2?"
        
        logger.info(f"Testing with context and prompt...")
        response_with_context = llm.generate_response(test_prompt_with_context, context=test_context)
        
        if "error" in response_with_context.lower():
            logger.error(f"LLM generation with context failed: {response_with_context}")
            return False
            
        logger.info(f"LLM Response with context: {response_with_context}")
        
        logger.info("LLM Handler test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"LLM Handler test failed: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def test_system():
    """Run all system tests"""
    try:
        # Test components one by one
        github_result = test_github_crawler()
        if github_result:
            logger.info("GitHub Crawler test passed")
        
        embedding_result = test_embedding_generator()
        if embedding_result:
            logger.info("Embedding Generator test passed")
        
        llm_result = test_llm_handler()
        if llm_result:
            logger.info("LLM Handler test passed")
        
        logger.info("All tests completed")
    except Exception as e:
        logger.error(f"System test failed: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    try:
        test_system()
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)