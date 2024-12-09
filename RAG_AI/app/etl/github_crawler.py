# app/etl/github_crawler.py
import os
import requests
import logging
from bs4 import BeautifulSoup
from pymongo import MongoClient
from typing import List, Dict, Any
from datetime import datetime
import base64
from github import Github
from urllib.parse import urljoin

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GitHubCrawler:
    def __init__(self, mongodb_uri: str):
        """Initialize GitHub crawler with MongoDB connection"""
        self.client = MongoClient(mongodb_uri)
        self.db = self.client.rag_ros2
        self.github_token = os.getenv('GITHUB_TOKEN')
        self.github_client = Github(self.github_token)
        
        # ROS2 repositories to crawl
        self.repos = [
            'ros2/ros2_documentation',
            'ros-planning/navigation2',
            'ros2/ros2',
            'ros2/demos'
        ]
        
        # Initialize DB collections if they don't exist
        self._init_collections()
    
    def _init_collections(self):
        """Initialize MongoDB collections with indexes"""
        if 'raw_documents' not in self.db.list_collection_names():
            self.db.create_collection('raw_documents')
            self.db.raw_documents.create_index([('url', 1)], unique=True)
            self.db.raw_documents.create_index([('last_updated', 1)])

    def crawl_repository(self, repo_name: str, branch: str = 'master') -> List[Dict[str, Any]]:
        """Crawl a GitHub repository for documentation"""
        try:
            logger.info(f"Crawling repository: {repo_name}")
            repo = self.github_client.get_repo(repo_name)
            documents = []

            # Get all markdown and RST files
            contents = repo.get_contents("")
            while contents:
                file_content = contents.pop(0)
                if file_content.type == "dir":
                    contents.extend(repo.get_contents(file_content.path))
                elif file_content.path.endswith(('.md', '.rst', '.txt')):
                    try:
                        # Get file content
                        content = base64.b64decode(file_content.content).decode('utf-8')
                        doc = {
                            'url': file_content.html_url,
                            'path': file_content.path,
                            'repo': repo_name,
                            'content': content,
                            'last_updated': datetime.utcnow(),
                            'metadata': {
                                'type': 'documentation',
                                'file_type': file_content.path.split('.')[-1],
                                'size': file_content.size,
                                'sha': file_content.sha
                            }
                        }
                        
                        # Store in MongoDB
                        self.db.raw_documents.update_one(
                            {'url': doc['url']},
                            {'$set': doc},
                            upsert=True
                        )
                        documents.append(doc)
                        logger.info(f"Processed file: {file_content.path}")
                        
                    except Exception as e:
                        logger.error(f"Error processing file {file_content.path}: {str(e)}")

            return documents
            
        except Exception as e:
            logger.error(f"Error crawling repository {repo_name}: {str(e)}")
            raise

    def crawl_all_repositories(self) -> List[Dict[str, Any]]:
        """Crawl all configured ROS2 repositories"""
        all_documents = []
        for repo in self.repos:
            try:
                documents = self.crawl_repository(repo)
                all_documents.extend(documents)
            except Exception as e:
                logger.error(f"Error crawling {repo}: {str(e)}")
                continue
        return all_documents

    def get_navigation_docs(self) -> List[Dict[str, Any]]:
        """Get navigation-specific documentation"""
        return list(self.db.raw_documents.find({
            '$or': [
                {'path': {'$regex': 'nav.*', '$options': 'i'}},
                {'content': {'$regex': 'navigation|path.*planning|motion', '$options': 'i'}},
            ]
        }))

    def __del__(self):
        """Cleanup connection"""
        self.client.close()
        
        
# # app/etl/github_crawler.py
# import os
# import requests
# from bs4 import BeautifulSoup
# from pymongo import MongoClient
# import logging

# logger = logging.getLogger(__name__)

# class GitHubCrawler:
#     def __init__(self, mongodb_uri, use_clearml=False):
#         """
#         Initialize GitHub crawler
        
#         Args:
#             mongodb_uri: MongoDB connection URI
#             use_clearml: Whether to use ClearML for tracking (default: False)
#         """
#         self.client = MongoClient(mongodb_uri)
#         self.db = self.client.rag_ros2
#         self.github_token = os.getenv('GITHUB_TOKEN')
        
#         # Initialize ClearML only if requested
#         if use_clearml:
#             try:
#                 from clearml import Task
#                 self.task = Task.init(project_name="RAG-ROS2", task_name="GitHub-ETL")
#             except Exception as e:
#                 logger.warning(f"Failed to initialize ClearML: {e}")
#                 self.task = None
#         else:
#             self.task = None

#     def crawl_ros2_docs(self, repo_url):
#         """
#         Crawl ROS2 documentation from GitHub repository
#         """
#         try:
#             headers = {'Authorization': f'token {self.github_token}'} if self.github_token else {}
            
#             logger.info(f"Crawling {repo_url}")
#             response = requests.get(repo_url, headers=headers)
#             response.raise_for_status()
            
#             soup = BeautifulSoup(response.text, 'html.parser')
            
#             content = {
#                 'url': repo_url,
#                 'content': soup.get_text(),
#                 'metadata': {
#                     'source': 'github',
#                     'type': 'documentation'
#                 }
#             }
            
#             # Store in MongoDB
#             logger.info("Storing content in MongoDB")
#             self.db.raw_data.insert_one(content)
            
#             # Log to ClearML if available
#             if self.task:
#                 self.task.logger.report_text(f"Crawled {repo_url}")
            
#             return content
            
#         except Exception as e:
#             logger.error(f"Error crawling GitHub: {e}")
#             raise

#     def __del__(self):
#         """Cleanup when object is destroyed"""
#         try:
#             self.client.close()
#         except:
#             pass