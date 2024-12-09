# app/etl/document_processor.py
import re
import logging
from typing import List, Dict, Any
from bs4 import BeautifulSoup
import markdown
from pymongo import MongoClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, mongodb_uri: str):
        """Initialize document processor"""
        self.client = MongoClient(mongodb_uri)
        self.db = self.client.rag_ros2
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        # Convert markdown to text if needed
        if any(marker in text for marker in ['##', '**', '__', '```']):
            text = markdown.markdown(text)
            # Remove HTML tags
            soup = BeautifulSoup(text, 'html.parser')
            text = soup.get_text()
        
        # Basic cleaning
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces
        text = re.sub(r'\n+', '\n', text)  # Replace multiple newlines
        text = text.strip()
        
        return text
    
    def extract_code_blocks(self, text: str) -> List[str]:
        """Extract code blocks from text"""
        code_blocks = []
        # Match markdown code blocks
        markdown_blocks = re.findall(r'```(?:\w+)?\n(.*?)\n```', text, re.DOTALL)
        code_blocks.extend(markdown_blocks)
        
        # Match inline code
        inline_code = re.findall(r'`([^`]+)`', text)
        code_blocks.extend(inline_code)
        
        return code_blocks
    
    def process_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single document"""
        try:
            content = document['content']
            
            # Extract code blocks before cleaning
            code_blocks = self.extract_code_blocks(content)
            
            # Clean the main content
            cleaned_content = self.clean_text(content)
            
            # Create processed document
            processed_doc = {
                'original_url': document['url'],
                'content': cleaned_content,
                'code_blocks': code_blocks,
                'metadata': {
                    'source': document['repo'],
                    'file_type': document['metadata']['file_type'],
                    'type': 'processed_documentation'
                }
            }
            
            # Store processed document
            self.db.processed_documents.update_one(
                {'original_url': document['url']},
                {'$set': processed_doc},
                upsert=True
            )
            
            return processed_doc
            
        except Exception as e:
            logger.error(f"Error processing document {document.get('url', 'unknown')}: {str(e)}")
            raise
    
    def process_all_documents(self) -> List[Dict[str, Any]]:
        """Process all raw documents"""
        raw_documents = self.db.raw_documents.find({})
        processed_docs = []
        
        for doc in raw_documents:
            try:
                processed_doc = self.process_document(doc)
                processed_docs.append(processed_doc)
            except Exception as e:
                logger.error(f"Error processing document: {str(e)}")
                continue
        
        return processed_docs
    
    def get_processed_navigation_docs(self) -> List[Dict[str, Any]]:
        """Get processed navigation-specific documents"""
        return list(self.db.processed_documents.find({
            '$or': [
                {'content': {'$regex': 'navigation|path.*planning|motion', '$options': 'i'}},
                {'metadata.source': 'ros-planning/navigation2'}
            ]
        }))

    def __del__(self):
        """Cleanup connection"""
        self.client.close()