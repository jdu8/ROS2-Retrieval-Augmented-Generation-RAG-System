# utils/helpers.py
import re
from typing import List, Dict, Any

class TextProcessor:
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        # Remove special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text.lower()
    
    @staticmethod
    def extract_code_blocks(text: str) -> List[str]:
        """Extract code blocks from text"""
        code_blocks = re.findall(r'```[\s\S]*?```', text)
        return [block.strip('`') for block in code_blocks]
    
    @staticmethod
    def extract_ros2_commands(text: str) -> List[str]:
        """Extract ROS2 specific commands"""
        # Pattern for common ROS2 commands
        ros2_pattern = r'ros2\s+\w+(?:\s+[\w/-]+)*'
        commands = re.findall(ros2_pattern, text)
        return commands

class DataValidator:
    @staticmethod
    def validate_document(doc: Dict[str, Any]) -> bool:
        """Validate document before insertion"""
        required_fields = ['content', 'source', 'type']
        return all(field in doc for field in required_fields)
    
    @staticmethod
    def validate_embedding(embedding: List[float]) -> bool:
        """Validate embedding vector"""
        return len(embedding) > 0 and all(isinstance(x, float) for x in embedding)
