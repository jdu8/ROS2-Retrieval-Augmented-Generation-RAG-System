# featurization/chunking.py
class DocumentChunker:
    def __init__(self, chunk_size=1000, overlap=200):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_document(self, text):
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Adjust chunk end to not split words
            if end < len(text):
                end = text.rfind(' ', start, end)
            
            chunks.append(text[start:end])
            start = end - self.overlap
            
        return chunks