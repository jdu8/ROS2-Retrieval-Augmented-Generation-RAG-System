# etl/youtube_crawler.py
# etl/youtube_crawler.py
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube
from pymongo import MongoClient
from clearml import Task
import logging

logger = logging.getLogger(__name__)

class YouTubeCrawler:
    def __init__(self, mongodb_uri):
        self.client = MongoClient(mongodb_uri)
        self.db = self.client.rag_ros2
        self.task = Task.current_task()

    def get_video_transcript(self, video_id):
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            # Combine transcript text
            text = ' '.join([entry['text'] for entry in transcript])
            
            # Get video metadata
            yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
            
            content = {
                'video_id': video_id,
                'content': text,
                'original_url': f"https://www.youtube.com/watch?v={video_id}",
                'metadata': {
                    'source': 'youtube',
                    'type': 'video',
                    'title': yt.title
                }
            }
            self.db.raw_data.insert_one(content)
            return content
        except Exception as e:
            logger.error(f"Error fetching transcript: {e}")
            return None

    def crawl_videos(self, video_ids):
        """Crawl multiple videos"""
        results = []
        for video_id in video_ids:
            content = self.get_video_transcript(video_id)
            if content:
                results.append(content)
                logger.info(f"Processed video: {video_id}")
        return results