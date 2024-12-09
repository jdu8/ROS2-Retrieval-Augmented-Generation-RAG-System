# etl/linkedin_crawler.py
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from pymongo import MongoClient
from clearml import Task
import time

class LinkedInCrawler:
    def __init__(self, mongodb_uri, linkedin_email, linkedin_password):
        self.client = MongoClient(mongodb_uri)
        self.db = self.client.rag_ros2
        self.email = linkedin_email
        self.password = linkedin_password
        self.task = Task.init(project_name="RAG-ROS2", task_name="LinkedIn-ETL")
        
        # Setup Selenium with Chrome in headless mode
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        self.driver = webdriver.Chrome(options=options)
        
    def login(self):
        """Login to LinkedIn"""
        self.driver.get('https://www.linkedin.com/login')
        
        # Wait for and fill in email
        email_elem = WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.ID, "username"))
        )
        email_elem.send_keys(self.email)
        
        # Fill in password and submit
        password_elem = self.driver.find_element(By.ID, "password")
        password_elem.send_keys(self.password)
        password_elem.submit()
        
        # Wait for login to complete
        time.sleep(10)
    
    def scrape_posts(self, search_terms):
        """Scrape LinkedIn posts related to ROS2"""
        self.login()
        
        for term in search_terms:
            search_url = f'https://www.linkedin.com/search/results/content/?keywords={term}'
            self.driver.get(search_url)
            
            # Wait for posts to load
            time.sleep(3)
            
            # Find all posts
            posts = self.driver.find_elements(By.CLASS_NAME, "feed-shared-update-v2")
            
            for post in posts:
                try:
                    content = post.find_element(By.CLASS_NAME, "feed-shared-text").text
                    
                    # Store in MongoDB
                    self.db.raw_data.insert_one({
                        'content': content,
                        'search_term': term,
                        'source': 'linkedin',
                        'type': 'post',
                        'timestamp': time.time()
                    })
                except:
                    continue
    
    def close(self):
        """Close the browser"""
        self.driver.quit()