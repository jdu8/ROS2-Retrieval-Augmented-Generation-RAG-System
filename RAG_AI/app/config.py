# config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    CLEARML_HOST = os.getenv("CLEARML_HOST", "localhost")
    HF_TOKEN = os.getenv("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")
    GITHUB_TOKEN = os.getenv("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")
    LINKEDIN_EMAIL = os.getenv("rambhapatel1010@gmail.com")
    LINKEDIN_PASSWORD = os.getenv("Rambha10@")
    CLEARML_ACCESS_KEY = os.getenv("EKFYKRET42301PIUBZ3SVKWC1NQ86Q")
    CLEARML_SECRET_KEY = os.getenv("M8wROIDU3e2X23A9dRLBpJ0qQVqyi1siFtGignb33tRBovpGr2eZo3VM9qlLdDCGsSY")
