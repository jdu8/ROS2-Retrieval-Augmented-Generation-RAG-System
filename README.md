# Riyam Patel (rp4334)
# Ishan Yadav (iy2159)

# CSGY-6613_Project_RAG

### ROS2 RAG System
#### Overview
A Retrieval-Augmented Generation (RAG) system specifically designed for ROS2 navigation queries. The system integrates multiple data sources including GitHub repositories, documentation, and YouTube tutorials to provide accurate, context-aware responses to ROS2 related questions.
Features

#### Multi-Source Data Integration:
GitHub Repository Crawling

Repositroy Used:

           1. 'ros2/ros2_documentation',
           2. 'ros-planning/navigation2',
           3. 'ros2/ros2',
           4. 'ros2/demos'
           
YouTube Tutorial Transcript Processing:

           1. 'https://www.youtube.com/watch?v=Gg25GfA456o&t=24s'
           
MongoDB for Raw Data Storage

Qdrant Vector Database for Semantic Search


#### LLM Integration:
Supports both Ollama and Fine-tuned Models
Custom Fine-tuned Model available on HuggingFace
Context-aware Response Generation
ROS2-specific Knowledge Base


#### Web Interface:
Built with Gradio
Pre-defined Question Templates
Custom Query Support
Model Selection (Ollama/HuggingFace)
Source Attribution


#### Monitoring and Logging:
ClearML Integration for Experiment Tracking
Pipeline Statistics
Error Handling and Logging

### Architecture

RAG_AI/

├── app/

│   ├── api/                # API endpoints

│   ├── etl/                # Data extraction and processing

│   ├── featurization/      # Vector embeddings and search

│   ├── models/             # LLM handlers

│   └── utils/              # Helper functions

├── docker/                 # Docker configuration

└── tests/                  # Unit tests

--------------------------------------------------------------------------------------------------------------------------------------

### Key Components
#### ETL Pipeline

GitHub Crawler: Extracts ROS2 navigation-related content from specified repositories
Document Processor: Cleans and structures raw text data
YouTube Crawler: Extracts and processes video transcripts

#### Feature Pipeline

Chunking: Splits documents into manageable segments
Embeddings: Generates vector embeddings using sentence-transformers
Vector Store: Manages embeddings in Qdrant for efficient similarity search

#### LLM Integration

##### Dual Model Support:
  Ollama for local inference
  Fine-tuned HuggingFace model for specialized responses
  
##### Context Enhancement: 
Combines retrieved documents with queries

#### Setup and Installation
##### Prerequisites
  Docker and Docker Compose
  NVIDIA GPU (recommended)
  HuggingFace Account (for fine-tuned model)

#### Installation
1. Clone the repository:
      git clone https://github.com/RiyamPatel2001/CSGY-6613_Project_RAG.git
      cd CSGY-6613_Project_RAG
2. Set up environment variables:
      cp .env.example .env
      Edit .env with your configurations (We haven't erased our .env files, so kindly keep it in mind)
3. Build and run with Docker:
      docker-compose up --build

#### Usage
Web Interface
  1. Access the Gradio interface at http://localhost:8000
  2. Choose between predefined questions or enter custom queries

--------------------------------------------------------------------------------------------------------------------------------------

### Key Concepts:

RAG Architecture
  The system uses a Retrieval-Augmented Generation approach:
  
  1. Retrieval: Query is used to find relevant documents in the vector store
  2. Augmentation: Retrieved context is combined with the query
  3. Generation: LLM generates response using the augmented context

Vector Search
  1. Documents are converted to vector embeddings
  2. Semantic similarity used for retrieval
  3. Cosine similarity for ranking results

Fine-tuning
  1. The model is fine-tuned specifically for ROS2 navigation:
  2. Uses LoRA (Low-Rank Adaptation)
  3. Trained on curated ROS2 QA pairs
  4. Optimized for technical accuracy

Contributing
  1. Fork the repository
  2. Create a feature branch
  3. Submit a pull request
