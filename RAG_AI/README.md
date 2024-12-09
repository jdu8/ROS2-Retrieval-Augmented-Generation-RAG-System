Team Members:
Riyam Patel (rp4334)

Ishan Yadav (iy2159)



# ROS2 Navigation Assistant

A RAG-based system for answering questions about ROS2 navigation, built with Ollama, MongoDB, and Qdrant.

## Prerequisites

- Docker and Docker Compose
- NVIDIA GPU (optional, for faster inference)
- NVIDIA Container Toolkit (if using GPU)

## Environment Setup

1. Create a `.env` file in the project root:
```bash
# GitHub token for accessing repositories
GITHUB_TOKEN=your_github_token_here

# Hugging Face token (optional)
HF_TOKEN=your_huggingface_token_here
```

2. If you have an NVIDIA GPU and want to use it:
```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## Running the System

1. Start the services:
```bash
docker-compose up --build
```

2. Access the interface:
- Open a web browser
- Go to http://localhost:8000

3. Fine-tuned model:
- It doesn't have a seperate option for pulling the model from hub, just click the option of "Huggingface" in the web-interface.
- First time running the fine-tuning will take time (a good amount of time, depending on your internet speed) because it will pull the model from the hub.

## Features

- Pre-populated questions about ROS2 navigation
- Custom question input
- Code generation for navigation tasks
- Context-aware responses using RAG
- Source attribution for responses

## Usage

1. Select a predefined question or enter your own
2. Optionally specify a Hugging Face model ID
3. Click "Get Answer"
4. View the response and sources

## System Components

- Ollama: Local LLM serving
- MongoDB: Document storage
- Qdrant: Vector database
- Gradio: Web interface

## ClearML
- Used ClearML as hosted service, not in the docker container, but seperately. It is working, you can check it in the Web interface of ClearML.

## Troubleshooting

If you encounter issues:

1. Check the logs:
```bash
docker-compose logs
```

2. Verify services are running:
```bash
docker-compose ps
```

3. Reset the system:
```bash
docker-compose down -v
docker-compose up --build
```
