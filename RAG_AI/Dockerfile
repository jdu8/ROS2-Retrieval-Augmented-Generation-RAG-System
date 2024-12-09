FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Set locale
RUN apt-get update && apt-get install -y locales
RUN locale-gen en_US en_US.UTF-8
RUN update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gnupg2 \
    lsb-release \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    software-properties-common \
    wget \
    netcat \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create ClearML config directory
RUN mkdir -p /root/.clearml

# Create ClearML configuration
RUN echo 'api {\n\
    web_server: ${CLEARML_WEB_HOST}\n\
    api_server: ${CLEARML_API_HOST}\n\
    files_server: ${CLEARML_FILES_HOST}\n\
    credentials: {\n\
        "access_key": "${CLEARML_API_ACCESS_KEY}",\n\
        "secret_key": "${CLEARML_API_SECRET_KEY}"\n\
    }\n\
}\n\
sdk {\n\
    development_mode: true\n\
}' > /root/.clearml/config.conf

# Install PyTorch with CUDA support using the recommended method
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Ollama
RUN wget -q https://github.com/ollama/ollama/releases/download/v0.1.17/ollama-linux-amd64 -O /usr/local/bin/ollama \
    && chmod +x /usr/local/bin/ollama

# Copy requirements and install Python dependencies
COPY requirements.txt .
# Remove torch from requirements.txt since we installed it separately
RUN sed -i '/torch/d' requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ .

# Create entrypoint script
RUN printf '#!/bin/bash\n\
echo "Starting services..."\n\
\n\
# Verify CUDA availability\n\
python3 -c "import torch; print(f'\''CUDA available: {torch.cuda.is_available()}'\'')"\n\
\n\
# Start Ollama service\n\
echo "Starting Ollama..."\n\
/usr/local/bin/ollama serve &\n\
sleep 10\n\
\n\
# Pull the model\n\
echo "Pulling Llama2 model..."\n\
/usr/local/bin/ollama pull llama2\n\
\n\
# Initialize the database\n\
echo "Running data pipeline..."\n\
python3 pipeline_orchestrator.py\n\
\n\
# Start the Gradio interface\n\
echo "Starting Gradio interface..."\n\
python3 gradio_app.py\n' > /entrypoint.sh \
    && chmod +x /entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# With NVIDIA-GPU support


# FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# # Avoid prompts from apt
# ENV DEBIAN_FRONTEND=noninteractive

# # Set locale
# RUN apt-get update && apt-get install -y locales
# RUN locale-gen en_US en_US.UTF-8
# RUN update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
# ENV LANG=en_US.UTF-8

# # Install basic dependencies
# RUN apt-get update && apt-get install -y \
#     curl \
#     gnupg2 \
#     lsb-release \
#     python3-pip \
#     python3-dev \
#     build-essential \
#     git \
#     software-properties-common \
#     wget \
#     netcat \
#     && rm -rf /var/lib/apt/lists/*

# WORKDIR /app

# # Install Ollama
# RUN wget -q https://github.com/ollama/ollama/releases/download/v0.1.17/ollama-linux-amd64 -O /usr/local/bin/ollama \
#     && chmod +x /usr/local/bin/ollama

# # Copy requirements and install Python dependencies
# COPY requirements.txt .

# # Install PyTorch with CUDA support
# RUN pip3 install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cu121

# # Install other requirements
# RUN pip3 install --no-cache-dir -r requirements.txt

# # Copy application code
# COPY app/ .

# # Create entrypoint script with better service checks
# RUN printf '#!/bin/bash\n\
# \n\
# echo "Checking GPU availability..."\n\
# if command -v nvidia-smi &> /dev/null; then\n\
#     nvidia-smi\n\
#     echo "GPU is available"\n\
# else\n\
#     echo "nvidia-smi not found, running in CPU mode"\n\
# fi\n\
# \n\
# echo "Starting Ollama service..."\n\
# /usr/local/bin/ollama serve &\n\
# \n\
# # Wait for Ollama to be ready\n\
# echo "Waiting for Ollama to start..."\n\
# for i in {1..30}; do\n\
#     if curl -s http://localhost:11434/api/tags &>/dev/null; then\n\
#         echo "Ollama is ready"\n\
#         break\n\
#     fi\n\
#     echo "Waiting for Ollama... $i/30"\n\
#     sleep 2\n\
# done\n\
# \n\
# echo "Pulling Llama2 model..."\n\
# /usr/local/bin/ollama pull llama2\n\
# \n\
# echo "Starting the application..."\n\
# python3 test_system.py\n\
# python3 api/routes.py\n' > /entrypoint.sh \
#     && chmod +x /entrypoint.sh

# # Set entrypoint
# ENTRYPOINT ["/entrypoint.sh"]

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Without NVIDIA-GPU support

# FROM ubuntu:22.04

# # Avoid prompts from apt
# ENV DEBIAN_FRONTEND=noninteractive

# # Set locale
# RUN apt-get update && apt-get install -y locales
# RUN locale-gen en_US en_US.UTF-8
# RUN update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
# ENV LANG=en_US.UTF-8

# # Install basic dependencies
# RUN apt-get update && apt-get install -y \
#     curl \
#     gnupg2 \
#     lsb-release \
#     python3-pip \
#     python3-dev \
#     build-essential \
#     git \
#     software-properties-common \
#     wget \
#     && rm -rf /var/lib/apt/lists/*

# # Set up ROS2 repository
# RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
# RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

# # Install ROS2 Humble
# RUN apt-get update && apt-get install -y \
#     ros-humble-ros-base \
#     python3-colcon-common-extensions \
#     python3-rosdep \
#     && rm -rf /var/lib/apt/lists/*

# WORKDIR /app

# # Install Ollama manually
# RUN wget -q https://github.com/ollama/ollama/releases/download/v0.1.17/ollama-linux-amd64 -O /usr/local/bin/ollama \
#     && chmod +x /usr/local/bin/ollama

# # Upgrade pip first
# RUN python3 -m pip install --no-cache-dir --upgrade pip

# # Copy requirements and install Python dependencies
# COPY requirements.txt .
# RUN pip3 install --no-cache-dir -r requirements.txt

# # Copy application code
# COPY app/ .

# # Create entrypoint script
# RUN printf '#!/bin/bash\n\
# source /opt/ros/humble/setup.bash\n\
# \n\
# echo "Starting Ollama service..."\n\
# /usr/local/bin/ollama serve &\n\
# \n\
# # Wait for Ollama to start\n\
# echo "Waiting for Ollama service to initialize..."\n\
# for i in {1..30}; do\n\
#     if curl -s http://127.0.0.1:11434/api/tags > /dev/null; then\n\
#         echo "Ollama service is ready"\n\
#         break\n\
#     fi\n\
#     echo "Waiting for Ollama service... ($i/30)"\n\
#     sleep 2\n\
# done\n\
# \n\
# echo "Pulling Llama2 model..."\n\
# curl -X POST http://127.0.0.1:11434/api/pull -d '"'"'{"name": "llama2"}'"'"'\n\
# \n\
# echo "Starting the application..."\n\
# python3 test_system.py\n\
# python3 api/routes.py\n' > /entrypoint.sh && \
#     chmod +x /entrypoint.sh

# # Set entrypoint
# ENTRYPOINT ["/entrypoint.sh"]
