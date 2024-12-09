# app/models/llm.py

import requests
import time
import logging
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMHandler:
    def __init__(self, model_name: str = "llama2", model_source: str = "ollama", max_retries: int = 3):
        self.model_name = model_name
        self.model_source = model_source  # "ollama" or "huggingface"
        self.max_retries = max_retries
        self.base_url = "http://127.0.0.1:11434"
        self.iteration_counter = 0
        
        # Try to initialize ClearML if available
        try:
            from clearml import Task, Logger
            self.task = Task.init(
                project_name="ROS2-RAG",
                task_name=f"LLM-Inference-{model_name}",
                auto_connect_frameworks=False
            )
            self.logger = Logger.current_logger()
            self.use_clearml = True
        except:
            logger.info("ClearML not configured, continuing without tracking")
            self.use_clearml = False

        # Initialize based on source
        if self.model_source == "huggingface":
            self.initialize_hf_model()
        else:
            self.ensure_model_availability()
            
    def ensure_model_availability(self) -> None:
        """Ensure the model is available and loaded"""
        for attempt in range(self.max_retries):
            try:
                # Check if Ollama service is ready
                response = requests.get(f"{self.base_url}/api/tags")
                if response.status_code == 200:
                    logger.info("Ollama service is ready")
                    
                    # Pull the model if needed
                    pull_response = requests.post(
                        f"{self.base_url}/api/pull",
                        json={"name": self.model_name}
                    )
                    pull_response.raise_for_status()
                    logger.info(f"Model {self.model_name} is ready")
                    return
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(5)
                else:
                    raise Exception(f"Failed to initialize Ollama after {self.max_retries} attempts")


    def initialize_hf_model(self):
        """Initialize HuggingFace model"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from peft import PeftModel
            import torch

            # Configure quantization with CPU offloading
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                llm_int8_enable_fp32_cpu_offload=True  # Enable CPU offloading
            )

            # Create custom device map
            device_map = {
                'model.embed_tokens': 'gpu',
                'model.norm': 'gpu',
                'lm_head': 'gpu',
                'model.layers': 'cpu',  # Offload large layers to CPU
            }

            # Load base model
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
            base_model = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Llama-2-7b-hf",
                quantization_config=bnb_config,
                device_map=device_map,  # Use custom device map
                torch_dtype=torch.float16
            )
            
            # Load LoRA adapter
            self.model = PeftModel.from_pretrained(
                base_model,
                self.model_name,
                device_map=device_map  # Use same device map for adapter
            )
            
            logger.info(f"Successfully loaded HuggingFace model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error loading HuggingFace model: {e}")
            raise

    def generate_response(self, prompt: str, context: Optional[str] = None) -> str:
        try:
            # Add ROS2-specific context to every prompt
            ros2_context = """You are a ROS2 navigation expert. Provide detailed, complete answers focusing on ROS2 robotics and navigation.
                            Always include code examples where relevant and explain concepts thoroughly. Do not cut off explanations mid-way."""

            full_prompt = (
                f"{ros2_context}\n\n"
                f"Context from ROS2 documentation: {context}\n\n"
                f"Question: {prompt}\n"
                "Answer (focusing only on ROS2-specific implementation):"
            ) if context else (
                f"{ros2_context}\n\n"
                f"Question: {prompt}\n"
                "Answer (focusing only on ROS2-specific implementation):"
            )
            
            if self.use_clearml:
                self.logger.report_text(f"Prompt: {prompt}")
                if context:
                    self.logger.report_text(f"Context: {context}")
            
            start_time = time.time()
            
            if self.model_source == "huggingface":
                # Generate using HuggingFace model
                inputs = self.tokenizer(full_prompt, return_tensors="pt").to("cuda")
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=500,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                # Generate using Ollama
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": full_prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "num_predict": 2048,
                            "stop": ["Question:", "\n\n"],  # Add stop sequences
                            "top_k": 40,
                            "top_p": 0.9,
                            "repeat_penalty": 1.1,  # Prevent repetitions
                            "stop": ["\n\nQuestion:", "\n\nHuman:", "\n\nSystem:"],  # Clear stop sequences
                            "presence_penalty": 0.6,  # Encourage longer responses
                            "frequency_penalty": 0.3,  # Reduce repetition
                        }
                    }
                )
                response.raise_for_status()
                result = response.json().get('response', '')
            
            if self.use_clearml:
                generation_time = time.time() - start_time
                self.iteration_counter += 1
                self.logger.report_scalar(
                    "timing", 
                    "generation_time", 
                    generation_time,
                    iteration=self.iteration_counter
                )
                self.logger.report_text(f"Response: {result[:200]}...")
            
            return result
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            logger.error(error_msg)
            if self.use_clearml:
                self.logger.report_text(f"Error: {str(e)}")
            return error_msg
        
# class LLMHandler:
#     def __init__(self, model_name: str = "llama2", max_retries: int = 3):
#         self.model_name = model_name
#         self.max_retries = max_retries
#         self.base_url = "http://127.0.0.1:11434"
#         self.iteration_counter = 0  # Add counter for iterations
        
#         # Try to initialize ClearML if available
#         try:
#             from clearml import Task, Logger
#             self.task = Task.init(
#                 project_name="ROS2-RAG",
#                 task_name=f"LLM-Inference-{model_name}",
#                 auto_connect_frameworks=False
#             )
#             self.logger = Logger.current_logger()
#             self.use_clearml = True
#         except:
#             logger.info("ClearML not configured, continuing without tracking")
#             self.use_clearml = False
        
#         self.ensure_model_availability()

#     def ensure_model_availability(self) -> None:
#         """Ensure the model is available and loaded"""
#         for attempt in range(self.max_retries):
#             try:
#                 # Check if Ollama service is ready
#                 response = requests.get(f"{self.base_url}/api/tags")
#                 if response.status_code == 200:
#                     logger.info("Ollama service is ready")
                    
#                     # Pull the model if needed
#                     pull_response = requests.post(
#                         f"{self.base_url}/api/pull",
#                         json={"name": self.model_name}
#                     )
#                     pull_response.raise_for_status()
#                     logger.info(f"Model {self.model_name} is ready")
#                     return
                    
#             except Exception as e:
#                 logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
#                 if attempt < self.max_retries - 1:
#                     time.sleep(5)
#                 else:
#                     raise Exception(f"Failed to initialize Ollama after {self.max_retries} attempts")

#     def generate_response(self, prompt: str, context: Optional[str] = None) -> str:
#         """Generate response with optional ClearML tracking"""
#         try:
#             # Add explicit ROS2 context and constraints
#             system_context = (
#                 "You are a ROS2 expert assistant. Your purpose is to help users with ROS2 navigation and related tasks. "
#                 "You should ONLY provide information about ROS2 and its ecosystem. "
#                 "If you cannot find relevant ROS2-specific information in the context, say so and ask for clarification about the ROS2 aspects of the question."
#             )
            
#             if context:
#                 full_prompt = (
#                     f"{system_context}\n\n"
#                     f"Using the following ROS2 documentation and code examples, answer the question:\n"
#                     f"Context:\n{context}\n\n"
#                     f"Question: {prompt}\n"
#                     "Answer with specific focus on ROS2 implementation:"
#                 )
#             else:
#                 full_prompt = (
#                     f"{system_context}\n\n"
#                     f"Question: {prompt}\n"
#                     "Answer with specific focus on ROS2 implementation:"
#                 )
                
#             # Track with ClearML if available
#             if self.use_clearml:
#                 self.logger.report_text(f"Prompt: {prompt}")
#                 if context:
#                     self.logger.report_text(f"Context: {context}")
            
#             start_time = time.time()
            
#             response = requests.post(
#                 f"{self.base_url}/api/generate",
#                 json={
#                     "model": self.model_name,
#                     "prompt": full_prompt,
#                     "stream": False,
#                     "options": {
#                         "temperature": 0.7,
#                         "num_predict": 500,
#                     }
#                 }
#             )
            
#             if self.use_clearml:
#                 generation_time = time.time() - start_time
#                 self.iteration_counter += 1
#                 self.logger.report_scalar(
#                     "timing", 
#                     "generation_time", 
#                     generation_time,
#                     iteration=self.iteration_counter
#                 )
            
#             response.raise_for_status()
#             result = response.json().get('response', '')
            
#             if self.use_clearml:
#                 self.logger.report_text(f"Response: {result[:200]}...")
            
#             return result
            
#         except Exception as e:
#             error_msg = f"Error generating response: {str(e)}"
#             logger.error(error_msg)
#             if self.use_clearml:
#                 self.logger.report_text(f"Error: {str(e)}")
#             return error_msg

# import requests
# import time
# import logging
# import json
# from typing import Dict, Any, Optional

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class LLMHandler:
#     def __init__(self, model_name: str = "llama2", max_retries: int = 3):
#         self.model_name = model_name
#         self.max_retries = max_retries
#         self.base_url = "http://127.0.0.1:11434"
#         self._ensure_model_availability()

#     def _wait_for_ollama(self, timeout: int = 30) -> bool:
#         """Wait for Ollama service to be ready"""
#         start_time = time.time()
#         while time.time() - start_time < timeout:
#             try:
#                 response = requests.get(f"{self.base_url}/api/tags")
#                 if response.status_code == 200:
#                     return True
#             except requests.exceptions.ConnectionError:
#                 logger.info("Waiting for Ollama service...")
#                 time.sleep(2)
#         return False

#     def _ensure_model_availability(self) -> None:
#         """Ensure the model is available"""
#         if not self._wait_for_ollama():
#             raise Exception("Ollama service failed to start")

#         for attempt in range(self.max_retries):
#             try:
#                 logger.info(f"Checking model availability, attempt {attempt + 1}")
                
#                 # Pull the model if needed
#                 logger.info(f"Pulling model {self.model_name}")
#                 pull_response = requests.post(
#                     f"{self.base_url}/api/pull",
#                     json={"name": self.model_name}
#                 )
#                 pull_response.raise_for_status()
#                 logger.info("Model pulled successfully")
#                 return
                
#             except Exception as e:
#                 logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
#                 if attempt < self.max_retries - 1:
#                     time.sleep(5)
#                 else:
#                     raise Exception(f"Failed to initialize Ollama after {self.max_retries} attempts")

#     def generate_response(self, prompt: str, context: Optional[str] = None) -> str:
#         """Generate a response using the LLM"""
#         try:
#             # Format the prompt
#             system_prompt = "You are a helpful assistant with expertise in ROS2 robotics."
#             if context:
#                 full_prompt = f"{system_prompt}\n\nContext: {context}\n\nQuestion: {prompt}\nAnswer:"
#             else:
#                 full_prompt = f"{system_prompt}\n\nQuestion: {prompt}\nAnswer:"

#             logger.info(f"Generating response for prompt: {full_prompt}")
            
#             # Make request to Ollama with stream=True
#             response = requests.post(
#                 f"{self.base_url}/api/generate",
#                 json={
#                     "model": self.model_name,
#                     "prompt": full_prompt,
#                     "stream": False,
#                     "raw": True,
#                     "options": {
#                         "temperature": 0.7,
#                         "num_predict": 500,
#                     }
#                 },
#                 stream=True
#             )
#             response.raise_for_status()

#             # Collect the complete response
#             full_response = ""
#             for line in response.iter_lines():
#                 if line:
#                     try:
#                         # Parse each line as a separate JSON object
#                         json_response = json.loads(line)
#                         if 'response' in json_response:
#                             full_response += json_response['response']
#                     except json.JSONDecodeError as e:
#                         logger.warning(f"Failed to parse line: {line}")
#                         continue

#             logger.info(f"Generated response: {full_response[:200]}...")  # Log first 200 chars
#             return full_response.strip()
            
#         except Exception as e:
#             logger.error(f"Error generating response: {str(e)}")
#             return f"Error generating response: {str(e)}"

#     def get_model_info(self) -> Dict[str, Any]:
#         """Get information about the loaded model"""
#         try:
#             response = requests.get(f"{self.base_url}/api/tags")
#             response.raise_for_status()
#             models = response.json()
#             model_info = next((model for model in models.get('models', []) 
#                              if model['name'] == self.model_name), None)
#             return model_info if model_info else {"error": f"Model {self.model_name} not found"}
#         except Exception as e:
#             logger.error(f"Error getting model info: {str(e)}")
#             return {"error": str(e)}