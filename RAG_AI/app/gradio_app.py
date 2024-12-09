# app/gradio_app.py
import gradio as gr
import os
import time
from clearml import Task
from models.llm import LLMHandler
from pipeline_orchestrator import PipelineOrchestrator
import logging
from typing import Tuple, Optional
from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ROS2Assistant:
    def __init__(self):
        # Use existing task if available
        self.task = Task.current_task()
        if not self.task:
            timestamp = int(time.time())
            self.task = Task.create(
                project_name="ROS2-RAG",
                task_name=f"Pipeline-Execution-{timestamp}",
                task_type="data_processing"  # Changed from 'pipeline' to 'data_processing'
            )

        # Initialize components
        self.llm = LLMHandler()
        
        # Initialize orchestrator with environment variables
        mongodb_uri = os.getenv('MONGODB_URI', 'mongodb://mongodb:27017')
        qdrant_host = os.getenv('QDRANT_HOST', 'qdrant')
        self.orchestrator = PipelineOrchestrator(mongodb_uri, qdrant_host)
        
        # Predefined questions for navigation tasks
        self.navigation_questions = [
            "Tell me how can I navigate to a specific pose- include replanning aspects in your answer.",
            "Can you provide me with code for this task of navigating to specific pose in ROS2?"
        ]

    def load_model(self, model_id: str) -> str:
        """Load a model from Hugging Face Hub"""
        try:
            logger.info(f"Loading model: {model_id}")
            model_path = snapshot_download(
                repo_id=model_id,
                use_auth_token=os.getenv('HF_TOKEN')
            )
            return f"Successfully loaded model: {model_id}"
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return f"Error loading model: {str(e)}"

    def generate_response(
        self,
        question: str,
        use_predefined: bool,
        custom_question: Optional[str],
        model_id: Optional[str]
    ) -> Tuple[str, str]:
        """Generate response for the question"""
        try:
            # Determine the actual question to use
            actual_question = question if use_predefined else custom_question
            if not actual_question:
                return "Please select a question or enter your own.", ""

            # Load custom model if specified
            if model_id:
                load_result = self.load_model(model_id)
                if "Error" in load_result:
                    return load_result, ""

            # Get relevant context from vector store
            context = self.orchestrator.search_navigation_content(actual_question)
            context_text = "\n".join([doc['text'] for doc in context])

            # Generate response
            response = self.llm.generate_response(actual_question, context=context_text)

            # Format sources
            sources = "\n\nSources:\n" + "\n".join([
                f"- {doc['source']}" for doc in context
            ]) if context else ""

            return response, sources

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: {str(e)}", ""

    def create_interface(self) -> gr.Blocks:
        with gr.Blocks(title="ROS2 Navigation Assistant") as interface:
            with gr.Row():
                with gr.Column(scale=2):
                    # Question selection
                    use_predefined = gr.Checkbox(label="Use predefined questions", value=True)
                    predefined_question = gr.Dropdown(
                        choices=self.navigation_questions,
                        label="Select a predefined question",
                        interactive=True
                    )
                    custom_question = gr.Textbox(
                        label="Or type your own question",
                        placeholder="Enter your question about ROS2 navigation...",
                        lines=3,
                        interactive=True,
                        visible=False
                    )

                with gr.Column(scale=1):
                    # Model selection
                    model_source = gr.Radio(
                        choices=["Ollama", "HuggingFace"],
                        label="Select Model Source",
                        value="Ollama"
                    )
                    model_id = gr.Textbox(
                        label="HuggingFace Model ID",
                        value="riyampatel2001/llama_ros2",  # Your model
                        visible=False
                    )
                    submit_btn = gr.Button("Get Answer", variant="primary")

            # Output areas
            response_box = gr.Markdown(label="Response")
            sources_box = gr.Markdown(label="Sources")

            # Show/hide model ID based on source selection
            model_source.change(
                fn=lambda x: gr.update(visible=x == "HuggingFace"),
                inputs=[model_source],
                outputs=[model_id]
            )

            # Handle question type visibility
            use_predefined.change(
                fn=lambda x: [gr.update(visible=x), gr.update(visible=not x)],
                inputs=[use_predefined],
                outputs=[predefined_question, custom_question]
            )

            # Handle submission
            submit_btn.click(
                fn=self.handle_query,
                inputs=[
                    predefined_question,
                    use_predefined,
                    custom_question,
                    model_source,
                    model_id
                ],
                outputs=[response_box, sources_box]
            )

        return interface

    def handle_query(
        self,
        predefined: str,
        use_predefined: bool,
        custom: str,
        model_source: str,
        model_id: str
    ) -> Tuple[str, str]:
        try:
            question = predefined if use_predefined else custom
            if not question:
                return "Please select a question or enter your own.", ""

            # Initialize appropriate model
            if model_source == "HuggingFace" and model_id:
                self.llm = LLMHandler(model_name=model_id, model_source="huggingface")
            else:
                self.llm = LLMHandler(model_name="llama2", model_source="ollama")

            # Get context and generate response
            context = self.orchestrator.search_navigation_content(question)
            context_text = "\n".join([doc['text'] for doc in context])
            response = self.llm.generate_response(question, context=context_text)

            # Format sources
            sources = "\n\nSources:\n" + "\n".join([
                f"- {doc['source']}" for doc in context
            ]) if context else ""

            return response, sources

        except Exception as e:
            logger.error(f"Error handling query: {e}")
            return f"Error: {str(e)}", ""

def main():
    # Create and launch the interface
    assistant = ROS2Assistant()
    interface = assistant.create_interface()
    
    # Launch the interface
    interface.launch(
        server_name="0.0.0.0",
        server_port=8000,
        share=True
    )

if __name__ == "__main__":
    main()