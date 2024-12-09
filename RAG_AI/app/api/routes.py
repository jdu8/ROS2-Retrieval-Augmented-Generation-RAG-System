# api/routes.py
from fastapi import FastAPI, HTTPException
from gradio import Interface
import gradio as gr

app = FastAPI()

def create_gradio_interface():
    def process_query(query, domain=None):
        # Initialize components
        llm = LLMHandler()
        embedder = EmbeddingGenerator()
        
        # Get relevant context
        query_embedding = embedder.create_embedding(query)
        relevant_docs = embedder.qdrant.search(
            collection_name="ros2_embeddings",
            query_vector=query_embedding,
            limit=5
        )
        
        # Generate response
        context = "\n".join([doc.payload['text'] for doc in relevant_docs])
        response = llm.generate_response(query, context)
        
        return response
    
    interface = gr.Interface(
        fn=process_query,
        inputs=[
            gr.Textbox(label="Question"),
            gr.Dropdown(choices=["ROS2", "Nav2", "MoveIt2", "Gazebo"], label="Domain")
        ],
        outputs=gr.Textbox(label="Answer"),
        title="ROS2 RAG System",
        description="Ask questions about ROS2 robotics development"
    )
    
    return interface