from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import base64
import os
from typing import List, Optional, Iterator
import asyncio
import json
import uuid
import io
import hashlib
import traceback
from dotenv import load_dotenv
import uvicorn

try:
    import pymupdf as fitz  # PyMuPDF
except ImportError:
    print("pymupdf not found, using fitz directly.")
    import fitz

from PIL import Image
import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext, Settings, SimpleDirectoryReader
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import ImageNode, NodeWithScore, MetadataMode, TextNode
from llama_index.core.prompts import PromptTemplate
from llama_index.core.base.response.schema import Response

# Original imports for agent functionality
from google import genai
from google.genai import types
from agno.agent import Agent
from agno.models.google import Gemini
from db.mongodb import connect_to_mongo, close_mongo_connection

load_dotenv()  # Load from current directory first
load_dotenv("../../.env")  # Then load from parent for additional keys

# Set up environment variables for backward compatibility
if not os.getenv("GEMINI_API_KEY") and os.getenv("GOOGLE_API_KEY"):
    os.environ["GEMINI_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Create uploads directory if it doesn't exist
UPLOADS_DIR = "./uploads"
ARTIFACTS_DIR = "./artifacts"
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

app = FastAPI(title="Multimodal Document Processing API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and include schema routes
from routes.schemas import router as schema_router
from routes.extraction import router as extraction_router
from routes.two_way_match import router as two_way_match_router
from routes.auth import router as auth_router
app.include_router(schema_router, prefix="/api")
app.include_router(extraction_router, prefix="/api")
app.include_router(two_way_match_router, prefix="/api")
app.include_router(auth_router, prefix="/api/v1")

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize connections on startup"""
    try:
        await connect_to_mongo()
        print("✅ Application startup completed")
    except Exception as e:
        print(f"❌ Startup failed: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Close connections on shutdown"""
    try:
        await close_mongo_connection()
        print("✅ Application shutdown completed")
    except Exception as e:
        print(f"❌ Shutdown error: {e}")

# Initialize Gemini client
try:
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
except Exception as e:
    print(f"Warning: Gemini client initialization failed: {e}")
    client = None

# Initialize embedding models for multimodal pipeline
embed_model_text = None
embed_model_image = None
storage_context = None

def restore_global_index():
    """Restore global index from existing Qdrant collections"""
    global global_index
    
    try:
        if not storage_context:
            print("❌ Storage context not available for index restoration")
            return False
            
        if global_index is not None:
            print("✅ Global index already exists")
            return True
            
        # Check if collections have data
        print("🔍 Checking Qdrant collections...")
        client_instance = storage_context.vector_store._client
        collections = client_instance.get_collections()
        print(f"Found {len(collections.collections)} collections: {[col.name for col in collections.collections]}")
        
        text_collection_exists = any(col.name == "text_collection" for col in collections.collections)
        image_collection_exists = any(col.name == "image_collection" for col in collections.collections)
        
        print(f"Text collection exists: {text_collection_exists}")
        print(f"Image collection exists: {image_collection_exists}")
        
        if not (text_collection_exists and image_collection_exists):
            print("❌ Required collections don't exist")
            return False
            
        # Check if collections have data
        text_info = client_instance.get_collection("text_collection")
        image_info = client_instance.get_collection("image_collection")
        
        print(f"Text collection points: {text_info.points_count}")
        print(f"Image collection points: {image_info.points_count}")
        
        if text_info.points_count == 0 and image_info.points_count == 0:
            print("❌ Collections exist but have no data")
            return False
            
        print(f"📊 Found existing data: {text_info.points_count} text points, {image_info.points_count} image points")
        
        # Create index from existing vector stores
        print("🔄 Restoring global index from existing Qdrant data...")
        
        # Create a simple index that connects to existing vector stores
        global_index = MultiModalVectorStoreIndex(
            [],  # No documents needed since we're connecting to existing data
            storage_context=storage_context,
        )
        
        print("✅ Global index restored successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to restore global index: {e}")
        import traceback
        traceback.print_exc()
        return False

def initialize_embeddings():
    """Initialize embedding models lazily when needed"""
    global embed_model_text, embed_model_image, storage_context
    
    if embed_model_text is None:
        try:
            print("Initializing text embedding model...")
            embed_model_text = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
            print("✅ Text embedding model loaded")
        except Exception as e:
            print(f"Warning: Text embedding model initialization failed: {e}")
            return False
            
    if embed_model_image is None:
        try:
            print("Initializing image embedding model...")
            embed_model_image = HuggingFaceEmbedding(model_name="sentence-transformers/clip-ViT-B-32")
            Settings.chunk_size = 512
            Settings.embed_model = embed_model_image
            print("✅ Image embedding model loaded")
        except Exception as e:
            print(f"Warning: Image embedding model initialization failed: {e}")
            return False
    
    # Initialize Qdrant client after embeddings are ready
    if storage_context is None:
        try:
            print("Initializing Qdrant vector stores...")
            qdrant_client_instance = qdrant_client.QdrantClient(path="./qdrant_db")
            
            # Ensure collections exist with proper configuration
            from qdrant_client.models import Distance, VectorParams
            
            # Create text collection if it doesn't exist
            try:
                qdrant_client_instance.get_collection("text_collection")
                print("✅ Text collection already exists")
            except Exception:
                print("Creating text collection...")
                qdrant_client_instance.create_collection(
                    collection_name="text_collection",
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE)  # BGE model dimension
                )
                print("✅ Text collection created")
            
            # Create image collection if it doesn't exist  
            try:
                qdrant_client_instance.get_collection("image_collection")
                print("✅ Image collection already exists")
            except Exception:
                print("Creating image collection...")
                qdrant_client_instance.create_collection(
                    collection_name="image_collection", 
                    vectors_config=VectorParams(size=512, distance=Distance.COSINE)  # CLIP model dimension
                )
                print("✅ Image collection created")
            
            # Create text vector store
            text_store = QdrantVectorStore(
                client=qdrant_client_instance,
                collection_name="text_collection",
            )
            
            # Create image vector store  
            image_store = QdrantVectorStore(
                client=qdrant_client_instance,
                collection_name="image_collection",
            )
            
            # Set global embedding models for LlamaIndex
            Settings.embed_model = embed_model_text  # Default to text embeddings
            
            storage_context = StorageContext.from_defaults(
                vector_store=text_store,
                image_store=image_store,
            )
            print("✅ Storage context initialized")
            
            # Try to restore global index from existing data
            restore_global_index()
            
            return True
        except Exception as e:
            print(f"Warning: Storage context initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        # Storage context already exists, try to restore global index
        restore_global_index()
    
    return True

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    filename: Optional[str] = None
    top_k: Optional[int] = 3
    use_llm_for_answer: Optional[bool] = True

class StreamQueryRequest(BaseModel):
    query: str
    filename: Optional[str] = None
    top_k: Optional[int] = 3

class Structure(BaseModel):
    text_response: str = Field(description="Text response from the LLM")
    file_name_used: List[str] = Field(description="List of image file names used", default=[])

# Utility functions for new pipeline
def make_filename(base_name, page, suffix, ext="jpg"):
    """Create deterministic filename for extracted images"""
    return f"{os.path.splitext(base_name)[0]}_p{page}_{suffix}.{ext}"

def extract_images(pdf_path, method="both", dpi=200):
    """
    Extract images from PDF.
    method = "embedded" | "fullpage" | "both"
    Saves images into /artifacts/embedded and /artifacts/fullpage
    """
    pdf_document = fitz.open(pdf_path)
    file_name = os.path.basename(pdf_path)
    
    # Create subdirectories for this specific document
    doc_name = os.path.splitext(file_name)[0]
    doc_artifacts_dir = os.path.join(ARTIFACTS_DIR, doc_name)
    embedded_dir = os.path.join(doc_artifacts_dir, "embedded")
    fullpage_dir = os.path.join(doc_artifacts_dir, "fullpage")
    
    if method in ["embedded", "both"]:
        os.makedirs(embedded_dir, exist_ok=True)
    if method in ["fullpage", "both"]:
        os.makedirs(fullpage_dir, exist_ok=True)
    
    extracted_files = []
    
    for page_num, page in enumerate(pdf_document):
        page_number = page_num + 1
        
        # --- Embedded images ---
        if method in ["embedded", "both"]:
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                filename = make_filename(file_name, page_number, f"emb{img_index+1}", image_ext)
                filepath = os.path.join(embedded_dir, filename)
                
                with open(filepath, "wb") as f:
                    f.write(image_bytes)
                extracted_files.append(filepath)
        
        # --- Full page render ---
        if method in ["fullpage", "both"]:
            zoom = dpi / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_bytes))
            filename = make_filename(file_name, page_number, "page", "jpg")
            page_img_path = os.path.join(fullpage_dir, filename)
            img.convert("RGB").save(page_img_path, "JPEG", quality=90)
            extracted_files.append(page_img_path)
    
    pdf_document.close()
    return extracted_files, doc_artifacts_dir

# Query engine for multimodal RAG
QA_PROMPT_TMPL = """Below we give parsed text and images as context.
Use both the parsed text and images to answer the question.
Write your response in markdown format.
Note: Don't Put Images Used: in the text_response

---------------------
{context_str}
---------------------

Given the context information and not prior knowledge, answer the query. Explain your reasoning based on the text or image, and if there are discrepancies, mention your reasoning for the answer.

Query: {query_str}
Answer: """

QA_PROMPT = PromptTemplate(QA_PROMPT_TMPL)

class MultimodalGeminiEngine(CustomQueryEngine):
    """Multimodal query engine with robust image processing and error handling"""
    
    qa_prompt: PromptTemplate
    retriever: BaseRetriever
    
    def __init__(self, qa_prompt: Optional[PromptTemplate] = None, **kwargs) -> None:
        """Initialize."""
        super().__init__(qa_prompt=qa_prompt or QA_PROMPT, **kwargs)
    
    def _get_image_mime_type(self, image_path: str) -> str:
        """Determine MIME type from file extension."""
        extension = image_path.lower().split('.')[-1]
        mime_types = {
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg', 
            'png': 'image/png',
            'gif': 'image/gif',
            'webp': 'image/webp',
            'bmp': 'image/bmp'
        }
        return mime_types.get(extension, 'image/jpeg')
    
    def _process_image_node(self, image_node: ImageNode) -> Optional[types.Part]:
        """Process a single image node into a GenAI Part."""
        methods = [
            self._try_base64_image,
            self._try_resolve_image,
            self._try_file_path,
            self._try_image_url
        ]
        
        for method in methods:
            try:
                part = method(image_node)
                if part is not None:
                    return part
            except Exception as e:
                continue
        
        print(f"Warning: Could not process ImageNode {image_node.id_}")
        return None
    
    def _try_base64_image(self, image_node: ImageNode) -> Optional[types.Part]:
        """Try to get image from base64 encoded data."""
        try:
            if hasattr(image_node, 'image') and image_node.image:
                image_bytes = base64.b64decode(image_node.image)
                return types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
            return None
        except Exception as e:
            return None
    
    def _try_resolve_image(self, image_node: ImageNode) -> Optional[types.Part]:
        """Try to get image using resolve_image method."""
        if hasattr(image_node, 'resolve_image'):
            image_buffer = image_node.resolve_image()
            image_bytes = image_buffer.getvalue()
            return types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
        return None
    
    def _try_file_path(self, image_node: ImageNode) -> Optional[types.Part]:
        """Try to get image from file path."""
        if hasattr(image_node, 'image_path') and image_node.image_path:
            with open(image_node.image_path, 'rb') as f:
                image_bytes = f.read()
            mime_type = self._get_image_mime_type(image_node.image_path)
            return types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        return None
    
    def _try_image_url(self, image_node: ImageNode) -> Optional[types.Part]:
        """Try to get image from URL."""
        if hasattr(image_node, 'image_url') and image_node.image_url:
            return types.Part.from_uri(
                file_uri=image_node.image_url,
                mime_type="image/jpeg"
            )
        return None
    
    def custom_query(self, query_str: str):
        """Execute the query with robust image processing - exact implementation from PDF."""
        # Retrieve nodes
        nodes = self.retriever.retrieve(query_str)
        img_nodes = [n for n in nodes if isinstance(n.node, ImageNode)]
        text_nodes = [n for n in nodes if isinstance(n.node, TextNode)]
        
        print(f"Image Node : {len(img_nodes)}")
        print(f"Text Node : {len(text_nodes)}")
        # Create context string
        context_str = "\n\n".join(
            [r.get_content(metadata_mode=MetadataMode.LLM) for r in nodes]
        )
        fmt_prompt = self.qa_prompt.format(context_str=context_str, query_str=query_str)
        
        # Prepare content parts
        content_parts = [fmt_prompt]
        
        # Process image nodes
        successful_images = 0
        for img_node in img_nodes:
            image_part = self._process_image_node(img_node.node)
            if image_part:
                content_parts.append(img_node.get_content(metadata_mode=MetadataMode.LLM))
                content_parts.append(image_part)
                successful_images += 1
        
        print(f"Successfully processed {successful_images}/{len(img_nodes)} images")
        
        try:
            # Generate content with structured output
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=content_parts,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": Structure,
                }
            )
            structured_response = response.parsed
            full_response = f"{structured_response.text_response}"
        except Exception as e:
            print(f"Structured output failed: {e}")
            # Fallback to regular response
            try:
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=content_parts,
                )
                full_response = response.text
                structured_response = None
            except Exception as fallback_error:
                print(f"Fallback failed: {fallback_error}")
                full_response = f"Error: {str(e)}"
                structured_response = None
        
        return Response(
            response=full_response,
            source_nodes=nodes,
            metadata={
                "text_nodes": text_nodes,
                "image_nodes": img_nodes,
                "successful_images": successful_images,
                "total_images": len(img_nodes),
                "structured_response": structured_response
            }
        )

# Global index variable
global_index = None

# Initialize agent for legacy compatibility
load_dotenv("./.env")
api = os.getenv("GOOGLE_API_KEY")

idp_agent = Agent(
    name="IDP AGENT",
    model=Gemini(id="gemini-2.0-flash", api_key=api),
    tools=[],
    add_history_to_messages=True,
    num_history_runs=2,
    instructions=[
        "You are an IDP Agent i.e. Intelligent Document Processing Agent. Working for InnuxAI",
        "Use the PDF conversion tools to extract and analyze information from documents.",
        "Handle page ranges or specific pages as specified by the user.",
        "If rate limiting occurs, inform the user about partial results.",
        "Always save the output to a file if specified, and return a confirmation message.",
        "For queries, parse the PDF path, range/specific pages, and output file from the user input.",
        "When analyzing documents, provide structured responses with clear reasoning."
    ],
    show_tool_calls=True,
    markdown=True
)

two_way_match_agent = Agent(
    name="Two Way Match Agent",
    model=Gemini(id="gemini-2.0-flash", api_key=api),
    tools=[],
    add_history_to_messages=True,
    num_history_runs=2,
    instructions=[
        "You are a Two Way Match Agent. Working for InnuxAI",
        "You will be given 2 json files. One is the PO (Purchase Order) and the other is the INV (Invoice)",
        "Your task is to match each line item in the INV to the corresponding line item in the PO",
        "If for an item in INV you cannot find a match in the PO, you should mark it as 'No Match Found'",
        "As the line items name will not be an exact match, use your reasoning to find the best match with your confidence score.",
        "return a json array with the following fields: item in INV, matched item in PO, confidence score (0-100), reasoning",
        "When analyzing, provide structured responses with clear reasoning.",
        "If rate limiting occurs, inform the user about it.",
    ],
    show_tool_calls=True,
    markdown=True
)

# API Endpoints
@app.get("/")
async def root():
    return {"message": "Multimodal Document Processing API", "status": "running"}

@app.post("/reindex-documents")
async def reindex_documents():
    """Reindex all existing documents in the uploads folder"""
    global global_index
    
    try:
        # Initialize embeddings if not already done
        if not initialize_embeddings():
            raise HTTPException(status_code=500, detail="Failed to initialize embedding models")
        
        if not storage_context:
            raise HTTPException(status_code=500, detail="Storage context not available")
        
        # Get all PDF files from uploads directory
        if not os.path.exists(UPLOADS_DIR):
            return {"message": "No uploads directory found", "processed": 0}
        
        pdf_files = [f for f in os.listdir(UPLOADS_DIR) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            return {"message": "No PDF files found to reindex", "processed": 0}
        
        processed_docs = 0
        all_documents = []
        
        for filename in pdf_files:
            try:
                file_path = os.path.join(UPLOADS_DIR, filename)
                print(f"Reindexing: {filename}")
                
                # Extract images using new pipeline
                extracted_files, doc_artifacts_dir = extract_images(file_path, method="both")
                print(f"Extracted {len(extracted_files)} images for {filename}")
                
                # Load the original PDF document
                pdf_documents = SimpleDirectoryReader(
                    input_files=[file_path]
                ).load_data()
                
                documents = pdf_documents.copy()
                
                # Load extracted images if they exist
                if os.path.exists(doc_artifacts_dir):
                    try:
                        artifacts_documents = SimpleDirectoryReader(
                            input_dir=doc_artifacts_dir,
                            recursive=True
                        ).load_data()
                        documents.extend(artifacts_documents)
                        print(f"Loaded {len(artifacts_documents)} image documents for {filename}")
                    except Exception as e:
                        print(f"Warning: Failed to load image documents for {filename}: {e}")
                
                all_documents.extend(documents)
                processed_docs += 1
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
        
        if not all_documents:
            return {"message": "No documents could be processed", "processed": 0}
        
        # Create new multimodal index with all documents
        print(f"Creating multimodal index with {len(all_documents)} total documents...")
        global_index = MultiModalVectorStoreIndex.from_documents(
            all_documents,
            storage_context=storage_context,
            show_progress=True,
        )
        print("✅ Multimodal index created successfully")
        
        return {
            "message": f"Successfully reindexed {processed_docs} PDF files",
            "processed": processed_docs,
            "total_documents": len(all_documents),
            "status": "success"
        }
        
    except Exception as e:
        print(f"Reindexing error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug-qdrant")
async def debug_qdrant():
    """Debug Qdrant connection and data"""
    try:
        if not storage_context:
            return {"error": "Storage context not available"}
        
        client_instance = storage_context.vector_store._client
        collections = client_instance.get_collections()
        
        result = {
            "storage_context_available": storage_context is not None,
            "client_available": client_instance is not None,
            "collections_count": len(collections.collections),
            "collections": []
        }
        
        for col in collections.collections:
            try:
                info = client_instance.get_collection(col.name)
                result["collections"].append({
                    "name": col.name,
                    "points": info.points_count,
                    "vector_size": info.config.params.vectors.size if hasattr(info.config.params, 'vectors') else 'unknown'
                })
            except Exception as e:
                result["collections"].append({
                    "name": col.name,
                    "error": str(e)
                })
        
        return result
    except Exception as e:
        return {"error": str(e), "traceback": str(e.__traceback__)}

@app.post("/restore-index")
async def restore_index():
    """Manually restore global index from existing Qdrant data"""
    try:
        if not initialize_embeddings():
            raise HTTPException(status_code=500, detail="Failed to initialize embeddings")
        
        success = restore_global_index()
        if success:
            return {
                "message": "Global index restored successfully",
                "global_index_available": global_index is not None
            }
        else:
            return {
                "message": "Failed to restore global index - no existing data found",
                "global_index_available": global_index is not None
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Index restoration failed: {str(e)}")

@app.post("/initialize-system")
async def initialize_system():
    """Manually initialize embeddings and storage context"""
    try:
        success = initialize_embeddings()
        if success:
            return {
                "message": "System initialized successfully",
                "embeddings_initialized": embed_model_text is not None and embed_model_image is not None,
                "storage_context_available": storage_context is not None
            }
        else:
            return {
                "message": "System initialization failed",
                "embeddings_initialized": embed_model_text is not None and embed_model_image is not None,
                "storage_context_available": storage_context is not None
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Initialization failed: {str(e)}")

@app.get("/qdrant-status")
async def qdrant_status():
    """Check Qdrant database status and collections"""
    try:
        if not storage_context:
            return {"error": "Storage context not initialized"}
        
        client_instance = storage_context.vector_store._client
        collections = client_instance.get_collections()
        
        status = {
            "database_path": "./qdrant_db",
            "total_collections": len(collections.collections),
            "collections": []
        }
        
        for collection in collections.collections:
            try:
                collection_info = client_instance.get_collection(collection.name)
                status["collections"].append({
                    "name": collection.name,
                    "points_count": collection_info.points_count,
                    "vector_size": collection_info.config.params.vectors.size,
                    "distance": collection_info.config.params.vectors.distance.value
                })
            except Exception as e:
                status["collections"].append({
                    "name": collection.name,
                    "error": str(e)
                })
        
        return status
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
async def health_check():
    try:
        # Check if qdrant collections exist
        qdrant_info = {}
        if storage_context and storage_context.vector_store:
            try:
                client_instance = storage_context.vector_store._client
                collections = client_instance.get_collections()
                qdrant_info = {
                    "collections": [col.name for col in collections.collections],
                    "text_collection_exists": any(col.name == "text_collection" for col in collections.collections),
                    "image_collection_exists": any(col.name == "image_collection" for col in collections.collections)
                }
                
                # Get collection info if they exist
                for collection_name in ["text_collection", "image_collection"]:
                    try:
                        collection_info = client_instance.get_collection(collection_name)
                        qdrant_info[f"{collection_name}_points"] = collection_info.points_count
                        qdrant_info[f"{collection_name}_vector_size"] = collection_info.config.params.vectors.size
                    except Exception:
                        qdrant_info[f"{collection_name}_points"] = 0
                        
            except Exception as e:
                qdrant_info = {"error": str(e)}
        
        return {
            "gemini_available": client is not None,
            "embeddings_initialized": embed_model_text is not None and embed_model_image is not None,
            "storage_context_available": storage_context is not None,
            "global_index_available": global_index is not None,
            "qdrant_info": qdrant_info
        }
    except Exception as e:
        return {
            "gemini_available": client is not None,
            "embeddings_initialized": embed_model_text is not None and embed_model_image is not None,
            "storage_context_available": storage_context is not None,
            "global_index_available": global_index is not None,
            "error": str(e)
        }

@app.post("/upload-document/")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process document using new multimodal pipeline"""
    global global_index
    
    try:
        # Initialize embeddings if not already done
        if not initialize_embeddings():
            raise HTTPException(status_code=500, detail="Failed to initialize embedding models")
        
        if not storage_context:
            raise HTTPException(status_code=500, detail="Storage context not available")
        
        # Read file content
        file_bytes = await file.read()
        file_hash = hashlib.md5(file_bytes).hexdigest()
        doc_id = f"{file_hash[:8]}"
        
        # Save the original file
        file_path = os.path.join(UPLOADS_DIR, f"{doc_id}_{file.filename}")
        with open(file_path, "wb") as f:
            f.write(file_bytes)
        
        if file.content_type == "application/pdf":
            print(f"Processing PDF: {file.filename}")
            
            # Extract images using new pipeline
            extracted_files, doc_artifacts_dir = extract_images(file_path, method="both")
            print(f"Extracted {len(extracted_files)} images to {doc_artifacts_dir}")
            
            # Load the original PDF document
            print("Loading PDF text content...")
            pdf_documents = SimpleDirectoryReader(
                input_files=[file_path]
            ).load_data()
            
            documents = pdf_documents.copy()
            
            # Load extracted images if they exist
            if os.path.exists(doc_artifacts_dir):
                print("Loading extracted images...")
                try:
                    artifacts_documents = SimpleDirectoryReader(
                        input_dir=doc_artifacts_dir,
                        recursive=True
                    ).load_data()
                    documents.extend(artifacts_documents)
                    print(f"Loaded {len(artifacts_documents)} image documents")
                except Exception as e:
                    print(f"Warning: Failed to load image documents: {e}")
            
            if not documents:
                raise HTTPException(status_code=500, detail="No documents found to process")
            
            print(f"Total documents to index: {len(documents)}")
            
            # Create or update multimodal index
            if global_index is None:
                print("Creating new multimodal index...")
                global_index = MultiModalVectorStoreIndex.from_documents(
                    documents,
                    storage_context=storage_context,
                    show_progress=True,
                )
                print("✅ Multimodal index created")
            else:
                print("Adding documents to existing index...")
                # Add new documents to existing index
                for i, doc in enumerate(documents):
                    print(f"Indexing document {i+1}/{len(documents)}")
                    global_index.insert(doc)
                print("✅ Documents added to existing index")
            
            return {
                "message": "Document uploaded and processed successfully",
                "doc_id": doc_id,
                "extracted_images": len(extracted_files),
                "total_documents": len(documents),
                "status": "success"
            }
        else:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type: {file.content_type}. Only PDF files are supported."
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query-documents/")
async def query_documents(request: QueryRequest):
    """Query documents using new multimodal pipeline"""
    global global_index
    
    try:
        if not global_index:
            raise HTTPException(status_code=400, detail="No documents uploaded yet. Please upload documents first.")
        
        if not client:
            raise HTTPException(status_code=500, detail="Gemini client not available")
        
        # Ensure embeddings are initialized
        if not initialize_embeddings():
            raise HTTPException(status_code=500, detail="Failed to initialize embedding models")
        
        # Create retriever
        retriever = global_index.as_retriever(
            similarity_top_k=request.top_k * 7,  # Get more text nodes
            image_similarity_top_k=request.top_k * 2  # Increased to get more image diversity
        )
        
        # Create multimodal engine
        engine = MultimodalGeminiEngine(retriever=retriever)
        
        # Execute query
        response = engine.custom_query(request.query)
        
        # Extract source information including images
        sources = []
        image_sources = []
        
        for node in response.source_nodes:
            if isinstance(node.node, ImageNode):
                image_info = {
                    "type": "image",
                    "content": node.get_content(metadata_mode=MetadataMode.LLM)[:200] + "..." if len(node.get_content(metadata_mode=MetadataMode.LLM)) > 200 else node.get_content(metadata_mode=MetadataMode.LLM),
                    "metadata": node.node.metadata if hasattr(node.node, 'metadata') else {},
                }
                
                # Add image path if available for frontend rendering
                if hasattr(node.node, 'image_path'):
                    image_info["image_path"] = node.node.image_path
                    image_info["filename"] = os.path.basename(node.node.image_path)
                
                image_sources.append(image_info)
            else:
                sources.append({
                    "type": "text",
                    "content": node.get_content(metadata_mode=MetadataMode.LLM)[:300] + "..." if len(node.get_content(metadata_mode=MetadataMode.LLM)) > 300 else node.get_content(metadata_mode=MetadataMode.LLM),
                    "metadata": node.node.metadata if hasattr(node.node, 'metadata') else {},
                })
        
        return {
            "query": request.query,
            "answer": response.response,
            "sources": sources,
            "image_sources": image_sources,
            "metadata": response.metadata,
            "method": "multimodal_rag"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query-documents-stream")
async def query_documents_stream(request: StreamQueryRequest):
    """Stream multimodal query responses"""
    global global_index
    
    async def generate_stream():
        try:
            if not global_index:
                yield f"data: {json.dumps({'type': 'error', 'content': 'No documents uploaded yet'})}\n\n"
                return
            
            # Initialize embeddings if needed
            if not initialize_embeddings():
                yield f"data: {json.dumps({'type': 'error', 'content': 'Failed to initialize embedding models'})}\n\n"
                return
            
            yield f"data: {json.dumps({'type': 'status', 'content': 'Searching documents...'})}\n\n"
            
            # Create retriever with exact values from PDF
            retriever = global_index.as_retriever(
                similarity_top_k=70,  # Exact value from PDF
                image_similarity_top_k=10  # Exact value from PDF
            )
            
            # Create multimodal engine
            engine = MultimodalGeminiEngine(retriever=retriever)
            
            yield f"data: {json.dumps({'type': 'status', 'content': 'Processing multimodal content...'})}\n\n"
            
            # Execute query
            response = engine.custom_query(request.query)
            
            # Send sources first
            sources = []
            image_sources = []
            
            # Get file names used from metadata (this is the manually tracked list)
            files_used_in_response = []
            if hasattr(response, 'metadata') and response.metadata:
                # First try structured response with correct field name
                structured_response = response.metadata.get('structured_response')
                if structured_response and hasattr(structured_response, 'file_name_used'):
                    files_used_in_response = structured_response.file_name_used
                else:
                    # Fallback to manually tracked image_files_used
                    files_used_in_response = response.metadata.get('image_files_used', [])
            
            for node in response.source_nodes:
                if isinstance(node.node, ImageNode):
                    image_info = {
                        "type": "image",
                        "content": node.get_content(metadata_mode=MetadataMode.LLM)[:200] + "...",
                        "metadata": node.node.metadata if hasattr(node.node, 'metadata') else {},
                    }
                    if hasattr(node.node, 'image_path'):
                        image_info["image_path"] = node.node.image_path
                        image_info["filename"] = os.path.basename(node.node.image_path)
                        # Mark if this image was used in the response
                        if image_info["filename"] in files_used_in_response:
                            image_info["used_in_response"] = True
                    image_sources.append(image_info)
                else:
                    sources.append({
                        "type": "text", 
                        "content": node.get_content(metadata_mode=MetadataMode.LLM)[:300] + "...",
                        "metadata": node.node.metadata if hasattr(node.node, 'metadata') else {},
                    })
            
            # Add images used in response to sources if not already included
            for filename in files_used_in_response:
                # Check if this filename is already in image_sources
                already_included = any(img.get("filename") == filename for img in image_sources)
                if not already_included:
                    # Try to find the full path for this filename
                    for node in response.source_nodes:
                        if isinstance(node.node, ImageNode) and hasattr(node.node, 'image_path'):
                            if os.path.basename(node.node.image_path) == filename:
                                image_sources.append({
                                    "type": "image",
                                    "content": f"Image used in response: {filename}",
                                    "metadata": node.node.metadata if hasattr(node.node, 'metadata') else {},
                                    "image_path": node.node.image_path,
                                    "filename": filename,
                                    "used_in_response": True
                                })
                                break
            
            yield f"data: {json.dumps({'type': 'sources', 'content': {'text_sources': sources, 'image_sources': image_sources, 'files_used_in_response': files_used_in_response}})}\n\n"
            
            # Stream the response
            yield f"data: {json.dumps({'type': 'response', 'content': response.response})}\n\n"
            yield f"data: {json.dumps({'type': 'complete', 'content': 'Response completed'})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': f'Error: {str(e)}'})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive", 
            "Content-Type": "text/event-stream",
        }
    )

@app.get("/get-image/{image_path:path}")
async def get_image(image_path: str):
    """Serve extracted images for frontend rendering"""
    try:
        # Construct full path
        full_path = os.path.join(ARTIFACTS_DIR, image_path)
        
        if not os.path.exists(full_path):
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Determine media type
        ext = os.path.splitext(full_path)[1].lower()
        media_type_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.webp': 'image/webp'
        }
        media_type = media_type_map.get(ext, 'image/jpeg')
        
        return FileResponse(
            path=full_path,
            media_type=media_type,
            filename=os.path.basename(full_path)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list-documents/")
async def list_documents():
    """List all uploaded documents"""
    try:
        uploaded_files = []
        
        if os.path.exists(UPLOADS_DIR):
            for filename in os.listdir(UPLOADS_DIR):
                filepath = os.path.join(UPLOADS_DIR, filename)
                if os.path.isfile(filepath):
                    file_stats = os.stat(filepath)
                    
                    # Determine content type from file extension
                    content_type = "application/pdf" if filename.lower().endswith('.pdf') else "application/msword"
                    
                    uploaded_files.append({
                        "filename": filename,
                        "content_type": content_type,
                        "file_size": file_stats.st_size,
                        "created_at": file_stats.st_mtime
                    })
        
        return {
            "total_documents": len(uploaded_files),
            "documents": uploaded_files
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/view-document/{filename}")
async def view_document(filename: str):
    """Get the original document file for viewing"""
    try:
        file_path = os.path.join(UPLOADS_DIR, filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Document not found")
        
        # For PDFs, set Content-Disposition to inline to display in browser
        response = FileResponse(
            path=file_path,
            media_type="application/pdf",
            filename=filename
        )
        response.headers["Content-Disposition"] = f'inline; filename="{filename}"'
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete-document/{filename}")
async def delete_document(filename: str):
    """Delete a specific document and its artifacts"""
    try:
        file_path = os.path.join(UPLOADS_DIR, filename)
        
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Also remove associated artifacts
        doc_name = os.path.splitext(filename)[0]
        artifacts_path = os.path.join(ARTIFACTS_DIR, doc_name)
        
        if os.path.exists(artifacts_path):
            import shutil
            shutil.rmtree(artifacts_path)
        
        return {
            "message": f"Document {filename} and its artifacts deleted successfully",
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/validate-file")
async def validate_file(file: UploadFile = File(...)):
    """
    Validate uploaded file for size and page count limits
    """
    try:
        # Check file size (5MB limit)
        MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB in bytes
        file_content = await file.read()
        file_size = len(file_content)
        
        if file_size > MAX_FILE_SIZE:
            return {
                "valid": False,
                "error": "file_size",
                "message": f"File size ({file_size / 1024 / 1024:.1f}MB) exceeds the 5MB limit",
                "file_size_mb": round(file_size / 1024 / 1024, 1),
                "max_size_mb": 5
            }
        
        # Check if it's a PDF and count pages
        if file.content_type == "application/pdf" or file.filename.lower().endswith('.pdf'):
            try:
                import io
                import   fitz  # PyMuPDF
                
                # Create a file-like object from the content
                pdf_stream = io.BytesIO(file_content)
                pdf_document = fitz.open(stream=pdf_stream, filetype="pdf")
                page_count = pdf_document.page_count
                pdf_document.close()
                
                MAX_PAGES = 100
                if page_count > MAX_PAGES:
                    return {
                        "valid": False,
                        "error": "page_count",
                        "message": f"PDF has {page_count} pages, which exceeds the {MAX_PAGES} page limit",
                        "page_count": page_count,
                        "max_pages": MAX_PAGES
                    }
                
                return {
                    "valid": True,
                    "message": "File validation successful",
                    "file_size_mb": round(file_size / 1024 / 1024, 1),
                    "page_count": page_count
                }
                
            except Exception as e:
                return {
                    "valid": False,
                    "error": "pdf_processing",
                    "message": f"Error processing PDF: {str(e)}"
                }
        else:
            # For non-PDF files, just check size
            return {
                "valid": True,
                "message": "File validation successful",
                "file_size_mb": round(file_size / 1024 / 1024, 1),
                "page_count": None
            }
            
    except Exception as e:
        return {
            "valid": False,
            "error": "validation_error",
            "message": f"File validation error: {str(e)}"
        }

@app.get("/dashboard/metrics")
async def get_dashboard_metrics():
    """
    Get dashboard metrics including document counts, processing stats, and daily metrics
    """
    try:
        from db.sqlite_db import db
        from datetime import datetime, date
        import os
        
        # Get document types count by file extension
        doc_type_counts = {}
        upload_files = os.listdir(UPLOADS_DIR) if os.path.exists(UPLOADS_DIR) else []
        
        for filename in upload_files:
            if filename.startswith('.'):  # Skip hidden files
                continue
            ext = os.path.splitext(filename)[1].lower()
            if ext == '.pdf':
                doc_type_counts['PDF'] = doc_type_counts.get('PDF', 0) + 1
            elif ext in ['.doc', '.docx']:
                doc_type_counts['Word'] = doc_type_counts.get('Word', 0) + 1
            elif ext in ['.txt']:
                doc_type_counts['Text'] = doc_type_counts.get('Text', 0) + 1
            elif ext in ['.jpg', '.jpeg', '.png', '.gif']:
                doc_type_counts['Image'] = doc_type_counts.get('Image', 0) + 1
            else:
                doc_type_counts['Other'] = doc_type_counts.get('Other', 0) + 1
        
        # Get today's date
        today = date.today().isoformat()
        
        # Get extraction stats for today
        extractions = db.get_extractions()
        today_extractions = [e for e in extractions if e['created_at'].startswith(today)]
        today_count = len(today_extractions)
        today_success = len([e for e in today_extractions if e.get('is_approved', False)])
        today_failed = today_count - today_success
        
        # Get total document count
        total_docs = len(upload_files)
        
        # Get total schema count
        schemas = db.get_schemas()
        total_schemas = len(schemas)
        
        # Get processing stats for the last 30 days for chart
        from collections import defaultdict
        from datetime import datetime, timedelta
        
        daily_stats = defaultdict(lambda: {'processed': 0, 'successful': 0, 'failed': 0})
        
        # Calculate stats for last 30 days
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=29)
        
        for extraction in extractions:
            try:
                extraction_date = datetime.fromisoformat(extraction['created_at'].replace('Z', '+00:00')).date()
                if start_date <= extraction_date <= end_date:
                    date_str = extraction_date.isoformat()
                    daily_stats[date_str]['processed'] += 1
                    if extraction.get('is_approved', False):
                        daily_stats[date_str]['successful'] += 1
                    else:
                        daily_stats[date_str]['failed'] += 1
            except Exception as e:
                print(f"Error parsing date {extraction.get('created_at')}: {e}")
                continue
        
        # Fill in missing dates with zeros
        current_date = start_date
        chart_data = []
        while current_date <= end_date:
            date_str = current_date.isoformat()
            stats = daily_stats[date_str]
            chart_data.append({
                'date': date_str,
                'processed': stats['processed'],
                'successful': stats['successful'],
                'failed': stats['failed']
            })
            current_date += timedelta(days=1)
        
        return {
            'doc_type_counts': doc_type_counts,
            'today_stats': {
                'total': today_count,
                'successful': today_success,
                'failed': today_failed
            },
            'overview_stats': {
                'total_documents': total_docs,
                'total_schemas': total_schemas,
                'total_extractions': len(extractions),
                'success_rate': round((sum([1 for e in extractions if e.get('is_approved', False)]) / max(len(extractions), 1)) * 100, 1)
            },
            'chart_data': chart_data
        }
        
    except Exception as e:
        print(f"Error getting dashboard metrics: {e}")
        import traceback
        traceback.print_exc()
        return {
            'doc_type_counts': {},
            'today_stats': {'total': 0, 'successful': 0, 'failed': 0},
            'overview_stats': {'total_documents': 0, 'total_schemas': 0, 'total_extractions': 0, 'success_rate': 0},
            'chart_data': []
        }

# Legacy endpoint for backward compatibility
@app.post("/query-documents1/")
async def query_documents_legacy(request: QueryRequest):
    """Legacy query endpoint using IDP agent"""
    global idp_agent
    
    try:
        # Use IDP agent to process query
        response = await idp_agent.arun(f"Query: {request.query}")
        
        # Extract only serializable content from agent response
        if hasattr(response, 'content') and response.content:
            answer = response.content
        elif isinstance(response, str):
            answer = response
        else:
            answer = str(response)
        
        return {
            "query": request.query,
            "answer": answer,
            "sources": [],
            "method": "legacy_agent"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
