from fastapi import APIRouter, File, UploadFile, HTTPException, status, Form
from typing import List, Optional
import os
import uuid
import shutil
from models.extraction_schema import (
    ExtractionRequest,
    ExtractionResponse,
    DataLibraryEntry,
    DataLibraryResponse,
    UpdateExtractionRequest
)
from db.sqlite_db import db
from services.extraction_service import extraction_service
import asyncio

router = APIRouter(prefix="/extraction", tags=["field-extraction"])

@router.post("/json-extraction", response_model=ExtractionResponse)
async def extract_json_from_pdf(
    file: UploadFile = File(...),
    schema_id: int = Form(...)
):
    """Extract JSON fields from PDF using specified schema"""
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PDF files are supported"
            )
        
        # Validate schema exists
        schema = db.get_schema_by_id(schema_id)
        if not schema:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Schema with ID {schema_id} not found"
            )
        
        # Create uploads directory if it doesn't exist
        uploads_dir = "./uploads"
        os.makedirs(uploads_dir, exist_ok=True)
        
        # Save uploaded file with unique name
        file_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename)[1]
        temp_filename = f"{file_id}{file_extension}"
        temp_file_path = os.path.join(uploads_dir, temp_filename)
        
        # Save file to disk
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        try:
            # Extract fields using the service
            result = await extraction_service.extract_fields_from_pdf(temp_file_path, schema_id)
            
            # Save extraction to data library
            extraction_id = db.save_extraction(
                schema_id=schema_id,
                filename=file.filename,
                pdf_path=temp_file_path,
                extracted_data=result["extracted_data"]
            )
            
            return ExtractionResponse(
                extraction_id=extraction_id,
                extracted_data=result["extracted_data"],
                confidence_scores=result.get("confidence_scores"),
                processing_time=result.get("processing_time")
            )
            
        except Exception as e:
            # Clean up temp file if extraction fails
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            raise e
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to extract fields: {str(e)}"
        )

@router.get("/data-library", response_model=DataLibraryResponse)
async def get_data_library(schema_id: Optional[int] = None):
    """Get all extractions from data library, optionally filtered by schema"""
    try:
        extractions = db.get_extractions(schema_id)
        
        entries = [DataLibraryEntry(**extraction) for extraction in extractions]
        
        return DataLibraryResponse(
            entries=entries,
            total=len(entries)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve data library: {str(e)}"
        )

@router.get("/data-library/{extraction_id}", response_model=DataLibraryEntry)
async def get_extraction(extraction_id: int):
    """Get a specific extraction by ID"""
    try:
        extraction = db.get_extraction_by_id(extraction_id)
        if not extraction:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Extraction with ID {extraction_id} not found"
            )
        
        return DataLibraryEntry(**extraction)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve extraction: {str(e)}"
        )

@router.put("/data-library/{extraction_id}", response_model=DataLibraryEntry)
async def update_extraction(extraction_id: int, update_request: UpdateExtractionRequest):
    """Update extraction data and approval status"""
    try:
        # Check if extraction exists
        existing_extraction = db.get_extraction_by_id(extraction_id)
        if not existing_extraction:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Extraction with ID {extraction_id} not found"
            )
        
        # Update extraction
        success = db.update_extraction(
            extraction_id=extraction_id,
            extracted_data=update_request.extracted_data,
            is_approved=update_request.is_approved or False
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update extraction"
            )
        
        # Return updated extraction
        updated_extraction = db.get_extraction_by_id(extraction_id)
        return DataLibraryEntry(**updated_extraction)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update extraction: {str(e)}"
        )

@router.delete("/data-library/{extraction_id}")
async def delete_extraction(extraction_id: int):
    """Delete an extraction from data library"""
    try:
        # Check if extraction exists
        extraction = db.get_extraction_by_id(extraction_id)
        if not extraction:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Extraction with ID {extraction_id} not found"
            )
        
        # Delete PDF file if it exists
        pdf_path = extraction.get('pdf_path')
        if pdf_path and os.path.exists(pdf_path):
            try:
                os.remove(pdf_path)
            except Exception as e:
                print(f"Warning: Could not delete PDF file {pdf_path}: {e}")
        
        # Delete from database
        success = db.delete_extraction(extraction_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete extraction"
            )
        
        return {"message": "Extraction deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete extraction: {str(e)}"
        )

@router.get("/pdf/{extraction_id}")
async def get_pdf_file(extraction_id: int):
    """Get the PDF file for a specific extraction"""
    from fastapi.responses import FileResponse
    
    try:
        extraction = db.get_extraction_by_id(extraction_id)
        if not extraction:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Extraction with ID {extraction_id} not found"
            )
        
        pdf_path = extraction.get('pdf_path')
        if not pdf_path or not os.path.exists(pdf_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="PDF file not found"
            )
        
        return FileResponse(
            path=pdf_path,
            media_type='application/pdf',
            filename=extraction.get('filename', 'document.pdf'),
            headers={
                "Content-Disposition": f'inline; filename="{extraction.get("filename", "document.pdf")}"'
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve PDF: {str(e)}"
        )
