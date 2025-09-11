from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum

class FieldType(str, Enum):
    text = "text"
    number = "number"
    date = "date"
    boolean = "boolean"
    email = "email"
    url = "url"
    dropdown = "dropdown"
    table = "table"

class TableColumn(BaseModel):
    name: str = Field(..., description="Column name")
    label: str = Field(..., description="Column display label")
    type: str = Field(..., description="Column data type (text, number, date, boolean)")

class FieldDefinition(BaseModel):
    name: str = Field(..., description="Field name")
    type: FieldType = Field(..., description="Field type")
    label: str = Field(..., description="Field display label")
    required: Optional[bool] = Field(False, description="Whether field is required")
    options: Optional[List[str]] = Field(None, description="Options for dropdown type")
    description: Optional[str] = Field(None, description="Field description")
    tableColumns: Optional[List[TableColumn]] = Field(None, description="Table columns for table type")

class CreateSchemaRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Schema name")
    description: Optional[str] = Field(None, max_length=500, description="Schema description")
    field_definitions: List[FieldDefinition] = Field(..., min_items=1, description="Field definitions")
    
    @validator('name')
    def validate_name(cls, v):
        if not v.strip():
            raise ValueError('Schema name cannot be empty')
        return v.strip()

class UpdateSchemaRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Schema name")
    description: Optional[str] = Field(None, max_length=500, description="Schema description")
    field_definitions: List[FieldDefinition] = Field(..., min_items=1, description="Field definitions")
    
    @validator('name')
    def validate_name(cls, v):
        if not v.strip():
            raise ValueError('Schema name cannot be empty')
        return v.strip()

class SchemaResponse(BaseModel):
    id: int
    json_name: str
    json_description: Optional[str]
    json_string: Dict[str, Any]
    created_at: str
    field_count: Optional[int] = None
    required_field_count: Optional[int] = None

class SchemaListResponse(BaseModel):
    schemas: List[SchemaResponse]
    total: int

class SchemaValidationResponse(BaseModel):
    valid: bool
    errors: List[str] = []

# Extraction models
class ExtractionRequest(BaseModel):
    schema_id: int = Field(..., description="ID of the schema to use for extraction")

class ExtractionResponse(BaseModel):
    extraction_id: int
    extracted_data: Dict[str, Any]
    confidence_scores: Optional[Dict[str, float]] = None
    processing_time: Optional[float] = None

class DataLibraryEntry(BaseModel):
    id: int
    schema_id: int
    schema_name: str
    filename: str
    pdf_path: str
    extracted_data: Dict[str, Any]
    is_approved: bool
    created_at: str
    updated_at: str

class DataLibraryResponse(BaseModel):
    entries: List[DataLibraryEntry]
    total: int

class UpdateExtractionRequest(BaseModel):
    extracted_data: Dict[str, Any] = Field(..., description="Updated extracted data")
    is_approved: Optional[bool] = Field(False, description="Whether the extraction is approved")