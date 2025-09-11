from fastapi import APIRouter, HTTPException, status
from typing import List
from models.extraction_schema import (
    CreateSchemaRequest, 
    UpdateSchemaRequest, 
    SchemaResponse, 
    SchemaListResponse,
    SchemaValidationResponse
)
from db.sqlite_db import db
import json
from datetime import datetime

router = APIRouter(prefix="/schemas", tags=["extraction-schemas"])

@router.post("/", response_model=SchemaResponse, status_code=status.HTTP_201_CREATED)
async def create_schema(schema_request: CreateSchemaRequest):
    """Create a new extraction schema"""
    try:
        # Check if name already exists
        if db.schema_name_exists(schema_request.name):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Schema with name '{schema_request.name}' already exists"
            )
        
        # Prepare schema JSON
        schema_json = {
            "field_definitions": [field.dict() for field in schema_request.field_definitions]
        }
        
        # Save to database
        schema_id = db.save_schema(
            name=schema_request.name,
            description=schema_request.description,
            schema_json=schema_json
        )
        
        # Retrieve and return the created schema
        created_schema = db.get_schema_by_id(schema_id)
        if not created_schema:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve created schema"
            )
        
        # Add computed fields
        field_definitions = created_schema['json_string'].get('field_definitions', [])
        created_schema['field_count'] = len(field_definitions)
        created_schema['required_field_count'] = len([f for f in field_definitions if f.get('required', False)])
        
        return SchemaResponse(**created_schema)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create schema: {str(e)}"
        )

@router.get("/", response_model=SchemaListResponse)
async def get_schemas():
    """Get all extraction schemas"""
    try:
        schemas = db.get_schemas()
        
        # Add computed fields for each schema
        for schema in schemas:
            field_definitions = schema['json_string'].get('field_definitions', [])
            schema['field_count'] = len(field_definitions)
            schema['required_field_count'] = len([f for f in field_definitions if f.get('required', False)])
        
        return SchemaListResponse(
            schemas=[SchemaResponse(**schema) for schema in schemas],
            total=len(schemas)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve schemas: {str(e)}"
        )

@router.get("/{schema_id}", response_model=SchemaResponse)
async def get_schema(schema_id: int):
    """Get a specific extraction schema by ID"""
    try:
        schema = db.get_schema_by_id(schema_id)
        if not schema:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Schema with ID {schema_id} not found"
            )
        
        # Add computed fields
        field_definitions = schema['json_string'].get('field_definitions', [])
        schema['field_count'] = len(field_definitions)
        schema['required_field_count'] = len([f for f in field_definitions if f.get('required', False)])
        
        return SchemaResponse(**schema)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve schema: {str(e)}"
        )

@router.get("/name/{schema_name}", response_model=SchemaResponse)
async def get_schema_by_name(schema_name: str):
    """Get a specific extraction schema by name"""
    try:
        schema = db.get_schema_by_name(schema_name)
        if not schema:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Schema with name '{schema_name}' not found"
            )
        
        # Add computed fields
        field_definitions = schema['json_string'].get('field_definitions', [])
        schema['field_count'] = len(field_definitions)
        schema['required_field_count'] = len([f for f in field_definitions if f.get('required', False)])
        
        return SchemaResponse(**schema)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve schema: {str(e)}"
        )

@router.put("/{schema_id}", response_model=SchemaResponse)
async def update_schema(schema_id: int, schema_request: UpdateSchemaRequest):
    """Update an existing extraction schema"""
    try:
        # Check if schema exists
        existing_schema = db.get_schema_by_id(schema_id)
        if not existing_schema:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Schema with ID {schema_id} not found"
            )
        
        # Check if new name conflicts with existing schemas (excluding current one)
        if db.schema_name_exists(schema_request.name, exclude_id=schema_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Schema with name '{schema_request.name}' already exists"
            )
        
        # Prepare schema JSON
        schema_json = {
            "field_definitions": [field.dict() for field in schema_request.field_definitions]
        }
        
        # Update in database
        success = db.update_schema(
            schema_id=schema_id,
            name=schema_request.name,
            description=schema_request.description,
            schema_json=schema_json
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update schema"
            )
        
        # Retrieve and return the updated schema
        updated_schema = db.get_schema_by_id(schema_id)
        if not updated_schema:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve updated schema"
            )
        
        # Add computed fields
        field_definitions = updated_schema['json_string'].get('field_definitions', [])
        updated_schema['field_count'] = len(field_definitions)
        updated_schema['required_field_count'] = len([f for f in field_definitions if f.get('required', False)])
        
        return SchemaResponse(**updated_schema)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update schema: {str(e)}"
        )

@router.delete("/{schema_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_schema(schema_id: int):
    """Delete an extraction schema"""
    try:
        # Check if schema exists
        existing_schema = db.get_schema_by_id(schema_id)
        if not existing_schema:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Schema with ID {schema_id} not found"
            )
        
        # Delete from database
        success = db.delete_schema(schema_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete schema"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete schema: {str(e)}"
        )

@router.post("/validate", response_model=SchemaValidationResponse)
async def validate_schema(schema_request: CreateSchemaRequest):
    """Validate a schema without saving it"""
    try:
        errors = []
        
        # Check if name already exists
        if db.schema_name_exists(schema_request.name):
            errors.append(f"Schema with name '{schema_request.name}' already exists")
        
        # Validate field definitions
        field_names = set()
        for i, field in enumerate(schema_request.field_definitions):
            if field.name in field_names:
                errors.append(f"Duplicate field name '{field.name}' at position {i + 1}")
            field_names.add(field.name)
            
            # Validate dropdown fields have options
            if field.type == "dropdown" and (not field.options or len(field.options) == 0):
                errors.append(f"Dropdown field '{field.name}' must have at least one option")
            
            # Validate table fields have columns
            if field.type == "table" and (not field.tableColumns or len(field.tableColumns) == 0):
                errors.append(f"Table field '{field.name}' must have at least one column")
        
        return SchemaValidationResponse(
            valid=len(errors) == 0,
            errors=errors
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to validate schema: {str(e)}"
        )