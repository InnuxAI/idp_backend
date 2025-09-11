import json
import os
import time
from typing import Dict, Any, List, Optional
import pymupdf as fitz
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from db.sqlite_db import db
import traceback

class ExtractionService:
    def __init__(self):
        """Initialize the extraction service with Google Gemini LLM"""
        self.llm = None
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the Google Gemini LLM"""
        try:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")
            
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=api_key,
                temperature=0.1,
                max_tokens=4096
            )
            print("✅ Google Gemini LLM initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize Google Gemini LLM: {e}")
            raise e
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using PyMuPDF"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
                text += "\n\n"  # Add page separator
            
            doc.close()
            return text.strip()
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            raise e
    
    def create_extraction_prompt(self, schema_fields: List[Dict[str, Any]], pdf_text: str) -> str:
        """Create a prompt for LLM extraction based on schema fields and PDF text"""
        
        # Build field descriptions
        field_descriptions = []
        for field in schema_fields:
            desc = f"- {field['name']} ({field['type']}): {field['label']}"
            if field.get('required'):
                desc += " [REQUIRED]"
            if field.get('description'):
                desc += f" - {field['description']}"
            if field.get('options'):
                desc += f" (Options: {', '.join(field['options'])})"
            field_descriptions.append(desc)
        
        # Create the extraction schema for JSON output
        json_schema = {}
        for field in schema_fields:
            field_info = {
                "type": field['type'],
                "description": field.get('description', field['label'])
            }
            if field.get('required'):
                field_info["required"] = True
            if field.get('options'):
                field_info["enum"] = field['options']
            json_schema[field['name']] = field_info
        
        prompt = f"""
You are an expert document analysis AI. Your task is to extract specific information from the provided PDF text according to the given schema.

EXTRACTION SCHEMA:
{chr(10).join(field_descriptions)}

INSTRUCTIONS:
1. Carefully read through the entire document text
2. Extract the requested information for each field
3. Return ONLY a valid JSON object with the extracted data
4. Use null for fields where information is not found
5. Ensure data types match the schema (text, number, date, boolean, etc.)
6. For dropdown fields, use only the provided options
7. For required fields, make your best effort to find the information

JSON SCHEMA STRUCTURE:
{json.dumps(json_schema, indent=2)}

DOCUMENT TEXT:
{pdf_text}

EXTRACTED DATA (JSON only):
"""
        return prompt
    
    async def extract_fields_from_pdf(self, pdf_path: str, schema_id: int) -> Dict[str, Any]:
        """Extract fields from PDF using the specified schema"""
        start_time = time.time()
        
        try:
            # Get schema from database
            schema = db.get_schema_by_id(schema_id)
            if not schema:
                raise ValueError(f"Schema with ID {schema_id} not found")
            
            # Extract field definitions
            field_definitions = schema['json_string'].get('field_definitions', [])
            if not field_definitions:
                raise ValueError("No field definitions found in schema")
            
            # Extract text from PDF
            print(f"Extracting text from PDF: {pdf_path}")
            pdf_text = self.extract_text_from_pdf(pdf_path)
            
            if not pdf_text.strip():
                raise ValueError("No text could be extracted from the PDF")
            
            # Create extraction prompt
            prompt = self.create_extraction_prompt(field_definitions, pdf_text)
            
            # Call LLM for extraction
            print("Calling LLM for field extraction...")
            message = HumanMessage(content=prompt)
            response = await self.llm.ainvoke([message])
            
            # Parse JSON response
            try:
                extracted_data = json.loads(response.content.strip())
            except json.JSONDecodeError:
                # Try to extract JSON from response if it's wrapped in other text
                content = response.content.strip()
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = content[start_idx:end_idx]
                    extracted_data = json.loads(json_str)
                else:
                    raise ValueError("Could not parse JSON from LLM response")
            
            processing_time = time.time() - start_time
            
            # Validate extracted data against schema
            validated_data = self._validate_extracted_data(extracted_data, field_definitions)
            
            print(f"✅ Field extraction completed in {processing_time:.2f} seconds")
            
            return {
                "extracted_data": validated_data,
                "processing_time": processing_time,
                "confidence_scores": {}  # Could be enhanced with confidence scoring
            }
            
        except Exception as e:
            print(f"❌ Error during field extraction: {e}")
            traceback.print_exc()
            raise e
    
    def _validate_extracted_data(self, extracted_data: Dict[str, Any], field_definitions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate and clean extracted data according to schema"""
        validated_data = {}
        
        for field in field_definitions:
            field_name = field['name']
            field_type = field['type']
            field_value = extracted_data.get(field_name)
            
            # Handle null/empty values
            if field_value is None or field_value == "":
                if field.get('required'):
                    # For required fields, try to provide a default value
                    if field_type == 'boolean':
                        validated_data[field_name] = False
                    elif field_type == 'number':
                        validated_data[field_name] = 0
                    else:
                        validated_data[field_name] = ""
                else:
                    validated_data[field_name] = None
                continue
            
            # Type validation and conversion
            try:
                if field_type == 'number':
                    # Try to convert to number
                    if isinstance(field_value, str):
                        # Remove any non-numeric characters except decimal point and minus
                        import re
                        numeric_str = re.sub(r'[^\d.-]', '', str(field_value))
                        validated_data[field_name] = float(numeric_str) if '.' in numeric_str else int(numeric_str)
                    else:
                        validated_data[field_name] = field_value
                        
                elif field_type == 'boolean':
                    if isinstance(field_value, str):
                        validated_data[field_name] = field_value.lower() in ['true', 'yes', '1', 'on']
                    else:
                        validated_data[field_name] = bool(field_value)
                        
                elif field_type == 'dropdown':
                    # Ensure value is in allowed options
                    options = field.get('options', [])
                    if options and field_value not in options:
                        # Try to find closest match
                        closest_match = None
                        for option in options:
                            if str(field_value).lower() in str(option).lower() or str(option).lower() in str(field_value).lower():
                                closest_match = option
                                break
                        validated_data[field_name] = closest_match or options[0]
                    else:
                        validated_data[field_name] = field_value
                        
                else:  # text, email, url, date
                    validated_data[field_name] = str(field_value)
                    
            except (ValueError, TypeError):
                # If conversion fails, use string representation
                validated_data[field_name] = str(field_value) if field_value is not None else ""
        
        return validated_data

from dotenv import load_dotenv
load_dotenv()
# Global extraction service instance
extraction_service = ExtractionService()
