from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import json
import os
from db.sqlite_db import db

# Agent imports
from agno.agent import Agent
from agno.models.google import Gemini
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

# Initialize the two-way match agent
api_key = os.getenv("GOOGLE_API_KEY")

two_way_match_agent = Agent(
    name="Two Way Match Agent",
    model=Gemini(id="gemini-2.0-flash", api_key=api_key),
    tools=[],
    instructions=[
        "You are a Two Way Match Agent. Working for InnuxAI",
        "You will be given 2 json files. One is the PO (Purchase Order) and the other is the INV (Invoice)",
        "Your task is to match each line item in the INV to the corresponding line item in the PO",
        "If for an item in INV you cannot find a match in the PO, you should mark it as 'No Match Found'",
        "As the line items name will not be an exact match, use your reasoning to find the best match with your confidence score.",
        "return a json array with the following fields: item_in_inv, matched_item_in_po, confidence_score (0-100), reasoning",
        "When analyzing, provide structured responses with clear reasoning.",
        "If rate limiting occurs, inform the user about it.",
        "Always return valid JSON format that can be parsed."
    ],
    show_tool_calls=True,
    markdown=True
)

class TwoWayMatchRequest(BaseModel):
    po_extraction_id: int = Field(..., description="Purchase Order extraction ID")
    invoice_extraction_id: int = Field(..., description="Invoice extraction ID")

class MatchResult(BaseModel):
    item_in_inv: str = Field(..., description="Item from invoice")
    matched_item_in_po: str = Field(..., description="Matched item from PO")
    confidence_score: int = Field(..., description="Confidence score 0-100")
    reasoning: str = Field(..., description="Reasoning for the match")

class TwoWayMatchResponse(BaseModel):
    po_filename: str
    invoice_filename: str
    match_results: List[MatchResult]
    total_invoice_items: int
    matched_items: int
    unmatched_items: int
    processing_time: Optional[float] = None

@router.post("/two-way-match", response_model=TwoWayMatchResponse)
async def perform_two_way_match(request: TwoWayMatchRequest):
    """
    Perform two-way matching between a Purchase Order and an Invoice.
    """
    import time
    start_time = time.time()
    
    try:
        # Fetch PO extraction
        po_extraction = db.get_extraction_by_id(request.po_extraction_id)
        
        if not po_extraction:
            raise HTTPException(
                status_code=404, 
                detail=f"Purchase Order extraction with ID {request.po_extraction_id} not found"
            )
        
        # Fetch Invoice extraction
        invoice_extraction = db.get_extraction_by_id(request.invoice_extraction_id)
        
        if not invoice_extraction:
            raise HTTPException(
                status_code=404, 
                detail=f"Invoice extraction with ID {request.invoice_extraction_id} not found"
            )
        
        # Prepare data for the agent
        po_data = po_extraction['extracted_data']
        invoice_data = invoice_extraction['extracted_data']
        
        # Create prompt for the agent
        prompt = f"""
        Please perform a two-way match between the following Purchase Order and Invoice data:

        PURCHASE ORDER DATA:
        {json.dumps(po_data, indent=2)}

        INVOICE DATA:
        {json.dumps(invoice_data, indent=2)}

        Please match each line item in the invoice to the corresponding line item in the purchase order.
        Return a JSON array with the following structure for each invoice item:
        {{
            "item_in_inv": "description of invoice item",
            "matched_item_in_po": "description of matched PO item or 'No Match Found'",
            "confidence_score": 85,
            "reasoning": "explanation of why this match was made"
        }}

        RULES:
        - Be strict in matching, but use reasoning to find the best possible match.
        - If no suitable match is found, use "No Match Found" for matched_item_in_po.
        - Confidence score should be between 0 and 100.
        - Always return valid JSON format that can be parsed.
        - If rate limiting occurs, inform the user about it.
        - Note that if there are numerical metrics (like measurements) in Description then to be to be concluded as a match, those measurements should match exactly (like 3.0 is not equal to 3 or 3.02) even their precising must be same.
        - Note that there may be some abbreviations or slight variations in item descriptions between the PO and Invoice. Use your best judgment to determine matches based on context and details provided.
        """
        
        # Run the agent
        try:
            response = two_way_match_agent.run(prompt)
            agent_response = response.content
            
            # Extract JSON from agent response
            # The agent response might contain markdown formatting, so we need to extract the JSON
            import re
            
            # Try to find JSON in the response
            json_pattern = r'```(?:json)?\s*(\[.*?\])\s*```'
            json_match = re.search(json_pattern, agent_response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
            else:
                # If no markdown code block, try to find JSON array directly
                json_pattern = r'(\[.*?\])'
                json_match = re.search(json_pattern, agent_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    raise ValueError("No valid JSON found in agent response")
            
            # Parse the JSON response
            try:
                match_results_raw = json.loads(json_str)
            except json.JSONDecodeError as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to parse agent response as JSON: {str(e)}"
                )
            
            # Validate and convert to MatchResult objects
            match_results = []
            for item in match_results_raw:
                try:
                    match_result = MatchResult(
                        item_in_inv=str(item.get('item_in_inv', '')),
                        matched_item_in_po=str(item.get('matched_item_in_po', '')),
                        confidence_score=int(item.get('confidence_score', 0)),
                        reasoning=str(item.get('reasoning', ''))
                    )
                    match_results.append(match_result)
                except Exception as e:
                    # If individual item parsing fails, create a default entry
                    match_results.append(MatchResult(
                        item_in_inv=str(item),
                        matched_item_in_po="Parse Error",
                        confidence_score=0,
                        reasoning=f"Failed to parse result: {str(e)}"
                    ))
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Agent processing failed: {str(e)}"
            )
        
        # Calculate statistics
        total_items = len(match_results)
        matched_items = sum(1 for result in match_results 
                          if result.matched_item_in_po != "No Match Found" 
                          and result.matched_item_in_po != "Parse Error")
        unmatched_items = total_items - matched_items
        
        processing_time = time.time() - start_time
        
        return TwoWayMatchResponse(
            po_filename=po_extraction['filename'],
            invoice_filename=invoice_extraction['filename'],
            match_results=match_results,
            total_invoice_items=total_items,
            matched_items=matched_items,
            unmatched_items=unmatched_items,
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/extractions/list")
async def list_extractions_for_matching():
    """
    Get list of extractions suitable for two-way matching.
    Returns approved extractions with their basic info.
    """
    try:
        extractions = db.get_extractions()
        
        # Filter only approved extractions
        approved_extractions = [
            extraction for extraction in extractions 
            if extraction.get('is_approved', False)
        ]
        
        extraction_list = []
        for extraction in approved_extractions:
            extraction_list.append({
                "id": extraction['id'],
                "filename": extraction['filename'],
                "schema_name": extraction['schema_name'],
                "created_at": extraction['created_at'],
                "updated_at": extraction['updated_at']
            })
        
        return {
            "extractions": extraction_list,
            "total": len(extraction_list)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch extractions: {str(e)}"
        )
