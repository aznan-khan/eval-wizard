"""
Survey API routes
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from typing import List, Dict, Any, Optional
import json
import logging

from app.models.survey import SurveyResponse
from app.services.data_processor import DataProcessor
from app.database.database import get_database

logger = logging.getLogger(__name__)
router = APIRouter()
data_processor = DataProcessor()

@router.get("/responses", response_model=List[SurveyResponse])
async def get_responses(
    course_type: Optional[str] = None,
    limit: Optional[int] = 100,
    db=Depends(get_database)
):
    """Get survey responses with optional filtering"""
    try:
        responses = await db.get_survey_responses(course_type=course_type, limit=limit)
        return responses
    except Exception as e:
        logger.error(f"Failed to retrieve responses: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve responses")

@router.post("/responses/upload")
async def upload_responses(
    file: UploadFile = File(...),
    db=Depends(get_database)
):
    """Upload survey responses from file"""
    try:
        # Read file content
        content = await file.read()
        
        # Parse JSON
        try:
            file_data = json.loads(content)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format")
        
        # Validate and process responses
        responses = await data_processor.validate_survey_responses(file_data)
        
        if not responses:
            raise HTTPException(status_code=400, detail="No valid responses found in file")
        
        # Store in database
        stored_count = await db.store_survey_responses(responses)
        
        return {
            "message": "Responses uploaded successfully",
            "total_responses": len(responses),
            "stored_responses": stored_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.post("/responses/generate-mock")
async def generate_mock_data(
    num_responses: int = 17,
    db=Depends(get_database)
):
    """Generate mock survey data for testing"""
    try:
        responses = await data_processor.generate_mock_data(num_responses)
        stored_count = await db.store_survey_responses(responses)
        
        return {
            "message": "Mock data generated successfully",
            "generated_responses": len(responses),
            "stored_responses": stored_count
        }
        
    except Exception as e:
        logger.error(f"Mock data generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Mock data generation failed: {str(e)}")

@router.get("/statistics")
async def get_survey_statistics(db=Depends(get_database)):
    """Get basic survey statistics"""
    try:
        stats = await db.get_response_statistics()
        return stats
    except Exception as e:
        logger.error(f"Statistics retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")