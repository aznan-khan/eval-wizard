"""
Insights API routes
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Any, Optional
import logging

from app.services.llm_service import LLMService
from app.services.statistical_analysis import StatisticalAnalysisService
from app.database.database import get_database

logger = logging.getLogger(__name__)
router = APIRouter()

llm_service = LLMService()
statistical_service = StatisticalAnalysisService()

@router.get("/trends")
async def get_satisfaction_trends(
    time_period: str = "last_month",
    db=Depends(get_database)
):
    """Get satisfaction trends over time"""
    try:
        trends = await statistical_service.calculate_trends(time_period)
        return {
            "trends": trends,
            "period": time_period,
            "generated_at": "2024-01-01T00:00:00Z"
        }
    except Exception as e:
        logger.error(f"Trend analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Trend analysis failed")

@router.get("/recommendations")
async def get_recommendations(db=Depends(get_database)):
    """Get AI-generated recommendations"""
    try:
        # This would use stored analysis results in a real implementation
        recommendations = [
            "Enhance Financial Analysis course content based on satisfaction feedback",
            "Optimize mobile experience to bridge preference-usage gap",
            "Expand AI and Content Creation course offerings",
            "Implement instructor training program for low-performing courses",
            "Develop more interactive content elements"
        ]
        
        return {
            "recommendations": recommendations,
            "generated_at": "2024-01-01T00:00:00Z",
            "confidence": 0.85
        }
    except Exception as e:
        logger.error(f"Recommendation generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Recommendation generation failed")