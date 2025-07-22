"""
Analysis API routes
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import List, Dict, Any, Optional
import logging

from app.models.survey import AnalysisRequest, AnalysisResult, SurveyResponse
from app.services.statistical_analysis import StatisticalAnalysisService
from app.services.llm_service import LLMService
from app.services.data_processor import DataProcessor
from app.database.database import get_database

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize services
statistical_service = StatisticalAnalysisService()
llm_service = LLMService()
data_processor = DataProcessor()

@router.post("/run", response_model=AnalysisResult)
async def run_analysis(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    db=Depends(get_database)
):
    """Run comprehensive analysis on survey data"""
    try:
        logger.info(f"Starting analysis for {len(request.responses)} responses")
        
        # Process data
        processed_data = await data_processor.process_survey_data(request.responses)
        
        # Run statistical analysis
        statistical_results = await statistical_service.run_comprehensive_analysis(processed_data)
        
        # Generate LLM insights if requested
        llm_insights = None
        if request.include_llm_insights:
            llm_insights = await llm_service.generate_insights(processed_data, statistical_results)
        
        # Perform segmentation if requested
        segmentation_results = None
        if request.include_segmentation:
            segmentation_results = await statistical_service.perform_segmentation(processed_data)
        
        # Generate recommendations
        recommendations = []
        if llm_insights:
            recommendations = await llm_service.generate_recommendations(statistical_results, llm_insights)
        
        # Compile results
        from datetime import datetime
        analysis_result = AnalysisResult(
            overview=statistical_results["overview"],
            course_metrics=statistical_results["course_metrics"],
            demographics=statistical_results["demographics"],
            statistical_tests=statistical_results["tests"],
            llm_insights=llm_insights or LLMInsights(
                sentiment_analysis={},
                topic_modeling={},
                insights=[],
                recommendations=[],
                quality_score=0.0
            ),
            segmentation=segmentation_results or Segmentation(
                segments=[],
                segment_characteristics={},
                satisfaction_by_segment={},
                recommended_actions={}
            ),
            recommendations=recommendations,
            timestamp=datetime.utcnow(),
            metadata={
                "total_responses": len(request.responses),
                "analysis_type": request.analysis_type,
                "version": "1.0.0"
            }
        )
        
        # Store results in background
        background_tasks.add_task(store_analysis_result, analysis_result, db)
        
        return analysis_result
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.get("/history")
async def get_analysis_history(
    limit: int = 10,
    offset: int = 0,
    db=Depends(get_database)
):
    """Get analysis history"""
    try:
        history = await db.get_analysis_history(limit=limit, offset=offset)
        return {
            "analyses": history,
            "count": len(history),
            "has_more": len(history) == limit
        }
    except Exception as e:
        logger.error(f"Failed to retrieve analysis history: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve analysis history")

@router.get("/quick")
async def quick_analysis(db=Depends(get_database)):
    """Run quick analysis on existing data"""
    try:
        # Get recent responses
        responses = await db.get_survey_responses(limit=50)
        
        if not responses:
            # Generate mock data if no responses exist
            responses = await data_processor.generate_mock_data(17)
            await db.store_survey_responses(responses)
        
        # Run basic analysis
        processed_data = await data_processor.process_survey_data(responses)
        statistical_results = await statistical_service.run_comprehensive_analysis(processed_data)
        
        return {
            "message": "Quick analysis completed",
            "overview": statistical_results["overview"],
            "course_count": len(statistical_results["course_metrics"]),
            "response_count": len(responses)
        }
        
    except Exception as e:
        logger.error(f"Quick analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Quick analysis failed: {str(e)}")

async def store_analysis_result(result: AnalysisResult, db):
    """Background task to store analysis results"""
    try:
        await db.store_analysis_result(result)
        logger.info("Analysis result stored successfully")
    except Exception as e:
        logger.error(f"Failed to store analysis result: {str(e)}")