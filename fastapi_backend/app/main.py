"""
FastAPI Survey Analysis Application
Main application entry point for statistical analysis of course evaluation surveys
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import asyncio
from typing import List, Dict, Any, Optional
import json
import logging
from datetime import datetime

from app.models.survey import SurveyResponse, AnalysisRequest, AnalysisResult
from app.services.statistical_analysis import StatisticalAnalysisService
from app.services.llm_service import LLMService
from app.services.data_processor import DataProcessor
from app.config.settings import Settings
from app.database.database import get_database
from app.api import auth, survey, analysis, insights

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Survey Analysis API",
    description="Comprehensive statistical analysis system for course evaluation surveys with LLM integration",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Settings
settings = Settings()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["authentication"])
app.include_router(survey.router, prefix="/api/survey", tags=["survey"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])
app.include_router(insights.router, prefix="/api/insights", tags=["insights"])

# Global services
statistical_service = StatisticalAnalysisService()
llm_service = LLMService()
data_processor = DataProcessor()

@app.on_event("startup")
async def startup_event():
    """Initialize services and database connections"""
    logger.info("Starting Survey Analysis API...")
    
    # Initialize database
    await get_database().connect()
    
    # Initialize LLM service
    await llm_service.initialize()
    
    logger.info("API startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources"""
    logger.info("Shutting down Survey Analysis API...")
    
    # Close database connections
    await get_database().disconnect()
    
    logger.info("Shutdown complete")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Survey Analysis API",
        "version": "1.0.0",
        "status": "active",
        "timestamp": datetime.utcnow().isoformat(),
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "analysis": "/api/analysis",
            "insights": "/api/insights"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check database connection
        db_status = await get_database().ping()
        
        # Check LLM service
        llm_status = await llm_service.health_check()
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "database": "healthy" if db_status else "unhealthy",
                "llm": "healthy" if llm_status else "unhealthy",
                "statistical_analysis": "healthy"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unavailable")

@app.post("/api/analyze", response_model=AnalysisResult)
async def run_analysis(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    db=Depends(get_database)
):
    """
    Main analysis endpoint - processes survey data and returns comprehensive results
    Implements Phase 1-8 of the work plan
    """
    try:
        logger.info(f"Starting analysis for {len(request.responses)} responses")
        
        # Phase 1-2: Data Preparation and EDA
        processed_data = await data_processor.process_survey_data(request.responses)
        
        # Phase 3: Statistical Testing
        statistical_results = await statistical_service.run_comprehensive_analysis(processed_data)
        
        # Phase 4: LLM Integration
        llm_insights = await llm_service.generate_insights(processed_data, statistical_results)
        
        # Phase 5: Advanced Analytics
        segmentation_results = await statistical_service.perform_segmentation(processed_data)
        
        # Phase 6: Compile Results
        analysis_result = AnalysisResult(
            overview=statistical_results["overview"],
            course_metrics=statistical_results["course_metrics"],
            demographics=statistical_results["demographics"],
            statistical_tests=statistical_results["tests"],
            llm_insights=llm_insights,
            segmentation=segmentation_results,
            recommendations=await llm_service.generate_recommendations(statistical_results, llm_insights),
            timestamp=datetime.utcnow(),
            metadata={
                "total_responses": len(request.responses),
                "analysis_version": "1.0.0",
                "processing_time_seconds": 0  # Will be calculated
            }
        )
        
        # Store results in background
        background_tasks.add_task(store_analysis_result, analysis_result, db)
        
        logger.info("Analysis completed successfully")
        return analysis_result
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/upload-survey-data")
async def upload_survey_data(
    file_data: Dict[str, Any],
    db=Depends(get_database)
):
    """Upload and validate survey response data"""
    try:
        # Validate data structure
        validated_responses = await data_processor.validate_survey_responses(file_data)
        
        # Store in database
        stored_count = await db.store_survey_responses(validated_responses)
        
        return {
            "message": "Survey data uploaded successfully",
            "responses_stored": stored_count,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Upload failed: {str(e)}")

@app.get("/api/analysis/history")
async def get_analysis_history(
    limit: int = 10,
    offset: int = 0,
    db=Depends(get_database)
):
    """Retrieve historical analysis results"""
    try:
        results = await db.get_analysis_history(limit=limit, offset=offset)
        return {
            "analyses": results,
            "count": len(results),
            "has_more": len(results) == limit
        }
    except Exception as e:
        logger.error(f"Failed to retrieve history: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve analysis history")

@app.get("/api/courses/{course_id}/analysis")
async def get_course_analysis(
    course_id: str,
    db=Depends(get_database)
):
    """Get detailed analysis for a specific course"""
    try:
        course_data = await db.get_course_responses(course_id)
        if not course_data:
            raise HTTPException(status_code=404, detail="Course not found")
        
        # Run course-specific analysis
        analysis = await statistical_service.analyze_course(course_data)
        return analysis
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Course analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Course analysis failed")

@app.get("/api/insights/trends")
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
            "generated_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Trend analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Trend analysis failed")

async def store_analysis_result(result: AnalysisResult, db):
    """Background task to store analysis results"""
    try:
        await db.store_analysis_result(result)
        logger.info("Analysis result stored successfully")
    except Exception as e:
        logger.error(f"Failed to store analysis result: {str(e)}")

# Exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    logger.error(f"ValueError: {str(exc)}")
    return JSONResponse(
        status_code=400,
        content={"detail": f"Invalid input: {str(exc)}"}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )