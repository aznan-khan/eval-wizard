"""
Data models for survey analysis system
Implements comprehensive data structures for the statistical analysis pipeline
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum

class GenderType(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    PREFER_NOT_TO_SAY = "prefer_not_to_say"

class RoleType(str, Enum):
    STUDENT = "student"
    PROFESSIONAL = "professional"
    EDUCATOR = "educator"
    RESEARCHER = "researcher"

class CourseType(str, Enum):
    AI_FUNDAMENTALS = "ai_fundamentals"
    DIGITAL_MARKETING = "digital_marketing"
    CONTENT_CREATION = "content_creation"
    APP_DEVELOPMENT = "app_development"
    FINANCIAL_ANALYSIS = "financial_analysis"

class LikertScale(int, Enum):
    VERY_DISSATISFIED = 1
    DISSATISFIED = 2
    NEUTRAL = 3
    SATISFIED = 4
    VERY_SATISFIED = 5

class NPSScale(int, Enum):
    DETRACTOR = 0  # 0-6
    PASSIVE = 7    # 7-8
    PROMOTER = 9   # 9-10

# Core Survey Response Model
class SurveyResponse(BaseModel):
    """Individual survey response model based on comprehensive question schema"""
    
    # Response metadata
    response_id: str = Field(..., description="Unique response identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    course_type: CourseType = Field(..., description="Course being evaluated")
    
    # Q1: Demographics - Gender
    gender: Optional[GenderType] = Field(None, description="Respondent gender")
    
    # Q2a-c: Demographics - Age, Location, Background
    age_group: Optional[str] = Field(None, description="Age group (e.g., '20-29')")
    location: Optional[str] = Field(None, description="Geographic location")
    professional_background: Optional[str] = Field(None, description="Professional background")
    
    # Q3: Course Discovery
    discovery_method: Optional[str] = Field(None, description="How they found the course")
    
    # Q4: Role
    primary_role: Optional[RoleType] = Field(None, description="Primary role/occupation")
    
    # Q5-6: Course Evaluation (Likert Scale 1-5)
    course_quality: Optional[LikertScale] = Field(None, description="Overall course quality rating")
    course_relevance: Optional[LikertScale] = Field(None, description="Course relevance to goals")
    
    # Q7-8: Feature Assessment
    feature_importance: Optional[Dict[str, int]] = Field(
        None, 
        description="Feature importance ratings (1-5 scale)"
    )
    feature_usage: Optional[Dict[str, int]] = Field(
        None, 
        description="Feature usage frequency (1-5 scale)"
    )
    
    # Q9a-d: Instructor Rating (Multi-dimensional)
    instructor_clarity: Optional[LikertScale] = Field(None, description="Instructor clarity rating")
    instructor_knowledge: Optional[LikertScale] = Field(None, description="Instructor knowledge rating")
    instructor_engagement: Optional[LikertScale] = Field(None, description="Instructor engagement rating")
    instructor_responsiveness: Optional[LikertScale] = Field(None, description="Instructor responsiveness rating")
    
    # Q10-13: Additional Course Metrics
    learning_objectives_met: Optional[LikertScale] = Field(None, description="Learning objectives achievement")
    course_organization: Optional[LikertScale] = Field(None, description="Course organization quality")
    content_difficulty: Optional[LikertScale] = Field(None, description="Content difficulty appropriateness")
    learning_improvement: Optional[LikertScale] = Field(None, description="Self-assessed learning improvement")
    
    # Q14: NPS Score (0-10)
    nps_score: Optional[int] = Field(None, ge=0, le=10, description="Net Promoter Score")
    
    # Q15: Future Interest
    future_course_interest: Optional[List[str]] = Field(None, description="Interest in future courses")
    
    # Q16: Platform Usage
    platform_preference: Optional[str] = Field(None, description="Platform preference (mobile/desktop)")
    
    # Q17: Open-ended Feedback
    improvement_suggestions: Optional[str] = Field(None, description="Improvement suggestions")
    
    # Q18: Overall Satisfaction
    overall_satisfaction: Optional[LikertScale] = Field(None, description="Overall satisfaction rating")
    
    @validator('nps_score')
    def validate_nps(cls, v):
        if v is not None and not (0 <= v <= 10):
            raise ValueError('NPS score must be between 0 and 10')
        return v
    
    @validator('feature_importance', 'feature_usage')
    def validate_feature_ratings(cls, v):
        if v is not None:
            for feature, rating in v.items():
                if not (1 <= rating <= 5):
                    raise ValueError(f'Feature rating for {feature} must be between 1 and 5')
        return v

class AnalysisRequest(BaseModel):
    """Request model for running comprehensive analysis"""
    responses: List[SurveyResponse] = Field(..., description="Survey responses to analyze")
    analysis_type: str = Field(default="comprehensive", description="Type of analysis to run")
    include_llm_insights: bool = Field(default=True, description="Include LLM-generated insights")
    include_segmentation: bool = Field(default=True, description="Include customer segmentation")
    
    @validator('responses')
    def validate_minimum_responses(cls, v):
        if len(v) < 1:
            raise ValueError('At least 1 response is required for analysis')
        return v

# Analysis Result Models
class OverviewMetrics(BaseModel):
    """Overview statistics for the analysis"""
    total_responses: int
    completion_rate: float
    avg_satisfaction: float
    nps_score: float
    response_rate_by_course: Dict[str, int]
    
class CourseMetrics(BaseModel):
    """Detailed metrics for individual courses"""
    course_name: str
    response_count: int
    quality_score: float
    satisfaction_score: float
    nps_score: float
    instructor_ratings: Dict[str, float]
    completion_rate: float
    
class DemographicAnalysis(BaseModel):
    """Demographic breakdown and analysis"""
    gender_distribution: Dict[str, int]
    role_distribution: Dict[str, int]
    age_distribution: Dict[str, int]
    satisfaction_by_demographic: Dict[str, Dict[str, float]]

class StatisticalTests(BaseModel):
    """Results from statistical hypothesis testing"""
    anova_results: Dict[str, Any]
    correlation_matrix: Dict[str, Dict[str, float]]
    chi_square_tests: Dict[str, Dict[str, Any]]
    regression_analysis: Dict[str, Any]
    reliability_metrics: Dict[str, float]

class LLMInsight(BaseModel):
    """Individual insight generated by LLM"""
    type: str = Field(..., description="Type of insight (positive, concern, opportunity)")
    title: str = Field(..., description="Insight title")
    description: str = Field(..., description="Detailed description")
    priority: str = Field(..., description="Priority level (high, medium, low)")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    supporting_data: Dict[str, Any] = Field(default_factory=dict)

class LLMInsights(BaseModel):
    """Collection of LLM-generated insights"""
    sentiment_analysis: Dict[str, Any]
    topic_modeling: Dict[str, Any]
    insights: List[LLMInsight]
    recommendations: List[str]
    quality_score: float

class Segmentation(BaseModel):
    """Customer segmentation results"""
    segments: List[Dict[str, Any]]
    segment_characteristics: Dict[str, Any]
    satisfaction_by_segment: Dict[str, float]
    recommended_actions: Dict[str, List[str]]

class AnalysisResult(BaseModel):
    """Comprehensive analysis result model"""
    overview: OverviewMetrics
    course_metrics: List[CourseMetrics]
    demographics: DemographicAnalysis
    statistical_tests: StatisticalTests
    llm_insights: LLMInsights
    segmentation: Segmentation
    recommendations: List[str]
    timestamp: datetime
    metadata: Dict[str, Any]
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

# Feature Models
class FeatureImportance(BaseModel):
    """Feature importance vs usage analysis"""
    feature_name: str
    importance_score: float
    usage_score: float
    gap_score: float  # importance - usage
    priority_level: str

class TrendAnalysis(BaseModel):
    """Trend analysis over time"""
    metric_name: str
    time_series_data: List[Dict[str, Any]]
    trend_direction: str  # increasing, decreasing, stable
    significance: float
    forecast: Optional[List[Dict[str, Any]]] = None

# Database Models
class StoredAnalysis(BaseModel):
    """Stored analysis result in database"""
    analysis_id: str
    created_at: datetime
    analysis_result: AnalysisResult
    version: str
    tags: List[str] = Field(default_factory=list)

class Course(BaseModel):
    """Course information model"""
    course_id: str
    course_name: str
    course_type: CourseType
    instructor_id: str
    created_at: datetime
    is_active: bool = True
    
class Instructor(BaseModel):
    """Instructor information model"""
    instructor_id: str
    name: str
    email: str
    specialization: List[str]
    courses: List[str] = Field(default_factory=list)

# Configuration Models
class AnalysisConfig(BaseModel):
    """Configuration for analysis parameters"""
    confidence_level: float = Field(default=0.95, ge=0.8, le=0.99)
    minimum_sample_size: int = Field(default=5, ge=1)
    llm_model: str = Field(default="gpt-4.1-2025-04-14")
    statistical_tests: List[str] = Field(default_factory=lambda: ["anova", "correlation", "chi_square"])
    segmentation_method: str = Field(default="kmeans")
    num_clusters: int = Field(default=4, ge=2, le=10)