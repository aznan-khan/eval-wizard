"""
Data Processing Service
Implements Phase 1-2 of the work plan: Data preparation, validation, and EDA setup
"""

import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import asyncio

from app.models.survey import SurveyResponse, CourseType, GenderType, RoleType

logger = logging.getLogger(__name__)

class DataProcessor:
    """Service for processing and validating survey data"""
    
    def __init__(self):
        self.required_fields = ['response_id', 'course_type']
        self.optional_fields = [
            'gender', 'age_group', 'primary_role', 'course_quality',
            'overall_satisfaction', 'nps_score', 'instructor_clarity'
        ]
    
    async def process_survey_data(self, responses: List[SurveyResponse]) -> List[SurveyResponse]:
        """
        Process and clean survey data
        Implements Phase 1 data preparation
        """
        try:
            logger.info(f"Processing {len(responses)} survey responses")
            
            # Data validation
            validated_responses = await self._validate_responses(responses)
            
            # Data cleaning
            cleaned_responses = await self._clean_responses(validated_responses)
            
            # Quality checks
            quality_report = await self._generate_quality_report(cleaned_responses)
            logger.info(f"Data quality report: {quality_report}")
            
            return cleaned_responses
            
        except Exception as e:
            logger.error(f"Data processing failed: {str(e)}")
            raise
    
    async def validate_survey_responses(self, file_data: Dict[str, Any]) -> List[SurveyResponse]:
        """
        Validate survey responses from uploaded file
        """
        try:
            responses = []
            
            # Handle different file formats
            if 'responses' in file_data:
                raw_responses = file_data['responses']
            elif isinstance(file_data, list):
                raw_responses = file_data
            else:
                raise ValueError("Invalid file format. Expected 'responses' key or array of responses")
            
            # Validate each response
            for i, raw_response in enumerate(raw_responses):
                try:
                    # Add response_id if missing
                    if 'response_id' not in raw_response:
                        raw_response['response_id'] = f"response_{i+1}_{datetime.utcnow().timestamp()}"
                    
                    # Add timestamp if missing
                    if 'timestamp' not in raw_response:
                        raw_response['timestamp'] = datetime.utcnow()
                    
                    # Validate and create response object
                    response = SurveyResponse(**raw_response)
                    responses.append(response)
                    
                except Exception as e:
                    logger.warning(f"Skipping invalid response {i+1}: {str(e)}")
                    continue
            
            logger.info(f"Validated {len(responses)} out of {len(raw_responses)} responses")
            return responses
            
        except Exception as e:
            logger.error(f"Response validation failed: {str(e)}")
            raise
    
    async def _validate_responses(self, responses: List[SurveyResponse]) -> List[SurveyResponse]:
        """Validate individual survey responses"""
        try:
            validated_responses = []
            
            for response in responses:
                # Check required fields
                if not response.response_id or not response.course_type:
                    logger.warning(f"Skipping response with missing required fields: {response.response_id}")
                    continue
                
                # Validate enum values
                if response.course_type and response.course_type not in CourseType:
                    logger.warning(f"Invalid course type: {response.course_type}")
                    continue
                
                if response.gender and response.gender not in GenderType:
                    logger.warning(f"Invalid gender: {response.gender}")
                    continue
                
                if response.primary_role and response.primary_role not in RoleType:
                    logger.warning(f"Invalid role: {response.primary_role}")
                    continue
                
                # Validate rating scales
                if response.nps_score is not None and not (0 <= response.nps_score <= 10):
                    logger.warning(f"Invalid NPS score: {response.nps_score}")
                    response.nps_score = None
                
                # Validate Likert scales (1-5)
                likert_fields = [
                    'course_quality', 'course_relevance', 'overall_satisfaction',
                    'instructor_clarity', 'instructor_knowledge', 'instructor_engagement',
                    'instructor_responsiveness', 'learning_objectives_met',
                    'course_organization', 'content_difficulty', 'learning_improvement'
                ]
                
                for field in likert_fields:
                    value = getattr(response, field, None)
                    if value is not None and not (1 <= value <= 5):
                        logger.warning(f"Invalid {field} score: {value}")
                        setattr(response, field, None)
                
                validated_responses.append(response)
            
            logger.info(f"Validated {len(validated_responses)} responses")
            return validated_responses
            
        except Exception as e:
            logger.error(f"Response validation failed: {str(e)}")
            raise
    
    async def _clean_responses(self, responses: List[SurveyResponse]) -> List[SurveyResponse]:
        """Clean and standardize response data"""
        try:
            cleaned_responses = []
            
            for response in responses:
                # Clean text fields
                if response.improvement_suggestions:
                    response.improvement_suggestions = self._clean_text(response.improvement_suggestions)
                
                if response.professional_background:
                    response.professional_background = self._clean_text(response.professional_background)
                
                if response.location:
                    response.location = self._clean_text(response.location)
                
                # Standardize age groups
                if response.age_group:
                    response.age_group = self._standardize_age_group(response.age_group)
                
                # Clean feature ratings
                if response.feature_importance:
                    response.feature_importance = self._clean_feature_ratings(response.feature_importance)
                
                if response.feature_usage:
                    response.feature_usage = self._clean_feature_ratings(response.feature_usage)
                
                cleaned_responses.append(response)
            
            return cleaned_responses
            
        except Exception as e:
            logger.error(f"Data cleaning failed: {str(e)}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """Clean and standardize text fields"""
        if not text:
            return text
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove special characters that might cause issues
        text = text.strip()
        
        return text
    
    def _standardize_age_group(self, age_group: str) -> str:
        """Standardize age group format"""
        if not age_group:
            return age_group
        
        # Convert to standard format
        age_mapping = {
            '18-25': '18-25',
            '26-35': '26-35',
            '36-45': '36-45',
            '46-55': '46-55',
            '56-65': '56-65',
            '65+': '65+',
            'under 18': 'Under 18',
            'over 65': '65+'
        }
        
        age_lower = age_group.lower().strip()
        return age_mapping.get(age_lower, age_group)
    
    def _clean_feature_ratings(self, feature_ratings: Dict[str, int]) -> Dict[str, int]:
        """Clean and validate feature ratings"""
        if not feature_ratings:
            return feature_ratings
        
        cleaned_ratings = {}
        for feature, rating in feature_ratings.items():
            # Validate rating is in range 1-5
            if isinstance(rating, (int, float)) and 1 <= rating <= 5:
                cleaned_ratings[feature.strip().lower()] = int(rating)
            else:
                logger.warning(f"Invalid feature rating for {feature}: {rating}")
        
        return cleaned_ratings
    
    async def _generate_quality_report(self, responses: List[SurveyResponse]) -> Dict[str, Any]:
        """Generate data quality report"""
        try:
            if not responses:
                return {"error": "No responses to analyze"}
            
            total_responses = len(responses)
            
            # Completeness analysis
            completeness = {}
            field_counts = {}
            
            for response in responses:
                response_dict = response.dict()
                for field, value in response_dict.items():
                    if field not in field_counts:
                        field_counts[field] = 0
                    
                    if value is not None and value != "" and value != []:
                        field_counts[field] += 1
            
            # Calculate completeness percentages
            for field, count in field_counts.items():
                completeness[field] = round((count / total_responses) * 100, 2)
            
            # Data consistency checks
            consistency_issues = []
            
            # Check for responses with very high/low ratings across all dimensions
            for response in responses:
                ratings = []
                likert_fields = ['course_quality', 'overall_satisfaction', 'instructor_clarity']
                
                for field in likert_fields:
                    value = getattr(response, field, None)
                    if value is not None:
                        ratings.append(value)
                
                if len(ratings) >= 2:
                    if all(r == 5 for r in ratings):
                        consistency_issues.append(f"Response {response.response_id}: All maximum ratings")
                    elif all(r == 1 for r in ratings):
                        consistency_issues.append(f"Response {response.response_id}: All minimum ratings")
            
            # Course distribution
            course_distribution = {}
            for response in responses:
                course = response.course_type
                course_distribution[course] = course_distribution.get(course, 0) + 1
            
            return {
                "total_responses": total_responses,
                "completeness": completeness,
                "consistency_issues": len(consistency_issues),
                "course_distribution": course_distribution,
                "data_quality_score": self._calculate_quality_score(completeness, consistency_issues, total_responses)
            }
            
        except Exception as e:
            logger.error(f"Quality report generation failed: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_quality_score(self, completeness: Dict[str, float], consistency_issues: List[str], total_responses: int) -> float:
        """Calculate overall data quality score (0-100)"""
        try:
            # Base score from completeness of key fields
            key_fields = ['course_type', 'overall_satisfaction', 'course_quality']
            key_completeness = [completeness.get(field, 0) for field in key_fields if field in completeness]
            
            if not key_completeness:
                return 0.0
            
            avg_completeness = sum(key_completeness) / len(key_completeness)
            
            # Penalty for consistency issues
            consistency_penalty = min(20, (len(consistency_issues) / total_responses) * 100)
            
            quality_score = max(0, avg_completeness - consistency_penalty)
            
            return round(quality_score, 2)
            
        except Exception as e:
            logger.error(f"Quality score calculation failed: {str(e)}")
            return 50.0
    
    async def generate_mock_data(self, num_responses: int = 17) -> List[SurveyResponse]:
        """Generate mock survey data for testing"""
        try:
            import random
            from datetime import timedelta
            
            responses = []
            courses = list(CourseType)
            genders = list(GenderType)
            roles = list(RoleType)
            age_groups = ['18-25', '26-35', '36-45', '46-55', '56-65']
            
            for i in range(num_responses):
                # Generate realistic ratings with some correlation
                base_satisfaction = random.choice([3, 4, 4, 4, 5])  # Bias toward positive
                
                # Correlated ratings (satisfied users tend to rate everything higher)
                satisfaction_modifier = (base_satisfaction - 3) * 0.3
                
                response = SurveyResponse(
                    response_id=f"mock_response_{i+1}",
                    timestamp=datetime.utcnow() - timedelta(days=random.randint(0, 30)),
                    course_type=random.choice(courses),
                    gender=random.choice(genders),
                    age_group=random.choice(age_groups),
                    primary_role=random.choice(roles),
                    course_quality=max(1, min(5, int(base_satisfaction + random.uniform(-0.5, 0.5) + satisfaction_modifier))),
                    course_relevance=max(1, min(5, int(base_satisfaction + random.uniform(-0.5, 0.5) + satisfaction_modifier))),
                    instructor_clarity=max(1, min(5, int(base_satisfaction + random.uniform(-0.5, 0.5) + satisfaction_modifier))),
                    instructor_knowledge=max(1, min(5, int(base_satisfaction + random.uniform(-0.5, 0.5) + satisfaction_modifier))),
                    instructor_engagement=max(1, min(5, int(base_satisfaction + random.uniform(-0.5, 0.5) + satisfaction_modifier))),
                    instructor_responsiveness=max(1, min(5, int(base_satisfaction + random.uniform(-0.5, 0.5) + satisfaction_modifier))),
                    learning_objectives_met=max(1, min(5, int(base_satisfaction + random.uniform(-0.5, 0.5) + satisfaction_modifier))),
                    course_organization=max(1, min(5, int(base_satisfaction + random.uniform(-0.5, 0.5) + satisfaction_modifier))),
                    content_difficulty=random.randint(2, 4),  # Difficulty is independent
                    learning_improvement=max(1, min(5, int(base_satisfaction + random.uniform(-0.5, 0.5) + satisfaction_modifier))),
                    nps_score=max(0, min(10, int((base_satisfaction - 1) * 2.5 + random.uniform(-1, 1)))),
                    overall_satisfaction=base_satisfaction,
                    improvement_suggestions=random.choice([
                        "More interactive content would be helpful",
                        "Course pacing could be improved",
                        "Excellent course, keep it up!",
                        "More real-world examples needed",
                        "Great instructor and content",
                        None
                    ]) if random.random() > 0.3 else None,
                    platform_preference=random.choice(['mobile', 'desktop', 'both']),
                    feature_importance={
                        'video_quality': random.randint(3, 5),
                        'interactive_exercises': random.randint(3, 5),
                        'downloadable_resources': random.randint(2, 5),
                        'discussion_forums': random.randint(2, 4),
                        'mobile_access': random.randint(3, 5)
                    },
                    feature_usage={
                        'video_quality': random.randint(2, 5),
                        'interactive_exercises': random.randint(2, 4),
                        'downloadable_resources': random.randint(1, 4),
                        'discussion_forums': random.randint(1, 3),
                        'mobile_access': random.randint(2, 5)
                    }
                )
                
                responses.append(response)
            
            logger.info(f"Generated {len(responses)} mock responses")
            return responses
            
        except Exception as e:
            logger.error(f"Mock data generation failed: {str(e)}")
            raise