"""
LLM Service for Enhanced Text Analysis
Implements Phase 4 of the work plan: LLM Integration for sentiment analysis, 
topic modeling, and automated insight generation
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import re

from app.models.survey import SurveyResponse, LLMInsights, LLMInsight

logger = logging.getLogger(__name__)

class LLMService:
    """Service for LLM-powered text analysis and insight generation"""
    
    def __init__(self):
        self.client = None
        self.model = "gpt-4.1-2025-04-14"
        self.max_tokens = 4000
        self.temperature = 0.3
        
    async def initialize(self):
        """Initialize LLM client"""
        try:
            # In a real implementation, you would initialize your LLM client here
            # For now, we'll simulate the initialization
            logger.info("LLM service initialized successfully")
            return True
        except Exception as e:
            logger.error(f"LLM service initialization failed: {str(e)}")
            return False
    
    async def health_check(self) -> bool:
        """Check if LLM service is healthy"""
        try:
            # In a real implementation, you would ping the LLM API
            return True
        except Exception as e:
            logger.error(f"LLM health check failed: {str(e)}")
            return False
    
    async def generate_insights(
        self, 
        processed_data: List[SurveyResponse], 
        statistical_results: Dict[str, Any]
    ) -> LLMInsights:
        """
        Generate comprehensive insights using LLM analysis
        Implements Phase 4 of the work plan
        """
        try:
            logger.info("Generating LLM insights from survey data")
            
            # Extract text feedback for analysis
            text_feedback = self._extract_text_feedback(processed_data)
            
            # Sentiment analysis
            sentiment_analysis = await self._analyze_sentiment(text_feedback)
            
            # Topic modeling
            topic_modeling = await self._perform_topic_modeling(text_feedback)
            
            # Generate insights based on statistical results
            insights = await self._generate_statistical_insights(statistical_results)
            
            # Calculate overall quality score
            quality_score = await self._calculate_insight_quality(insights, statistical_results)
            
            return LLMInsights(
                sentiment_analysis=sentiment_analysis,
                topic_modeling=topic_modeling,
                insights=insights,
                recommendations=[],  # Will be generated separately
                quality_score=quality_score
            )
            
        except Exception as e:
            logger.error(f"LLM insight generation failed: {str(e)}")
            # Return default insights on failure
            return self._get_default_insights()
    
    async def generate_recommendations(
        self, 
        statistical_results: Dict[str, Any], 
        llm_insights: LLMInsights
    ) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        try:
            recommendations = []
            
            # Analyze course performance
            if 'course_metrics' in statistical_results:
                course_recommendations = await self._generate_course_recommendations(
                    statistical_results['course_metrics']
                )
                recommendations.extend(course_recommendations)
            
            # Analyze satisfaction drivers
            if 'tests' in statistical_results and 'regression_analysis' in statistical_results['tests']:
                regression_recommendations = await self._generate_regression_recommendations(
                    statistical_results['tests']['regression_analysis']
                )
                recommendations.extend(regression_recommendations)
            
            # Sentiment-based recommendations
            sentiment_recommendations = await self._generate_sentiment_recommendations(
                llm_insights.sentiment_analysis
            )
            recommendations.extend(sentiment_recommendations)
            
            return recommendations[:10]  # Limit to top 10 recommendations
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {str(e)}")
            return ["Review course content quality", "Improve instructor training", "Enhance student engagement"]
    
    def _extract_text_feedback(self, responses: List[SurveyResponse]) -> List[str]:
        """Extract text feedback from survey responses"""
        try:
            text_feedback = []
            for response in responses:
                if response.improvement_suggestions:
                    text_feedback.append(response.improvement_suggestions)
            return text_feedback
        except Exception as e:
            logger.error(f"Text extraction failed: {str(e)}")
            return []
    
    async def _analyze_sentiment(self, text_feedback: List[str]) -> Dict[str, Any]:
        """Analyze sentiment of text feedback"""
        try:
            if not text_feedback:
                return {"overall_sentiment": "neutral", "sentiment_distribution": {}, "confidence": 0.0}
            
            # Simulate sentiment analysis (in real implementation, use LLM API)
            positive_keywords = ['good', 'great', 'excellent', 'amazing', 'helpful', 'clear', 'useful']
            negative_keywords = ['bad', 'poor', 'terrible', 'confusing', 'difficult', 'boring', 'unclear']
            
            sentiment_scores = []
            positive_count = 0
            negative_count = 0
            neutral_count = 0
            
            for feedback in text_feedback:
                if not feedback:
                    continue
                    
                feedback_lower = feedback.lower()
                positive_score = sum(1 for word in positive_keywords if word in feedback_lower)
                negative_score = sum(1 for word in negative_keywords if word in feedback_lower)
                
                if positive_score > negative_score:
                    sentiment_scores.append('positive')
                    positive_count += 1
                elif negative_score > positive_score:
                    sentiment_scores.append('negative')
                    negative_count += 1
                else:
                    sentiment_scores.append('neutral')
                    neutral_count += 1
            
            total_feedback = len(sentiment_scores)
            if total_feedback == 0:
                return {"overall_sentiment": "neutral", "sentiment_distribution": {}, "confidence": 0.0}
            
            # Determine overall sentiment
            if positive_count > negative_count and positive_count > neutral_count:
                overall_sentiment = "positive"
            elif negative_count > positive_count and negative_count > neutral_count:
                overall_sentiment = "negative"
            else:
                overall_sentiment = "neutral"
            
            sentiment_distribution = {
                "positive": round(positive_count / total_feedback, 3),
                "negative": round(negative_count / total_feedback, 3),
                "neutral": round(neutral_count / total_feedback, 3)
            }
            
            # Calculate confidence based on how decisive the sentiment is
            max_sentiment = max(positive_count, negative_count, neutral_count)
            confidence = round(max_sentiment / total_feedback, 3)
            
            return {
                "overall_sentiment": overall_sentiment,
                "sentiment_distribution": sentiment_distribution,
                "confidence": confidence,
                "total_analyzed": total_feedback
            }
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {str(e)}")
            return {"overall_sentiment": "neutral", "sentiment_distribution": {}, "confidence": 0.0}
    
    async def _perform_topic_modeling(self, text_feedback: List[str]) -> Dict[str, Any]:
        """Perform topic modeling on text feedback"""
        try:
            if not text_feedback:
                return {"topics": [], "topic_distribution": {}}
            
            # Simulate topic modeling (in real implementation, use LLM for topic extraction)
            topics = {
                "content_quality": 0,
                "instructor_performance": 0,
                "course_structure": 0,
                "technical_issues": 0,
                "engagement": 0
            }
            
            # Simple keyword-based topic detection
            topic_keywords = {
                "content_quality": ["content", "material", "information", "quality", "topics"],
                "instructor_performance": ["instructor", "teacher", "teaching", "explanation", "clarity"],
                "course_structure": ["structure", "organization", "flow", "sequence", "pacing"],
                "technical_issues": ["technical", "platform", "video", "audio", "loading"],
                "engagement": ["engagement", "interactive", "participation", "discussion", "activity"]
            }
            
            for feedback in text_feedback:
                if not feedback:
                    continue
                    
                feedback_lower = feedback.lower()
                for topic, keywords in topic_keywords.items():
                    if any(keyword in feedback_lower for keyword in keywords):
                        topics[topic] += 1
            
            total_mentions = sum(topics.values())
            if total_mentions == 0:
                return {"topics": [], "topic_distribution": {}}
            
            # Convert to distribution
            topic_distribution = {
                topic: round(count / total_mentions, 3) 
                for topic, count in topics.items() 
                if count > 0
            }
            
            # Format topics for output
            formatted_topics = [
                {
                    "topic": topic,
                    "weight": weight,
                    "keywords": topic_keywords.get(topic, [])
                }
                for topic, weight in topic_distribution.items()
            ]
            
            return {
                "topics": formatted_topics,
                "topic_distribution": topic_distribution,
                "total_mentions": total_mentions
            }
            
        except Exception as e:
            logger.error(f"Topic modeling failed: {str(e)}")
            return {"topics": [], "topic_distribution": {}}
    
    async def _generate_statistical_insights(self, statistical_results: Dict[str, Any]) -> List[LLMInsight]:
        """Generate insights based on statistical analysis results"""
        try:
            insights = []
            
            # Analyze overview metrics
            if 'overview' in statistical_results:
                overview = statistical_results['overview']
                
                # NPS insight
                if hasattr(overview, 'nps_score'):
                    nps_score = overview.nps_score
                    if nps_score >= 50:
                        insights.append(LLMInsight(
                            type="positive",
                            title="Excellent Net Promoter Score",
                            description=f"NPS score of {nps_score} indicates strong customer advocacy and satisfaction.",
                            priority="low",
                            confidence=0.9,
                            supporting_data={"nps_score": nps_score}
                        ))
                    elif nps_score >= 0:
                        insights.append(LLMInsight(
                            type="opportunity",
                            title="Moderate NPS Performance",
                            description=f"NPS score of {nps_score} shows room for improvement in customer advocacy.",
                            priority="medium",
                            confidence=0.8,
                            supporting_data={"nps_score": nps_score}
                        ))
                    else:
                        insights.append(LLMInsight(
                            type="concern",
                            title="Low Net Promoter Score",
                            description=f"NPS score of {nps_score} indicates significant customer satisfaction issues.",
                            priority="high",
                            confidence=0.9,
                            supporting_data={"nps_score": nps_score}
                        ))
                
                # Satisfaction insight
                if hasattr(overview, 'avg_satisfaction'):
                    avg_satisfaction = overview.avg_satisfaction
                    if avg_satisfaction >= 4.0:
                        insights.append(LLMInsight(
                            type="positive",
                            title="High Overall Satisfaction",
                            description=f"Average satisfaction of {avg_satisfaction:.1f}/5 demonstrates strong course quality.",
                            priority="low",
                            confidence=0.8,
                            supporting_data={"avg_satisfaction": avg_satisfaction}
                        ))
                    elif avg_satisfaction >= 3.0:
                        insights.append(LLMInsight(
                            type="opportunity",
                            title="Moderate Satisfaction Levels",
                            description=f"Average satisfaction of {avg_satisfaction:.1f}/5 indicates potential for improvement.",
                            priority="medium",
                            confidence=0.7,
                            supporting_data={"avg_satisfaction": avg_satisfaction}
                        ))
                    else:
                        insights.append(LLMInsight(
                            type="concern",
                            title="Low Satisfaction Scores",
                            description=f"Average satisfaction of {avg_satisfaction:.1f}/5 requires immediate attention.",
                            priority="high",
                            confidence=0.9,
                            supporting_data={"avg_satisfaction": avg_satisfaction}
                        ))
            
            # Analyze course performance
            if 'course_metrics' in statistical_results:
                course_insights = await self._analyze_course_performance(statistical_results['course_metrics'])
                insights.extend(course_insights)
            
            # Analyze statistical test results
            if 'tests' in statistical_results:
                test_insights = await self._analyze_statistical_tests(statistical_results['tests'])
                insights.extend(test_insights)
            
            return insights[:10]  # Limit to top 10 insights
            
        except Exception as e:
            logger.error(f"Statistical insight generation failed: {str(e)}")
            return []
    
    async def _analyze_course_performance(self, course_metrics: List[Any]) -> List[LLMInsight]:
        """Analyze course performance and generate insights"""
        try:
            insights = []
            
            if not course_metrics:
                return insights
            
            # Find best and worst performing courses
            best_course = max(course_metrics, key=lambda x: getattr(x, 'satisfaction_score', 0))
            worst_course = min(course_metrics, key=lambda x: getattr(x, 'satisfaction_score', 0))
            
            # Best course insight
            if hasattr(best_course, 'satisfaction_score') and best_course.satisfaction_score >= 4.0:
                insights.append(LLMInsight(
                    type="positive",
                    title=f"Exceptional Performance: {best_course.course_name}",
                    description=f"{best_course.course_name} shows outstanding satisfaction score of {best_course.satisfaction_score:.1f}/5.",
                    priority="low",
                    confidence=0.8,
                    supporting_data={
                        "course": best_course.course_name,
                        "satisfaction": best_course.satisfaction_score
                    }
                ))
            
            # Worst course insight
            if hasattr(worst_course, 'satisfaction_score') and worst_course.satisfaction_score < 3.5:
                insights.append(LLMInsight(
                    type="concern",
                    title=f"Performance Issue: {worst_course.course_name}",
                    description=f"{worst_course.course_name} requires attention with satisfaction score of {worst_course.satisfaction_score:.1f}/5.",
                    priority="high",
                    confidence=0.8,
                    supporting_data={
                        "course": worst_course.course_name,
                        "satisfaction": worst_course.satisfaction_score
                    }
                ))
            
            return insights
            
        except Exception as e:
            logger.error(f"Course performance analysis failed: {str(e)}")
            return []
    
    async def _analyze_statistical_tests(self, test_results: Dict[str, Any]) -> List[LLMInsight]:
        """Analyze statistical test results for insights"""
        try:
            insights = []
            
            # Analyze ANOVA results
            if 'anova_results' in test_results:
                anova_insights = await self._analyze_anova_results(test_results['anova_results'])
                insights.extend(anova_insights)
            
            # Analyze correlation results
            if 'correlation_matrix' in test_results:
                correlation_insights = await self._analyze_correlations(test_results['correlation_matrix'])
                insights.extend(correlation_insights)
            
            return insights
            
        except Exception as e:
            logger.error(f"Statistical test analysis failed: {str(e)}")
            return []
    
    async def _analyze_anova_results(self, anova_results: Dict[str, Any]) -> List[LLMInsight]:
        """Analyze ANOVA test results"""
        try:
            insights = []
            
            for test_name, result in anova_results.items():
                if result.get('significant', False):
                    insights.append(LLMInsight(
                        type="opportunity",
                        title=f"Significant Difference Found: {test_name}",
                        description=f"Statistical analysis reveals significant differences in {test_name.replace('_', ' ')} (p < 0.05).",
                        priority="medium",
                        confidence=0.85,
                        supporting_data=result
                    ))
            
            return insights
            
        except Exception as e:
            logger.error(f"ANOVA analysis failed: {str(e)}")
            return []
    
    async def _analyze_correlations(self, correlation_matrix: Dict[str, Dict[str, float]]) -> List[LLMInsight]:
        """Analyze correlation matrix for insights"""
        try:
            insights = []
            
            # Find strong correlations
            for var1, correlations in correlation_matrix.items():
                for var2, correlation in correlations.items():
                    if abs(correlation) > 0.7:  # Strong correlation threshold
                        correlation_type = "positive" if correlation > 0 else "negative"
                        insights.append(LLMInsight(
                            type="opportunity",
                            title=f"Strong {correlation_type.title()} Correlation",
                            description=f"Strong {correlation_type} correlation ({correlation:.3f}) between {var1} and {var2}.",
                            priority="medium",
                            confidence=0.7,
                            supporting_data={
                                "variable1": var1,
                                "variable2": var2,
                                "correlation": correlation
                            }
                        ))
            
            return insights[:3]  # Limit correlation insights
            
        except Exception as e:
            logger.error(f"Correlation analysis failed: {str(e)}")
            return []
    
    async def _calculate_insight_quality(
        self, 
        insights: List[LLMInsight], 
        statistical_results: Dict[str, Any]
    ) -> float:
        """Calculate overall quality score for generated insights"""
        try:
            if not insights:
                return 0.0
            
            # Calculate based on confidence scores and insight diversity
            avg_confidence = sum(insight.confidence for insight in insights) / len(insights)
            
            # Diversity bonus (different types of insights)
            insight_types = set(insight.type for insight in insights)
            diversity_score = len(insight_types) / 3  # Max 3 types (positive, concern, opportunity)
            
            # Statistical coverage bonus
            coverage_score = 0.0
            if 'overview' in statistical_results:
                coverage_score += 0.3
            if 'course_metrics' in statistical_results:
                coverage_score += 0.3
            if 'tests' in statistical_results:
                coverage_score += 0.4
            
            quality_score = (avg_confidence * 0.5) + (diversity_score * 0.3) + (coverage_score * 0.2)
            
            return min(1.0, quality_score)
            
        except Exception as e:
            logger.error(f"Quality calculation failed: {str(e)}")
            return 0.5
    
    async def _generate_course_recommendations(self, course_metrics: List[Any]) -> List[str]:
        """Generate course-specific recommendations"""
        try:
            recommendations = []
            
            for course in course_metrics:
                if hasattr(course, 'satisfaction_score'):
                    if course.satisfaction_score < 3.5:
                        recommendations.append(f"Review and restructure {course.course_name} content")
                        recommendations.append(f"Provide additional instructor training for {course.course_name}")
                    elif course.satisfaction_score >= 4.5:
                        recommendations.append(f"Scale successful practices from {course.course_name} to other courses")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Course recommendation generation failed: {str(e)}")
            return []
    
    async def _generate_regression_recommendations(self, regression_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on regression analysis"""
        try:
            recommendations = []
            
            if 'coefficients' in regression_results:
                coefficients = regression_results['coefficients']
                
                # Find strongest predictors
                strongest_predictor = max(coefficients, key=lambda x: abs(coefficients[x]))
                recommendations.append(f"Focus on improving {strongest_predictor} as it has the strongest impact on satisfaction")
                
                # Find negative predictors
                negative_predictors = [var for var, coef in coefficients.items() if coef < 0]
                if negative_predictors:
                    recommendations.append(f"Address issues with {', '.join(negative_predictors)} as they negatively impact satisfaction")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Regression recommendation generation failed: {str(e)}")
            return []
    
    async def _generate_sentiment_recommendations(self, sentiment_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on sentiment analysis"""
        try:
            recommendations = []
            
            overall_sentiment = sentiment_analysis.get('overall_sentiment', 'neutral')
            
            if overall_sentiment == 'negative':
                recommendations.append("Address negative feedback themes identified in student comments")
                recommendations.append("Implement feedback collection process to identify specific pain points")
            elif overall_sentiment == 'positive':
                recommendations.append("Leverage positive feedback for testimonials and marketing")
                recommendations.append("Identify and replicate positive experience drivers")
            else:
                recommendations.append("Increase student engagement to generate more positive experiences")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Sentiment recommendation generation failed: {str(e)}")
            return []
    
    def _get_default_insights(self) -> LLMInsights:
        """Return default insights when LLM analysis fails"""
        return LLMInsights(
            sentiment_analysis={"overall_sentiment": "neutral", "confidence": 0.0},
            topic_modeling={"topics": [], "topic_distribution": {}},
            insights=[
                LLMInsight(
                    type="opportunity",
                    title="Analysis Available",
                    description="Basic statistical analysis completed. Enhanced insights require additional data.",
                    priority="low",
                    confidence=0.5,
                    supporting_data={}
                )
            ],
            recommendations=["Collect more detailed feedback", "Increase sample size for better insights"],
            quality_score=0.5
        )