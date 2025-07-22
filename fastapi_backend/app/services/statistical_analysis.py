"""
Statistical Analysis Service
Implements Phases 2-3 and 5 of the work plan: EDA, Statistical Testing, and Advanced Analytics
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, classification_report
from typing import Dict, List, Any, Tuple, Optional
import logging
from datetime import datetime, timedelta

from app.models.survey import (
    SurveyResponse, OverviewMetrics, CourseMetrics, 
    DemographicAnalysis, StatisticalTests, Segmentation
)

logger = logging.getLogger(__name__)

class StatisticalAnalysisService:
    """Comprehensive statistical analysis service for survey data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    async def run_comprehensive_analysis(self, responses: List[SurveyResponse]) -> Dict[str, Any]:
        """
        Run comprehensive statistical analysis covering Phases 2-3 of work plan
        """
        try:
            logger.info(f"Starting comprehensive analysis for {len(responses)} responses")
            
            # Convert to DataFrame for analysis
            df = self._responses_to_dataframe(responses)
            
            # Phase 2: Exploratory Data Analysis
            overview = await self._calculate_overview_metrics(df)
            course_metrics = await self._analyze_courses(df)
            demographics = await self._analyze_demographics(df)
            
            # Phase 3: Statistical Testing
            statistical_tests = await self._run_statistical_tests(df)
            
            return {
                "overview": overview,
                "course_metrics": course_metrics,
                "demographics": demographics,
                "tests": statistical_tests
            }
            
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {str(e)}")
            raise
    
    async def _calculate_overview_metrics(self, df: pd.DataFrame) -> OverviewMetrics:
        """Calculate overview metrics for dashboard"""
        try:
            total_responses = len(df)
            completion_rate = (df.notna().all(axis=1).sum() / total_responses) * 100
            
            # Calculate average satisfaction (Q18)
            avg_satisfaction = df['overall_satisfaction'].mean() if 'overall_satisfaction' in df.columns else 0
            
            # Calculate NPS score
            nps_score = await self._calculate_nps(df)
            
            # Response rate by course
            response_by_course = df['course_type'].value_counts().to_dict()
            
            return OverviewMetrics(
                total_responses=total_responses,
                completion_rate=completion_rate,
                avg_satisfaction=avg_satisfaction,
                nps_score=nps_score,
                response_rate_by_course=response_by_course
            )
            
        except Exception as e:
            logger.error(f"Overview metrics calculation failed: {str(e)}")
            raise
    
    async def _calculate_nps(self, df: pd.DataFrame) -> float:
        """Calculate Net Promoter Score"""
        try:
            if 'nps_score' not in df.columns:
                return 0.0
            
            nps_scores = df['nps_score'].dropna()
            if len(nps_scores) == 0:
                return 0.0
            
            promoters = (nps_scores >= 9).sum()
            detractors = (nps_scores <= 6).sum()
            total = len(nps_scores)
            
            nps = ((promoters - detractors) / total) * 100
            return round(nps, 2)
            
        except Exception as e:
            logger.error(f"NPS calculation failed: {str(e)}")
            return 0.0
    
    async def _analyze_courses(self, df: pd.DataFrame) -> List[CourseMetrics]:
        """Analyze metrics for each course"""
        try:
            course_metrics = []
            
            for course in df['course_type'].unique():
                course_data = df[df['course_type'] == course]
                
                # Basic metrics
                response_count = len(course_data)
                quality_score = course_data['course_quality'].mean() if 'course_quality' in course_data.columns else 0
                satisfaction_score = course_data['overall_satisfaction'].mean() if 'overall_satisfaction' in course_data.columns else 0
                
                # Course-specific NPS
                course_nps = await self._calculate_nps(course_data)
                
                # Instructor ratings
                instructor_ratings = {}
                instructor_cols = ['instructor_clarity', 'instructor_knowledge', 'instructor_engagement', 'instructor_responsiveness']
                for col in instructor_cols:
                    if col in course_data.columns:
                        instructor_ratings[col] = course_data[col].mean()
                
                # Completion rate (assume 100% for now, can be enhanced)
                completion_rate = 100.0
                
                course_metrics.append(CourseMetrics(
                    course_name=course,
                    response_count=response_count,
                    quality_score=quality_score,
                    satisfaction_score=satisfaction_score,
                    nps_score=course_nps,
                    instructor_ratings=instructor_ratings,
                    completion_rate=completion_rate
                ))
            
            return course_metrics
            
        except Exception as e:
            logger.error(f"Course analysis failed: {str(e)}")
            raise
    
    async def _analyze_demographics(self, df: pd.DataFrame) -> DemographicAnalysis:
        """Analyze demographic distributions and satisfaction by demographic"""
        try:
            # Gender distribution
            gender_dist = df['gender'].value_counts().to_dict() if 'gender' in df.columns else {}
            
            # Role distribution
            role_dist = df['primary_role'].value_counts().to_dict() if 'primary_role' in df.columns else {}
            
            # Age distribution
            age_dist = df['age_group'].value_counts().to_dict() if 'age_group' in df.columns else {}
            
            # Satisfaction by demographic
            satisfaction_by_demo = {}
            
            if 'overall_satisfaction' in df.columns:
                # Satisfaction by gender
                if 'gender' in df.columns:
                    satisfaction_by_demo['gender'] = df.groupby('gender')['overall_satisfaction'].mean().to_dict()
                
                # Satisfaction by role
                if 'primary_role' in df.columns:
                    satisfaction_by_demo['role'] = df.groupby('primary_role')['overall_satisfaction'].mean().to_dict()
            
            return DemographicAnalysis(
                gender_distribution=gender_dist,
                role_distribution=role_dist,
                age_distribution=age_dist,
                satisfaction_by_demographic=satisfaction_by_demo
            )
            
        except Exception as e:
            logger.error(f"Demographic analysis failed: {str(e)}")
            raise
    
    async def _run_statistical_tests(self, df: pd.DataFrame) -> StatisticalTests:
        """Run comprehensive statistical tests (Phase 3 of work plan)"""
        try:
            # ANOVA tests
            anova_results = await self._run_anova_tests(df)
            
            # Correlation analysis
            correlation_matrix = await self._calculate_correlations(df)
            
            # Chi-square tests
            chi_square_tests = await self._run_chi_square_tests(df)
            
            # Regression analysis
            regression_analysis = await self._run_regression_analysis(df)
            
            # Reliability metrics (Cronbach's Alpha)
            reliability_metrics = await self._calculate_reliability(df)
            
            return StatisticalTests(
                anova_results=anova_results,
                correlation_matrix=correlation_matrix,
                chi_square_tests=chi_square_tests,
                regression_analysis=regression_analysis,
                reliability_metrics=reliability_metrics
            )
            
        except Exception as e:
            logger.error(f"Statistical tests failed: {str(e)}")
            raise
    
    async def _run_anova_tests(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run ANOVA tests comparing satisfaction across groups"""
        try:
            anova_results = {}
            
            # Test satisfaction differences across courses
            if 'overall_satisfaction' in df.columns and 'course_type' in df.columns:
                groups = [group['overall_satisfaction'].dropna() for name, group in df.groupby('course_type')]
                if len(groups) > 1 and all(len(g) > 0 for g in groups):
                    f_stat, p_value = stats.f_oneway(*groups)
                    anova_results['satisfaction_by_course'] = {
                        'f_statistic': f_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
            
            # Test satisfaction differences by role
            if 'overall_satisfaction' in df.columns and 'primary_role' in df.columns:
                groups = [group['overall_satisfaction'].dropna() for name, group in df.groupby('primary_role')]
                if len(groups) > 1 and all(len(g) > 0 for g in groups):
                    f_stat, p_value = stats.f_oneway(*groups)
                    anova_results['satisfaction_by_role'] = {
                        'f_statistic': f_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
            
            return anova_results
            
        except Exception as e:
            logger.error(f"ANOVA tests failed: {str(e)}")
            return {}
    
    async def _calculate_correlations(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate correlation matrix for numerical variables"""
        try:
            # Select numerical columns for correlation
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numerical_cols) < 2:
                return {}
            
            correlation_matrix = df[numerical_cols].corr().to_dict()
            
            # Clean up correlation matrix (remove self-correlations and NaN values)
            cleaned_matrix = {}
            for col1 in correlation_matrix:
                cleaned_matrix[col1] = {}
                for col2 in correlation_matrix[col1]:
                    if col1 != col2 and not np.isnan(correlation_matrix[col1][col2]):
                        cleaned_matrix[col1][col2] = round(correlation_matrix[col1][col2], 3)
            
            return cleaned_matrix
            
        except Exception as e:
            logger.error(f"Correlation calculation failed: {str(e)}")
            return {}
    
    async def _run_chi_square_tests(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Run chi-square tests for categorical variables"""
        try:
            chi_square_results = {}
            
            # Test course preference by gender
            if 'course_type' in df.columns and 'gender' in df.columns:
                contingency_table = pd.crosstab(df['course_type'], df['gender'])
                if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
                    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                    chi_square_results['course_by_gender'] = {
                        'chi2_statistic': chi2,
                        'p_value': p_value,
                        'degrees_of_freedom': dof,
                        'significant': p_value < 0.05
                    }
            
            # Test role distribution across courses
            if 'course_type' in df.columns and 'primary_role' in df.columns:
                contingency_table = pd.crosstab(df['course_type'], df['primary_role'])
                if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
                    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                    chi_square_results['role_by_course'] = {
                        'chi2_statistic': chi2,
                        'p_value': p_value,
                        'degrees_of_freedom': dof,
                        'significant': p_value < 0.05
                    }
            
            return chi_square_results
            
        except Exception as e:
            logger.error(f"Chi-square tests failed: {str(e)}")
            return {}
    
    async def _run_regression_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run multiple regression analysis to predict satisfaction"""
        try:
            if 'overall_satisfaction' not in df.columns:
                return {}
            
            # Select predictor variables
            predictor_cols = [
                'course_quality', 'instructor_clarity', 'learning_objectives_met', 'course_organization'
            ]
            available_predictors = [col for col in predictor_cols if col in df.columns]
            
            if len(available_predictors) < 2:
                return {}
            
            # Prepare data
            X = df[available_predictors].dropna()
            y = df.loc[X.index, 'overall_satisfaction']
            
            if len(X) < 5:  # Minimum sample size
                return {}
            
            # Fit regression model
            model = LinearRegression()
            model.fit(X, y)
            
            # Calculate metrics
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            
            # Coefficients
            coefficients = dict(zip(available_predictors, model.coef_))
            
            return {
                'r_squared': r2,
                'coefficients': coefficients,
                'intercept': model.intercept_,
                'predictors_used': available_predictors,
                'sample_size': len(X)
            }
            
        except Exception as e:
            logger.error(f"Regression analysis failed: {str(e)}")
            return {}
    
    async def _calculate_reliability(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate Cronbach's Alpha for scale reliability"""
        try:
            reliability_metrics = {}
            
            # Instructor rating scale reliability
            instructor_cols = ['instructor_clarity', 'instructor_knowledge', 'instructor_engagement', 'instructor_responsiveness']
            available_instructor_cols = [col for col in instructor_cols if col in df.columns]
            
            if len(available_instructor_cols) >= 2:
                instructor_data = df[available_instructor_cols].dropna()
                if len(instructor_data) > 2:
                    alpha = self._cronbach_alpha(instructor_data)
                    reliability_metrics['instructor_scale'] = alpha
            
            return reliability_metrics
            
        except Exception as e:
            logger.error(f"Reliability calculation failed: {str(e)}")
            return {}
    
    def _cronbach_alpha(self, df: pd.DataFrame) -> float:
        """Calculate Cronbach's Alpha"""
        try:
            # Number of items
            k = len(df.columns)
            
            # Variance of each item
            item_variances = df.var(axis=0, ddof=1).sum()
            
            # Variance of the total score
            total_variance = df.sum(axis=1).var(ddof=1)
            
            # Cronbach's Alpha
            alpha = (k / (k - 1)) * (1 - item_variances / total_variance)
            
            return round(alpha, 3)
            
        except Exception as e:
            logger.error(f"Cronbach's Alpha calculation failed: {str(e)}")
            return 0.0
    
    async def perform_segmentation(self, responses: List[SurveyResponse]) -> Segmentation:
        """
        Perform customer segmentation analysis (Phase 5 of work plan)
        """
        try:
            df = self._responses_to_dataframe(responses)
            
            # Prepare features for clustering
            feature_cols = [
                'overall_satisfaction', 'course_quality', 'nps_score',
                'instructor_clarity', 'learning_objectives_met'
            ]
            available_features = [col for col in feature_cols if col in df.columns]
            
            if len(available_features) < 2:
                return Segmentation(
                    segments=[],
                    segment_characteristics={},
                    satisfaction_by_segment={},
                    recommended_actions={}
                )
            
            # Prepare data for clustering
            clustering_data = df[available_features].dropna()
            
            if len(clustering_data) < 4:  # Minimum for clustering
                return Segmentation(
                    segments=[],
                    segment_characteristics={},
                    satisfaction_by_segment={},
                    recommended_actions={}
                )
            
            # Scale features
            scaled_features = self.scaler.fit_transform(clustering_data)
            
            # Determine optimal number of clusters (up to 4)
            n_clusters = min(4, len(clustering_data) // 2)
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(scaled_features)
            
            # Add cluster labels to dataframe
            clustering_data['cluster'] = cluster_labels
            
            # Analyze segments
            segments = []
            segment_characteristics = {}
            satisfaction_by_segment = {}
            recommended_actions = {}
            
            for cluster_id in range(n_clusters):
                cluster_data = clustering_data[clustering_data['cluster'] == cluster_id]
                
                # Segment characteristics
                characteristics = {}
                for feature in available_features:
                    characteristics[feature] = {
                        'mean': cluster_data[feature].mean(),
                        'std': cluster_data[feature].std()
                    }
                
                # Segment description
                segment_name = f"Segment_{cluster_id + 1}"
                avg_satisfaction = cluster_data['overall_satisfaction'].mean() if 'overall_satisfaction' in cluster_data.columns else 0
                
                segments.append({
                    'id': cluster_id,
                    'name': segment_name,
                    'size': len(cluster_data),
                    'avg_satisfaction': avg_satisfaction,
                    'characteristics': characteristics
                })
                
                segment_characteristics[segment_name] = characteristics
                satisfaction_by_segment[segment_name] = avg_satisfaction
                
                # Generate recommendations based on satisfaction level
                if avg_satisfaction >= 4.0:
                    recommended_actions[segment_name] = [
                        "Maintain high quality",
                        "Leverage as advocates",
                        "Gather success stories"
                    ]
                elif avg_satisfaction >= 3.0:
                    recommended_actions[segment_name] = [
                        "Identify improvement areas",
                        "Increase engagement",
                        "Address specific concerns"
                    ]
                else:
                    recommended_actions[segment_name] = [
                        "Urgent intervention required",
                        "Deep dive into issues",
                        "Consider course restructuring"
                    ]
            
            return Segmentation(
                segments=segments,
                segment_characteristics=segment_characteristics,
                satisfaction_by_segment=satisfaction_by_segment,
                recommended_actions=recommended_actions
            )
            
        except Exception as e:
            logger.error(f"Segmentation analysis failed: {str(e)}")
            return Segmentation(
                segments=[],
                segment_characteristics={},
                satisfaction_by_segment={},
                recommended_actions={}
            )
    
    async def analyze_course(self, course_responses: List[SurveyResponse]) -> Dict[str, Any]:
        """Analyze a specific course in detail"""
        try:
            if not course_responses:
                return {}
            
            df = self._responses_to_dataframe(course_responses)
            
            # Basic metrics
            basic_metrics = {
                'response_count': len(df),
                'completion_rate': 100.0,  # Assuming all responses are complete
                'avg_satisfaction': df['overall_satisfaction'].mean() if 'overall_satisfaction' in df.columns else 0,
                'avg_quality': df['course_quality'].mean() if 'course_quality' in df.columns else 0
            }
            
            # Instructor analysis
            instructor_metrics = {}
            instructor_cols = ['instructor_clarity', 'instructor_knowledge', 'instructor_engagement', 'instructor_responsiveness']
            for col in instructor_cols:
                if col in df.columns:
                    instructor_metrics[col] = {
                        'mean': df[col].mean(),
                        'std': df[col].std()
                    }
            
            # Strengths and weaknesses
            strengths = []
            weaknesses = []
            
            if 'course_quality' in df.columns and df['course_quality'].mean() >= 4.0:
                strengths.append("High content quality")
            
            if 'overall_satisfaction' in df.columns and df['overall_satisfaction'].mean() >= 4.0:
                strengths.append("High overall satisfaction")
            
            # Feature analysis (if available)
            feature_analysis = {}
            if 'feature_importance' in df.columns and 'feature_usage' in df.columns:
                # This would require more complex analysis of feature dictionaries
                pass
            
            return {
                'basic_metrics': basic_metrics,
                'instructor_metrics': instructor_metrics,
                'strengths': strengths,
                'weaknesses': weaknesses,
                'feature_analysis': feature_analysis
            }
            
        except Exception as e:
            logger.error(f"Course analysis failed: {str(e)}")
            return {}
    
    async def calculate_trends(self, time_period: str) -> Dict[str, Any]:
        """Calculate satisfaction trends over time"""
        try:
            # This would require historical data storage
            # For now, return mock trend data
            return {
                'satisfaction_trend': {
                    'direction': 'stable',
                    'change_percentage': 2.3,
                    'significance': 0.15
                },
                'nps_trend': {
                    'direction': 'increasing',
                    'change_percentage': 5.7,
                    'significance': 0.03
                },
                'course_quality_trend': {
                    'direction': 'increasing',
                    'change_percentage': 3.1,
                    'significance': 0.08
                }
            }
            
        except Exception as e:
            logger.error(f"Trend calculation failed: {str(e)}")
            return {}
    
    def _responses_to_dataframe(self, responses: List[SurveyResponse]) -> pd.DataFrame:
        """Convert survey responses to pandas DataFrame"""
        try:
            data = []
            for response in responses:
                row = response.dict()
                # Flatten any nested dictionaries
                flattened_row = {}
                for key, value in row.items():
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            flattened_row[f"{key}_{sub_key}"] = sub_value
                    else:
                        flattened_row[key] = value
                data.append(flattened_row)
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"DataFrame conversion failed: {str(e)}")
            return pd.DataFrame()