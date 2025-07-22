"""
Database service for storing survey data and analysis results
"""

import asyncio
import json
import sqlite3
import aiosqlite
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from app.models.survey import SurveyResponse, AnalysisResult, StoredAnalysis

logger = logging.getLogger(__name__)

class DatabaseService:
    """Database service for survey analysis system"""
    
    def __init__(self, database_path: str = "survey_analysis.db"):
        self.database_path = database_path
        self._connection = None
    
    async def connect(self):
        """Initialize database connection and create tables"""
        try:
            self._connection = await aiosqlite.connect(self.database_path)
            await self._create_tables()
            logger.info("Database connected successfully")
        except Exception as e:
            logger.error(f"Database connection failed: {str(e)}")
            raise
    
    async def disconnect(self):
        """Close database connection"""
        if self._connection:
            await self._connection.close()
            logger.info("Database disconnected")
    
    async def ping(self) -> bool:
        """Check database connectivity"""
        try:
            if not self._connection:
                return False
            cursor = await self._connection.execute("SELECT 1")
            await cursor.fetchone()
            return True
        except Exception:
            return False
    
    async def _create_tables(self):
        """Create database tables"""
        try:
            # Survey responses table
            await self._connection.execute("""
                CREATE TABLE IF NOT EXISTS survey_responses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    response_id TEXT UNIQUE NOT NULL,
                    course_type TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    data TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Analysis results table
            await self._connection.execute("""
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id TEXT UNIQUE NOT NULL,
                    version TEXT NOT NULL,
                    result_data TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    tags TEXT
                )
            """)
            
            # Courses table
            await self._connection.execute("""
                CREATE TABLE IF NOT EXISTS courses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    course_id TEXT UNIQUE NOT NULL,
                    course_name TEXT NOT NULL,
                    course_type TEXT NOT NULL,
                    instructor_id TEXT,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Instructors table
            await self._connection.execute("""
                CREATE TABLE IF NOT EXISTS instructors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    instructor_id TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    email TEXT,
                    specialization TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            await self._connection.commit()
            logger.info("Database tables created successfully")
            
        except Exception as e:
            logger.error(f"Table creation failed: {str(e)}")
            raise
    
    async def store_survey_responses(self, responses: List[SurveyResponse]) -> int:
        """Store survey responses in database"""
        try:
            stored_count = 0
            
            for response in responses:
                try:
                    # Convert response to JSON
                    response_data = response.json()
                    
                    await self._connection.execute("""
                        INSERT OR REPLACE INTO survey_responses 
                        (response_id, course_type, timestamp, data)
                        VALUES (?, ?, ?, ?)
                    """, (
                        response.response_id,
                        response.course_type,
                        response.timestamp,
                        response_data
                    ))
                    
                    stored_count += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to store response {response.response_id}: {str(e)}")
                    continue
            
            await self._connection.commit()
            logger.info(f"Stored {stored_count} survey responses")
            return stored_count
            
        except Exception as e:
            logger.error(f"Survey response storage failed: {str(e)}")
            raise
    
    async def get_survey_responses(self, course_type: Optional[str] = None, limit: Optional[int] = None) -> List[SurveyResponse]:
        """Retrieve survey responses from database"""
        try:
            query = "SELECT data FROM survey_responses"
            params = []
            
            if course_type:
                query += " WHERE course_type = ?"
                params.append(course_type)
            
            query += " ORDER BY timestamp DESC"
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            cursor = await self._connection.execute(query, params)
            rows = await cursor.fetchall()
            
            responses = []
            for row in rows:
                try:
                    response_data = json.loads(row[0])
                    response = SurveyResponse(**response_data)
                    responses.append(response)
                except Exception as e:
                    logger.warning(f"Failed to parse response: {str(e)}")
                    continue
            
            return responses
            
        except Exception as e:
            logger.error(f"Response retrieval failed: {str(e)}")
            return []
    
    async def store_analysis_result(self, result: AnalysisResult) -> str:
        """Store analysis result in database"""
        try:
            analysis_id = f"analysis_{datetime.utcnow().timestamp()}"
            result_data = result.json()
            
            await self._connection.execute("""
                INSERT INTO analysis_results 
                (analysis_id, version, result_data, tags)
                VALUES (?, ?, ?, ?)
            """, (
                analysis_id,
                "1.0.0",
                result_data,
                json.dumps([])  # Empty tags for now
            ))
            
            await self._connection.commit()
            logger.info(f"Stored analysis result: {analysis_id}")
            return analysis_id
            
        except Exception as e:
            logger.error(f"Analysis result storage failed: {str(e)}")
            raise
    
    async def get_analysis_history(self, limit: int = 10, offset: int = 0) -> List[StoredAnalysis]:
        """Retrieve analysis history"""
        try:
            cursor = await self._connection.execute("""
                SELECT analysis_id, created_at, result_data, version, tags
                FROM analysis_results
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """, (limit, offset))
            
            rows = await cursor.fetchall()
            analyses = []
            
            for row in rows:
                try:
                    result_data = json.loads(row[2])
                    analysis_result = AnalysisResult(**result_data)
                    
                    stored_analysis = StoredAnalysis(
                        analysis_id=row[0],
                        created_at=datetime.fromisoformat(row[1]),
                        analysis_result=analysis_result,
                        version=row[3],
                        tags=json.loads(row[4]) if row[4] else []
                    )
                    
                    analyses.append(stored_analysis)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse analysis {row[0]}: {str(e)}")
                    continue
            
            return analyses
            
        except Exception as e:
            logger.error(f"Analysis history retrieval failed: {str(e)}")
            return []
    
    async def get_course_responses(self, course_id: str) -> List[SurveyResponse]:
        """Get responses for a specific course"""
        try:
            cursor = await self._connection.execute("""
                SELECT data FROM survey_responses
                WHERE course_type = ?
                ORDER BY timestamp DESC
            """, (course_id,))
            
            rows = await cursor.fetchall()
            responses = []
            
            for row in rows:
                try:
                    response_data = json.loads(row[0])
                    response = SurveyResponse(**response_data)
                    responses.append(response)
                except Exception as e:
                    logger.warning(f"Failed to parse course response: {str(e)}")
                    continue
            
            return responses
            
        except Exception as e:
            logger.error(f"Course response retrieval failed: {str(e)}")
            return []
    
    async def get_response_statistics(self) -> Dict[str, Any]:
        """Get basic response statistics"""
        try:
            # Total responses
            cursor = await self._connection.execute("SELECT COUNT(*) FROM survey_responses")
            total_responses = (await cursor.fetchone())[0]
            
            # Responses by course
            cursor = await self._connection.execute("""
                SELECT course_type, COUNT(*) 
                FROM survey_responses 
                GROUP BY course_type
            """)
            course_counts = dict(await cursor.fetchall())
            
            # Recent responses (last 30 days)
            cursor = await self._connection.execute("""
                SELECT COUNT(*) FROM survey_responses
                WHERE timestamp >= datetime('now', '-30 days')
            """)
            recent_responses = (await cursor.fetchone())[0]
            
            return {
                "total_responses": total_responses,
                "responses_by_course": course_counts,
                "recent_responses": recent_responses
            }
            
        except Exception as e:
            logger.error(f"Statistics retrieval failed: {str(e)}")
            return {}

# Global database instance
_database_instance = None

def get_database() -> DatabaseService:
    """Get global database instance"""
    global _database_instance
    if _database_instance is None:
        _database_instance = DatabaseService()
    return _database_instance