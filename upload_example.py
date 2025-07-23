#!/usr/bin/env python3
"""
Example script showing how to upload your survey data to the FastAPI backend
"""

import requests
import json

# Your survey data format
survey_data = [
    {
        "SrNO": 1,
        "ResponseNum": "1",
        "Response Attributes": {
            "LastUpdate": "2025-07-21T05:47:50.577"
        },
        "Responses": {
            "Q1_Please_enter_your_full_name": "Aznan Khan",
            "Q2a_Gender": "Male",
            "Q2b_Date_of_Birth": "10/27/2000",
            "Q2c_Email_Address": "akhan4@sogolytics.com",
            "Q3_Please_enter_the_course_name_you_are_evaluating": "Artificial Intelligence",
            "Q4_What_is_your_role": "Student",
            "Q5_How_would_you_rate_the_quality_of_the_course_content": "Excellent",
            "Q6_How_relevant_was_the_course_content_to_your_needs": "Highly Relevant",
            "Q7a_Search": "Very Important",
            "Q7b_Notifications": "Neutral",
            "Q7c_Profile_Customization": "Very Important",
            "Q7d_Dark_Mode": "Important",
            "Q7e_Saved_Items": "Important",
            "Q8a_Search": "Used on Mobile , Used on Desktop",
            "Q8b_Notifications": "Used on Desktop",
            "Q8c_Profile_Customization": "Used on Mobile , Used on Desktop",
            "Q8d_Dark_Mode": "Used on Mobile , Used on Desktop",
            "Q8e_Saved_Items": "Never Used",
            "Q9a_Clarity": "Excellent",
            "Q9b_Subject_Knowledge": "Very Good",
            "Q9c_Engagement": "Good",
            "Q9d_Responsiveness": "Very Good",
            "Q10_How_would_you_rate_the_overall_organization_and_structure_of_the_course": "Excellent",
            "Q11_Was_the_course_schedule_clearly_communicated": "Yes",
            "Q12_How_clearly_were_the_learning_objectives_presented": "Extremely Clear",
            "Q13_To_what_extent_did_the_course_improve_your_skills_and_knowledge": "Extensively",
            "Q14_How_likely_are_you_to_recommend_this_course_to_a_friend_or_colleague": "9",
            "Q15_Which_of_the_following_topics_would_you_be_interested_in_for_future_courses_Select_all_that_apply": "Advanced Techniques , Practical Workshops , Research Methods",
            "Q16_How_would_you_rate_the_overall_difficulty_level_of_the_course": "Just Right",
            "Q17_What_improvements_would_you_suggest_for_future_courses": "More hand on",
            "Q18_Overall_how_satisfied_are_you_with_the_course_experience": "Very Satisfied"
        }
    }
]

def upload_via_file():
    """Method 1: Upload as JSON file"""
    # Save to temporary file
    with open('temp_survey_data.json', 'w') as f:
        json.dump(survey_data, f)
    
    # Upload file
    with open('temp_survey_data.json', 'rb') as f:
        files = {'file': ('survey_data.json', f, 'application/json')}
        response = requests.post('http://localhost:8000/api/survey/responses/upload', files=files)
    
    print("File Upload Response:", response.json())

def upload_via_direct_post():
    """Method 2: Direct POST to analysis endpoint"""
    payload = {
        "responses": survey_data,
        "include_segmentation": True,
        "generate_insights": True
    }
    
    response = requests.post('http://localhost:8000/api/analysis/run', json=payload)
    print("Direct Analysis Response:", response.json())

def quick_test():
    """Method 3: Quick analysis with existing data"""
    response = requests.get('http://localhost:8000/api/analysis/quick')
    print("Quick Analysis Response:", response.json())

if __name__ == "__main__":
    print("=== Survey Data Upload Examples ===")
    print("Make sure your FastAPI server is running on http://localhost:8000")
    print()
    
    try:
        # Test the API is running
        health = requests.get('http://localhost:8000/health')
        print(f"API Health: {health.json()}")
        print()
        
        # Upload your data
        print("1. Uploading via file...")
        upload_via_file()
        print()
        
        print("2. Running direct analysis...")
        upload_via_direct_post()
        print()
        
        print("3. Quick analysis test...")
        quick_test()
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to FastAPI server.")
        print("Make sure to run: cd fastapi_backend && python run.py")