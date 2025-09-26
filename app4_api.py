from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any
import pandas as pd
import json
import io

app = FastAPI(title="LinkedIn Analysis API - Multi-Format Support", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load fallback CSV data
CSV_FILE = "linkedin_comments_with_themes.csv"
fallback_df = pd.DataFrame()

try:
    fallback_df = pd.read_csv(CSV_FILE)
    print(f"✅ Loaded {len(fallback_df)} fallback comments from CSV")
except Exception as e:
    print(f"⚠️ Fallback CSV not found: {e}")

# Classification mappings
SUPER_THEME_LABELS = {
    "suggestions_ideas": "Suggestions",
    "support_enthusiasm": "Support", 
    "positive_feedback": "Complementary",
    "negative_feedback": "Negatives",
    "neutral": "Neutral"
}

PERSONA_LABELS = {
    "industry_exec": "CEOs",
    "industry_lead": "Industry Leaders", 
    "academic_leadership": "Academic Leaders",
    "academic_staff": "Academic Staff",
    "unknown": "Others",
}

class JSONUploadRequest(BaseModel):
    data: Any
    filename: str = "uploaded.json"

def process_json_upload(data: list) -> pd.DataFrame:
    """Process uploaded JSON file and apply classification"""
    try:
        upload_df = pd.json_normalize(data)
        
        # Map JSON columns (standard LinkedIn export format)
        if 'commentary' in upload_df.columns:
            upload_df['text'] = upload_df['commentary']
        if 'actor.name' in upload_df.columns:
            upload_df['author'] = upload_df['actor.name']
        if 'engagement.comments' in upload_df.columns:
            upload_df['replies'] = pd.to_numeric(upload_df['engagement.comments'], errors='coerce').fillna(0)
        if 'actor.position' in upload_df.columns:
            upload_df['author_title'] = upload_df['actor.position']
        if 'linkedinUrl' in upload_df.columns:
            upload_df['comment_url'] = upload_df['linkedinUrl']
        if 'createdAt' in upload_df.columns:
            upload_df['created_at'] = pd.to_datetime(upload_df['createdAt'])
        
        # Add missing columns with defaults
        for col in ['text', 'author', 'author_title', 'replies', 'shares']:
            if col not in upload_df.columns:
                upload_df[col] = '' if col in ['text', 'author', 'author_title'] else 0
        
        # Apply basic classification for JSON (since no LLM classifications)
        def classify_theme_basic(text):
            if not isinstance(text, str):
                return 'neutral'
            text_lower = text.lower()
            
            if any(word in text_lower for word in ['suggest', 'should', 'could', 'recommend', 'idea']):
                return 'suggestions_ideas'
            elif any(word in text_lower for word in ['great', 'excellent', 'amazing', 'love', 'support']):
                return 'support_enthusiasm'
            elif any(word in text_lower for word in ['collaborate', 'partnership', 'together', 'join']):
                return 'positive_feedback'
            elif any(word in text_lower for word in ['concern', 'issue', 'problem', 'worry']):
                return 'negative_feedback'
            else:
                return 'neutral'
        
        def classify_persona_basic(title):
            if not isinstance(title, str):
                return 'unknown'
            title_lower = title.lower()
            
            if any(word in title_lower for word in ['ceo', 'chief executive', 'founder', 'president']):
                return 'industry_exec'
            elif any(word in title_lower for word in ['director', 'head', 'manager', 'lead']):
                return 'industry_lead'
            elif any(word in title_lower for word in ['professor', 'academic', 'researcher']):
                return 'academic_leadership'
            elif any(word in title_lower for word in ['lecturer', 'instructor', 'teacher']):
                return 'academic_staff'
            else:
                return 'unknown'
        
        upload_df['super_theme'] = upload_df['text'].apply(classify_theme_basic)
        upload_df['persona'] = upload_df['author_title'].apply(classify_persona_basic)
        upload_df['sentiment'] = 0.8
        
        return upload_df
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"JSON processing failed: {str(e)}")

def process_csv_upload(df: pd.DataFrame) -> pd.DataFrame:
    """Process uploaded CSV file with LLM classifications"""
    try:
        processed_df = df.copy()
        
        # Map CSV columns to expected format
        processed_df['text'] = processed_df['commentary']
        processed_df['author'] = processed_df['actor_name'] 
        processed_df['author_title'] = processed_df['actor_position']
        processed_df['replies'] = processed_df['comments_count'].fillna(0).astype(int)
        processed_df['created_at'] = pd.to_datetime(processed_df['createdAt'])
        processed_df['comment_url'] = processed_df['linkedinUrl']
        processed_df['shares'] = processed_df['shares'].fillna(0).astype(int)
        
        # Map LLM classifications from your CSV
        theme_mapping = {
            "suggestion_idea": "suggestions_ideas",
            "praise_support": "support_enthusiasm",
            "offer_collaboration": "positive_feedback", 
            "critique_risk": "negative_feedback",
            "neutral_info": "neutral"
        }
        processed_df['super_theme'] = processed_df['comment_theme'].map(theme_mapping).fillna('neutral')
        
        profession_mapping = {
            "leadership_exec": "industry_exec",
            "product_strategy_ops": "industry_lead",
            "academia": "academic_leadership", 
            "data_ai_ml": "academic_staff",
            "entrepreneur_advisor": "unknown"
        }
        processed_df['persona'] = processed_df['profession_bucket'].map(profession_mapping).fillna('unknown')
        
        # Use LLM confidence scores
        processed_df['sentiment'] = processed_df['theme_confidence']
        
        return processed_df
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV processing failed: {str(e)}")

def create_analysis_response(df: pd.DataFrame) -> dict:
    """Create standardized analysis response from processed DataFrame"""
    try:
        # Count themes and personas
        theme_counts = df['super_theme'].value_counts().to_dict()
        persona_counts = df['persona'].value_counts().to_dict()
        
        # Create treemap data structure
        treemap_data = {
            "complementary": {"breakdown": {}},
            "suggestions": {"breakdown": {}}, 
            "negatives": {"breakdown": {}},
            "neutral": {"breakdown": {}}
        }
        
        # Map themes to treemap categories
        theme_to_category = {
            "suggestions_ideas": "suggestions",
            "support_enthusiasm": "complementary",
            "positive_feedback": "complementary",
            "negative_feedback": "negatives", 
            "neutral": "neutral"
        }
        
        for theme_code, count in theme_counts.items():
            category = theme_to_category.get(theme_code, "neutral")
            theme_df = df[df['super_theme'] == theme_code]
            persona_breakdown = theme_df['persona'].value_counts().to_dict()
            
            for persona_code, persona_count in persona_breakdown.items():
                display_name = PERSONA_LABELS.get(persona_code, "Others")
                if display_name in treemap_data[category]["breakdown"]:
                    treemap_data[category]["breakdown"][display_name] += persona_count
                else:
                    treemap_data[category]["breakdown"][display_name] = persona_count
        
        # Create radar chart data
        max_theme_count = max(theme_counts.values()) if theme_counts else 1
        theme_radar = {}
        for code, label in SUPER_THEME_LABELS.items():
            count = theme_counts.get(code, 0)
            normalized = min(10, (count / max_theme_count * 10))
            # Map to frontend radar labels
            if label == "Support":
                theme_radar["Support"] = round(normalized, 1)
            elif label == "Suggestions":
                theme_radar["Ideas"] = round(normalized, 1)
            elif label == "Complementary":  
                theme_radar["Complements"] = round(normalized, 1)
            elif label == "Negatives":
                theme_radar["Negative"] = round(normalized, 1)
            else:
                theme_radar["Theme"] = round(normalized, 1)
        
        max_persona_count = max(persona_counts.values()) if persona_counts else 1
        profession_radar = {}
        for code, label in PERSONA_LABELS.items():
            count = persona_counts.get(code, 0)
            normalized = min(10, (count / max_persona_count * 10))
            profession_radar[label] = round(normalized, 1)
        
        return {
            "status": "success",
            "message": f"Analysis complete with {len(df)} comments",
            "total_comments": len(df),
            "treemap_data": treemap_data,
            "theme_radar": theme_radar,
            "profession_radar": profession_radar,
            "source": "csv_llm_classifications"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis creation failed: {str(e)}")

@app.get("/")
def root():
    return {
        "message": "LinkedIn Analysis API - Multi-Format Support",
        "fallback_data": len(fallback_df) if not fallback_df.empty else 0,
        "supported_formats": ["CSV (with LLM classifications)", "JSON (LinkedIn export)"],
        "endpoints": ["/app4-analysis", "/upload-file"]
    }

@app.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):
    """Handle CSV or JSON file upload"""
    try:
        contents = await file.read()
        
        if file.filename.lower().endswith('.csv'):
            # Process CSV file
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
            processed_df = process_csv_upload(df)
            
        elif file.filename.lower().endswith('.json'):
            # Process JSON file
            data = json.loads(contents.decode('utf-8'))
            if isinstance(data, dict) and 'data' in data:
                data = data['data']
            processed_df = process_json_upload(data)
            
        else:
            raise HTTPException(status_code=400, detail="Only CSV and JSON files are supported")
        
        # Create analysis response
        analysis_result = create_analysis_response(processed_df)
        
        return analysis_result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

@app.get("/app4-analysis")
async def get_app4_analysis():
    """Get analysis using fallback CSV data"""
    try:
        if fallback_df.empty:
            raise HTTPException(status_code=404, detail="No fallback data available")
        
        # Process fallback data
        processed_df = process_csv_upload(fallback_df.copy())
        
        # Create analysis response
        analysis_result = create_analysis_response(processed_df)
        analysis_result["source"] = "fallback_csv_data"
        
        return analysis_result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"App4 analysis failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)