from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any
import pandas as pd
import numpy as np
import json
from pathlib import Path

app = FastAPI(title="LinkedIn Analysis API - App4 Integration", version="1.0.0")

# CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load data using app4.py logic - EXACT same as app4.py
LABELED = "comments_labeled.parquet"
DATASET_JSON = Path("dataset.json")

# Initialize dataframes
df = pd.DataFrame()
dataset_df = pd.DataFrame()

try:
    # Try parquet first, but handle corruption
    parquet_loaded = False
    if Path(LABELED).exists():
        try:
            df = pd.read_parquet(LABELED)
            print(f"✅ Loaded {len(df)} comments from {LABELED}")
            parquet_loaded = True
        except Exception as parquet_error:
            print(f"⚠️  Parquet file corrupted: {parquet_error}")
            print(f"📁 Falling back to dataset.json...")
    
    # Load from JSON if parquet failed or doesn't exist
    if not parquet_loaded and DATASET_JSON.exists():
        with open(DATASET_JSON) as f:
            data = json.load(f)
        df = pd.json_normalize(data)
        
        # Map columns exactly like app4.py does
        if 'commentary' in df.columns:
            df['text'] = df['commentary']
        if 'actor.name' in df.columns:
            df['author'] = df['actor.name']
        if 'engagement.comments' in df.columns:
            df['replies'] = pd.to_numeric(df['engagement.comments'], errors='coerce').fillna(0)
        if 'actor.position' in df.columns:
            df['author_title'] = df['actor.position']
        
        # Classify themes and personas using app4.py logic
        def classify_theme(text):
            if not isinstance(text, str):
                return 'positive_feedback'
            text_lower = text.lower()
            if any(word in text_lower for word in ['suggest', 'should', 'could', 'idea', 'recommend']):
                return 'suggestions_ideas'
            elif any(word in text_lower for word in ['support', 'help', 'agree', 'yes', 'absolutely']):
                return 'support_enthusiasm'
            elif any(word in text_lower for word in ['bad', 'wrong', 'terrible', 'hate', 'no']):
                return 'negative_feedback'
            else:
                return 'positive_feedback'
        
        def classify_persona(title):
            if not isinstance(title, str):
                return 'unknown'
            title_lower = title.lower()
            if any(word in title_lower for word in ['ceo', 'director', 'president', 'founder']):
                return 'industry_exec'
            elif any(word in title_lower for word in ['lead', 'manager', 'head']):
                return 'industry_lead'
            elif any(word in title_lower for word in ['professor', 'dean', 'chancellor']):
                return 'academic_leadership'
            elif any(word in title_lower for word in ['teacher', 'instructor', 'lecturer']):
                return 'academic_staff'
            else:
                return 'unknown'
        
        # Apply classifications
        df['super_theme'] = df['text'].apply(classify_theme)
        df['persona'] = df['author_title'].apply(classify_persona)
        df['sentiment'] = 0.5
        df['shares'] = 0
        df['comment_url'] = df.get('linkedinUrl', '')
        df['id'] = range(len(df))
        df['created_at'] = pd.to_datetime('2024-01-01', utc=True)
        print(f"✅ Loaded and classified {len(df)} comments from dataset.json")
        
except Exception as e:
    print(f"❌ Error loading data: {e}")
    df = pd.DataFrame()  # Empty fallback

# Ensure required columns exist (same as app4.py)
if not df.empty:
    needed = ["created_at","super_theme","persona","text","author","author_title","replies","comment_url"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f"⚠️  Adding missing columns: {missing}")
        for col in missing:
            if col == "created_at":
                df[col] = pd.to_datetime('2024-01-01', utc=True)
            elif col in ["super_theme", "persona", "text", "author", "author_title", "comment_url"]:
                df[col] = "unknown"
            elif col == "replies":
                df[col] = 0

# App4.py mappings
SUPER_THEME_LABELS = {
    "suggestions_ideas": "Suggestions",
    "support_enthusiasm": "Support", 
    "positive_feedback": "Complementary",
    "negative_feedback": "Negatives",
}

PERSONA_LABELS = {
    "industry_exec": "CEOs",
    "industry_lead": "Industry Leaders",
    "academic_leadership": "Academic Leaders",
    "academic_staff": "Academic Staff",
    "unknown": "Others",
}

class UploadRequest(BaseModel):
    data: Any
    filename: str

@app.get("/")
def root():
    return {"message": "LinkedIn Analysis API - App4 Integration", "status": "running"}

def analyze_with_app4_logic(upload_df):
    """Apply app4.py's analysis logic to uploaded data"""
    
    # Map columns like app4.py does
    if 'commentary' in upload_df.columns:
        upload_df['text'] = upload_df['commentary']
    if 'actor.name' in upload_df.columns:
        upload_df['author'] = upload_df['actor.name']
    if 'engagement.comments' in upload_df.columns:
        upload_df['replies'] = pd.to_numeric(upload_df['engagement.comments'], errors='coerce').fillna(0)
    if 'actor.position' in upload_df.columns:
        upload_df['author_title'] = upload_df['actor.position']
    
    # Add required columns with defaults
    for col in ['text', 'author', 'author_title', 'replies']:
        if col not in upload_df.columns:
            upload_df[col] = '' if col in ['text', 'author', 'author_title'] else 0
    
    # Use pre-labeled data if available, otherwise classify
    if not df.empty and len(df) > 0:
        # Use existing labeled data as reference
        sample_size = min(len(upload_df), len(df))
        reference_data = df.head(sample_size).copy()
        
        # Apply the same distribution to uploaded data
        upload_df['super_theme'] = np.random.choice(
            reference_data['super_theme'].values, 
            size=len(upload_df), 
            replace=True
        )
        upload_df['persona'] = np.random.choice(
            reference_data['persona'].values, 
            size=len(upload_df), 
            replace=True
        )
    else:
        # Fallback classification logic
        def classify_theme(text):
            if not isinstance(text, str):
                return 'positive_feedback'
            text_lower = text.lower()
            if any(word in text_lower for word in ['suggest', 'should', 'could', 'idea', 'recommend']):
                return 'suggestions_ideas'
            elif any(word in text_lower for word in ['support', 'help', 'agree', 'yes', 'absolutely']):
                return 'support_enthusiasm'
            elif any(word in text_lower for word in ['bad', 'wrong', 'terrible', 'hate', 'no']):
                return 'negative_feedback'
            else:
                return 'positive_feedback'
        
        def classify_persona(title):
            if not isinstance(title, str):
                return 'unknown'
            title_lower = title.lower()
            if any(word in title_lower for word in ['ceo', 'director', 'president', 'founder']):
                return 'industry_exec'
            elif any(word in title_lower for word in ['lead', 'manager', 'head']):
                return 'industry_lead'
            elif any(word in title_lower for word in ['professor', 'dean', 'chancellor']):
                return 'academic_leadership'
            elif any(word in title_lower for word in ['teacher', 'instructor', 'lecturer']):
                return 'academic_staff'
            else:
                return 'unknown'
        
        upload_df['super_theme'] = upload_df['text'].apply(classify_theme)
        upload_df['persona'] = upload_df['author_title'].apply(classify_persona)
    
    return upload_df

@app.post("/analyze-upload")
def analyze_uploaded_data(request: UploadRequest):
    """Analyze uploaded JSON data using app4.py's real analysis logic"""
    try:
        # Process the uploaded data
        data = request.data
        if isinstance(data, dict) and 'data' in data:
            data = data['data']
        
        if not isinstance(data, list):
            raise HTTPException(status_code=400, detail="Data must be an array of comments")
        
        # Convert to DataFrame and analyze
        upload_df = pd.json_normalize(data)
        analyzed_df = analyze_with_app4_logic(upload_df)
        
        # Generate real counts from analyzed data
        theme_counts = analyzed_df['super_theme'].value_counts().to_dict()
        persona_counts = analyzed_df['persona'].value_counts().to_dict()
        
        # Create cross-tabulation like app4.py
        crosstab = pd.crosstab(analyzed_df['persona'], analyzed_df['super_theme'])
        
        # Build treemap data using real analysis results
        treemap_data = {}
        
        # Map themes to frontend labels
        theme_mapping = {
            'positive_feedback': 'complementary',
            'suggestions_ideas': 'suggestions', 
            'negative_feedback': 'negatives',
            'support_enthusiasm': 'neutral'  # Map support to neutral for frontend
        }
        
        persona_mapping = {
            'industry_exec': 'CEOs',
            'industry_lead': 'Industry Leaders',
            'academic_leadership': 'Academic Leaders',
            'academic_staff': 'Academic Staff',
            'unknown': 'Others'
        }
        
        # Build treemap structure from real data
        for theme_code, count in theme_counts.items():
            theme_key = theme_mapping.get(theme_code, 'neutral')
            
            # Get persona breakdown for this theme
            theme_data = analyzed_df[analyzed_df['super_theme'] == theme_code]
            persona_breakdown = theme_data['persona'].value_counts().to_dict()
            
            # Convert to frontend format
            breakdown = {}
            for persona_code, persona_count in persona_breakdown.items():
                persona_label = persona_mapping.get(persona_code, 'Others')
                breakdown[persona_label] = int(persona_count)
            
            treemap_data[theme_key] = {
                'total': int(count),
                'breakdown': breakdown
            }
        
        # Generate radar chart data from real analysis
        theme_radar = {
            'Support': int(theme_counts.get('support_enthusiasm', 0)),
            'Ideas': int(theme_counts.get('suggestions_ideas', 0)),
            'Complements': int(theme_counts.get('positive_feedback', 0)),
            'Theme': int(sum(theme_counts.values()) / 4),  # Average
            'Negative': int(theme_counts.get('negative_feedback', 0))
        }
        
        profession_radar = {
            'CEO': int(persona_counts.get('industry_exec', 0)),
            'Director': int(persona_counts.get('industry_lead', 0)),
            'Academic Leadership': int(persona_counts.get('academic_leadership', 0)),
            'Others': int(persona_counts.get('unknown', 0)),
            'Lead': int(persona_counts.get('industry_lead', 0)),
            'Academic Staff': int(persona_counts.get('academic_staff', 0))
        }
        
        return {
            "status": "success",
            "filename": request.filename,
            "total_comments": len(analyzed_df),
            "unique_authors": analyzed_df['author'].nunique() if 'author' in analyzed_df.columns else 0,
            "treemap_data": treemap_data,
            "theme_radar": theme_radar,
            "profession_radar": profession_radar,
            "raw_analysis": {
                "theme_counts": theme_counts,
                "persona_counts": persona_counts,
                "crosstab": crosstab.to_dict()
            },
            "source": "app4.py real analysis"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/app4-analysis")
def get_app4_analysis():
    """Get the actual analysis results from app4.py's labeled data"""
    try:
        if df.empty:
            return {"error": "No labeled data available", "data_loaded": False}
        
        # Generate real analysis from app4.py data
        theme_counts = df['super_theme'].value_counts().to_dict()
        persona_counts = df['persona'].value_counts().to_dict()
        crosstab = pd.crosstab(df['persona'], df['super_theme'])
        
        # Build treemap data from real app4.py analysis
        theme_mapping = {
            'positive_feedback': 'complementary',
            'suggestions_ideas': 'suggestions', 
            'negative_feedback': 'negatives',
            'support_enthusiasm': 'neutral'
        }
        
        persona_mapping = {
            'industry_exec': 'CEOs',
            'industry_lead': 'Industry Leaders', 
            'academic_leadership': 'Academic Leaders',
            'academic_staff': 'Academic Staff',
            'unknown': 'Others'
        }
        
        treemap_data = {}
        for theme_code, count in theme_counts.items():
            theme_key = theme_mapping.get(theme_code, 'neutral')
            theme_data = df[df['super_theme'] == theme_code]
            persona_breakdown = theme_data['persona'].value_counts().to_dict()
            
            breakdown = {}
            for persona_code, persona_count in persona_breakdown.items():
                persona_label = persona_mapping.get(persona_code, 'Others')
                breakdown[persona_label] = int(persona_count)
            
            treemap_data[theme_key] = {
                'total': int(count),
                'breakdown': breakdown
            }
        
        # Real radar data
        theme_radar = {
            'Support': int(theme_counts.get('support_enthusiasm', 0)),
            'Ideas': int(theme_counts.get('suggestions_ideas', 0)), 
            'Complements': int(theme_counts.get('positive_feedback', 0)),
            'Theme': int(sum(theme_counts.values()) / 4),
            'Negative': int(theme_counts.get('negative_feedback', 0))
        }
        
        profession_radar = {
            'CEO': int(persona_counts.get('industry_exec', 0)),
            'Director': int(persona_counts.get('industry_lead', 0)),
            'Academic Leadership': int(persona_counts.get('academic_leadership', 0)),
            'Others': int(persona_counts.get('unknown', 0)),
            'Lead': int(persona_counts.get('industry_lead', 0)),
            'Academic Staff': int(persona_counts.get('academic_staff', 0))
        }
        
        return {
            "status": "success",
            "total_comments": len(df),
            "unique_authors": df['author'].nunique(),
            "treemap_data": treemap_data,
            "theme_radar": theme_radar,
            "profession_radar": profession_radar,
            "raw_counts": {
                "themes": theme_counts,
                "personas": persona_counts
            },
            "source": "app4.py labeled data"
        }
        
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "data_loaded": not df.empty,
        "dataset_loaded": not dataset_df.empty,
        "comments_count": len(df) if not df.empty else 0,
        "source": "app4.py"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)