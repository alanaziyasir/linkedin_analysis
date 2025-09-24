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
        
        # Enhanced classification logic
        def classify_theme(text):
            if not isinstance(text, str):
                return 'positive_feedback'
            text_lower = text.lower()
            
            suggestion_words = ['suggest', 'should', 'could', 'would', 'idea', 'recommend', 'propose', 'add', 'include', 'consider', 'curriculum', 'training', 'course', 'program', 'module', 'focus on', 'teach']
            if any(word in text_lower for word in suggestion_words):
                return 'suggestions_ideas'
            
            support_words = ['great', 'excellent', 'amazing', 'fantastic', 'brilliant', 'excited', 'support', 'help', 'agree', 'yes', 'absolutely', 'love', 'perfect', 'good', 'best', 'awesome']
            if any(word in text_lower for word in support_words):
                return 'support_enthusiasm'
            
            negative_words = ['bad', 'wrong', 'terrible', 'hate', 'no', 'not', 'never', 'difficult', 'problem', 'issue', 'concern', 'fail', 'poor']
            if any(word in text_lower for word in negative_words):
                return 'negative_feedback'
            
            return 'positive_feedback'
        
        def classify_persona(title):
            if not isinstance(title, str):
                return 'unknown'
            title_lower = title.lower()
            
            exec_words = ['ceo', 'chief executive', 'president', 'founder', 'co-founder', 'director', 'managing director', 'executive director', 'chairman']
            if any(word in title_lower for word in exec_words):
                return 'industry_exec'
            
            lead_words = ['lead', 'leader', 'manager', 'head', 'senior', 'principal', 'vice president', 'vp', 'supervisor', 'coordinator', 'specialist']
            if any(word in title_lower for word in lead_words):
                return 'industry_lead'
            
            academic_lead_words = ['professor', 'prof', 'dean', 'chancellor', 'provost', 'chair', 'department head', 'academic director']
            if any(word in title_lower for word in academic_lead_words):
                return 'academic_leadership'
            
            academic_staff_words = ['teacher', 'instructor', 'lecturer', 'assistant professor', 'associate professor', 'researcher', 'phd', 'dr.']
            if any(word in title_lower for word in academic_staff_words):
                return 'academic_staff'
            
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

@app.post("/analyze-upload")
def analyze_uploaded_data(request: UploadRequest):
    """Analyze uploaded JSON data using app4.py's classification logic"""
    try:
        data = request.data
        if isinstance(data, dict) and 'data' in data:
            data = data['data']
        
        if not isinstance(data, list):
            raise HTTPException(status_code=400, detail="Data must be an array of comments")
        
        # Convert to DataFrame and analyze
        upload_df = pd.json_normalize(data)
        
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
        
        # Enhanced classification logic
        def classify_theme(text):
            if not isinstance(text, str):
                return 'positive_feedback'
            text_lower = text.lower()
            
            suggestion_words = ['suggest', 'should', 'could', 'would', 'idea', 'recommend', 'propose', 'add', 'include', 'consider', 'curriculum', 'training', 'course', 'program', 'module', 'focus on', 'teach']
            if any(word in text_lower for word in suggestion_words):
                return 'suggestions_ideas'
            
            support_words = ['great', 'excellent', 'amazing', 'fantastic', 'brilliant', 'excited', 'support', 'help', 'agree', 'yes', 'absolutely', 'love', 'perfect', 'good', 'best', 'awesome']
            if any(word in text_lower for word in support_words):
                return 'support_enthusiasm'
            
            negative_words = ['bad', 'wrong', 'terrible', 'hate', 'no', 'not', 'never', 'difficult', 'problem', 'issue', 'concern', 'fail', 'poor']
            if any(word in text_lower for word in negative_words):
                return 'negative_feedback'
            
            return 'positive_feedback'
        
        def classify_persona(title):
            if not isinstance(title, str):
                return 'unknown'
            title_lower = title.lower()
            
            exec_words = ['ceo', 'chief executive', 'president', 'founder', 'co-founder', 'director', 'managing director', 'executive director', 'chairman']
            if any(word in title_lower for word in exec_words):
                return 'industry_exec'
            
            lead_words = ['lead', 'leader', 'manager', 'head', 'senior', 'principal', 'vice president', 'vp', 'supervisor', 'coordinator', 'specialist']
            if any(word in title_lower for word in lead_words):
                return 'industry_lead'
            
            academic_lead_words = ['professor', 'prof', 'dean', 'chancellor', 'provost', 'chair', 'department head', 'academic director']
            if any(word in title_lower for word in academic_lead_words):
                return 'academic_leadership'
            
            academic_staff_words = ['teacher', 'instructor', 'lecturer', 'assistant professor', 'associate professor', 'researcher', 'phd', 'dr.']
            if any(word in title_lower for word in academic_staff_words):
                return 'academic_staff'
            
            return 'unknown'
        
        upload_df['super_theme'] = upload_df['text'].apply(classify_theme)
        upload_df['persona'] = upload_df['author_title'].apply(classify_persona)
        
        # Generate counts
        theme_counts = upload_df['super_theme'].value_counts().to_dict()
        persona_counts = upload_df['persona'].value_counts().to_dict()
        
        # Create treemap data
        treemap_data = {}
        for theme_code, theme_label in SUPER_THEME_LABELS.items():
            if theme_code in theme_counts:
                theme_comments = upload_df[upload_df['super_theme'] == theme_code]
                persona_breakdown = theme_comments['persona'].value_counts().to_dict()
                
                breakdown_display = {}
                for persona_code, count in persona_breakdown.items():
                    persona_label = PERSONA_LABELS.get(persona_code, persona_code)
                    breakdown_display[persona_label] = count
                
                treemap_data[theme_label.lower().replace(' ', '_')] = {
                    'total': theme_counts[theme_code],
                    'breakdown': breakdown_display
                }
        
        # Create radar chart data
        theme_radar = {}
        for code, label in SUPER_THEME_LABELS.items():
            count = theme_counts.get(code, 0)
            normalized = min(10, (count / max(theme_counts.values()) * 10) if theme_counts else 0)
            theme_radar[label.split(' ')[0]] = round(normalized, 1)
        
        profession_radar = {}
        for code, label in PERSONA_LABELS.items():
            count = persona_counts.get(code, 0)
            normalized = min(10, (count / max(persona_counts.values()) * 10) if persona_counts else 0)
            profession_radar[label.split(' ')[0]] = round(normalized, 1)
        
        return {
            "status": "success",
            "message": f"Successfully analyzed {len(upload_df)} comments",
            "total_comments": len(upload_df),
            "treemap_data": treemap_data,
            "theme_radar": theme_radar,
            "profession_radar": profession_radar,
            "filename": request.filename,
            "source": "app4.py classification logic"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/app4-analysis")
def get_app4_analysis():
    """Get analysis using the real app4.py data and classification logic"""
    try:
        if df.empty:
            raise HTTPException(status_code=404, detail="No data available. Load dataset.json first.")
        
        # Use the loaded and classified data from app4.py logic
        theme_counts = df['super_theme'].value_counts().to_dict()
        persona_counts = df['persona'].value_counts().to_dict()
        
        # Create treemap data structure
        treemap_data = {}
        for theme_code, theme_label in SUPER_THEME_LABELS.items():
            if theme_code in theme_counts:
                theme_comments = df[df['super_theme'] == theme_code]
                persona_breakdown = theme_comments['persona'].value_counts().to_dict()
                
                breakdown_display = {}
                for persona_code, count in persona_breakdown.items():
                    persona_label = PERSONA_LABELS.get(persona_code, persona_code)
                    breakdown_display[persona_label] = count
                
                treemap_data[theme_label.lower().replace(' ', '_')] = {
                    'total': theme_counts[theme_code],
                    'breakdown': breakdown_display
                }
        
        # Create radar chart data
        theme_radar = {}
        for code, label in SUPER_THEME_LABELS.items():
            count = theme_counts.get(code, 0)
            normalized = min(10, (count / max(theme_counts.values()) * 10) if theme_counts else 0)
            theme_radar[label.split(' ')[0]] = round(normalized, 1)
        
        profession_radar = {}
        for code, label in PERSONA_LABELS.items():
            count = persona_counts.get(code, 0)
            normalized = min(10, (count / max(persona_counts.values()) * 10) if persona_counts else 0)
            profession_radar[label.split(' ')[0]] = round(normalized, 1)
        
        return {
            "status": "success",
            "message": f"Real app4.py analysis with {len(df)} comments",
            "total_comments": len(df),
            "treemap_data": treemap_data,
            "theme_radar": theme_radar,
            "profession_radar": profession_radar,
            "source": "app4.py real data"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get app4 analysis: {str(e)}")