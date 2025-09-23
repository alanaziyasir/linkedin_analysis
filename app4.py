# app.py — Softer Donuts + Persona vs Theme Heatmap + RAG (no likes; “Complementary Comments”)

from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
import sklearn  # optional for top_ngrams
import streamlit.components.v1 as components
from pathlib import Path
import json



# --- make_html_fullwidth.py style helper (inline here for convenience) ---
import re
from bs4 import BeautifulSoup  # pip install beautifulsoup4

FIX_CLASS_PATTERNS = [
    r"^w-\d+$", r"^w-(px|sm|md|lg|xl|2xl)$", r"^max-w-.*", r"^min-w-.*",
    r"^container$", r"^Container$", r"^page(-container)?$", r"^wrapper$",
]

OVERRIDE_CSS = """
/* Force common wrappers to stretch */
html, body { width:100% !important; max-width:100% !important; margin:0 !important; padding:0 !important; }
.container, .container-fluid, .content, .wrapper, .wrap, .page, .page-container,
.grid, .row, [class*="container"], [class*="Container"], [class*="max-w-"] {
  width:100% !important; max-width:100% !important; margin-left:0 !important; margin-right:0 !important;
}
section, article, .card, .panel { width:100% !important; max-width:100% !important; }
img, video, canvas, svg { max-width:100% !important; height:auto; }
"""

def make_fullwidth_html(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html, "html.parser")

    # 1) strip inline width/max-width from wrappers
    for tag in soup.find_all(style=True):
        style = tag.get("style", "")
        # drop width / max-width declarations
        style = re.sub(r"(?:max-)?width\s*:\s*[^;]+;?", "", style, flags=re.I)

        #style = re.sub(r"(?:^|;)\s*(?:min-|max-)?width\s*:\s*[^;]+;?", ";", style, flags=re.I)

        tag["style"] = style

    # 2) remove fixed-width utility classes
    class_re_list = [re.compile(pat) for pat in FIX_CLASS_PATTERNS]
    for tag in soup.find_all(class_=True):
        keep = []
        for c in tag.get("class", []):
            if not any(r.match(c) for r in class_re_list):
                keep.append(c)
        if keep:
            tag["class"] = keep
        else:
            del tag["class"]

    # 3) inject override stylesheet in <head> (or at top)
    head = soup.head or soup.new_tag("head")
    style_tag = soup.new_tag("style")
    style_tag.string = OVERRIDE_CSS
    head.insert(0, style_tag)
    if not soup.head:
        soup.html.insert(0, head) if soup.html else soup.insert(0, head)

    return str(soup)


def hr():
    st.markdown("<hr style='margin:12px 0;border:none;border-top:1px solid #e5e7eb;'/>", unsafe_allow_html=True)

st.set_page_config(page_title="LinkedIn Comment Intelligence", layout="wide")
st.markdown("""
<style>
:root, body, .stApp, [data-testid="stAppViewContainer"] { font-size: 1.06rem; }
.big-label { font-size: 2.1rem; font-weight: 800; margin: .25rem 0 .75rem 0; letter-spacing: .2px; }
.small-muted { color: #6b7280; font-size: 1.0rem; }
.quote { font-style: italic; }
[data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 2rem; }
[data-testid="stMetric"] [data-testid="stMetricLabel"] { font-size: 1.05rem; }
</style>
""", unsafe_allow_html=True)

LABELED = "comments_labeled.parquet"
try:
    df = pd.read_parquet(LABELED)
except Exception as e:
    st.error(f"Error loading parquet file: {e}")
    st.info("Loading from dataset.json instead...")
    # Load from JSON as fallback
    with open("dataset.json") as f:
        data = json.load(f)
    df = pd.json_normalize(data)
    # Map columns
    if 'commentary' in df.columns:
        df['text'] = df['commentary']
    if 'actor.name' in df.columns:
        df['author'] = df['actor.name']
    if 'engagement.comments' in df.columns:
        df['replies'] = pd.to_numeric(df['engagement.comments'], errors='coerce').fillna(0)
    # Add missing columns
    df['super_theme'] = 'suggestions_ideas'
    df['persona'] = 'unknown'
    df['sentiment'] = 0.5
    df['shares'] = 0
    df['comment_url'] = df.get('linkedinUrl', '')
    df['author_title'] = df.get('actor.position', '')
    df['id'] = range(len(df))
    df['created_at'] = pd.to_datetime('2024-01-01', utc=True)  # Default date
    if 'createdAt' in df.columns:
        df['created_at'] = pd.to_datetime(df['createdAt'], utc=True, errors='coerce')
    elif 'createdAtTimestamp' in df.columns:
        df['created_at'] = pd.to_datetime(df['createdAtTimestamp'], unit='ms', utc=True, errors='coerce')

# Required cols - check after loading
needed = ["created_at","super_theme","persona","text","author","author_title","replies","comment_url"]
if not df.empty:
    missing = [c for c in needed if c not in df.columns]
    if missing:
        st.warning(f"Some columns missing: {missing}. Using defaults.")
        # Add missing columns with defaults
        for col in missing:
            if col == "created_at":
                df[col] = pd.to_datetime('2024-01-01', utc=True)
            elif col in ["super_theme", "persona", "text", "author", "author_title", "comment_url"]:
                df[col] = "unknown"
            elif col == "replies":
                df[col] = 0

# Hygiene (no likes anywhere)
for c in ["replies","shares"]:
    if c not in df.columns: df[c] = 0
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
if "sentiment" not in df.columns: df["sentiment"] = 0.0
df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce").fillna(0.0)

# engagement score from replies + shares only
w_replies, w_shares = 1.5, 2.0
df["engagement_score"] = (df.get("replies",0)*w_replies + df.get("shares",0)*w_shares).astype(float)

# ---------- Mappings ----------
SUPER_THEME_ORDER = ["suggestions_ideas","support_enthusiasm","positive_feedback","negative_feedback"]
SUPER_THEME_LABELS = {
    "suggestions_ideas":"Suggestions & Ideas",
    "support_enthusiasm":"Support & Enthusiasm",
    "positive_feedback":"Complementary Comments",   # renamed
    "negative_feedback":"Negative Feedback",
}
THEME_COLORS = {
    "Suggestions & Ideas": "#6BAED6",
    "Support & Enthusiasm": "#C6DBEF",
    "Complementary Comments": "#C7E9C0",  # renamed key
    "Negative Feedback": "#F7C4C0",
}

PERSONA_ORDER = ["industry_exec","industry_lead","academic_leadership","academic_staff","unknown"]
PERSONA_LABELS = {
    "industry_exec":"CEOs / Directors (Industry)",
    "industry_lead":"Lead Roles (Industry)",
    "academic_leadership":"Academic Leadership",
    "academic_staff":"Academic Staff",
    "unknown":"Non-Leadership",  # renamed display
}
PERSONA_COLORS = {
    "CEOs / Directors (Industry)": "#6BAED6",
    "Lead Roles (Industry)": "#C6DBEF",
    "Academic Leadership": "#C7E9C0",
    "Academic Staff": "#F7C4C0",
    "Non-Leadership": "#E5E7EB",   # renamed key
}
PERSONA_ICONS = {
    "industry_exec":"🏢",
    "industry_lead":"👔",
    "academic_leadership":"🎓",
    "academic_staff":"👩‍🏫",
    "unknown":"❓",
}

# ---- Helper: top keyphrases from Suggestions (kept for future use) ----
def top_ngrams(texts, top_n=12):
    import re
    try:
        from sklearn.feature_extraction.text import CountVectorizer
        vec = CountVectorizer(ngram_range=(1,2), min_df=2, stop_words="english")
        X = vec.fit_transform([t for t in texts if isinstance(t, str)])
        terms = vec.get_feature_names_out()
        counts = X.sum(axis=0).A1
        dfk = pd.DataFrame({"term": terms, "count": counts})
        dfk = dfk[dfk["term"].str.len() >= 3]
        return dfk.sort_values("count", ascending=False).head(top_n).reset_index(drop=True)
    except Exception:
        from collections import Counter
        toks, bigs = [], []
        for t in texts:
            if not isinstance(t, str): 
                continue
            t_low = t.lower()
            toks += re.findall(r"[A-Za-z\u0600-\u06FF]{3,}", t_low)
            words = re.findall(r"[A-Za-z\u0600-\u06FF]{3,}", t_low)
            bigs += [" ".join(x) for x in zip(words, words[1:])]
        cnt = Counter(toks + bigs)
        terms, counts = zip(*cnt.most_common(top_n)) if cnt else ([], [])
        return pd.DataFrame({"term": terms, "count": counts})

# ---------- Sidebar Filters ----------
with st.sidebar:
    st.header("Filters")

    LOCAL_TZ = "Asia/Riyadh"
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    if df["created_at"].isna().all():
        st.warning("No valid timestamps in the data."); st.stop()

    df["created_local"] = df["created_at"].dt.tz_convert(LOCAL_TZ)
    df["created_local_naive"] = df["created_local"].dt.tz_localize(None)

    date_min = df["created_local_naive"].min().to_pydatetime()
    date_max = df["created_local_naive"].max().to_pydatetime()
    start, end = st.slider("Date range", min_value=date_min, max_value=date_max, value=(date_min, date_max))

    # For counts in sidebar
    df_date = df[(df["created_local_naive"] >= start) & (df["created_local_naive"] <= end)]

    # Professions
    st.subheader("Professions")
    core_personas = ["industry_exec","industry_lead","academic_leadership","academic_staff"]
    if "persona_selected" not in st.session_state:
        st.session_state.persona_selected = set(core_personas)

    selected_codes = []
    for code in core_personas:
        count = int((df_date["persona"] == code).sum())
        label = f"{PERSONA_ICONS[code]} {PERSONA_LABELS[code]}  ({count})"
        checked = st.checkbox(label, key=f"cb_prof_{code}", value=(code in st.session_state.persona_selected))
        if checked: selected_codes.append(code)
    st.session_state.persona_selected = set(selected_codes)

    include_unknown = st.checkbox(f"{PERSONA_ICONS['unknown']} Non-Leadership", value=True, key="cb_prof_unknown")

    hr()

    # Themes
    st.subheader("Themes")
    if "theme_selected" not in st.session_state:
        st.session_state.theme_selected = set(SUPER_THEME_ORDER)

    selected_theme_codes = []
    for code in SUPER_THEME_ORDER:
        label = SUPER_THEME_LABELS[code]
        count = int((df_date["super_theme"] == code).sum())
        checked = st.checkbox(f"{label}  ({count})", key=f"cb_theme_{code}", value=(code in st.session_state.theme_selected))
        if checked: selected_theme_codes.append(code)
    st.session_state.theme_selected = set(selected_theme_codes)
    sel_themes_codes = set(st.session_state.theme_selected)

# Final filters
mask_date = (df["created_local_naive"] >= start) & (df["created_local_naive"] <= end)
sel_personas_codes = set(st.session_state.persona_selected)
if include_unknown: sel_personas_codes.add("unknown")
mask_persona = df["persona"].isin(sel_personas_codes) if sel_personas_codes else True
mask_theme = df["super_theme"].isin(sel_themes_codes) if sel_themes_codes else True
df_f = df.loc[mask_date & mask_persona & mask_theme].copy()
if df_f.empty:
    st.info("No comments match the selected filters."); st.stop()

# ---------- Chart helper ----------
def soft_donut(df_counts, names_col, values_col, color_map, height=None, y_shift=0.10):
    import plotly.express as px
    fig = px.pie(
        df_counts,
        names=names_col,
        values=values_col,
        hole=0.58,
        template="simple_white",
        color=names_col,
        color_discrete_map=color_map,
    )
    # Percent labels on slices; no text outside
    fig.update_traces(
        textinfo="percent",
        textposition="inside",
        textfont_size=14,
        marker=dict(line=dict(color="#FFFFFF", width=2)),
        hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
        domain=dict(y=[y_shift, 1.0])  # nudges donut a bit upward
    )
    # Legend: vertical, to the RIGHT of the donut, centered vertically
    fig.update_layout(
        legend=dict(
            orientation="v",
            x=1.02, xanchor="left",   # place legend just to the right
            y=0.5,  yanchor="middle",
            font=dict(size=18),
            traceorder="normal",
        ),
        font=dict(size=17),
        margin=dict(l=10, r=170, t=24, b=8),  # right margin leaves room for legend
    )
    if height:
        fig.update_layout(height=height)
    return fig






# ---------- Layout ----------
# tab1, tab2, tab3 = st.tabs(["Overview", "Browse", "Q&A"])
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Insights", "Q&A (LLM)", "Browse"])


# Overview: donuts + heatmap
# ---------- Overview (50/50: left heatmap aligned to two donuts on right) ----------
# ---------- Overview (50/50: heatmap | donuts; aligned heights & tighter margins) ----------
with tab1:
    # KPI row
    c1, c2 = st.columns(2)
    c1.metric("Comments", len(df_f))
    c2.metric("Authors", df_f["author"].nunique())

    # Prep data
    counts_theme = (df_f["super_theme"].value_counts()
                    .reindex(SUPER_THEME_ORDER).fillna(0)
                    .rename(index=SUPER_THEME_LABELS).reset_index())
    counts_theme.columns = ["Theme","Comments"]

    counts_pers = (df_f["persona"].value_counts()
                   .reindex(PERSONA_ORDER).fillna(0)
                   .rename(index=PERSONA_LABELS).reset_index())
    counts_pers.columns = ["Persona","Comments"]


        # Pin consistent heights so left == two donuts combined
    DONUT_H = 360      # same for both
    GAP_PX  = 12
    HEAT_H  = DONUT_H * 2 + GAP_PX + 14

    fig1 = soft_donut(counts_theme, "Theme", "Comments", THEME_COLORS, height=DONUT_H, y_shift=0.10)
    fig2 = soft_donut(counts_pers,  "Persona", "Comments", PERSONA_COLORS, height=DONUT_H, y_shift=0.10)



    COMMON_MARGINS = dict(l=140, r=10, t=36, b=8)  # match legend space
    fig1.update_layout(height=DONUT_H, margin=COMMON_MARGINS)
    fig2.update_layout(height=DONUT_H, margin=COMMON_MARGINS)


    # Heatmap data
    tmp = df_f.assign(
        theme_label=lambda d: d["super_theme"].map(SUPER_THEME_LABELS),
        persona_label=lambda d: d["persona"].map(PERSONA_LABELS),
    )
    ordered_themes   = [SUPER_THEME_LABELS[t] for t in SUPER_THEME_ORDER if t in df_f["super_theme"].unique()]
    ordered_personas = [PERSONA_LABELS[p]     for p in PERSONA_ORDER     if p in df_f["persona"].unique()]
    heat = (tmp.pivot_table(index="persona_label", columns="theme_label", values="id",
                            aggfunc="count", fill_value=0)
              .reindex(index=ordered_personas, columns=ordered_themes))

    # One row: 50/50 split
    left, right = st.columns(2, gap="large")

    with left:
        st.subheader("Persona vs Theme")
        if heat.size == 0:
            st.info("No data for the selected filters.")
        else:
            fig_hm = px.imshow(
                heat.values,
                x=list(heat.columns),
                y=list(heat.index),
                text_auto=True,
                aspect="auto",
                color_continuous_scale="Blues",
                origin="upper"
            )
            fig_hm.update_traces(textfont=dict(size=15))
            fig_hm.update_layout(
                template="simple_white",
                font=dict(size=17),
                height=HEAT_H,                                # align to two donuts
                margin=dict(l=10, r=10, t=10, b=0),           # <-- remove extra bottom margin
                xaxis_title="Theme",
                yaxis_title="Profession",
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_hm, use_container_width=True)

    with right:
        st.subheader("Comments by Theme")
        st.plotly_chart(fig1, use_container_width=True)
        # tiny spacer instead of hr() to preserve tight alignment
        st.markdown(f"<div style='height:{GAP_PX}px'></div>", unsafe_allow_html=True)
        st.subheader("Comments by Profession")
        st.plotly_chart(fig2, use_container_width=True)


# Browse: keep, but no Avg sentiment / Replies KPIs and no likes anywhere
with tab4:
    st.subheader("Browse by Theme")
    themes_present = [t for t in SUPER_THEME_ORDER if t in df_f["super_theme"].unique()]
    if not themes_present:
        st.info("No themes present after filtering."); st.stop()

    sel_theme_disp = st.selectbox("Theme", [SUPER_THEME_LABELS[t] for t in themes_present])
    sel_theme_code = next(k for k,v in SUPER_THEME_LABELS.items() if v == sel_theme_disp)
    df_t = df_f[df_f["super_theme"] == sel_theme_code].copy()

    st.markdown(f"<div class='big-label'>{SUPER_THEME_LABELS[sel_theme_code]}</div>", unsafe_allow_html=True)
    st.metric("Comments", len(df_t))

    eng = pd.to_numeric(df_t.get("engagement_score", 0), errors="coerce").fillna(0.0).astype(float).values
    score = np.log1p(eng)
    if sel_theme_code == "suggestions_ideas":
        length = df_t["text"].str.len().clip(40, 600).astype(float).values
        score = score + 0.15 * (length - length.min()) / (length.max() - length.min() + 1e-9)
    order = score.argsort()[::-1]
    topN = min(5, len(df_t))

    st.markdown("**Representative comments**")
    for idx in order[:topN]:
        row = df_t.iloc[idx]
        st.write(f"> {row['text']}")
        meta = f"— {row.get('author','Anonymous')}"
        if isinstance(row.get("author_title"), str) and row["author_title"].strip():
            meta += f", {row['author_title']}"
        meta += f"  | 💬 {int(row.get('replies',0))}"
        st.caption(meta)
        if isinstance(row.get("comment_url"), str) and row["comment_url"].strip():
            st.write(f"[Open comment]({row['comment_url']})")
        hr()

    sort_pref = [c for c in ["engagement_score","replies"] if c in df_t.columns]
    df_t_sorted = df_t.sort_values(sort_pref, ascending=False) if sort_pref else df_t.copy()

    display_cols = ["author","author_title","text","replies","sentiment","persona","created_local_naive","comment_url"]
    present_cols = [c for c in display_cols if c in df_t_sorted.columns]
    st.markdown("**All comments in this theme**")
    st.dataframe(df_t_sorted[present_cols], use_container_width=True)

# ---------- Q&A (RAG over dataset.json) ----------
# ---------- Q&A (Grounded on LinkedIn comments) ----------

# ---------- Insights (HTML, full-width with true upscaling) ----------
# ---------- Insights (load HTML inside its own iframe and override CSS there) ----------
# ---------- Insights (preprocess fixed widths → full-width embed) ----------
with tab2:
    import streamlit.components.v1 as components
    from pathlib import Path

    # st.subheader("HUMAIN Academy — Tabbed Insights")

    html_path = Path("humain_academy_tabbed_insights_2x3.html")
    if not html_path.exists():
        alt = Path("/mnt/data/humain_academy_tabbed_insights_2x3.html")
        html_path = alt if alt.exists() else html_path

    if not html_path.exists():
        st.error("Could not find humain_academy_tabbed_insights_2x3.html")
    else:
        raw = html_path.read_text(encoding="utf-8")

        # >>> PREPROCESS: remove fixed widths & inject overrides
        fixed = make_fullwidth_html(raw)

        # Option 1 (preferred when HTML needs JS): iframe
        components.html(fixed, height=1400, scrolling=True, width=1600)  # width will match the Streamlit column
        #st.html(fixed, width="stretch")
with tab3:
    st.subheader("Q&A (LLM)")

    # ---------- Config / paths ----------
    from pathlib import Path
    DATASET_JSON = Path("dataset.json")

    # ---------- OpenAI key via Streamlit Secrets (docs-correct) ----------
    import os
    import streamlit as st

    OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", "").strip()
    OPENAI_PROJECT = st.secrets.get("OPENAI_PROJECT", "").strip()

    from openai import OpenAI
    OAICLIENT = OpenAI(api_key=OPENAI_KEY, project=OPENAI_PROJECT or None)

    # (Optional) quick status so you can see it’s wired:
    st.caption(f"LLM: {'enabled' if OPENAI_KEY else 'disabled'} • Project: {OPENAI_PROJECT or '—'}")




    # OPENAI_KEY = st.secrets["OPENAI_API_KEY"]
    # USE_LLM = False
    # OAICLIENT = None
    # LLM_STATUS = "disabled"
    # LLM_REASON = ""

    if OPENAI_KEY:
        try:
            from openai import OpenAI  # requires `openai>=1.0.0` in requirements.txt
            OAICLIENT = OpenAI(api_key=OPENAI_KEY)
            USE_LLM = True
            LLM_STATUS = "enabled"
        except ImportError:
            LLM_REASON = "The `openai` package is not installed. Add `openai>=1.0.0` to requirements.txt and redeploy."
        except Exception as e:
            LLM_REASON = f"{type(e).__name__}: {e}"
    else:
        LLM_REASON = "Secret `OPENAI_API_KEY` not found. Add it in Streamlit Cloud → Settings → Secrets."

    if not USE_LLM:
        st.caption(f"LLM status: {LLM_STATUS} — {LLM_REASON}")

    # ---------- Load + tidy dataset ----------
    import json, re, pandas as pd, numpy as np

    def _load_dataset_json(path: Path) -> pd.DataFrame:
        if not path.exists():
            st.error(f"Missing {path}. Place dataset.json in the app folder.")
            st.stop()
        txt = path.read_text(encoding="utf-8")
        try:
            raw = json.loads(txt)
            if not isinstance(raw, list):
                raw = raw.get("data", raw)
                if not isinstance(raw, list):
                    raise ValueError("JSON root is not a list")
        except Exception:
            raw = [json.loads(line) for line in txt.splitlines() if line.strip()]
        d0 = pd.json_normalize(raw)

        keep = {
            "id":"id","linkedinUrl":"comment_url","commentary":"text",
            "createdAt":"created_at","createdAtTimestamp":"created_ts","postId":"post_id",
            "engagement.comments":"replies",
            "actor.name":"author","actor.position":"author_title","actor.linkedinUrl":"author_url",
        }
        present = [k for k in keep if k in d0.columns]
        d0 = d0[present].rename(columns=keep)

        def clean(s: str) -> str:
            if not isinstance(s, str): return ""
            return re.sub(r"\s+"," ", s.replace("\u200b","")).strip()
        d0["text"] = d0["text"].map(clean)
        d0 = d0[d0["text"].str.len()>0].copy()

        # timestamps
        d0["created_at"] = pd.to_datetime(d0.get("created_at"), utc=True, errors="coerce")
        if "created_ts" in d0:
            d0["created_ts"] = pd.to_numeric(d0["created_ts"], errors="coerce")
        else:
            d0["created_ts"] = (d0["created_at"].astype("int64")//10**6).where(d0["created_at"].notna(), 0)

        d0["replies"] = pd.to_numeric(d0.get("replies"), errors="coerce").fillna(0).astype(int)

        # Bring over LLM labels from the main df if present
        dmerge = df[["id","super_theme","persona","sentiment"]].copy()
        dmerge["id"] = dmerge["id"].astype(str)
        d0["id"] = d0["id"].astype(str)
        d0 = d0.merge(dmerge, on="id", how="left")
        d0["super_theme"] = d0["super_theme"].fillna("unknown")
        d0["persona"] = d0["persona"].fillna("unknown")
        d0["sentiment"] = pd.to_numeric(d0["sentiment"], errors="coerce").fillna(0.0)
        return d0.reset_index(drop=True)

    data_for_index = _load_dataset_json(DATASET_JSON)

    # ---------- Embeddings (SentenceTransformers) with TF-IDF fallback ----------
    @st.cache_resource(show_spinner=False)
    def _load_st_model():
        try:
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        except Exception:
            return None

    st_model = _load_st_model()

    @st.cache_resource(show_spinner=False)
    def _build_embeddings(texts: list[str], use_st: bool):
        if use_st and st_model is not None:
            embs = st_model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
            return "st", np.asarray(embs, dtype="float32"), None
        else:
            # TF-IDF fallback
            from sklearn.feature_extraction.text import TfidfVectorizer
            vec = TfidfVectorizer(ngram_range=(1,2), min_df=1, stop_words="english")
            X = vec.fit_transform(texts)
            return "tfidf", X, vec

    method, DOC_EMB, TFIDF_VEC = _build_embeddings(
        data_for_index["text"].tolist(), use_st=(st_model is not None)
    )

    # ---------- Query helpers ----------
    from sklearn.metrics.pairwise import cosine_similarity

    def _mask_indices(persona=None, super_theme=None, start_ts=None, end_ts=None):
        m = pd.Series([True]*len(data_for_index))
        if persona:
            m &= data_for_index["persona"].isin(list(persona))
        if super_theme:
            m &= data_for_index["super_theme"].isin(list(super_theme))
        if start_ts is not None:
            m &= (data_for_index["created_ts"].fillna(0) >= int(start_ts))
        if end_ts is not None:
            m &= (data_for_index["created_ts"].fillna(0) <= int(end_ts))
        return np.where(m.to_numpy())[0]

    def ask(question, k=10, persona=None, super_theme=None, start_ts=None, end_ts=None):
        idx = _mask_indices(persona, super_theme, start_ts, end_ts)
        if idx.size == 0:
            return {"answer":"No relevant comments found.", "hits":[]}

        # Compute similarities
        if method == "st":
            q = st_model.encode([question], normalize_embeddings=True)
            sims = (DOC_EMB[idx] @ q[0]).astype("float32")     # cosine via dot product
        else:
            q = TFIDF_VEC.transform([question])
            sims = cosine_similarity(q, DOC_EMB[idx]).ravel()

        order = sims.argsort()[::-1][:k]
        hit_idx = idx[order]
        docs  = [data_for_index["text"].iat[i] for i in hit_idx]
        metas = data_for_index.iloc[hit_idx][["author","author_title","comment_url","replies"]].to_dict("records")
        scores = sims[order]

        # Build grounded context
        lines = []
        for t, m in zip(docs, metas):
            who  = m.get("author") or "Anonymous"
            role = m.get("author_title") or ""
            link = m.get("comment_url") or ""
            lines.append(f"- {t}\n  (by {who}{', '+role if role else ''}) {link}")
        ctx = "\n".join(lines)

        # Synthesis or extractive
        if USE_LLM and OAICLIENT is not None:
            prompt = f"""Answer the question using ONLY the comments below.
Cite by pasting the comment URLs in parentheses after claims.
If the comments are insufficient, say what's missing.

Question: {question}

Comments:
{ctx}
"""
            try:
                resp = OAICLIENT.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role":"user","content":prompt}],
                    temperature=0.1,
                )
                ans = resp.choices[0].message.content.strip()
            except Exception as e:
                st.warning(f"LLM call failed: {type(e).__name__}: {e}")
                ans = "Top matching comments:\n" + "\n".join([f"- {d}" for d in docs[:6]])
        else:
            ans = ("Top matching comments:\n" + "\n".join([f"- {d}" for d in docs[:6]]) +
                   "\n\n(LLM synthesis is disabled. See the LLM status message above to enable it.)")

        hits = []
        for m, s in zip(metas, scores):
            hits.append({
                "author": m.get("author"),
                "author_title": m.get("author_title"),
                "comment_url": m.get("comment_url"),
                "replies": m.get("replies"),
                "score": float(s),  # similarity score (0..1 if ST; cosine if TF-IDF)
            })
        return {"answer": ans, "hits": hits}

    # ---------- Apply current sidebar filters (date → UTC ms) ----------
    import pytz
    tz = pytz.timezone("Asia/Riyadh")
    start_ts = int(tz.localize(pd.to_datetime(start).to_pydatetime()).astimezone(pytz.utc).timestamp()*1000)
    end_ts   = int(tz.localize(pd.to_datetime(end).to_pydatetime()).astimezone(pytz.utc).timestamp()*1000)

    # ---------- Minimal UI ----------
    q = st.text_input("Ask a question", placeholder="e.g., What curriculum modules did people request?")
    if q:
        persona_filter = sel_personas_codes if len(sel_personas_codes)>0 else None
        theme_filter   = sel_themes_codes   if len(sel_themes_codes)>0   else None
        res = ask(q, k=10, persona=persona_filter, super_theme=theme_filter, start_ts=start_ts, end_ts=end_ts)

        st.markdown("### Answer")
        st.write(res["answer"])

        st.markdown("### Sources")
        if not res["hits"]:
            st.info("No sources.")
        else:
            for h in res["hits"]:
                with st.container():
                    meta = f"{h.get('author') or 'Anonymous'}"
                    if h.get("author_title"):
                        meta += f" — {h['author_title']}"
                    if h.get("score") is not None:
                        meta += f"  •  match: {float(h['score']):.2f}"
                    st.caption(meta)
                    if h.get("comment_url"):
                        st.write(f"[Open comment]({h['comment_url']})")
                    hr()
