# app.py — Fixed Super Themes + Improved Persona UX + Softer Donut Charts
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np

def hr():
    st.markdown(
        "<hr style='margin:12px 0;border:none;border-top:1px solid #e5e7eb;'/>",
        unsafe_allow_html=True
    )

st.set_page_config(page_title="LinkedIn Comment Intelligence", layout="wide")
st.markdown("""
<style>
.big-label { font-size: 2.0rem; font-weight: 800; margin: 0.25rem 0 0.75rem 0; letter-spacing: .2px; }
.small-muted { color: #6b7280; font-size: 0.9rem; }
.quote { font-style: italic; }
</style>
""", unsafe_allow_html=True)

ART = Path("./")
LABELED = ART / "comments_labeled.parquet"

# ---------- Load ----------
if not LABELED.exists():
    st.error(f"Missing {LABELED}. Run the LLM labeling step (Step 4) first.")
    st.stop()

df = pd.read_parquet(LABELED)

# Ensure needed columns exist
needed = ["created_at","super_theme","persona","text","author","author_title","likes","replies","comment_url"]
missing = [c for c in needed if c not in df.columns]
if missing:
    st.error(f"Required columns missing: {missing}")
    st.stop()

# Hygiene
for c in ["likes","replies","shares"]:
    if c not in df.columns: df[c] = 0
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
if "sentiment" not in df.columns: df["sentiment"] = 0.0
df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce").fillna(0.0)
if "engagement_score" not in df.columns:
    w_likes, w_replies, w_shares = 1.0, 1.5, 2.0
    df["engagement_score"] = df["likes"]*w_likes + df["replies"]*w_replies + df["shares"]*w_shares
df["engagement_score"] = pd.to_numeric(df["engagement_score"], errors="coerce").fillna(0.0)

# ---------- Mappings ----------
SUPER_THEME_ORDER = ["suggestions_ideas","support_enthusiasm","positive_feedback","negative_feedback"]
SUPER_THEME_LABELS = {
    "suggestions_ideas":"Suggestions & Ideas",
    "support_enthusiasm":"Support & Enthusiasm",
    "positive_feedback":"Positive Feedback",
    "negative_feedback":"Negative Feedback",
}
THEME_COLORS = {
    "Suggestions & Ideas": "#6BAED6",
    "Support & Enthusiasm": "#C6DBEF",
    "Positive Feedback": "#C7E9C0",
    "Negative Feedback": "#F7C4C0",
}

PERSONA_ORDER = ["industry_exec","industry_lead","academic_leadership","academic_staff","unknown"]
PERSONA_LABELS = {
    "industry_exec":"CEOs / Directors (Industry)",
    "industry_lead":"Lead Roles (Industry)",
    "academic_leadership":"Academic Leadership",
    "academic_staff":"Academic Staff",
    "unknown":"Unknown",
}
PERSONA_COLORS = {
    "CEOs / Directors (Industry)": "#6BAED6",
    "Lead Roles (Industry)": "#C6DBEF",
    "Academic Leadership": "#C7E9C0",
    "Academic Staff": "#F7C4C0",
    "Unknown": "#E5E7EB",
}
PERSONA_ICONS = {
    "industry_exec":"🏢",
    "industry_lead":"👔",
    "academic_leadership":"🎓",
    "academic_staff":"👩‍🏫",
    "unknown":"❓",
}

# ---------- Sidebar Filters (Professions + Themes stacked) ----------
with st.sidebar:
    st.header("Filters")

    LOCAL_TZ = "Asia/Riyadh"
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    if df["created_at"].isna().all():
        st.warning("No valid timestamps in dataset."); st.stop()

    df["created_local"] = df["created_at"].dt.tz_convert(LOCAL_TZ)
    df["created_local_naive"] = df["created_local"].dt.tz_localize(None)

    date_min = df["created_local_naive"].min().to_pydatetime()
    date_max = df["created_local_naive"].max().to_pydatetime()
    start, end = st.slider("Date range", min_value=date_min, max_value=date_max, value=(date_min, date_max))

    # Pre-filter by date for counts
    df_date = df[(df["created_local_naive"] >= start) & (df["created_local_naive"] <= end)]

    # ----- Professions -----
    st.subheader("Professions")
    core_personas = ["industry_exec","industry_lead","academic_leadership","academic_staff"]
    if "persona_selected" not in st.session_state:
        st.session_state.persona_selected = set(core_personas)

    # Quick presets (with unique keys)
    colA, colB, colC, colD = st.columns(4)
    # if colA.button("All", key="prof_all"):
    #     st.session_state.persona_selected = set(core_personas)
    # if colB.button("Industry", key="prof_industry"):
    #     st.session_state.persona_selected = {"industry_exec", "industry_lead"}
    # if colC.button("Academia", key="prof_academia"):
    #     st.session_state.persona_selected = {"academic_leadership", "academic_staff"}
    # if colD.button("Execs only", key="prof_execs"):
    #     st.session_state.persona_selected = {"industry_exec"}

    # Checkboxes with counts
    selected_codes = []
    for code in core_personas:
        count = int((df_date["persona"] == code).sum())
        label = f"{PERSONA_ICONS[code]} {PERSONA_LABELS[code]}  ({count})"
        checked = st.checkbox(label, key=f"cb_prof_{code}", value=(code in st.session_state.persona_selected))
        if checked:
            selected_codes.append(code)
    st.session_state.persona_selected = set(selected_codes)

    include_unknown = st.checkbox(f"{PERSONA_ICONS['unknown']} Include Unknown", value=False, key="cb_prof_unknown")

    hr()  # divider between Profession and Theme blocks

    # ----- Themes -----
    st.subheader("Themes")
    if "theme_selected" not in st.session_state:
        st.session_state.theme_selected = set(SUPER_THEME_ORDER)

    tA, tB, tC, tD = st.columns(4)
    # if tA.button("All", key="theme_all"):
    #     st.session_state.theme_selected = set(SUPER_THEME_ORDER)
    # if tB.button("Suggestions", key="theme_suggestions"):
    #     st.session_state.theme_selected = {"suggestions_ideas"}
    # if tC.button("Positive/Support", key="theme_pos_support"):
    #     st.session_state.theme_selected = {"support_enthusiasm", "positive_feedback"}
    # if tD.button("Critical", key="theme_critical"):
    #     st.session_state.theme_selected = {"negative_feedback"}

    selected_theme_codes = []
    for code in SUPER_THEME_ORDER:
        label = SUPER_THEME_LABELS[code]
        count = int((df_date["super_theme"] == code).sum())
        checked = st.checkbox(f"{label}  ({count})", key=f"cb_theme_{code}", value=(code in st.session_state.theme_selected))
        if checked:
            selected_theme_codes.append(code)
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

# ---------- Helper to make soft donut chart ----------
def soft_donut(df_counts, names_col, values_col, color_map, center_text):
    fig = px.pie(
        df_counts, names=names_col, values=values_col,
        hole=0.58, template="simple_white",
        color=names_col, color_discrete_map=color_map
    )
    fig.update_traces(
        textinfo="percent+label",
        textposition="inside",
        textfont_size=14,
        marker=dict(line=dict(color="#FFFFFF", width=2))
    )
    fig.update_layout(
        legend_title=None,
        font=dict(size=15),
        uniformtext_minsize=12,
        uniformtext_mode="hide",
        margin=dict(l=10, r=10, t=10, b=10),
    )
    fig.add_annotation(
        text=center_text, x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color="#374151")
    )
    return fig

# ---------- Layout ----------
# st.title("Comment Insights — Fixed Themes & Personas")

tab1, tab2 = st.tabs(["Overview", "Browse"])

# ---------- Overview (soft donut charts side-by-side) ----------
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Comments", len(df_f))
    c2.metric("Authors", df_f["author"].nunique())
    pos_pct = (df_f["super_theme"].isin(["support_enthusiasm","positive_feedback"]).mean()*100) if len(df_f) else 0
    c3.metric("Positive/Support %", f"{pos_pct:.0f}%")
    c4.metric("Suggestions", int((df_f["super_theme"]=="suggestions_ideas").sum()))

    counts_theme = (df_f["super_theme"].value_counts()
                    .reindex(SUPER_THEME_ORDER).fillna(0)
                    .rename(index=SUPER_THEME_LABELS).reset_index())
    counts_theme.columns = ["Theme","Comments"]

    counts_pers = (df_f["persona"].value_counts()
                   .reindex(PERSONA_ORDER).fillna(0)
                   .rename(index=PERSONA_LABELS).reset_index())
    counts_pers.columns = ["Persona","Comments"]

    fig1 = soft_donut(counts_theme, "Theme", "Comments", THEME_COLORS, f"N={len(df_f)}")
    fig2 = soft_donut(counts_pers, "Persona", "Comments", PERSONA_COLORS, "")

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.subheader("Share of Comments by Theme")
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.subheader("Share of Comments by Profession")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Theme stats")
    theme_stats = (
        df_f.groupby("super_theme")
            .agg(
                comments=("id","count"),
                avg_sent=("sentiment","mean"),
                # likes=("likes","sum"),
                replies=("replies","sum"),
                #ideas=("text", lambda s: int((df_f.loc[s.index,"super_theme"]=="suggestions_ideas").sum()))
            )
            .reindex(SUPER_THEME_ORDER)
            .reset_index()
    )
    theme_stats["super_theme"] = theme_stats["super_theme"].map(SUPER_THEME_LABELS)
    st.dataframe(theme_stats, use_container_width=True)

# ---------- Browse (larger theme heading) ----------
with tab2:
    st.subheader("Browse by Theme")
    themes_present = [t for t in SUPER_THEME_ORDER if t in df_f["super_theme"].unique()]
    if not themes_present:
        st.info("No themes present after filtering."); st.stop()

    sel_theme_disp = st.selectbox("Theme", [SUPER_THEME_LABELS[t] for t in themes_present])
    sel_theme_code = next(k for k,v in SUPER_THEME_LABELS.items() if v == sel_theme_disp)
    df_t = df_f[df_f["super_theme"] == sel_theme_code].copy()

    st.markdown(f"<div class='big-label'>{SUPER_THEME_LABELS[sel_theme_code]}</div>", unsafe_allow_html=True)

    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Comments", len(df_t))
    t2.metric("Avg sentiment", f"{df_t['sentiment'].mean():+.2f}")
    # t3.metric("Likes", int(df_t["likes"].sum()))
    t4.metric("Replies", int(df_t["replies"].sum()))

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
        meta += f"  | 👍 {int(row['likes'])} • 💬 {int(row['replies'])}"
        st.caption(meta)
        if isinstance(row.get("comment_url"), str) and row["comment_url"].strip():
            st.write(f"[Open comment]({row['comment_url']})")
        hr()

    sort_pref = [c for c in ["engagement_score","likes","replies"] if c in df_t.columns]
    df_t_sorted = df_t.sort_values(sort_pref, ascending=False) if sort_pref else df_t.copy()

    display_cols = ["author","author_title","text","likes","replies","sentiment","persona","created_local_naive","comment_url"]
    present_cols = [c for c in display_cols if c in df_t_sorted.columns]
    st.markdown("**All comments in this theme**")
    st.dataframe(df_t_sorted[present_cols], use_container_width=True)
