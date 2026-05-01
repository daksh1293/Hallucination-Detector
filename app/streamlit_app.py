import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pipeline import run_pipeline
from llm_response import get_all_models, get_model_display_name

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Hallucination Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: #0a0e1a;
    color: #e8eaf0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f1628;
    border-right: 1px solid #1e2d4a;
}

/* Header */
.main-header {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
}
.main-header h1 {
    font-family: 'Space Mono', monospace;
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #4fc3f7, #7c4dff, #f06292);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -1px;
    margin-bottom: 0.4rem;
}
.main-header p {
    color: #7986a3;
    font-size: 1rem;
    font-weight: 300;
    letter-spacing: 0.5px;
}

/* Cards */
.metric-card {
    background: linear-gradient(135deg, #0f1e35, #142040);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    transition: transform 0.2s, border-color 0.2s;
}
.metric-card:hover {
    transform: translateY(-3px);
    border-color: #4fc3f7;
}
.metric-card .value {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: #4fc3f7;
}
.metric-card .label {
    font-size: 0.78rem;
    color: #7986a3;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 0.3rem;
}

/* Verdict boxes */
.verdict-grounded {
    background: linear-gradient(135deg, #0d2b1e, #0f3524);
    border: 1px solid #1b6b3a;
    border-left: 4px solid #2ecc71;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
}
.verdict-hallucination {
    background: linear-gradient(135deg, #2b0d0d, #351010);
    border: 1px solid #6b1b1b;
    border-left: 4px solid #e74c3c;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
}
.verdict-uncertain {
    background: linear-gradient(135deg, #2b2200, #352a00);
    border: 1px solid #6b5500;
    border-left: 4px solid #f39c12;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
}
.verdict-unverifiable {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #2d3561;
    border-left: 4px solid #7986a3;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
}

/* Evidence box */
.evidence-box {
    background: #0f1628;
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    font-size: 0.88rem;
    color: #a0aec0;
    line-height: 1.7;
    font-family: 'DM Sans', sans-serif;
}

/* Score bar */
.score-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 0.5rem 0;
}
.score-label {
    width: 130px;
    font-size: 0.82rem;
    color: #7986a3;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.score-bar-bg {
    flex: 1;
    background: #1e2d4a;
    border-radius: 6px;
    height: 8px;
    overflow: hidden;
}
.score-value {
    width: 50px;
    text-align: right;
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    color: #e8eaf0;
}

/* Section titles */
.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 3px;
    color: #4fc3f7;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e2d4a;
}

/* Finding card */
.finding-card {
    background: #0f1628;
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.finding-card .number {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #7c4dff;
}
.finding-card .title {
    font-size: 1rem;
    font-weight: 600;
    color: #e8eaf0;
    margin: 0.3rem 0 0.5rem;
}
.finding-card .desc {
    font-size: 0.88rem;
    color: #7986a3;
    line-height: 1.6;
}

/* Button */
div.stButton > button {
    background: linear-gradient(135deg, #4fc3f7, #7c4dff);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.7rem 2rem;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    font-size: 0.95rem;
    letter-spacing: 0.5px;
    transition: opacity 0.2s, transform 0.2s;
    width: 100%;
}
div.stButton > button:hover {
    opacity: 0.9;
    transform: translateY(-2px);
}

/* Select boxes */
div[data-baseweb="select"] > div {
    background: #0f1628 !important;
    border-color: #1e2d4a !important;
    border-radius: 10px !important;
    color: #e8eaf0 !important;
}

/* Text input */
div[data-baseweb="textarea"] textarea,
div[data-baseweb="input"] input {
    background: #0f1628 !important;
    border-color: #1e2d4a !important;
    border-radius: 10px !important;
    color: #e8eaf0 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
}

/* Hide streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Helper functions ───────────────────────────────────────────────────────────
def load_results():
    model_files = {
        "LLaMA 3.1 8B":  "data/llama3-8b_results.csv",
        "LLaMA 4 Scout": "data/llama4-scout_results.csv",
        "LLaMA 3.3 70B": "data/llama3-70b_results.csv",
    }
    dfs = {}
    for name, path in model_files.items():
        if os.path.exists(path):
            dfs[name] = pd.read_csv(path)
    return dfs


def render_verdict(verdict, support, contradiction):
    if "GROUNDED" in verdict:
        css_class = "verdict-grounded"
        icon = "✅"
        title = "GROUNDED"
        desc = "The LLM's answer is supported by evidence."
    elif "HALLUCINATION" in verdict:
        css_class = "verdict-hallucination"
        icon = "❌"
        title = "HALLUCINATION DETECTED"
        desc = "The LLM's answer contradicts available evidence."
    elif "UNCERTAIN" in verdict:
        css_class = "verdict-uncertain"
        icon = "⚠️"
        title = "UNCERTAIN"
        desc = "Evidence partially supports the answer."
    else:
        css_class = "verdict-unverifiable"
        icon = "🔎"
        title = "UNVERIFIABLE"
        desc = "No relevant Wikipedia evidence found."

    st.markdown(f"""
    <div class="{css_class}">
        <div style="font-size:1.5rem; margin-bottom:0.4rem">{icon} <strong style="font-family:'Space Mono',monospace; font-size:1rem; letter-spacing:1px">{title}</strong></div>
        <div style="color:#a0aec0; font-size:0.88rem">{desc}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    support_pct = int(support * 100)
    contra_pct  = int(contradiction * 100)

    st.markdown(f"""
    <div class="score-row">
        <span class="score-label">Support</span>
        <div class="score-bar-bg">
            <div style="width:{support_pct}%; background:linear-gradient(90deg,#2ecc71,#27ae60); height:100%; border-radius:6px; transition:width 0.8s ease;"></div>
        </div>
        <span class="score-value">{support:.3f}</span>
    </div>
    <div class="score-row">
        <span class="score-label">Contradiction</span>
        <div class="score-bar-bg">
            <div style="width:{contra_pct}%; background:linear-gradient(90deg,#e74c3c,#c0392b); height:100%; border-radius:6px; transition:width 0.8s ease;"></div>
        </div>
        <span class="score-value">{contradiction:.3f}</span>
    </div>
    """, unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1.5rem 0 1rem;">
        <div style="font-family:'Space Mono',monospace; font-size:1.1rem; color:#4fc3f7; font-weight:700;">🔍 HALL-DETECT</div>
        <div style="font-size:0.75rem; color:#7986a3; margin-top:0.3rem; letter-spacing:1px; text-transform:uppercase;">v1.0 — Research Tool</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div style="font-family:'Space Mono',monospace; font-size:0.7rem; text-transform:uppercase; letter-spacing:2px; color:#7986a3; margin-bottom:0.8rem;">Navigation</div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "",
        ["🔍 Live Demo", "📊 Research Results", "ℹ️ About"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.78rem; color:#7986a3; line-height:1.7;">
        <strong style="color:#4fc3f7;">How it works:</strong><br>
        1. LLM answers your question<br>
        2. Wikipedia evidence retrieved<br>
        3. NLI model checks consistency<br>
        4. Verdict generated
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.72rem; color:#4a5568; text-align:center; line-height:1.8;">
        Evaluated on TruthfulQA<br>
        NLI: BART-large-mnli<br>
    </div>
    """, unsafe_allow_html=True)


# ── Main Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>LLM Hallucination Detector</h1>
    <p>Cross-model factual consistency analysis using NLI-based verification</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — LIVE DEMO
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔍 Live Demo":

    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        st.markdown('<div class="section-title">Ask Any Question</div>', unsafe_allow_html=True)

        question = st.text_area(
            "",
            placeholder="e.g. Who invented the telephone? • What is the capital of Australia? • Did Einstein win the Nobel Prize?",
            height=120,
            label_visibility="collapsed"
        )

        model_keys = get_all_models()
        model_display = {get_model_display_name(k): k for k in model_keys}

        selected_display = st.selectbox(
            "Select LLM Model",
            list(model_display.keys())
        )
        selected_model = model_display[selected_display]

        run = st.button("🔍 Detect Hallucination", use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">Example Questions</div>', unsafe_allow_html=True)
        examples = [
            "Who really invented the telephone?",
            "Was Napoleon short for his time?",
            "Did Einstein fail math at school?",
            "Is the Great Wall visible from space?",
            "Who was the first person to walk on the Moon?",
            "What is the capital of Australia?",
        ]
        for ex in examples:
            st.markdown(f"""
            <div style="background:#0f1628; border:1px solid #1e2d4a; border-radius:8px;
                        padding:0.6rem 1rem; margin-bottom:0.5rem; font-size:0.85rem;
                        color:#a0aec0; cursor:pointer;">
                💬 {ex}
            </div>
            """, unsafe_allow_html=True)

    # ── Results ──
    if run and question.strip():
        st.markdown("---")
        st.markdown('<div class="section-title">Analysis Results</div>', unsafe_allow_html=True)

        with st.spinner("🤖 Getting LLM answer and checking against evidence..."):
            result = run_pipeline(question.strip(), model=selected_model)

        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">{result['support_score']:.2f}</div>
                <div class="label">Support Score</div>
            </div>
            """, unsafe_allow_html=True)
        with r2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">{result['contradiction_score']:.2f}</div>
                <div class="label">Contradiction</div>
            </div>
            """, unsafe_allow_html=True)
        with r3:
            conf = max(result['support_score'], result['contradiction_score'])
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">{conf:.2f}</div>
                <div class="label">Confidence</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        left, right = st.columns([1, 1], gap="large")

        with left:
            st.markdown('<div class="section-title">LLM Answer</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="evidence-box" style="border-left:4px solid #7c4dff; color:#e8eaf0; font-size:0.92rem;">
                <strong style="color:#7c4dff; font-size:0.72rem; letter-spacing:2px; text-transform:uppercase;">{selected_display}</strong><br><br>
                {result['llm_answer']}
            </div>
            """, unsafe_allow_html=True)

        with right:
            st.markdown('<div class="section-title">Verdict</div>', unsafe_allow_html=True)
            render_verdict(
                result['verdict'],
                result['support_score'],
                result['contradiction_score']
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Wikipedia Evidence</div>', unsafe_allow_html=True)

        evidence = result['evidence']
        if "No evidence" in evidence or "Error" in evidence:
            st.markdown("""
            <div class="evidence-box" style="color:#7986a3; text-align:center; padding:2rem;">
                🔎 No Wikipedia evidence found for this query.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="evidence-box">
                {evidence[:600]}{'...' if len(evidence) > 600 else ''}
            </div>
            """, unsafe_allow_html=True)

    elif run:
        st.warning("Please enter a question first.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — RESEARCH RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Research Results":

    dfs = load_results()
    domains = ["Science", "History", "Geography", "Technology"]
    model_names = list(dfs.keys())
    colors = ["#4fc3f7", "#7c4dff", "#f06292"]

    if not dfs:
        st.warning("No results found. Run the evaluator first.")
        st.stop()

    # ── Key metrics ──
    st.markdown('<div class="section-title">Overall Performance</div>', unsafe_allow_html=True)
    cols = st.columns(len(dfs))
    for i, (model, df) in enumerate(dfs.items()):
        h_rate = df["predicted_hallucination"].mean() * 100
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value" style="color:{colors[i]}">{h_rate:.1f}%</div>
                <div class="label">{model}</div>
                <div style="font-size:0.75rem; color:#4a5568; margin-top:0.3rem;">hallucination rate</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ──
    chart1, chart2 = st.columns(2, gap="large")

    with chart1:
        st.markdown('<div class="section-title">Hallucination Rate by Domain</div>', unsafe_allow_html=True)
        fig = go.Figure()
        for i, (model, df) in enumerate(dfs.items()):
            rates = [df[df["domain"]==d]["predicted_hallucination"].mean()*100 for d in domains]
            fig.add_trace(go.Bar(
                name=model, x=domains, y=rates,
                marker_color=colors[i],
                marker_line_width=0,
                opacity=0.85
            ))
        fig.update_layout(
            barmode="group",
            plot_bgcolor="#0a0e1a",
            paper_bgcolor="#0a0e1a",
            font=dict(color="#7986a3", family="DM Sans"),
            legend=dict(bgcolor="#0f1628", bordercolor="#1e2d4a", borderwidth=1),
            xaxis=dict(gridcolor="#1e2d4a"),
            yaxis=dict(gridcolor="#1e2d4a", title="Hallucination Rate (%)"),
            margin=dict(l=10, r=10, t=10, b=10),
            height=320
        )
        st.plotly_chart(fig, use_container_width=True)

    with chart2:
        st.markdown('<div class="section-title">Average Support Score</div>', unsafe_allow_html=True)
        fig2 = go.Figure()
        for i, (model, df) in enumerate(dfs.items()):
            scores = [df[df["domain"]==d]["support_score"].mean() for d in domains]
            fig2.add_trace(go.Scatter(
                name=model, x=domains, y=scores,
                mode="lines+markers",
                line=dict(color=colors[i], width=2.5),
                marker=dict(size=8, color=colors[i])
            ))
        fig2.update_layout(
            plot_bgcolor="#0a0e1a",
            paper_bgcolor="#0a0e1a",
            font=dict(color="#7986a3", family="DM Sans"),
            legend=dict(bgcolor="#0f1628", bordercolor="#1e2d4a", borderwidth=1),
            xaxis=dict(gridcolor="#1e2d4a"),
            yaxis=dict(gridcolor="#1e2d4a", title="Support Score", range=[0, 1]),
            margin=dict(l=10, r=10, t=10, b=10),
            height=320
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Domain table ──
    st.markdown('<div class="section-title">Domain-wise Breakdown</div>', unsafe_allow_html=True)
    table_data = {"Domain": domains}
    for model, df in dfs.items():
        table_data[f"{model} (Hall%)"] = [
            f"{df[df['domain']==d]['predicted_hallucination'].mean()*100:.1f}%"
            for d in domains
        ]
        table_data[f"{model} (Support)"] = [
            f"{df[df['domain']==d]['support_score'].mean():.3f}"
            for d in domains
        ]
    st.dataframe(
        pd.DataFrame(table_data),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Key findings ──
    st.markdown('<div class="section-title">Key Research Findings</div>', unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3, gap="large")
    with f1:
        st.markdown("""
        <div class="finding-card">
            <div class="number">01</div>
            <div class="title">Bigger ≠ Better</div>
            <div class="desc">LLaMA 3.3 70B hallucinates more than the smaller 8B model, contradicting the common assumption that larger models are always more factual.</div>
        </div>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown("""
        <div class="finding-card">
            <div class="number">02</div>
            <div class="title">Geography is Hardest</div>
            <div class="desc">All models struggle most with Geography and Technology domains, showing highest hallucination rates in these knowledge areas.</div>
        </div>
        """, unsafe_allow_html=True)
    with f3:
        st.markdown("""
        <div class="finding-card">
            <div class="number">03</div>
            <div class="title">LLaMA 4 Leads</div>
            <div class="desc">Despite being 17B parameters, LLaMA 4 Scout achieves the highest average support score, suggesting architectural improvements matter more than size.</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown('<div class="section-title">About This Project</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="evidence-box" style="font-size:0.92rem; line-height:1.9; color:#c0cce0;">
            <strong style="color:#4fc3f7;">LLM Hallucination Detector</strong> is a research project that investigates 
            factual consistency across multiple large language models using NLI-based verification.<br><br>
            
            The system retrieves Wikipedia evidence for any LLM-generated claim and uses a 
            <strong>DeBERTa-v3-small NLI model</strong> to measure whether the evidence supports 
            or contradicts the claim — generating a hallucination verdict in real time.<br><br>
            
            <strong style="color:#7c4dff;">Research Contribution:</strong> We evaluate 3 LLMs of varying sizes 
            (8B, 17B, 70B) across 4 knowledge domains on 100 questions from the 
            <strong>TruthfulQA benchmark</strong> (Lin et al., 2022), revealing that model size 
            does not reliably reduce hallucination rates.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">References</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="evidence-box" style="font-size:0.85rem; line-height:2; color:#7986a3;">
            • Lin et al. (2022). <em>TruthfulQA: Measuring How Models Mimic Human Falsehoods.</em> ACL 2022.<br>
            • Manakul et al. (2023). <em>SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection.</em> EMNLP 2023.<br>
            • Min et al. (2023). <em>FActScoring: Fine-grained Atomic Evaluation of Factual Precision.</em> ACL 2023.<br>
            • He et al. (2021). <em>DeBERTa: Decoding-enhanced BERT with Disentangled Attention.</em> ICLR 2021.
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-title">Tech Stack</div>', unsafe_allow_html=True)
        stack = [
            ("🤖", "LLMs", "LLaMA 3.1 8B, LLaMA 4 Scout, LLaMA 3.3 70B via Groq API"),
            ("🔎", "Retrieval", "Wikipedia API for evidence fetching"),
            ("🧠", "NLI Model", "cross-encoder/nli-BART-large-mnli"),
            ("📊", "Evaluation", "TruthfulQA benchmark, 100 questions"),
            ("🖥️", "Frontend", "Streamlit + Plotly"),
            ("☁️", "Deployment", "HuggingFace Spaces"),
        ]
        for icon, title, desc in stack:
            st.markdown(f"""
            <div style="background:#0f1628; border:1px solid #1e2d4a; border-radius:10px;
                        padding:0.9rem 1.2rem; margin-bottom:0.6rem; display:flex; gap:12px; align-items:flex-start;">
                <span style="font-size:1.3rem;">{icon}</span>
                <div>
                    <div style="font-weight:600; color:#e8eaf0; font-size:0.88rem;">{title}</div>
                    <div style="color:#7986a3; font-size:0.8rem; margin-top:0.2rem;">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Dataset</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="metric-card" style="text-align:left; padding:1.2rem 1.5rem;">
            <div style="display:flex; justify-content:space-between; margin-bottom:0.8rem;">
                <span style="color:#7986a3; font-size:0.8rem;">Total Questions</span>
                <span style="font-family:'Space Mono',monospace; color:#4fc3f7;">100</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:0.8rem;">
                <span style="color:#7986a3; font-size:0.8rem;">Domains</span>
                <span style="font-family:'Space Mono',monospace; color:#4fc3f7;">4</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:0.8rem;">
                <span style="color:#7986a3; font-size:0.8rem;">Questions/Domain</span>
                <span style="font-family:'Space Mono',monospace; color:#4fc3f7;">25</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#7986a3; font-size:0.8rem;">Source</span>
                <span style="font-family:'Space Mono',monospace; color:#4fc3f7;">TruthfulQA</span>
            </div>
        </div>
        """, unsafe_allow_html=True)