import streamlit as st
import requests
import os

# Backend API URL config
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip('/')

# Page Config
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom Times New Roman / Theme-Aware Editorial Academic Styling
st.markdown("""
<style>
/* Main Serif Fonts (Times New Roman) - Respecting Dark/Light Theme Colors */
html, body, [class*="css"] {
    font-family: 'Times New Roman', Times, Georgia, serif !important;
}

/* Titles and Headers */
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    border-bottom: 2px solid currentColor;
    padding-bottom: 8px;
    margin-bottom: 0.5rem;
}
.sub-header {
    opacity: 0.7;
    font-style: italic;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}

/* Sidebar Border */
[data-testid="stSidebar"] {
    border-right: 1px solid rgba(128, 128, 128, 0.2) !important;
}

/* Minimalist Card Styling (Academic Format) */
.paper-card {
    border: 1px solid rgba(128, 128, 128, 0.5);
    padding: 16px;
    margin-bottom: 16px;
}
.badge {
    padding: 2px 8px;
    border: 1px solid rgba(128, 128, 128, 0.8);
    font-size: 0.8rem;
    display: inline-block;
    margin-right: 8px;
}
.badge-secondary {
    border: 1px solid currentColor;
    padding: 2px 8px;
    font-size: 0.8rem;
    display: inline-block;
}

/* Excerpt Sources Styling (Quotation Style) */
.source-card {
    border-left: 2px solid currentColor;
    padding: 12px 16px;
    margin: 8px 0;
    font-style: italic;
    opacity: 0.9;
}

/* Override input widgets styling to match minimal theme */
input, select, textarea {
    border-radius: 0px !important;
}

/* Override default button corners */
div.stButton > button {
    border-radius: 0px !important;
    transition: all 0.2s ease;
}
</style>
""", unsafe_allow_html=True)

# Helper function to check API status
def check_backend():
    try:
        # Strip trailing slashes to prevent double-slash endpoints (e.g. domain.com//)
        base_url = API_URL.rstrip('/')
        # 10s timeout: long enough that a waking server might respond mid-boot
        response = requests.get(f"{base_url}/", timeout=10)
        return response.ok
    except Exception:
        return False

# Render Warning / Auto-Wake-Up Loop if Backend is offline
if not check_backend():
    import time
    
    # Track how long we've been waiting across reruns
    if "wake_start" not in st.session_state:
        st.session_state["wake_start"] = time.time()
    
    elapsed = int(time.time() - st.session_state["wake_start"])
    max_wait = 120  # Render free tier can take up to ~120s
    progress = min(elapsed / max_wait, 0.95)
    
    st.markdown("### Starting Backend Server...")
    st.progress(progress, text=f"Connecting... ({elapsed}s elapsed, usually takes ~60s)")
    st.info(
        "The backend server goes to sleep after inactivity on Render's free tier. "
        "It is now booting up automatically. This typically takes **60–90 seconds**."
    )
    
    with st.spinner("Pinging backend..."):
        time.sleep(5)
    st.rerun()

# Backend is alive — clear the wake timer if it was set
if "wake_start" in st.session_state:
    del st.session_state["wake_start"]
# Initialize browser session ID for user-level isolation
import uuid
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

session_id = st.session_state["session_id"]
session_headers = {
    "X-Session-ID": session_id
}

# Load currently ingested papers
def get_papers():
    try:
        res = requests.get(f"{API_URL}/papers/", headers=session_headers)
        if res.ok:
            return res.json()
    except Exception:
        pass
    return []

papers = get_papers()

# ─── SIDEBAR: UPLOAD & INVENTORY & MODEL SELECTOR ──────────────────────
with st.sidebar:
    st.markdown("### Model Configuration")
    provider_options = {
        "Google Gemini Flash": ("gemini", "gemini-2.5-flash-lite"),
        "OpenRouter Qwen 72B": ("openrouter", "qwen/qwen-2.5-72b-instruct")
    }
    selected_option = st.selectbox(
        "Active LLM Model:",
        options=list(provider_options.keys()),
        index=0
    )
    active_provider, active_model = provider_options[selected_option]
    
    # HTTP Headers containing active model routing details and session ID
    llm_headers = {
        "X-LLM-Provider": active_provider,
        "X-LLM-Model": active_model,
        "X-Session-ID": session_id
    }

    st.markdown("---")
    st.markdown("### Upload Paper")
    uploaded_file = st.file_uploader("Select research paper PDF", type=["pdf"])
    
    if uploaded_file:
        if st.button("Ingest Paper", use_container_width=True):
            with st.spinner("Processing PDF and creating index..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    response = requests.post(f"{API_URL}/papers/upload", files=files, headers=session_headers)
                    if response.ok:
                        st.success("Ingested successfully.")
                        st.rerun()
                    else:
                        st.error(f"Failed to ingest: {response.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

    st.markdown("---")
    st.markdown(f"### Library ({len(papers)} papers)")
    
    for paper in papers:
        col_name, col_del = st.columns([3.5, 2])
        with col_name:
            st.markdown(f"**{paper.get('title') or paper.get('filename')}**")
            st.caption(f"Pages: {paper.get('num_pages')} | Chunks: {paper.get('num_chunks')}")
        with col_del:
            if st.button("Delete", key=f"del_{paper.get('id')}"):
                try:
                    res = requests.delete(f"{API_URL}/papers/{paper.get('id')}", headers=session_headers)
                    if res.ok:
                        st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
        st.markdown("<hr style='margin: 8px 0; border: 0; border-top: 1px solid rgba(128, 128, 128, 0.2);'/>", unsafe_allow_html=True)


# ─── MAIN APP PAGE ──────────────────────────────────────────────────
st.markdown("<div class='main-header'>AI Research Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Semantic Search, RAG Q&A, structured summaries, and paper comparisons.</div>", unsafe_allow_html=True)

if not papers:
    st.info("Welcome. Your library is empty. Upload a PDF paper in the sidebar to begin.")
    st.stop()

# Navigation Tabs (No symbols, plain text, removed Citation Generator)
tab_qa, tab_sum, tab_search, tab_compare = st.tabs([
    "Q&A Assistant", 
    "Summarizer", 
    "Semantic Search", 
    "Compare Papers"
])

# ─── TAB 1: RAG Q&A ──────────────────────────────────────────────────
with tab_qa:
    st.markdown("### Ask Questions")
    selected_paper = st.selectbox(
        "Select paper:",
        papers,
        format_func=lambda p: p.get('title') or p.get('filename'),
        key="qa_paper_select"
    )
    
    question = st.text_input("Enter question:", placeholder="What are the key results or contributions?")
    top_k = st.slider("Context size (chunks):", min_value=1, max_value=10, value=5, key="qa_top_k")
    
    if st.button("Submit Question", type="primary") and question:
        with st.spinner("Generating answer..."):
            try:
                payload = {"question": question, "top_k": top_k}
                res = requests.post(
                    f"{API_URL}/papers/{selected_paper.get('id')}/ask", 
                    json=payload, 
                    headers=llm_headers
                )
                if res.ok:
                    data = res.json()
                    st.markdown("#### Answer:")
                    st.write(data.get("answer"))
                    
                    st.markdown("---")
                    st.markdown("#### References:")
                    for idx, src in enumerate(data.get("sources", [])):
                        with st.expander(f"Excerpt {src['excerpt_num']} | Page {src['page']} | Score: {src['score']:.2f}"):
                            st.markdown(f"<div class='source-card'>{src['text']}</div>", unsafe_allow_html=True)
                else:
                    st.error(res.json().get("detail", "Error processing request."))
            except Exception as e:
                st.error(f"Connection error: {e}")


# ─── TAB 2: SUMMARIZER ───────────────────────────────────────────────
with tab_sum:
    st.markdown("### Summarizer")
    selected_sum_paper = st.selectbox(
        "Select paper to summarize:",
        papers,
        format_func=lambda p: p.get('title') or p.get('filename'),
        key="sum_paper_select"
    )
    
    if st.button("Generate Summary", type="primary"):
        with st.spinner("Generating structured summary..."):
            try:
                res = requests.get(
                    f"{API_URL}/papers/{selected_sum_paper.get('id')}/summarize", 
                    headers=llm_headers
                )
                if res.ok:
                    summary_text = res.json().get("summary")
                    st.markdown("#### Summary:")
                    st.markdown(summary_text)
                else:
                    st.error(res.json().get("detail", "Error generating summary."))
            except Exception as e:
                st.error(f"Connection error: {e}")


# ─── TAB 3: SEMANTIC SEARCH ──────────────────────────────────────────
with tab_search:
    st.markdown("### Cross-Paper Semantic Search")
    query = st.text_input("Enter search query:", placeholder="Query terms...")
    search_k = st.slider("Number of results:", min_value=1, max_value=20, value=5, key="search_k")
    
    if st.button("Search", type="primary") and query:
        with st.spinner("Searching..."):
            try:
                res = requests.get(f"{API_URL}/search/", params={"q": query, "top_k": search_k}, headers=session_headers)
                if res.ok:
                    data = res.json()
                    results = data.get("results", [])
                    st.markdown(f"Results: {len(results)} matches")
                    
                    for r in results:
                        st.markdown(f"""
                        <div class='paper-card'>
                            <span class='badge'>{r['paper']['title']}</span>
                            <span class='badge-secondary'>Page {r['page']}</span>
                            <div style='margin-top: 10px; line-height: 1.5;'>{r['text']}</div>
                            <div style='margin-top: 8px; text-align: right;'>
                                <small>Score: {r['score']:.2f}</small>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.error("Error executing search.")
            except Exception as e:
                st.error(f"Connection error: {e}")


# ─── TAB 4: COMPARE PAPERS ───────────────────────────────────────────
with tab_compare:
    st.markdown("### Multi-Paper Comparative Analysis")
    selected_pids = st.multiselect(
        "Select papers to compare (min 2):",
        papers,
        format_func=lambda p: p.get('title') or p.get('filename')
    )
    aspect = st.text_input("Comparison aspect:", value="methodology, findings, and contributions")
    
    if st.button("Compare", type="primary"):
        if len(selected_pids) < 2:
            st.warning("Please select at least 2 papers.")
        else:
            with st.spinner("Generating comparison..."):
                try:
                    payload = {
                        "paper_ids": [p.get("id") for p in selected_pids],
                        "aspect": aspect
                    }
                    res = requests.post(
                        f"{API_URL}/compare/", 
                        json=payload, 
                        headers=llm_headers
                    )
                    if res.ok:
                        data = res.json()
                        st.markdown("#### Comparison Report:")
                        st.markdown(data.get("comparison"))
                    else:
                        st.error(res.json().get("detail", "Error generating comparison."))
                except Exception as e:
                    st.error(f"Connection error: {e}")
