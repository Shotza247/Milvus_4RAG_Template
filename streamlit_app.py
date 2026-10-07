import os
import json
import requests
import pandas as pd
import streamlit as st

# ==============================================================================
# Page Configuration & Styling
# ==============================================================================
st.set_page_config(
    page_title="Milvus RAG Explorer & Evaluator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }
    .score-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .vector-score {
        background-color: #E0F2FE;
        color: #0369A1;
    }
    .rerank-score {
        background-color: #DCFCE7;
        color: #15803D;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# Sidebar - Configuration & Connection
# ==============================================================================
st.sidebar.image("https://milvus.io/static/images/milvus-logo.png", width=180)
st.sidebar.title("⚙️ RAG Settings")

API_BASE_URL = st.sidebar.text_input(
    "FastAPI Backend URL",
    value=os.getenv("API_BASE_URL", "http://localhost:8000"),
    help="Endpoint for the FastAPI RAG service",
)

# Health Check in Sidebar
def check_health():
    try:
        res = requests.get(f"{API_BASE_URL}/api/v1/health", timeout=3)
        if res.status_code == 200:
            return res.json()
        return None
    except Exception:
        return None

health_data = check_health()
if health_data:
    st.sidebar.success(f"🟢 Connected to {health_data.get('app_name', 'API')}")
    milvus_info = health_data.get("milvus", {})
    if milvus_info.get("connected"):
        st.sidebar.info(f"📊 Milvus: v{milvus_info.get('server_version', 'Connected')}")
    else:
        st.sidebar.warning("⚠️ Milvus instance unreachable (mock/detached mode)")
else:
    st.sidebar.error("🔴 Backend API Offline (Start via docker/podman or uvicorn)")

# ==============================================================================
# Helper API Functions
# ==============================================================================
def get_collections():
    try:
        res = requests.get(f"{API_BASE_URL}/api/v1/collections", timeout=5)
        if res.status_code == 200:
            return res.json().get("collections", [])
        return ["rag_documents"]
    except Exception:
        return ["rag_documents"]

def create_collection(name: str, dim: int, metric: str):
    payload = {"collection_name": name, "dimension": dim, "metric_type": metric}
    res = requests.post(f"{API_BASE_URL}/api/v1/collections", json=payload, timeout=10)
    return res.status_code == 200, res.json()

def upload_document_api(file_obj, collection_name: str, chunk_size: int, chunk_overlap: int):
    files = {"file": (file_obj.name, file_obj.getvalue(), file_obj.type)}
    data = {
        "collection_name": collection_name,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }
    res = requests.post(f"{API_BASE_URL}/api/v1/documents/upload", files=files, data=data, timeout=60)
    return res.status_code == 200, res.json()

def ingest_text_api(text: str, doc_id: str, collection_name: str, chunk_size: int, chunk_overlap: int):
    payload = {
        "collection_name": collection_name,
        "doc_id": doc_id,
        "text": text,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }
    res = requests.post(f"{API_BASE_URL}/api/v1/documents/ingest-text", json=payload, timeout=30)
    return res.status_code == 200, res.json()

def search_two_stage(query: str, collection_name: str, top_k: int, top_n: int, threshold: float):
    payload = {
        "collection_name": collection_name,
        "query": query,
        "top_k_candidates": top_k,
        "top_n_final": top_n,
        "rerank_score_threshold": threshold,
    }
    res = requests.post(f"{API_BASE_URL}/api/v1/search/two-stage", json=payload, timeout=30)
    return res.status_code == 200, res.json()

def search_vector_only(query: str, collection_name: str, top_k: int):
    payload = {
        "collection_name": collection_name,
        "query": query,
        "top_k": top_k,
        "rerank": False,
    }
    res = requests.post(f"{API_BASE_URL}/api/v1/search", json=payload, timeout=30)
    return res.status_code == 200, res.json()

def run_benchmark_api(top_k: int, top_n: int):
    payload = {"use_sample_dataset_file": True, "top_k": top_k, "top_n": top_n}
    res = requests.post(f"{API_BASE_URL}/api/v1/eval/benchmark", json=payload, timeout=60)
    return res.status_code == 200, res.json()

def generate_synthetic_api(text: str, doc_id: str, num_samples: int, use_llm: bool):
    payload = {
        "text": text,
        "doc_id": doc_id,
        "num_samples": num_samples,
        "use_llm": use_llm,
        "save_as_benchmark": True,
    }
    res = requests.post(f"{API_BASE_URL}/api/v1/eval/generate-synthetic", json=payload, timeout=60)
    return res.status_code == 200, res.json()

# ==============================================================================
# Sidebar - Ingestion Section
# ==============================================================================
available_collections = get_collections()
if not available_collections:
    available_collections = ["rag_documents"]

st.sidebar.divider()
st.sidebar.markdown("### 📄 Ingest Documents")

target_collection = st.sidebar.selectbox("Target Collection", options=available_collections, key="sidebar_ingest_col")
c_size = st.sidebar.slider("Chunk Size", min_value=100, max_value=2000, value=500, step=50)
c_overlap = st.sidebar.slider("Chunk Overlap", min_value=0, max_value=300, value=50, step=10)

uploaded_file = st.sidebar.file_uploader(
    "Upload file (PDF, TXT, MD, DOCX, JSON)",
    type=["pdf", "txt", "md", "docx", "json"],
    help="Supported formats: PDF, TXT, Markdown, Word, JSON",
)

if uploaded_file is not None:
    st.sidebar.caption(f"📁 {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")
    if st.sidebar.button("🚀 Ingest File to Milvus", type="primary", use_container_width=True):
        with st.spinner("Chunking, embedding & indexing into Milvus..."):
            success, resp = upload_document_api(uploaded_file, target_collection, c_size, c_overlap)
            if success:
                st.sidebar.success(f"✅ Ingested {resp.get('chunks_created', '')} chunks!")
            else:
                st.sidebar.error(f"❌ {resp.get('detail', 'Ingestion failed')}")

with st.sidebar.expander("✍️ Ingest Raw Text"):
    raw_doc_id = st.text_input("Doc ID", value="custom_notes_01")
    raw_text_input = st.text_area("Raw Text", height=100, placeholder="Paste text...")
    if st.button("Ingest Text", use_container_width=True):
        if not raw_text_input.strip():
            st.warning("Please provide text.")
        else:
            with st.spinner("Ingesting text..."):
                success, resp = ingest_text_api(raw_text_input, raw_doc_id, target_collection, c_size, c_overlap)
                if success:
                    st.success("✅ Text chunked and stored!")
                else:
                    st.error(f"❌ Failed: {resp.get('detail')}")

# ==============================================================================
# Main App Header & 3 Main Tabs
# ==============================================================================
st.markdown('<div class="main-title">⚡ Milvus RAG & Reranking Studio</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Test your local documents with 2-Stage Retrieval (Milvus Vector Search + Cross-Encoder Reranking) & RAG Triad Evaluation.</div>',
    unsafe_allow_html=True,
)

tabs = st.tabs([
    "🔍 Query & 2-Stage Search",
    "📊 RAG Triad & Benchmark Eval",
    "🗂️ Collection Manager",
])

# ------------------------------------------------------------------------------
# TAB 1: Query & 2-Stage Retrieval Testing
# ------------------------------------------------------------------------------
with tabs[0]:
    st.header("🔍 Query & Retrieval Evaluation")
    st.write("Compare raw Milvus Vector Similarity search vs 2-Stage Cross-Encoder Reranking in real time.")

    search_col = st.selectbox("Search in Collection", options=available_collections, key="search_col")
    
    query_text = st.text_input(
        "Enter your query / question:",
        value="What are the key architecture points and benefits mentioned in the document?",
        placeholder="Type a natural language query...",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        top_k_candidates = st.slider("Stage 1: Coarse Candidate Pool (Top-K)", min_value=2, max_value=50, value=15)
    with c2:
        top_n_final = st.slider("Stage 2: Reranked Final Results (Top-N)", min_value=1, max_value=15, value=5)
    with c3:
        score_thresh = st.slider("Reranker Score Cut-off Threshold", min_value=0.0, max_value=1.0, value=0.35, step=0.05)

    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        run_twostage = st.button("⚡ Run 2-Stage Search (Milvus + Reranker)", type="primary", use_container_width=True)
    with col_btn2:
        run_veconly = st.button("🔎 Run Vector-Only Search (Milvus Baseline)", use_container_width=True)

    if run_twostage or run_veconly:
        if not query_text.strip():
            st.warning("Please enter a query.")
        else:
            with st.spinner("Searching and ranking results..."):
                if run_twostage:
                    success, res = search_two_stage(query_text, search_col, top_k_candidates, top_n_final, score_thresh)
                else:
                    success, res = search_vector_only(query_text, search_col, top_n_final)

                if not success:
                    st.error(f"Search failed: {res.get('detail', 'Unknown error')}")
                else:
                    results = res.get("results", [])
                    st.subheader(f"Retrieved {len(results)} Chunks ({res.get('retrieval_stage')})")

                    if not results:
                        st.info("No matching chunks found above threshold.")
                    else:
                        for idx, chunk in enumerate(results, start=1):
                            with st.container():
                                st.markdown(f"#### #{idx} • Chunk ID: `{chunk.get('chunk_id')}` (Doc: `{chunk.get('doc_id')}`)")
                                
                                m_col1, m_col2, m_col3 = st.columns([1, 1, 3])
                                sim_score = chunk.get("similarity_score")
                                rerank_score = chunk.get("rerank_score")

                                with m_col1:
                                    if sim_score is not None:
                                        st.markdown(f'<span class="score-badge vector-score">Vector Sim: {sim_score:.4f}</span>', unsafe_allow_html=True)
                                with m_col2:
                                    if rerank_score is not None:
                                        st.markdown(f'<span class="score-badge rerank-score">Rerank Score: {rerank_score:.4f}</span>', unsafe_allow_html=True)

                                st.text_area(
                                    f"Content (#{idx})",
                                    value=chunk.get("text", ""),
                                    height=110,
                                    key=f"chunk_txt_{idx}",
                                    disabled=True,
                                )
                                with st.expander("Metadata"):
                                    st.json(chunk.get("metadata", {}))
                                st.markdown("---")

# ------------------------------------------------------------------------------
# TAB 2: RAG Triad & Benchmark Evaluation
# ------------------------------------------------------------------------------
with tabs[1]:
    st.header("📊 End-to-End RAG Triad & Benchmark Evaluation (Option B)")
    st.write(
        "Generate **custom synthetic test benchmarks** directly from your uploaded documents (Ragas/TruLens style) and evaluate real retrieval accuracy."
    )

    with st.expander("✨ 1. Generate Synthetic Benchmarks From Your Document", expanded=True):
        st.markdown("**Automatically create realistic test questions & ground-truth facts from your document chunks:**")
        syn_col1, syn_col2 = st.columns([2, 1])
        with syn_col1:
            syn_doc_id = st.text_input("Document Name / ID for Benchmark", value="my_uploaded_doc")
            syn_text = st.text_area(
                "Document Text to Benchmark (or paste key sections)",
                height=140,
                placeholder="Paste key sections of your document here to automatically derive test questions...",
            )
        with syn_col2:
            num_syn = st.slider("Number of Questions to Generate", min_value=2, max_value=15, value=5)
            use_llm_flag = st.checkbox("Use OpenAI LLM (requires OPENAI_API_KEY in .env)", value=False)
            st.caption("If disabled, uses smart heuristic fact-extraction (100% offline, zero-cost).")

        if st.button("🪄 Generate Document Benchmark Suite", type="secondary"):
            if not syn_text.strip():
                st.warning("Please paste or provide document text.")
            else:
                with st.spinner("Analyzing text and generating golden query-fact pairs..."):
                    success, syn_resp = generate_synthetic_api(syn_text, syn_doc_id, num_syn, use_llm_flag)
                    if success:
                        st.success(f"✅ {syn_resp.get('message')}")
                        st.session_state["generated_benchmarks"] = syn_resp.get("samples", [])
                    else:
                        st.error(f"❌ Failed: {syn_resp.get('detail')}")

        if "generated_benchmarks" in st.session_state:
            st.markdown("##### 📋 Generated Golden Test Samples:")
            for s in st.session_state["generated_benchmarks"]:
                st.markdown(f"- **Q:** *{s.get('query')}*")
                st.markdown(f"  - **Expected Answer:** `{s.get('ground_truth_answer')}`")
                st.markdown(f"  - **Key Facts:** `{', '.join(s.get('relevant_keywords', []))}`")

    st.divider()

    st.subheader("2. Run Evaluation against Milvus + Reranker")
    ev_col1, ev_col2 = st.columns(2)
    with ev_col1:
        b_top_k = st.number_input("Stage 1: Milvus Candidate Pool (Top-K)", min_value=2, max_value=50, value=10)
    with ev_col2:
        b_top_n = st.number_input("Stage 2: Reranked Context Size (Top-N)", min_value=1, max_value=20, value=5)

    if st.button("🧪 Execute Benchmark Evaluation Run", type="primary"):
        with st.spinner("Running evaluation over benchmark queries..."):
            success, eval_res = run_benchmark_api(b_top_k, b_top_n)
            if not success:
                st.error(f"Evaluation failed: {eval_res.get('detail')}")
            else:
                st.success("✅ Evaluation Complete!")
                st.info(f"💡 {eval_res.get('summary')}")

                # Performance Gains
                g_col1, g_col2 = st.columns(2)
                g_col1.metric("MRR Accuracy Gain", f"+{eval_res.get('accuracy_gain_mrr_percent')}%")
                g_col2.metric("NDCG Accuracy Gain", f"+{eval_res.get('accuracy_gain_ndcg_percent')}%")

                vec_eval = eval_res.get("vector_only_evaluation", {})
                rr_eval = eval_res.get("reranked_evaluation", {})

                # Comparison Table
                st.subheader("Stage 1 (Vector Only) vs Stage 2 (Vector + Rerank) Metrics")
                
                metrics_df = pd.DataFrame({
                    "Metric": [
                        "Hit Rate @ K",
                        "MRR @ K (Mean Reciprocal Rank)",
                        "NDCG @ K (Ranking Quality)",
                        "Precision @ K",
                        "Recall @ K",
                        "RAG Triad: Context Relevance",
                        "RAG Triad: Context Recall",
                        "RAG Triad: Faithfulness / Groundedness",
                    ],
                    "Stage 1: Vector Only": [
                        vec_eval.get("ir_metrics", {}).get("hit_rate_at_k", 0),
                        vec_eval.get("ir_metrics", {}).get("mrr_at_k", 0),
                        vec_eval.get("ir_metrics", {}).get("ndcg_at_k", 0),
                        vec_eval.get("ir_metrics", {}).get("precision_at_k", 0),
                        vec_eval.get("ir_metrics", {}).get("recall_at_k", 0),
                        vec_eval.get("rag_triad_metrics", {}).get("context_relevance", 0),
                        vec_eval.get("rag_triad_metrics", {}).get("context_recall", 0),
                        vec_eval.get("rag_triad_metrics", {}).get("faithfulness", 0),
                    ],
                    "Stage 2: Milvus + Cross-Encoder": [
                        rr_eval.get("ir_metrics", {}).get("hit_rate_at_k", 0),
                        rr_eval.get("ir_metrics", {}).get("mrr_at_k", 0),
                        rr_eval.get("ir_metrics", {}).get("ndcg_at_k", 0),
                        rr_eval.get("ir_metrics", {}).get("precision_at_k", 0),
                        rr_eval.get("ir_metrics", {}).get("recall_at_k", 0),
                        rr_eval.get("rag_triad_metrics", {}).get("context_relevance", 0),
                        rr_eval.get("rag_triad_metrics", {}).get("context_recall", 0),
                        rr_eval.get("rag_triad_metrics", {}).get("faithfulness", 0),
                    ],
                })

                st.dataframe(metrics_df, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 3: Collection Manager
# ------------------------------------------------------------------------------
with tabs[2]:
    st.header("🗂️ Milvus Collection Manager")

    st.subheader("Existing Collections")
    for col_name in available_collections:
        with st.expander(f"📁 {col_name}"):
            try:
                info_res = requests.get(f"{API_BASE_URL}/api/v1/collections/{col_name}")
                if info_res.status_code == 200:
                    st.json(info_res.json())
                else:
                    st.write("Could not retrieve collection stats.")
            except Exception as e:
                st.write(f"Error: {e}")

    st.divider()
    st.subheader("Create New Collection")
    new_col_name = st.text_input("New Collection Name", value="my_new_documents")
    new_col_dim = st.number_input("Embedding Dimension", value=384, step=1)
    new_col_metric = st.selectbox("Metric Type", options=["COSINE", "L2", "IP"])

    if st.button("➕ Create Collection"):
        if not new_col_name.strip():
            st.warning("Please provide a collection name.")
        else:
            success, res = create_collection(new_col_name, new_col_dim, new_col_metric)
            if success:
                st.success(f"Collection '{new_col_name}' created successfully!")
                st.rerun()
            else:
                st.error(f"Failed to create collection: {res.get('detail')}")
