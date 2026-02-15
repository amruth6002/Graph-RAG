import streamlit as st
import os
import time
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import sys
sys.path.append('src')
from util import (
    encode_pdf, build_knowledge_graph, rerank_documents,
    find_node_by_content, expand_context_via_graph, visualize_graph
)

# Page config
st.set_page_config(
    page_title="GraphRAG - Graph-Enhanced RAG",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# Try to get API key from Streamlit secrets first, then env
try:
    api_key = st.secrets["GROQ_API_KEY"]
    os.environ["GROQ_API_KEY"] = api_key
except:
    api_key = os.getenv("GROQ_API_KEY")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .stButton>button {
        background-color: #667eea;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #764ba2;
        border: none;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'graph_rag' not in st.session_state:
    st.session_state.graph_rag = None
if 'time_records' not in st.session_state:
    st.session_state.time_records = {}
if 'current_pdf_path' not in st.session_state:
    st.session_state.current_pdf_path = None
if 'query_history' not in st.session_state:
    st.session_state.query_history = []

class GraphRAG:
    def __init__(self, path, chunk_size=1000, chunk_overlap=200, n_retrieved=10):
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY")
        )

        start_time = time.time()
        self.vector_store, self.splits, self.embedding_model = encode_pdf(
            path, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.time_records = {'FAISS Indexing': time.time() - start_time}

        start_time = time.time()
        self.knowledge_graph = build_knowledge_graph(
            self.splits, self.llm, self.embedding_model
        )
        self.time_records['Graph Building'] = time.time() - start_time

        self.n_retrieved = n_retrieved
        self.chunks_query_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": n_retrieved}
        )

    def run(self, query):
        results = {}
        
        # Query Rewriting
        start_time = time.time()
        query_rewrite_template = """You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system. 
        Given the original query, rewrite it to be more specific, detailed, and likely to retrieve relevant information. Don't give anything else except the rewritten query.
        Original query: {query}
        Rewritten query:"""
        
        query_rewrite_prompt = PromptTemplate(
            input_variables=["query"],
            template=query_rewrite_template
        )
        query_rewriter = query_rewrite_prompt | self.llm
        changed_query = query_rewriter.invoke(query).content
        results['rewritten_query'] = changed_query
        self.time_records['Query Rewrite'] = time.time() - start_time

        # Vector Retrieval
        start_time = time.time()
        retrieved_docs = self.chunks_query_retriever.invoke(changed_query)
        results['retrieved_count'] = len(retrieved_docs)
        self.time_records['Vector Retrieval'] = time.time() - start_time

        # Reranking
        n_rerank = min(3, len(retrieved_docs))
        start_time = time.time()
        ranked_results = rerank_documents(changed_query, retrieved_docs, n_retrieved=n_rerank)
        results['reranked_count'] = len(ranked_results)
        self.time_records['Reranking'] = time.time() - start_time

        # Graph Expansion
        start_time = time.time()
        seed_nodes = []
        for doc, score in ranked_results:
            node_idx = find_node_by_content(self.knowledge_graph.graph, doc.page_content)
            if node_idx is not None:
                seed_nodes.append((node_idx, score))

        if seed_nodes:
            context_texts, traversal_path = expand_context_via_graph(
                self.knowledge_graph, seed_nodes, max_nodes=10
            )
            results['traversal_path'] = traversal_path
            results['nodes_visited'] = len(traversal_path)
            
            if traversal_path:
                visualize_graph(self.knowledge_graph, traversal_path, save_path="graph_traversal.png")
        else:
            context_texts = [doc.page_content for doc, _ in ranked_results]
            traversal_path = []
            results['traversal_path'] = []
            results['nodes_visited'] = 0

        self.time_records['Graph Expansion'] = time.time() - start_time

        # Answer Generation
        start_time = time.time()
        context_text = "\n\n".join(context_texts)
        prompt = f"Based on the following context, answer the question.\n\nContext:\n{context_text}\n\nQuestion: {changed_query}\n\nAnswer:"
        response = self.llm.invoke(prompt).content
        results['answer'] = response
        self.time_records['Answer Generation'] = time.time() - start_time

        return results

# Header
st.markdown('<h1 class="main-header">GraphRAG</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Graph-Enhanced Retrieval-Augmented Generation</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    # API Key Check
    if not api_key:
        st.error("GROQ_API_KEY not found! Please add it to Streamlit secrets.")
        api_key = st.text_input("Enter your Groq API Key:", type="password")
        if api_key:
            os.environ["GROQ_API_KEY"] = api_key
    else:
        st.success("API Key loaded successfully")

    st.divider()
    
    # File Upload
    st.subheader("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])
    
    # Advanced Settings
    with st.expander("🔧 Advanced Settings"):
        chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 100)
        chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200, 50)
        n_retrieved = st.slider("Top-K Retrieval", 5, 20, 10, 1)
        
        if st.button("🔄 Rebuild Indexes"):
            import shutil
            if os.path.exists("indexes"):
                shutil.rmtree("indexes")
                st.success("Indexes cleared! Upload a PDF to rebuild.")
                st.session_state.graph_rag = None

    st.divider()
    
    # System Info
    if st.session_state.graph_rag:
        st.subheader("System Status")
        graph = st.session_state.graph_rag.knowledge_graph.graph
        st.metric("Graph Nodes", len(graph.nodes))
        st.metric("Graph Edges", len(graph.edges))
        st.metric("Document Chunks", len(st.session_state.graph_rag.splits))

# Main Content
col1, col2 = st.columns([2, 1])

with col1:
    # PDF Processing
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        # Check if we need to process a new PDF
        if st.session_state.current_pdf_path != tmp_path or st.session_state.graph_rag is None:
            with st.spinner("🔄 Processing PDF... This may take a few minutes..."):
                try:
                    st.session_state.graph_rag = GraphRAG(
                        path=tmp_path,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        n_retrieved=n_retrieved
                    )
                    st.session_state.current_pdf_path = tmp_path
                    st.session_state.time_records = st.session_state.graph_rag.time_records
                    st.success("PDF processed successfully!")
                    
                    # Show ingestion metrics
                    st.markdown("### Ingestion Metrics")
                    cols = st.columns(2)
                    cols[0].metric("FAISS Indexing Time", f"{st.session_state.time_records.get('FAISS Indexing', 0):.2f}s")
                    cols[1].metric("Graph Building Time", f"{st.session_state.time_records.get('Graph Building', 0):.2f}s")
                    
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
                    st.session_state.graph_rag = None

    # Query Interface
    if st.session_state.graph_rag:
        st.markdown("---")
        st.markdown("### Ask a Question")
        
        # Example queries
        example_queries = [
            "What is the main cause of climate change?",
            "What are the effects of deforestation?",
            "Explain the greenhouse effect",
            "What are renewable energy sources?"
        ]
        
        selected_example = st.selectbox("Or choose an example:", ["Custom query..."] + example_queries)
        
        if selected_example != "Custom query...":
            query = selected_example
        else:
            query = st.text_input("Enter your question:", placeholder="e.g., What is climate change?")
        
        if st.button("Search", type="primary", use_container_width=True):
            if query:
                with st.spinner("Processing query..."):
                    try:
                        results = st.session_state.graph_rag.run(query)
                        
                        # Add to history
                        st.session_state.query_history.append({
                            'query': query,
                            'results': results,
                            'time': time.strftime("%H:%M:%S")
                        })
                        
                        # Display Results
                        st.markdown("---")
                        st.markdown("### Answer")
                        st.markdown(f"<div style='background-color: #f0f9ff; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #667eea;'>{results['answer']}</div>", unsafe_allow_html=True)
                        
                        # Pipeline Steps
                        st.markdown("---")
                        st.markdown("### 🔄 Pipeline Breakdown")
                        
                        pipeline_cols = st.columns(5)
                        pipeline_cols[0].metric("Query Rewrite", f"{st.session_state.graph_rag.time_records.get('Query Rewrite', 0):.2f}s")
                        pipeline_cols[1].metric("Vector Search", f"{st.session_state.graph_rag.time_records.get('Vector Retrieval', 0):.2f}s")
                        pipeline_cols[2].metric("Reranking", f"{st.session_state.graph_rag.time_records.get('Reranking', 0):.2f}s")
                        pipeline_cols[3].metric("Graph Expansion", f"{st.session_state.graph_rag.time_records.get('Graph Expansion', 0):.2f}s")
                        pipeline_cols[4].metric("Generation", f"{st.session_state.graph_rag.time_records.get('Answer Generation', 0):.2f}s")
                        
                        # Retrieval Info
                        st.markdown("---")
                        st.markdown("### Retrieval Details")
                        detail_cols = st.columns(3)
                        detail_cols[0].markdown(f"<div class='metric-card'><b>Retrieved Chunks:</b> {results['retrieved_count']}</div>", unsafe_allow_html=True)
                        detail_cols[1].markdown(f"<div class='metric-card'><b>Reranked Chunks:</b> {results['reranked_count']}</div>", unsafe_allow_html=True)
                        detail_cols[2].markdown(f"<div class='metric-card'><b>Graph Nodes Visited:</b> {results['nodes_visited']}</div>", unsafe_allow_html=True)
                        
                        with st.expander("View Rewritten Query"):
                            st.info(results['rewritten_query'])
                        
                        # Graph Visualization
                        if os.path.exists("graph_traversal.png"):
                            st.markdown("---")
                            st.markdown("### Knowledge Graph Traversal")
                            st.image("graph_traversal.png", use_container_width=True)
                            st.caption("Visualization shows the traversal path through the knowledge graph (green = start, red = end)")
                        
                    except Exception as e:
                        st.error(f"Error processing query: {str(e)}")
            else:
                st.warning("Please enter a question")
    else:
        st.info("Please upload a PDF file from the sidebar to get started")

with col2:
    st.markdown("### Query History")
    
    if st.session_state.query_history:
        for i, item in enumerate(reversed(st.session_state.query_history[-5:])):  # Show last 5
            with st.expander(f"{item['time']} - Query {len(st.session_state.query_history)-i}"):
                st.markdown(f"**Q:** {item['query']}")
                st.markdown(f"**A:** {item['results']['answer'][:200]}...")
    else:
        st.info("No queries yet. Start by asking a question!")
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    **GraphRAG** combines:
    - **Vector Search** (FAISS)
    - **Knowledge Graphs** (NetworkX)
    - **Cross-Encoder Reranking**
    - **LLM Generation** (Groq)
    
    Built with Python, LangChain, and Streamlit.
    
    [View on GitHub](https://github.com/amruth6002/GraphRAG)
    """)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #888;'>Built with Streamlit | Powered by Groq & FAISS</p>", unsafe_allow_html=True)
