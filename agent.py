# agent.py
"""
Full Agentic RAG backend using LangGraph, local Chroma, Serper web fallback, and Groq LLM.
Provides run_agentic_query(query: str) -> dict as the public entrypoint.
"""

import os
import time

from dotenv import load_dotenv

DEBUG = True

# Load .env for local development (safe no-op in cloud)
load_dotenv()


from typing import List, Dict, Any
from typing_extensions import TypedDict


# Optional Streamlit support
try:
    import streamlit as st
except ImportError:
    st = None

if st is not None:
    try:
        secrets = st.secrets
        print("in st is not none")
        os.environ.setdefault("GROQ_API_KEY", secrets.get("GROQ_API_KEY", ""))
        os.environ.setdefault("SERPER_API_KEY", secrets.get("SERPER_API_KEY", ""))
    except Exception:
        pass

# It will work for both cloud and streamlit. 
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set")

if not SERPER_API_KEY:
    raise RuntimeError("SERPER_API_KEY is not set")

print ("GROQ_API_KEY: ", GROQ_API_KEY)
print ("SERPER_API_KEY: ", SERPER_API_KEY)


# Third-party imports (ensure these are in requirements.txt)
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from langchain_community.utilities import GoogleSerperAPIWrapper
from langgraph.graph import StateGraph, START, END

# ----------------------------
# Config / Constants
# ----------------------------
CHROMA_PATH = "./chroma_db"  # persistent local folder
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"

# Collection names as you requested
COLLECTION_QNA = "medical_qna_collection"
COLLECTION_DEVICES = "medical_devices_collection"

# Retrieval and Context safe limit
MAX_DOCS = 3
MAX_CONTEXT_CHARS = 3000

# ----------------------------
# Module-level singletons
# ----------------------------
_embedding_model = None
_chroma_client = None
_medical_qna_collection = None
_medical_devices_collection = None
_search_tool = None
_groq_client = None

# ----------------------------
# Helpers: Debug Helper
# ----------------------------
def debug(msg: str):
    if DEBUG:
        print(f"[DEBUG] {msg}")

# ----------------------------
# Helpers: lazy singletons
# ----------------------------
def get_embedding_model():
    """Singleton loader for SentenceTransformer embedding model."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def get_chroma_client():
    global _chroma_client, _medical_qna_collection, _medical_devices_collection
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        # create named collections
        _medical_qna_collection = _chroma_client.get_collection(name=COLLECTION_QNA)
        _medical_devices_collection = _chroma_client.get_collection(name=COLLECTION_DEVICES )

    return _chroma_client


def get_search_tool():
    """Singleton Serper wrapper that reads SERPER_API_KEY from env (or streamlit secrets)."""
    global _search_tool
    if _search_tool is None:
        # GoogleSerperAPIWrapper reads os.environ["SERPER_API_KEY"]
        _search_tool = GoogleSerperAPIWrapper()
    return _search_tool

def get_groq_client():
    """Singleton Groq (OpenAI-compatible) client using GROQ_API_KEY env var."""
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not found in environment or Streamlit secrets.")
        _groq_client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    return _groq_client


# ----------------------------
# helper function to limit context
# ----------------------------

def build_safe_context(docs: list[str]) -> str:
    """
    Safely builds context with:
    - max docs
    - max total characters
    """
    context_parts = []
    total_chars = 0

    for i, doc in enumerate(docs[:MAX_DOCS]):
        chunk = f"[Document {i+1}]\n{doc.strip()}\n\n"
        if total_chars + len(chunk) > MAX_CONTEXT_CHARS:
            break
        context_parts.append(chunk)
        total_chars += len(chunk)

    return "".join(context_parts).strip()

# ----------------------------
# Computer Embedding confidence
# ----------------------------
def compute_embedding_confidence(distances: list[float]) -> dict:
    """
    Compute confidence score from Chroma distances.

    distances: list of cosine distances (0 = identical, 1 = opposite)
    """

    if not distances:
        return {
            "confidence": 0.0,
            "confidence_label": "no_evidence"
        }

    # Convert distances → similarities 
    # Cosine distance range: 0 → 1
    # Lower = more similar 
    similarities = [1 - d for d in distances]
    print ("similarities: ", similarities)
    print ("Len(similarities): ", len(similarities))
    avg_similarity = sum(similarities) / len(similarities)
    confidence = round(avg_similarity * 100, 2) # Convert to percentage

    # Confidence labeling
    if confidence >= 80:
        label = "high"
    elif confidence >= 65:
        label = "medium"
    else:
        label = "low"

    return {
        "confidence": confidence,
        "confidence_label": label,
        "avg_similarity": round(avg_similarity, 3)
    }


# ----------------------------
# LLM wrapper
# ----------------------------
def get_llm_response(prompt: str, model: str = GROQ_MODEL, temperature: float = 0.2, max_tokens: int = 512) -> str:
    """
    Call Groq (via OpenAI-compatible client) using Chat Completions and return the assistant text.
    """
    client = get_groq_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    try:
        return resp.choices[0].message.content
    except Exception:
        return str(resp)

# ----------------------------
# Ensure everything initialized (fail fast)
# ----------------------------
def _ensure_init():
    """
    Make sure embedding model, chroma client/collections, search tool and groq client are created.
    Called at the start of run_agentic_query to fail early when config/environment missing.
    """
    get_embedding_model()
    get_chroma_client()
    get_search_tool()
    get_groq_client()

# ----------------------------
# Chroma retrieval helper (manual embeddings)
# ----------------------------
def retrieve_from_chroma_collection(collection, query: str, top_k: int = MAX_DOCS):
    print("retrieve_from_chroma_collection")
    """
    Encode `query` with same embedding model used to index and query the given chroma collection.
    Returns (docs: List[str], metas: List[dict])
    """
    emb_model = get_embedding_model()
    q_emb = emb_model.encode([query])[0].tolist()
    results = collection.query(query_embeddings=[q_emb], n_results=top_k)
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0] # get embedding distance result
    confidence_data = compute_embedding_confidence(distances)   # It return confidence, confidence_label and avg similaries

    return docs, metas, confidence_data

# ----------------------------
# LangGraph GraphState type
# ----------------------------
class GraphState(TypedDict, total=False):
    query: str
    context: str
    prompt: str
    response: str
    source: str
    #is_relevant: str
    #iteration_count: int
    retrieval_success: bool
    confidence: float
    confidence_label: str

# ----------------------------
# Node implementations for StateGraph
# ----------------------------
def router(state: GraphState) -> GraphState:
    debug("➡️ Entered Router node")

    """
    LLM-based router: returns one of "Retrieve_QnA", "Retrieve_Device", "Web_Search" stored in state['source'].
    Includes simple fallback heuristics if LLM output unexpected.
    """
    _ensure_init()
    q = state.get("query", "").strip()
    if not q:
        raise ValueError("Empty query provided to router.")
    print("Query :", q)
    decision_prompt = f"""
You are a strict medical routing assistant. Choose EXACTLY ONE label: Retrieve_QnA, Retrieve_Device, or Web_Search.
Rules:
- If the question is about symptoms, diagnosis, treatment, disease knowledge -> Retrieve_QnA.
- If the question is about device manuals, calibration, operation or model numbers -> Retrieve_Device.
- If the question requests very recent regulation, trade data, prices, brand availability, or news -> Web_Search.

Query: "{q}"
Respond with the single token only.
"""
    decision_raw = get_llm_response(decision_prompt,  max_tokens=10).strip().split()[0] if q else "Retrieve_QnA"
    print(" decision :", decision_raw)
    decision = decision_raw if decision_raw in ("Retrieve_QnA", "Retrieve_Device", "Web_Search") else None
    
    # fallback heuristics
    if decision is None:
        debug("➡️ No router decision")
        low = q.lower()
        if any(k in low for k in ("manual", "device", "calib", "model", "pump", "insulin pump")):
            decision = "Retrieve_Device"
        elif any(k in low for k in ("news", "price", "export", "duty", "latest", "2025")):
            decision = "Web_Search"
        else:
            decision = "Retrieve_QnA"
    
    debug(f"Router decision: {decision}")
    state["source"] = decision
    # reset iterations for a new query
    #state["iteration_count"] = 0
    return state

def route_decision(state: GraphState) -> str:
    return state.get("source", "Retrieve_QnA")

def retrieve_context_q_n_a(state: GraphState) -> GraphState:
    debug("➡️ Entered Retrieve_QnA node")
    
    """Retrieve from medical_qna_collection"""
    #client = get_chroma_client()
    docs, metas, confidence_data  = retrieve_from_chroma_collection(_medical_qna_collection, state.get("query", ""), top_k=MAX_DOCS)
    # add source header e.g. 
    # [Document 1]
    # Hypertension is defined as persistently elevated arterial blood pressure...
    # [Document 2]
    # Blood pressure is typically measured using a sphygmomanometer...
    state["context"] = build_safe_context(docs)
    debug("➡️ Fetched document")
    state["source"] = "Retrieve_QnA"
    state["retrieval_success"] = bool(docs)
        # Store confidence measure
    state["confidence"] = confidence_data["confidence"]
    state["confidence_label"] = confidence_data["confidence_label"]

    print("➡️ retrieval_success flag", state["retrieval_success"])
    print("📘 QnA docs:", len(docs))
    print("📘 Confidence_label:", confidence_data["confidence_label"])
    return state

def retrieve_context_medical_device(state: GraphState) -> GraphState:
    """Retrieve from medical_devices_collection"""
    #client = get_chroma_client()
    #coll = client.get_or_create_collection(name=COLLECTION_DEVICES)
    #docs, metas = retrieve_from_chroma_collection(coll, state.get("query", ""), top_k=3)
    docs, metas, confidence_data  = retrieve_from_chroma_collection(_medical_devices_collection, state.get("query", ""), top_k=MAX_DOCS)
    state["context"] = "\n\n".join(docs) if docs else ""
    state["source"] = "Retrieve_Device"
    state["retrieval_success"] = bool(docs)
    # Store confidence measure
    state["confidence"] = confidence_data["confidence"]
    state["confidence_label"] = confidence_data["confidence_label"]
    print("➡️ retrieval_success flag", state["retrieval_success"])
    print("📘 QnA docs:", len(docs))
    print("📘 Confidence_label:", confidence_data["confidence_label"])
    return state

def web_search(state: GraphState) -> GraphState:
    print ("in web search")
    debug("➡️ Entered Web_Search node")
    """Call Serper web search; store truncated result into state['context']"""
    q = state.get("query", "")   
    if not any(k in q.lower() for k in ("medical", "drug", "device", "clinical", "health")):
        state["context"] = "Web search restricted to medical domain."
        return state

    search_tool = get_search_tool()

    try:
        raw = search_tool.run(q)
    except TypeError:
        raw = search_tool.run(query=q)
    except Exception as e:
        raw = f"[Web search failed: {str(e)}]"
    debug(f"Raw web search length: {len(raw)}")
    state["context"] = raw[:2500]
    state["source"] = "Web_Search"
    return state

#def check_context_relevance(state: GraphState) -> GraphState:
#    debug("➡️ Entered Relevance_Checker node")
#    """
#    Ask LLM to decide whether the retrieved context contain sufficient medical information to reasonably answer the question?.
#    Stores 'Yes' or 'No' in state['is_relevant'].
#    """
#    q = state.get("query", "")
#    context = state.get("context", "")[:1500]  # 🔐 HARD LIMIT
#    relevance_prompt = f"""
#Answer Yes or No only.
#Does the following context contain sufficient medical information to reasonably answer the question?

#Context:
#{context}

#User Query:
#{q}
#"""
#    debug(f"Relevance prompt length: {len(relevance_prompt)}")
#    decision = get_llm_response(relevance_prompt,max_tokens=5 ).strip().lower()
#    debug(f"Relevance decision: {decision}")
#    state["is_relevant"] = "Yes" if decision.startswith("y") else "No"
#    return state

#def increment_iteration(state: GraphState) -> GraphState:
#    """Node to increment iteration_count so mutations persist across graph steps."""
#    cnt = int(state.get("iteration_count", 0)) + 1
#    state["iteration_count"] = cnt
#    return state

#def relevance_decision(state: GraphState) -> str:
#    """
#    Pure router (no mutation): decide next route based on state['is_relevant'] and iteration_count.
#    Forces 'Yes' after max attempts to avoid infinite loops.
#    """

#    if state.get("retrieval_success"):
#        return "Yes"  # NEVER web-search if Chroma has content

#    return state.get("is_relevant", "Yes")
    
def build_prompt(state: GraphState) -> GraphState:
    print ("in build prompt")
    debug("➡️ Entered Build_Prompt node")
    """Compose RAG prompt for the final LLM call."""
    q = state.get("query", "")
    print("Query in build prompt: ", q)
    context = state.get("context", "")
    #raw_context = state.get("context", "")[:3000] # limit
    #context = limit_context(raw_context, max_chars=3000)
    prompt = f"""
            Answer the following question using the context below.
            Context:
            {context}
            Question: {q}
            please limit your answer in 50 words.
            """
    #prompt = f"""You are a conservative, evidence-first medical assistant. Use ONLY the context provided below. \
#If the context is insufficient, say 'I don't have enough evidence to answer safely.'

#CONTEXT:
#{context}

#QUESTION:
#{q}

#Instruction: Provide a concise answer (max 50 words) and a short 'sources' line.
#"""
    debug(f"Final prompt length: {len(prompt)}")
    state["prompt"] = prompt
    return state

def call_llm(state: GraphState) -> GraphState:
    print ("in call_llm")
    """Call the LLM and store the raw model output in state['response']."""
    prompt = state.get("prompt", "")
    print ("Final Prompt to generate LLM output: ", prompt)
    
    try:
        out = get_llm_response(prompt)
    except Exception as e:
        out = f"[LLM call failed: {str(e)}]"
    state["response"] = out
    return state

# ----------------------------
# Build LangGraph StateGraph
# ----------------------------
workflow = StateGraph(GraphState)

workflow.add_node("Router", router)
workflow.add_node("Retrieve_QnA", retrieve_context_q_n_a)
workflow.add_node("Retrieve_Device", retrieve_context_medical_device)
workflow.add_node("Web_Search", web_search)
#workflow.add_node("Relevance_Checker", check_context_relevance)
#workflow.add_node("Iteration_Tracker", increment_iteration)
workflow.add_node("Augment", build_prompt)
workflow.add_node("Generate", call_llm)

# wiring
workflow.add_edge(START, "Router")
# Router decides WHICH source
workflow.add_conditional_edges("Router", 
    route_decision, 
    {
    "Retrieve_QnA": "Retrieve_QnA",
    "Retrieve_Device": "Retrieve_Device",
    "Web_Search": "Web_Search",
    }
)

# After retrieval → check only if docs exist
workflow.add_conditional_edges(
    "Retrieve_QnA",
    lambda state: "Augment" if state["retrieval_success"] else "Web_Search",
    {
        "Augment": "Augment",
        "Web_Search": "Web_Search",
    }
)

workflow.add_conditional_edges(
    "Retrieve_Device",
    lambda state: "Augment" if state["retrieval_success"] else "Web_Search",
    {
        "Augment": "Augment",
        "Web_Search": "Web_Search",
    }
)

# Web search always goes forward
workflow.add_edge("Web_Search", "Augment")

# Final generation
workflow.add_edge("Augment", "Generate")
workflow.add_edge("Generate", END)

agentic_rag = workflow.compile()

#workflow.add_edge("Retrieve_QnA", "Relevance_Checker")
#workflow.add_edge("Retrieve_Device", "Relevance_Checker")
#workflow.add_edge("Web_Search", "Relevance_Checker")

# mutate iteration counter after relevance check
#workflow.add_edge("Relevance_Checker", "Iteration_Tracker")
# router uses iteration state to choose next edge
#workflow.add_conditional_edges("Iteration_Tracker", relevance_decision, {
#    "Yes": "Augment",
#    "No": "Web_Search",
#})

#workflow.add_edge("Augment", "Generate")
#workflow.add_edge("Generate", END)

# agentic_rag = workflow.compile()

# ----------------------------
# Public invocation helper
# ----------------------------
def run_agentic_query(query: str) -> Dict[str, Any]:
    """
    Public helper to invoke the compiled LangGraph agent.
    Returns the final state dictionary with at least: response, source, context, iteration_count.
    """
    _ensure_init()
    #initial_state: GraphState = {"query": query, "iteration_count": 0}
    initial_state: GraphState = {"query": query}
    # Some langgraph versions call .invoke(), some use .run(); try invoke then fallback
    #try:
    final_state = agentic_rag.invoke(initial_state)
    #except Exception:
    #    final_state = agentic_rag.run(initial_state)
    return dict(final_state)

# CLI debug entrypoint
if __name__ == "__main__":
    _ensure_init()
    print("Collections:", COLLECTION_QNA, COLLECTION_DEVICES)
    q = input("Enter a query: ")
    out = run_agentic_query(q)
    print("=== RESPONSE ===")
    print(out.get("response"))
    print("=== SOURCE ===")
    print(out.get("source"))
    print("=== Confidence ===")
    print(out.get("confidence"))
