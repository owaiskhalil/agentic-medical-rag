# app.py
# Streamlit front-end for Agentic RAG
import streamlit as st
from agent import run_agentic_query


st.set_page_config(page_title="Agentic Medical RAG", layout="centered")


st.title("🩺 Agentic Medical RAG Assistant")
st.write("Ask medical or device questions. The system prefers internal medical DBs; uses web search as fallback.")


query = st.text_input("Enter your medical question:")

if st.button("Ask"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            try:
                result = run_agentic_query(query)
                st.subheader("Answer")
                st.write(result.get("response", "No response"))


                st.subheader("Source")
                st.write(result.get("source", "Unknown"))


                st.subheader("Retrieved Context (preview)")
                st.write((result.get("context") or "")[:2000])

                st.subheader("Confidence")
                st.metric(
                    label="Retrieval Confidence",
                    value=f"{result.get('confidence', 0)}%",
                    delta=result.get("confidence_label", "")
                )


                #st.write("\nIteration count:", result.get("iteration_count"))

            except Exception as e:
                st.error(f"Error: {str(e)}")