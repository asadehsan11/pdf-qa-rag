"""Streamlit UI for PDF Q&A."""
import streamlit as st

from src.qa import answer

st.set_page_config(page_title="PDF Q&A", page_icon="📄")
st.title("📄 PDF Q&A")
st.caption("Ask questions about your PDFs. Answers include citations.")

if "messages" not in st.session_state:
    st.session_state.messages = []


def show_sources(sources):
    with st.expander("Sources"):
        for i, s in enumerate(sources, start=1):
            st.markdown(f"**[{i}] {s['source']} - page {s['page']}**")
            st.caption(s["text"][:300] + "...")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            show_sources(msg["sources"])

if question := st.chat_input("Ask a question about your PDFs"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = answer(question)
    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
    )
    st.rerun()