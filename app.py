"""Streamlit UI for PDF Q&A."""
import html

import streamlit as st

from src.ingest import get_collection, ingest_pdf
from src.qa import answer
from src.summarize import summarize_document

REPO_URL = "https://github.com/asadehsan11/pdf-qa-rag"
CARD_COLORS = ["#0E8F6E", "#F2A93B", "#2EC4A6", "#E07A5F", "#5B8DEF", "#9B6BDF"]


def clean(text):
    """Escape HTML and markdown characters so snippets show exactly as written."""
    return (
        html.escape(text)
        .replace("_", "&#95;")
        .replace("*", "&#42;")
        .replace("$", "&#36;")
    )


st.set_page_config(page_title="PDF Q&A Assistant", page_icon="📄", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --ink: #0B3D33;
        --brand: #0E8F6E;
        --brand-2: #0C8566;
        --brand-3: #0A7C5F;
        --tint: #D5F3E8;
        --tint-border: #B5E6D3;
        --tint-text: #0B6B52;
        --accent: #F2A93B;
        --muted: #5B6478;
        --shadow: rgba(11,61,51,.08);
        --glow: rgba(14,143,110,.25);
    }
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {padding-top: 1rem; max-width: 1200px;}

    .navbar {
        display: flex; align-items: center; justify-content: space-between;
        background: #fff; border-radius: 16px; padding: .8rem 1.4rem;
        box-shadow: 0 4px 18px var(--shadow); margin-bottom: 1.5rem;
    }
    .brand {font-weight: 800; font-size: 1.15rem; color: var(--ink);}
    .navbar a {
        color: var(--ink); text-decoration: none; font-weight: 600;
        background: var(--tint); padding: .35rem .9rem; border-radius: 999px;
    }

    .hero {
        background: linear-gradient(120deg, var(--brand) 0%, var(--brand-2) 50%, var(--brand-3) 100%);
        border-radius: 24px; padding: 2.6rem 2.4rem; margin-bottom: 1.8rem;
        box-shadow: 0 12px 30px var(--glow);
    }
    .hero h1 {font-size: 2.5rem; font-weight: 800; margin: 0 0 .5rem 0; color: #fff;}
    .hero p {font-size: 1.1rem; max-width: 680px; margin: 0 0 1.2rem 0; color: #fff;}
    .pill {
        display: inline-block; background: rgba(255,255,255,.22);
        border: 1px solid rgba(255,255,255,.45); color: #fff; padding: .3rem .9rem;
        border-radius: 999px; font-size: .85rem; font-weight: 600; margin-right: .5rem;
    }

    .card-title {font-weight: 700; font-size: 1.05rem; margin-bottom: .5rem; color: var(--ink);}
    .section-title {font-weight: 800; font-size: 1.4rem; color: var(--ink); margin: 2rem 0 .8rem 0;}
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: transparent !important; border-radius: 18px; background: #fff;
        box-shadow: 0 6px 22px var(--shadow);
    }
    div[data-testid="stForm"] {border: 0; padding: 0;}
    .source-card {
        background: #fff; border-left: 5px solid var(--accent); border-radius: 12px;
        padding: .8rem 1rem; margin-bottom: .7rem; box-shadow: 0 4px 14px var(--shadow);
    }
    .source-card b {color: var(--ink);}
    .source-card span {color: var(--muted); font-size: .9rem;}
    .chip {
        display: inline-block; background: var(--tint); border: 1px solid var(--tint-border);
        color: var(--tint-text); padding: .25rem .7rem; border-radius: 999px;
        font-size: .85rem; font-weight: 600; margin: 0 .35rem .35rem 0;
    }

    .play-card {
        background: #fff; border-radius: 18px; border-top: 6px solid var(--c);
        padding: 1.1rem 1.2rem; margin-bottom: 1rem; min-height: 230px;
        box-shadow: 0 6px 22px var(--shadow); display: flex; flex-direction: column;
    }
    .play-num {font-weight: 800; font-size: 1.7rem; color: var(--c); line-height: 1;}
    .play-title {font-weight: 700; font-size: 1.05rem; color: var(--ink); margin: .5rem 0 .4rem 0;}
    .play-text {color: var(--muted); font-size: .92rem; line-height: 1.55; flex: 1;}
    .play-pages {
        margin-top: .8rem; align-self: flex-start; background: var(--tint);
        color: var(--tint-text); font-size: .8rem; font-weight: 600;
        padding: .15rem .65rem; border-radius: 999px;
    }

    .stButton > button, .stFormSubmitButton > button {
        background: linear-gradient(90deg, var(--brand), var(--brand-2)); color: #fff;
        font-weight: 700; border: 0; border-radius: 12px; padding: .6rem 1.4rem;
    }
    .stButton > button p, .stFormSubmitButton > button p {color: #fff !important;}
    .footer {text-align: center; color: var(--muted); margin: 2.5rem 0 1rem 0; font-size: .9rem;}
    .footer a {color: var(--brand); font-weight: 600; text-decoration: none;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="navbar">
        <div class="brand">📄 PDF Q&amp;A Assistant</div>
        <a href="{REPO_URL}" target="_blank">GitHub</a>
    </div>
    <div class="hero">
        <h1>Ask your documents anything</h1>
        <p>Upload your PDFs and ask questions in plain English. Answers are grounded
        in your own documents and come with page-level citations.</p>
        <span class="pill">Multi-PDF</span>
        <span class="pill">Page citations</span>
        <span class="pill">Grounded answers</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if "history" not in st.session_state:
    st.session_state.history = []
if "cards" not in st.session_state:
    st.session_state.cards = {}

left, right = st.columns([1, 2], gap="large")

with left:
    with st.container(border=True):
        st.markdown('<div class="card-title">📤 Upload documents</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "PDF files", type="pdf", accept_multiple_files=True, label_visibility="collapsed"
        )
        if st.button("Process PDF", use_container_width=True):
            if not uploaded:
                st.warning("Choose at least one PDF first.")
            else:
                with st.spinner("Indexing..."):
                    for f in uploaded:
                        n = ingest_pdf(f, f.name)
                        st.success(f"{f.name}: {n} chunks")
        metas = get_collection().get(include=["metadatas"])["metadatas"]
        names = sorted({m["source"] for m in metas})
        st.markdown(
            '<div class="card-title" style="margin-top:1rem">📚 Indexed documents</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            "".join(f'<span class="chip">{html.escape(n)}</span>' for n in names)
            or "No documents yet.",
            unsafe_allow_html=True,
        )

with right:
    with st.container(border=True):
        st.markdown('<div class="card-title">❓ Ask a question</div>', unsafe_allow_html=True)
        with st.form("ask_form", clear_on_submit=True):
            question = st.text_input(
                "Your question",
                placeholder="Ask anything about your PDFs...",
                label_visibility="collapsed",
            )
            asked = st.form_submit_button("Ask")
    if asked and question.strip():
        with st.spinner("Thinking..."):
            result = answer(question.strip())
        st.session_state.history.append({"question": question.strip(), **result})

    if st.session_state.history:
        latest = st.session_state.history[-1]
        st.markdown(
            f'<div class="card-title" style="margin-top:1rem">💬 {html.escape(latest["question"])}</div>',
            unsafe_allow_html=True,
        )
        with st.container(border=True):
            st.markdown(latest["answer"])
        st.markdown(
            '<div class="card-title" style="margin-top:1rem">📎 Sources</div>',
            unsafe_allow_html=True,
        )
        for i, s in enumerate(latest["sources"], start=1):
            st.markdown(
                f'<div class="source-card"><b>[{i}] {html.escape(s["source"])} - page {s["page"]}</b><br>'
                f'<span>{clean(s["text"][:300])}...</span></div>',
                unsafe_allow_html=True,
            )
        older = st.session_state.history[:-1]
        if older:
            with st.expander(f"Chat history ({len(older)})"):
                for item in reversed(older):
                    st.markdown(f"**Q:** {item['question']}")
                    st.markdown(item["answer"])
                    st.divider()
    else:
        with st.container(border=True):
            st.markdown(
                "Your answer will appear here. Upload PDFs on the left, click "
                "**Process PDF**, then ask a question."
            )

st.markdown('<div class="section-title">🗂️ Summary cards</div>', unsafe_allow_html=True)
with st.container(border=True):
    if names:
        pick_col, btn_col = st.columns([3, 1], gap="medium")
        with pick_col:
            doc = st.selectbox("Document", names, label_visibility="collapsed")
        with btn_col:
            summarize_clicked = st.button("Summarize as cards", use_container_width=True)
        if summarize_clicked:
            with st.spinner("Reading the document and building cards..."):
                try:
                    st.session_state.cards[doc] = summarize_document(doc)
                except Exception as e:
                    st.error(f"Could not build the cards: {e}")
    else:
        doc = None
        st.markdown("Upload and process a PDF to build summary cards.")

cards = st.session_state.cards.get(doc) if doc else None
if cards:
    cols = st.columns(3, gap="medium")
    for i, c in enumerate(cards):
        pages = f'<div class="play-pages">Pages {clean(c["pages"])}</div>' if c["pages"] else ""
        with cols[i % 3]:
            st.markdown(
                f'<div class="play-card" style="--c:{CARD_COLORS[i % len(CARD_COLORS)]}">'
                f'<div class="play-num">{i + 1:02d}</div>'
                f'<div class="play-title">{clean(c["heading"])}</div>'
                f'<div class="play-text">{clean(c["summary"])}</div>'
                f"{pages}</div>",
                unsafe_allow_html=True,
            )

st.markdown(
    f'<div class="footer">Built by Asad Ehsan · <a href="{REPO_URL}" target="_blank">View source on GitHub</a></div>',
    unsafe_allow_html=True,
)