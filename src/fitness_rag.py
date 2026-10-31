"""Fitness & nutrition Retrieval-Augmented Generation (RAG) assistant.

The application builds an in-memory FAISS vector store from a locally supplied
PDF, retrieves relevant chunks, and asks Gemini to answer only from retrieved
context. The knowledge-source PDF and API key are intentionally not committed.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import gradio as gr
import pandas as pd
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from text_utils import clean_text, detect_section

PDF_PATH = Path("data/workout_nutrition_guide.pdf")
RESULTS_PATH = Path("results/evaluation_results.csv")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = 4

DEFAULT_EMBED_MODEL = "models/gemini-embedding-001"
DEFAULT_CHAT_MODEL = "gemini-2.5-flash"

EVAL_QUESTIONS = [
    "What does the document recommend for hydration around workouts?",
    "According to the document, what should a pre-workout meal focus on?",
    "What is recommended after a workout for recovery?",
    "Does the document mention sports drinks? If yes, what does it say?",
    "Give one example of a pre-workout snack mentioned in the document.",
    "Give one example of a post-workout meal mentioned in the document.",
    "What does the document say about carbohydrates and protein in workout meals?",
]


def require_api_key() -> None:
    """Load the Gemini key from the environment without embedding it in code."""
    load_dotenv()
    if not (os.getenv("GOOGLE_API_KEY") or "").strip():
        raise RuntimeError(
            "GOOGLE_API_KEY is not configured. Copy .env.example to .env and add your own key."
        )


def load_and_prepare_docs(pdf_path: Path):
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"Knowledge-source PDF not found at {pdf_path}. See data/README.md."
        )

    docs = PyPDFLoader(str(pdf_path)).load()
    for page_index, doc in enumerate(docs):
        doc.page_content = clean_text(doc.page_content)
        doc.metadata["source"] = pdf_path.name
        doc.metadata["page_number"] = page_index + 1
        doc.metadata["section"] = detect_section(page_index)
    return docs


def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


def build_vectorstore(chunks):
    """Build FAISS in memory instead of loading an untrusted pickle index."""
    embeddings = GoogleGenerativeAIEmbeddings(model=DEFAULT_EMBED_MODEL)
    return FAISS.from_documents(chunks, embeddings)


def get_llm():
    return ChatGoogleGenerativeAI(model=DEFAULT_CHAT_MODEL, temperature=0)


def format_context(docs) -> str:
    return "\n\n".join(
        f"[Page {doc.metadata.get('page_number')}] {doc.page_content}" for doc in docs
    )


def ask_rag(question: str, vectorstore, llm, k: int = TOP_K):
    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty.")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a document-grounded fitness and nutrition assistant. "
                "Answer only from the supplied context. If the answer is not present, "
                "say exactly: 'I could not find that in the provided document.' "
                "Do not add outside medical or nutrition advice.",
            ),
            ("human", "Question: {question}\n\nContext:\n{context}"),
        ]
    )

    retrieved_docs = vectorstore.similarity_search(question, k=k)
    context = format_context(retrieved_docs)
    answer = (prompt | llm).invoke({"question": question, "context": context}).content
    return retrieved_docs, answer


def source_summary(docs) -> str:
    """Show source metadata without dumping full document chunks into logs/files."""
    return "\n".join(
        f"- Page {doc.metadata.get('page_number')} | Section {doc.metadata.get('section')}"
        for doc in docs
    )


def run_eval(vectorstore, llm, output_path: Path):
    rows = []
    for question in EVAL_QUESTIONS:
        retrieved_docs, answer = ask_rag(question, vectorstore, llm)
        rows.append(
            {
                "question": question,
                "retrieved_pages": ",".join(
                    str(doc.metadata.get("page_number")) for doc in retrieved_docs
                ),
                "final_answer": answer,
                "manual_accuracy_label": "",
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_path, index=False)
    print(f"Saved local evaluation output to {output_path}")
    return frame


def run_cli_question(question: str, vectorstore, llm):
    docs, answer = ask_rag(question, vectorstore, llm)
    print(f"\nANSWER:\n{answer}\n")
    print("Retrieved sources:")
    print(source_summary(docs))


def launch_gradio(vectorstore, llm):
    def respond(question: str):
        try:
            docs, answer = ask_rag(question, vectorstore, llm)
            return answer, source_summary(docs)
        except Exception as exc:
            return f"Request failed: {exc}", ""

    demo = gr.Interface(
        fn=respond,
        inputs=gr.Textbox(label="Ask about workout nutrition"),
        outputs=[
            gr.Textbox(label="Grounded Answer"),
            gr.Textbox(label="Retrieved Sources"),
        ],
        title="Fitness RAG Assistant",
        description=(
            "Retrieval-augmented assistant that answers from a locally supplied "
            "fitness/nutrition document. Educational demo; not medical advice."
        ),
    )
    demo.launch()


def build_runtime():
    require_api_key()
    docs = load_and_prepare_docs(PDF_PATH)
    chunks = split_docs(docs)
    vectorstore = build_vectorstore(chunks)
    return vectorstore, get_llm(), len(chunks)


def main():
    parser = argparse.ArgumentParser(description="Fitness & Nutrition RAG Assistant")
    parser.add_argument(
        "--mode",
        choices=["build", "ask", "eval", "app"],
        default="build",
    )
    parser.add_argument(
        "--question",
        default="What is recommended after a workout for recovery?",
    )
    args = parser.parse_args()

    vectorstore, llm, chunk_count = build_runtime()

    if args.mode == "build":
        print(f"Ready. Indexed chunks in memory: {chunk_count}")
    elif args.mode == "ask":
        run_cli_question(args.question, vectorstore, llm)
    elif args.mode == "eval":
        run_eval(vectorstore, llm, RESULTS_PATH)
    else:
        launch_gradio(vectorstore, llm)


if __name__ == "__main__":
    main()
