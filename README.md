# Fitness RAG Assistant

A Retrieval-Augmented Generation (RAG) application that answers workout and nutrition questions from a locally supplied document using **LangChain, Gemini, FAISS, PyPDF, and Gradio**.

## Portfolio Context

This is a cleaned standalone portfolio version of an academic RAG project. The original implementation used a workout-meals PDF, Gemini embeddings, FAISS similarity search, Gemini generation, qualitative evaluation questions, and a Gradio interface. The public version removes credentials, local virtual environments, generated vector-store pickle files, screenshots/local-machine artifacts, and the source PDF.

## What It Demonstrates

- PDF ingestion with `PyPDFLoader`
- text cleanup and page/section metadata
- recursive chunking with overlap
- Gemini embeddings
- FAISS semantic similarity retrieval
- Top-K evidence retrieval
- grounded prompting that restricts answers to retrieved context
- Gemini response generation
- CLI question mode
- seven-question qualitative evaluation workflow
- Gradio web interface
- source/page visibility for retrieved evidence
- environment-based secret management
- lightweight unit tests + GitHub Actions

## RAG Architecture

```text
Authorized fitness/nutrition PDF
          |
          v
     PyPDFLoader
          |
          v
Clean + metadata enrichment
          |
          v
RecursiveCharacterTextSplitter
(chunk=800, overlap=120)
          |
          v
Gemini Embeddings
          |
          v
 In-memory FAISS index
          |
      similarity search
          |
       Top K = 4
          |
          v
Retrieved context + question
          |
          v
Grounded prompt -> Gemini 2.5 Flash
          |
          v
Answer + retrieved page/section evidence
```

## Security & Privacy

- No Gemini/API key is committed.
- `.env` is ignored; `.env.example` contains only a placeholder.
- The original source PDF is not redistributed.
- Generated FAISS/pickle files are ignored.
- The clean version does **not** load an untrusted serialized FAISS docstore with `allow_dangerous_deserialization=True`; it rebuilds the vector index in memory from the trusted local document.
- Full retrieved chunks are not persisted by the evaluation workflow.

If an API key from the original coursework ZIP was ever exposed outside your local machine, rotate/revoke it before reuse.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Add your own Gemini key to `.env`:

```text
GOOGLE_API_KEY=your_gemini_api_key_here
```

Place an authorized PDF at:

```text
data/workout_nutrition_guide.pdf
```

## Run

Build the in-memory index:

```bash
python src/fitness_rag.py --mode build
```

Ask one question:

```bash
python src/fitness_rag.py --mode ask --question "What should a pre-workout meal focus on?"
```

Run the qualitative evaluation set:

```bash
python src/fitness_rag.py --mode eval
```

Launch the Gradio UI:

```bash
python src/fitness_rag.py --mode app
```

## Evaluation

The supplied academic project contained seven evaluation prompts. The manual accuracy-label column was not completed, so this portfolio version does not invent an accuracy score. See `docs/EVALUATION.md` for the reproducible evaluation approach.

## Tech Stack

**Python · LangChain · Gemini · FAISS · PyPDF · Gradio · Pandas · RAG · Vector Search · Prompt Grounding**

## Responsible Use

This project demonstrates document-grounded retrieval and generation. Fitness/nutrition answers depend entirely on the supplied source document and should not be treated as personalized medical advice.
