# Knowledge-source document

The original academic project used a workout/nutrition PDF as its RAG knowledge source. That document is intentionally not redistributed in this public repository.

To run the project, place an authorized fitness/nutrition PDF at:

```text
data/workout_nutrition_guide.pdf
```

The application extracts page text locally, adds page/section metadata, splits the text into overlapping chunks, creates Gemini embeddings, and builds an in-memory FAISS index.

Only use documents you are authorized to process and redistribute.
