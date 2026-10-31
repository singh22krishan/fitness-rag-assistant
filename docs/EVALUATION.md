# Evaluation approach

The original project included seven qualitative questions covering hydration, pre-workout meals, post-workout recovery, sports drinks, snack examples, meal examples, and carbohydrate/protein guidance.

The supplied academic evaluation CSV contained generated answers and retrieved source-text previews, but its manual `accuracy_label` field had not been completed. This public portfolio version therefore does **not** claim an accuracy percentage.

`--mode eval` reproduces the same seven-question evaluation workflow locally and writes:

- the question
- retrieved page numbers
- the generated grounded answer
- a blank manual accuracy label for human review

Full retrieved document chunks are intentionally not written to the public repository.
