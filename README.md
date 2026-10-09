# Remember or Retrieve

Does a fine-tuned model *remember* the answer, or is it better to *retrieve* it at query time?

This project fine-tunes an open model on the Old School RuneScape Wiki and benchmarks it
head to head against the [`rag-with-receipts`](https://github.com/lofeodo/rag-with-receipts)
RAG pipeline on the same 55-question golden set.

**Status:** under construction. See [`docs/roadmap.md`](docs/roadmap.md) for the plan and
progress.

## Data attribution

The corpus is text from the [Old School RuneScape Wiki](https://oldschool.runescape.wiki/),
used under CC BY-NC-SA 3.0. See [`data/NOTICE.md`](data/NOTICE.md).

## License

Code: MIT. Corpus: CC BY-NC-SA 3.0 (see above).
