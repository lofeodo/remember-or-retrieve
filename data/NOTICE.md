# Data notice

`corpus/chunks.jsonl` contains text from the
[Old School RuneScape Wiki](https://oldschool.runescape.wiki/), chunked by
[`rag-with-receipts`](https://github.com/lofeodo/rag-with-receipts). The wiki text is
licensed under [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/):
attribution required, non-commercial use only, derivatives under the same license. Each chunk
records its source page URL. Old School RuneScape is a trademark of Jagex Ltd.; this project
is not affiliated with or endorsed by Jagex.

`eval/golden_set.json` is the 55-question golden set authored for `rag-with-receipts`. It is
evaluation-only and must never be used for training.

See `PROVENANCE.json` for where each file came from and its SHA-256.
