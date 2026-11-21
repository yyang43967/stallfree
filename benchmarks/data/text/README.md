# Benchmark prompt text

`pride_and_prejudice.txt` is the text of Jane Austen's *Pride and Prejudice*, which is in the
public domain (published 1813). The benchmark workload samples prompt token ids from it at random
offsets (see `benchmarks/workload.py`).

- Source: Project Gutenberg eBook #1342, downloaded on 2026-11-20 from the Project Gutenberg
  mirror at https://gutenberg.pglaf.org/1/3/4/1342/1342-0.txt (the canonical address
  https://www.gutenberg.org/cache/epub/1342/pg1342.txt returned HTTP 503 at the time).
- Kept: the novel itself, from "It is a truth universally acknowledged" to "the means of uniting
  them."
- Removed: the Project Gutenberg header, footer, and trademark text, the 1894 edition's title
  pages, preface, list of illustrations, and every illustration caption. Trailing spaces were
  stripped and runs of blank lines collapsed to one.
- SHA-256 of the committed file:
  `518e6f9e8e2efccefce84aeeff981df5c2cb016a75526c99f5609a56b5bd3209`

Every benchmark run records this hash in its `run.json`, and a test checks that the file still
matches it.
