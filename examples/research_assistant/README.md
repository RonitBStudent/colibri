# Research-paper structured extraction

This prototype turns a UTF-8 plain-text paper into grounded, schema-shaped JSON
through colibrì's OpenAI-compatible server. It uses only the Python standard
library. PDF extraction is deliberately out of scope for the first experiment:
convert a PDF to text with a tool you already trust, then pass the text file here.

The schema asks for the research question, methods, datasets, claims with verbatim
evidence, and limitations. The client checks both the JSON shape and whether every
`evidence_quote` occurs in the source text.

## Extract one paper

Start colibrì normally, then run:

```sh
python3 examples/research_assistant/extract.py \
  examples/research_assistant/sample_paper.txt \
  --endpoint http://127.0.0.1:8000/v1 \
  --model glm-5.2-colibri \
  --output analysis.json
```

Set `COLI_API_KEY` if the server requires authentication. The key is read from the
environment and is never written to a report.

## Reproducible schema A/B

Keep the paper, prompt, model, sampling settings, warm-up count, and run count
fixed. Change only whether the server process has `SCHEMA` armed.

### 1. Baseline

Start the server without `SCHEMA`:

```sh
COLI_MODEL=/path/to/model ./c/coli serve --port 8000 --model-id glm-5.2-colibri
```

In another terminal:

```sh
python3 examples/research_assistant/benchmark.py run \
  examples/research_assistant/sample_paper.txt \
  --label baseline --warmup 1 --runs 3 \
  --hardware "CPU / RAM / GPU" \
  --storage "disk / filesystem / iobench result" \
  --output baseline.json
```

Stop the server after the rung completes.

### 2. Schema drafting

Restart the same server with the one experimental knob:

```sh
SCHEMA=examples/research_assistant/paper.schema.json \
COLI_MODEL=/path/to/model ./c/coli serve --port 8000 --model-id glm-5.2-colibri
```

Run the identical client workload:

```sh
python3 examples/research_assistant/benchmark.py run \
  examples/research_assistant/sample_paper.txt \
  --label schema --warmup 1 --runs 3 \
  --hardware "CPU / RAM / GPU" \
  --storage "disk / filesystem / iobench result" \
  --output schema.json
```

Compare the machine-readable reports:

```sh
python3 examples/research_assistant/benchmark.py compare \
  baseline.json schema.json --output comparison.json
```

The client records the commit, hardware, storage, OS, latency, end-to-end completion
tok/s, OpenAI usage counts, JSON/schema validity, and grounded-evidence rate.
Preserve the two server logs as well: colibrì's end-of-run grammar acceptance,
expert hit rate, and profile split are required to explain a speed change. Report
the exact commands, warm-up policy, run count, and median—not only the fastest run.

## Local tests

The tests mock the HTTP boundary and do not require a model or network access:

```sh
python3 -m unittest discover -s examples/research_assistant/tests -v
```
