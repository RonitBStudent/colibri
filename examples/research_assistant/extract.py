#!/usr/bin/env python3
"""Extract a grounded, schema-shaped summary from research-paper text."""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


SYSTEM_PROMPT = """You extract research findings without inventing facts.
Return exactly one compact JSON object matching the supplied schema. Do not use
Markdown fences. Every evidence_quote must be copied verbatim from the paper.
Use an empty array when the paper does not report a method, dataset, claim, or
limitation. Judge claim strength from the evidence described in the paper."""


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chat_completions_url(endpoint):
    parts = urlsplit(endpoint)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ValueError("endpoint must be an http(s) URL")
    path = parts.path.rstrip("/")
    if path.endswith("/chat/completions"):
        final_path = path
    elif path.endswith("/v1"):
        final_path = path + "/chat/completions"
    else:
        final_path = path + "/v1/chat/completions"
    return urlunsplit((parts.scheme, parts.netloc, final_path, "", ""))


def build_user_prompt(paper_text, schema):
    compact_schema = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    return (
        "JSON schema:\n" + compact_schema +
        "\n\nPaper text begins:\n" + paper_text + "\nPaper text ends."
    )


def request_completion(endpoint, model, paper_text, schema, api_key=None,
                       timeout=1800, max_tokens=1200):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(paper_text, schema)},
        ],
        "temperature": 0,
        "top_p": 1,
        "max_completion_tokens": max_tokens,
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    request = Request(
        chat_completions_url(endpoint),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    started = time.monotonic()
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
            response_headers = dict(response.headers.items())
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"API returned HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"could not reach API: {exc.reason}") from exc
    elapsed = time.monotonic() - started
    try:
        body = json.loads(raw)
        content = body["choices"][0]["message"]["content"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("API response is not an OpenAI chat completion") from exc
    if not isinstance(content, str):
        raise RuntimeError("API response content is not text")
    return content, body, response_headers, elapsed


def parse_json_content(content):
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"model output is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("model output must be a JSON object")
    return value


def validate_schema(value, schema, path="$"):
    """Validate the JSON-Schema subset used by paper.schema.json."""
    errors = []
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: expected one of {schema['enum']!r}")
        return errors

    expected = schema.get("type")
    type_checks = {
        "object": lambda item: isinstance(item, dict),
        "array": lambda item: isinstance(item, list),
        "string": lambda item: isinstance(item, str),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "number": lambda item: isinstance(item, (int, float)) and not isinstance(item, bool),
        "boolean": lambda item: isinstance(item, bool),
        "null": lambda item: item is None,
    }
    if expected and (expected not in type_checks or not type_checks[expected](value)):
        errors.append(f"{path}: expected {expected}")
        return errors

    if expected == "object":
        properties = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in value:
                errors.append(f"{path}: missing required property {name!r}")
        if schema.get("additionalProperties") is False:
            for name in value:
                if name not in properties:
                    errors.append(f"{path}: unexpected property {name!r}")
        for name, child in value.items():
            if name in properties:
                errors.extend(validate_schema(child, properties[name], f"{path}.{name}"))
    elif expected == "array" and "items" in schema:
        for index, child in enumerate(value):
            errors.extend(validate_schema(child, schema["items"], f"{path}[{index}]"))
    return errors


def normalize_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()


def grounding_report(analysis, paper_text):
    source = normalize_whitespace(paper_text)
    results = []
    for index, claim in enumerate(analysis.get("claims", [])):
        quote = claim.get("evidence_quote", "") if isinstance(claim, dict) else ""
        grounded = bool(quote) and normalize_whitespace(quote) in source
        results.append({"claim_index": index, "quote": quote, "grounded": grounded})
    grounded = sum(item["grounded"] for item in results)
    return {"grounded": grounded, "total": len(results), "claims": results}


def analyze_paper(paper_text, schema, endpoint, model, api_key=None,
                  timeout=1800, max_tokens=1200):
    content, response, headers, elapsed = request_completion(
        endpoint, model, paper_text, schema, api_key, timeout, max_tokens
    )
    parse_error = None
    try:
        analysis = parse_json_content(content)
        schema_errors = validate_schema(analysis, schema)
        grounding = grounding_report(analysis, paper_text)
    except ValueError as exc:
        analysis = None
        schema_errors = [str(exc)]
        grounding = {"grounded": 0, "total": 0, "claims": []}
        parse_error = str(exc)
    usage = response.get("usage", {}) if isinstance(response, dict) else {}
    return {
        "analysis": analysis,
        "raw_content": content,
        "valid": analysis is not None and not schema_errors,
        "schema_errors": schema_errors,
        "parse_error": parse_error,
        "grounding": grounding,
        "metrics": {
            "elapsed_seconds": elapsed,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "queue_wait_ms": headers.get("x-colibri-queue-wait-ms"),
        },
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paper", type=Path, help="UTF-8 plain-text paper")
    parser.add_argument("--schema", type=Path,
                        default=Path(__file__).with_name("paper.schema.json"))
    parser.add_argument("--endpoint", default=os.environ.get("COLI_ENDPOINT", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--model", default=os.environ.get("COLI_MODEL_ID", "glm-5.2-colibri"))
    parser.add_argument("--api-key", default=os.environ.get("COLI_API_KEY"))
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--output", type=Path, default=Path("analysis.json"))
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    paper_text = args.paper.read_text(encoding="utf-8")
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    result = analyze_paper(
        paper_text, schema, args.endpoint, args.model, args.api_key,
        args.timeout, args.max_tokens,
    )
    output = result["analysis"] if result["analysis"] is not None else {
        "raw_content": result["raw_content"]
    }
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {
        "paper_sha256": sha256_text(paper_text),
        "schema_sha256": sha256_text(args.schema.read_text(encoding="utf-8")),
        "valid": result["valid"],
        "schema_errors": result["schema_errors"],
        "grounding": result["grounding"],
        "metrics": result["metrics"],
        "output": str(args.output),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    sys.exit(main())
