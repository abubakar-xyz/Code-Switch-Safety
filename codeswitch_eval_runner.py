#!/usr/bin/env python3
"""
CodeSwitch-Safety API Runner
============================

Runs the stripped-down model-facing prompt dataset against OpenAI and/or Gemini models.

Default model pair:
  - OpenAI:  gpt-4o
  - Gemini:  gemini-3.5-flash

Input:
  codeswitch_model_facing_prompts_clean.csv

Outputs:
  outputs_raw.csv
  run_summary_latest.csv

Environment variables:
  OPENAI_API_KEY=...
  GEMINI_API_KEY=...

Install:
  pip install pandas requests tqdm python-dotenv

Smoke test:
  python codeswitch_eval_runner.py --test --providers both

Full run:
  python codeswitch_eval_runner.py --providers both --resume
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable: Iterable, **_: Any) -> Iterable:
        return iterable

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
GEMINI_GENERATE_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

DEFAULT_OPENAI_MODELS = ["gpt-4o"]
DEFAULT_GEMINI_MODELS = ["gemini-3.5-flash"]

REQUIRED_PROMPT_COLUMNS = {
    "prompt_id", "triplet_id", "language_pair", "target_language", "harm_domain",
    "condition", "challenge_level_id", "challenge_level", "prompt"
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def read_prompts(path: str) -> pd.DataFrame:
    prompt_path = Path(path)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    df = pd.read_csv(prompt_path)
    missing = REQUIRED_PROMPT_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Prompt file missing required columns: {sorted(missing)}")
    if not df["prompt_id"].is_unique:
        duplicates = df.loc[df["prompt_id"].duplicated(), "prompt_id"].head(10).tolist()
        raise ValueError(f"prompt_id must be unique. Example duplicates: {duplicates}")
    if df["prompt"].isna().any() or (df["prompt"].astype(str).str.strip() == "").any():
        raise ValueError("Prompt file contains blank prompt values.")
    return df


def apply_filters(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    filtered = df.copy()
    if args.language_pairs:
        filtered = filtered[filtered["language_pair"].isin(args.language_pairs)]
    if args.harm_domains:
        filtered = filtered[filtered["harm_domain"].isin(args.harm_domains)]
    if args.conditions:
        filtered = filtered[filtered["condition"].isin(args.conditions)]
    if args.challenge_levels:
        filtered = filtered[filtered["challenge_level_id"].isin(args.challenge_levels)]
    if args.test:
        filtered = filtered.head(6)
    elif args.limit is not None:
        filtered = filtered.head(args.limit)
    return filtered.reset_index(drop=True)


def load_existing_outputs(path: str) -> pd.DataFrame:
    output_path = Path(path)
    if not output_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(output_path)
    except Exception:
        return pd.DataFrame()


def already_completed(existing: pd.DataFrame, prompt_id: str, provider: str, model: str) -> bool:
    required = {"prompt_id", "provider", "model", "status"}
    if existing.empty or not required.issubset(existing.columns):
        return False
    mask = (
        (existing["prompt_id"].astype(str) == str(prompt_id)) &
        (existing["provider"].astype(str) == provider) &
        (existing["model"].astype(str) == model) &
        (existing["status"].astype(str) == "ok")
    )
    return bool(mask.any())


def append_row_csv(path: str, row: Dict[str, Any]) -> None:
    output_path = Path(path)
    pd.DataFrame([row]).to_csv(
        output_path,
        mode="a",
        index=False,
        header=not output_path.exists(),
        encoding="utf-8",
    )


def exponential_backoff(attempt: int, base: float = 2.0, cap: float = 90.0) -> None:
    delay = min(cap, base * (2 ** attempt)) + random.uniform(0.0, 1.0)
    time.sleep(delay)


def is_retryable_error(error: Exception) -> bool:
    message = str(error).lower()
    retry_tokens = [
        "429", "rate", "resource_exhausted", "quota", "timeout",
        "temporarily", "503", "502", "504", "connection", "server error"
    ]
    return any(token in message for token in retry_tokens)


def call_openai(prompt: str, model: str, temperature: float, max_output_tokens: int, timeout: int) -> Tuple[str, Dict[str, Any]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Put it in your environment or a local .env file.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": model,
        "input": prompt,
        "max_output_tokens": max_output_tokens,
        "temperature": temperature,
    }

    response = requests.post(OPENAI_RESPONSES_URL, headers=headers, json=payload, timeout=timeout)

    # Some model families reject temperature. Retry once without it.
    if response.status_code >= 400 and "temperature" in response.text.lower():
        payload.pop("temperature", None)
        response = requests.post(OPENAI_RESPONSES_URL, headers=headers, json=payload, timeout=timeout)

    if response.status_code >= 400:
        raise RuntimeError(f"OpenAI HTTP {response.status_code}: {response.text[:2000]}")

    data = response.json()
    text = data.get("output_text")
    if text is None:
        chunks: List[str] = []
        for item in data.get("output", []) or []:
            for content in item.get("content", []) or []:
                if content.get("type") in {"output_text", "text"} and "text" in content:
                    chunks.append(content["text"])
        text = "\n".join(chunks).strip() if chunks else json.dumps(data, ensure_ascii=False)[:5000]
    return text, data.get("usage", {})


def call_gemini(prompt: str, model: str, temperature: float, max_output_tokens: int, timeout: int) -> Tuple[str, Dict[str, Any]]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Put it in your environment or a local .env file.")

    url = GEMINI_GENERATE_URL.format(model=model)
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": api_key,
    }
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        },
    }

    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if response.status_code >= 400:
        raise RuntimeError(f"Gemini HTTP {response.status_code}: {response.text[:2000]}")

    data = response.json()
    text_parts: List[str] = []
    for candidate in data.get("candidates", []) or []:
        content = candidate.get("content", {}) or {}
        for part in content.get("parts", []) or []:
            if "text" in part:
                text_parts.append(part["text"])
    text = "\n".join(text_parts).strip() if text_parts else json.dumps(data, ensure_ascii=False)[:5000]
    return text, data.get("usageMetadata", {})


def call_model_with_retries(
    provider: str,
    model: str,
    prompt: str,
    temperature: float,
    max_output_tokens: int,
    timeout: int,
    max_retries: int,
) -> Tuple[str, Dict[str, Any]]:
    for attempt in range(max_retries + 1):
        try:
            if provider == "openai":
                return call_openai(prompt, model, temperature, max_output_tokens, timeout)
            if provider == "gemini":
                return call_gemini(prompt, model, temperature, max_output_tokens, timeout)
            raise ValueError(f"Unknown provider: {provider}")
        except Exception as exc:
            if attempt < max_retries and is_retryable_error(exc):
                print(
                    f"Retryable error from {provider}/{model}: {exc}. "
                    f"Retrying {attempt + 1}/{max_retries}...",
                    file=sys.stderr,
                )
                exponential_backoff(attempt)
                continue
            raise


def selected_jobs(args: argparse.Namespace) -> List[Tuple[str, str]]:
    jobs: List[Tuple[str, str]] = []
    if args.providers in {"openai", "both"}:
        jobs.extend(("openai", model) for model in args.openai_models)
    if args.providers in {"gemini", "both"}:
        jobs.extend(("gemini", model) for model in args.gemini_models)
    if not jobs:
        raise ValueError("No provider/model jobs selected.")
    return jobs


def write_summary(output_path: str, summary_path: str) -> None:
    if not Path(output_path).exists():
        return
    df = pd.read_csv(output_path)
    if df.empty:
        return

    summary = (
        df.groupby(["provider", "model", "status"])
        .size()
        .reset_index(name="rows")
        .sort_values(["provider", "model", "status"])
    )
    summary.to_csv(summary_path, index=False)

    print("\nRun summary:")
    print(summary.to_string(index=False))

    ok = df[df["status"] == "ok"]
    if not ok.empty:
        coverage = (
            ok.groupby(["provider", "model"])["prompt_id"]
            .nunique()
            .reset_index(name="unique_prompts_completed")
        )
        print("\nPrompt coverage:")
        print(coverage.to_string(index=False))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CodeSwitch-Safety prompts against OpenAI/Gemini APIs.")
    parser.add_argument("--input", default="codeswitch_model_facing_prompts_clean.csv")
    parser.add_argument("--output", default="outputs_raw.csv")
    parser.add_argument("--summary-output", default="run_summary_latest.csv")
    parser.add_argument("--providers", choices=["openai", "gemini", "both"], default="both")
    parser.add_argument("--openai-models", nargs="*", default=DEFAULT_OPENAI_MODELS)
    parser.add_argument("--gemini-models", nargs="*", default=DEFAULT_GEMINI_MODELS)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--sleep", type=float, default=6.0, help="Seconds between calls; increase if rate-limited.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--test", action="store_true", help="Run first 6 prompts only.")
    parser.add_argument("--resume", action="store_true", help="Skip prompt/model rows already completed with status=ok.")
    parser.add_argument("--language-pairs", nargs="*", default=None)
    parser.add_argument("--harm-domains", nargs="*", default=None)
    parser.add_argument("--conditions", nargs="*", choices=["english", "target_language", "code_switched"], default=None)
    parser.add_argument("--challenge-levels", nargs="*", choices=["L1", "L2", "L3"], default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    prompts = apply_filters(read_prompts(args.input), args)
    jobs = selected_jobs(args)

    print("CodeSwitch-Safety eval runner")
    print(f"Input file: {args.input}")
    print(f"Prompt rows after filters: {len(prompts)}")
    print(f"Models: {jobs}")
    print(f"Output file: {args.output}")
    print("Reminder: keep API keys in .env and do not commit controlled materials.\n")

    existing = load_existing_outputs(args.output) if args.resume else pd.DataFrame()
    total_jobs = len(prompts) * len(jobs)
    completed_or_skipped = 0

    for _, prompt_row in tqdm(list(prompts.iterrows()), total=len(prompts), desc="prompts"):
        for provider, model in jobs:
            if args.resume and already_completed(existing, prompt_row["prompt_id"], provider, model):
                completed_or_skipped += 1
                continue

            started = utc_now()
            start_time = time.time()
            status = "ok"
            response_text = ""
            error = ""
            usage_json = ""

            try:
                response_text, usage = call_model_with_retries(
                    provider=provider,
                    model=model,
                    prompt=str(prompt_row["prompt"]),
                    temperature=args.temperature,
                    max_output_tokens=args.max_output_tokens,
                    timeout=args.timeout,
                    max_retries=args.max_retries,
                )
                usage_json = to_json(usage)
            except Exception as exc:
                status = "error"
                error = repr(exc) + "\n" + traceback.format_exc(limit=3)

            output_row = {
                "run_timestamp_utc": started,
                "provider": provider,
                "model": model,
                "temperature": args.temperature,
                "max_output_tokens": args.max_output_tokens,
                "status": status,
                "error": error,
                "latency_seconds": round(time.time() - start_time, 3),
                "prompt_id": prompt_row["prompt_id"],
                "triplet_id": prompt_row["triplet_id"],
                "language_pair": prompt_row["language_pair"],
                "target_language": prompt_row["target_language"],
                "harm_domain": prompt_row["harm_domain"],
                "condition": prompt_row["condition"],
                "challenge_level_id": prompt_row["challenge_level_id"],
                "challenge_level": prompt_row["challenge_level"],
                "abstract_harm_intent": prompt_row.get("abstract_harm_intent", ""),
                "operational_detail_level": prompt_row.get("operational_detail_level", ""),
                "expected_refusal_reason": prompt_row.get("expected_refusal_reason", ""),
                "prompt_sha256": prompt_row.get("prompt_sha256", ""),
                "prompt": prompt_row["prompt"],
                "response_text": response_text,
                "usage_json": usage_json,
            }
            append_row_csv(args.output, output_row)
            completed_or_skipped += 1
            time.sleep(args.sleep)

    write_summary(args.output, args.summary_output)
    print(f"\nDone. Completed/skipped {completed_or_skipped}/{total_jobs} prompt-model jobs.")


if __name__ == "__main__":
    main()
