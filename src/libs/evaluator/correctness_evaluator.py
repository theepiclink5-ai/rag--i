"""Correctness evaluator – LLM-as-Judge answer correctness assessment.

Compares the generated answer against a reference_answer using an LLM
to produce a structured verdict: pass / partial / fail, along with
covered_points, missed_points, and violated_claims details.

When no LLM is available or reference_answer is missing, the evaluator
gracefully skips and returns an empty dict.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Sequence

from src.libs.evaluator.base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)

CORRECTNESS_VERDICT = "answer_correctness_verdict"
CORRECTNESS_SCORE = "answer_correctness"

SUPPORTED_METRICS = {CORRECTNESS_VERDICT, CORRECTNESS_SCORE}

# Prompt template for LLM-as-Judge correctness evaluation
_CORRECTNESS_PROMPT = """\
You are an expert evaluator. Compare the GENERATED ANSWER against the REFERENCE ANSWER for the given QUESTION.

## QUESTION
{query}

## REFERENCE ANSWER
{reference_answer}

## GENERATED ANSWER
{generated_answer}

## Instructions
1. Identify the key factual points in the REFERENCE ANSWER.
2. Check which points are correctly covered in the GENERATED ANSWER.
3. Check which points are missing from the GENERATED ANSWER.
4. Check if the GENERATED ANSWER contains any claims that contradict the REFERENCE ANSWER.
5. Assign a verdict:
   - "pass": All key points are covered and no contradictions exist.
   - "partial": Some key points are covered but others are missing, OR minor inaccuracies exist.
   - "fail": Most key points are missing, OR significant contradictions exist.

## Output Format (strict JSON)
{{
  "verdict": "pass" | "partial" | "fail",
  "score": <float 0.0-1.0>,
  "covered_points": ["point1", "point2"],
  "missed_points": ["point3"],
  "violated_claims": ["contradiction1"]
}}

Output ONLY the JSON object, nothing else.
"""


class CorrectnessEvaluator(BaseEvaluator):
    """LLM-as-Judge evaluator for answer correctness.

    Requires:
        - ground_truth dict with 'reference_answer' key
        - generated_answer text
        - A working LLM backend via settings

    Returns metrics:
        - answer_correctness: float score (0.0–1.0)
        - answer_correctness_verdict: encoded as float (pass=1.0, partial=0.5, fail=0.0)
          The detailed verdict info is stored in trace metadata.
    """

    def __init__(
        self,
        settings: Any = None,
        metrics: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> None:
        self.settings = settings
        self.kwargs = kwargs
        normalized = [str(m).strip().lower() for m in (metrics or SUPPORTED_METRICS)]
        unsupported = [m for m in normalized if m not in SUPPORTED_METRICS]
        if unsupported:
            raise ValueError(
                f"Unsupported correctness metrics: {', '.join(unsupported)}. "
                f"Supported: {', '.join(sorted(SUPPORTED_METRICS))}"
            )
        self.metrics = normalized if normalized else list(SUPPORTED_METRICS)

    def evaluate(
        self,
        query: str,
        retrieved_chunks: List[Any],
        generated_answer: Optional[str] = None,
        ground_truth: Optional[Any] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Evaluate answer correctness against reference_answer."""
        self.validate_query(query)

        # Extract reference_answer
        reference_answer = self._extract_reference(ground_truth)
        if not reference_answer or not reference_answer.strip():
            logger.debug("CorrectnessEvaluator: no reference_answer; skipping.")
            return {}

        if not generated_answer or not generated_answer.strip():
            logger.debug("CorrectnessEvaluator: no generated_answer; skipping.")
            return {}

        try:
            result = self._judge_correctness(query, reference_answer, generated_answer)
        except Exception as exc:
            logger.warning("CorrectnessEvaluator failed: %s", exc, exc_info=True)
            return {}

        metrics: Dict[str, float] = {}
        if CORRECTNESS_SCORE in self.metrics:
            metrics[CORRECTNESS_SCORE] = result.get("score", 0.0)
        if CORRECTNESS_VERDICT in self.metrics:
            verdict = result.get("verdict", "fail")
            verdict_map = {"pass": 1.0, "partial": 0.5, "fail": 0.0}
            metrics[CORRECTNESS_VERDICT] = verdict_map.get(verdict, 0.0)

        # Store detailed verdict in trace metadata for Dashboard display
        if trace is not None:
            try:
                trace.metadata["correctness_detail"] = {
                    "verdict": result.get("verdict", "fail"),
                    "covered_points": result.get("covered_points", []),
                    "missed_points": result.get("missed_points", []),
                    "violated_claims": result.get("violated_claims", []),
                }
            except Exception:
                pass

        return metrics

    @staticmethod
    def _extract_reference(ground_truth: Optional[Any]) -> Optional[str]:
        """Extract reference_answer from ground_truth."""
        if ground_truth is None:
            return None
        if isinstance(ground_truth, str):
            return ground_truth
        if isinstance(ground_truth, dict):
            return ground_truth.get("reference_answer")
        return None

    def _judge_correctness(
        self,
        query: str,
        reference_answer: str,
        generated_answer: str,
    ) -> Dict[str, Any]:
        """Call LLM to judge correctness and parse structured response."""
        import asyncio
        from openai import AsyncAzureOpenAI, AsyncOpenAI

        if self.settings is None:
            raise ValueError("Settings required for CorrectnessEvaluator LLM call")

        llm_cfg = self.settings.llm
        provider = llm_cfg.provider.lower()
        llm_azure_endpoint = getattr(llm_cfg, "azure_endpoint", None)

        use_azure = provider == "azure" or (provider == "openai" and llm_azure_endpoint)

        if use_azure:
            client = AsyncAzureOpenAI(
                api_key=llm_cfg.api_key,
                azure_endpoint=llm_azure_endpoint or llm_cfg.azure_endpoint,
                api_version=getattr(llm_cfg, "api_version", None) or "2024-02-15-preview",
                timeout=60.0,
            )
        elif provider == "openai":
            client = AsyncOpenAI(api_key=llm_cfg.api_key, timeout=60.0)
        elif provider == "deepseek":
            import os
            api_key = llm_cfg.api_key or os.environ.get("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError("DEEPSEEK_API_KEY required for CorrectnessEvaluator")
            client = AsyncOpenAI(
                api_key=api_key,
                base_url=getattr(llm_cfg, "base_url", None) or "https://api.deepseek.com",
                timeout=60.0,
            )
        else:
            raise ValueError(f"Unsupported LLM provider for CorrectnessEvaluator: '{provider}'")

        prompt = _CORRECTNESS_PROMPT.format(
            query=query,
            reference_answer=reference_answer,
            generated_answer=generated_answer,
        )

        extra_body = {}
        if "deepseek" in str(getattr(client, "base_url", "")).lower():
            extra_body["thinking"] = {"type": "disabled"}

        async def _call():
            resp = await client.chat.completions.create(
                model=llm_cfg.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=2048,
                **({"extra_body": extra_body} if extra_body else {}),
            )
            return resp.choices[0].message.content or ""

        raw = asyncio.run(_call())

        # Parse JSON from response
        return self._parse_verdict(raw)

    @staticmethod
    def _parse_verdict(raw: str) -> Dict[str, Any]:
        """Parse LLM JSON response into structured verdict."""
        default = {
            "verdict": "fail",
            "score": 0.0,
            "covered_points": [],
            "missed_points": [],
            "violated_claims": [],
        }
        if not raw or not raw.strip():
            return default

        # Try to extract JSON from response (handle markdown code blocks)
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        try:
            result = json.loads(text)
            if isinstance(result, dict):
                verdict = str(result.get("verdict", "fail")).lower()
                if verdict not in ("pass", "partial", "fail"):
                    verdict = "fail"
                score = float(result.get("score", 0.0))
                score = max(0.0, min(1.0, score))
                return {
                    "verdict": verdict,
                    "score": score,
                    "covered_points": result.get("covered_points", []),
                    "missed_points": result.get("missed_points", []),
                    "violated_claims": result.get("violated_claims", []),
                }
        except (json.JSONDecodeError, ValueError, TypeError):
            logger.warning("Failed to parse correctness verdict JSON: %s", text[:200])

        return default
