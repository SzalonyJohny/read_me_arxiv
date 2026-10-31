"""Price estimates for one paper: the Claude Haiku 5.5 clean-up and every TTS engine.

Check the numbers before relying on them; each constant says where it comes from.
"""

from __future__ import annotations

from .models import EngineEstimate, Estimate, Paper
from .textfilter import apply_filter, speech_units

# ---------------------------------------------------------------- Claude Haiku 5.5 (clean-up)
# USD per million tokens, from the claude-api reference (shared/model-migration.md, "Migrating
# to Claude Haiku 5.5"), 2026-10-31. Two rate cards, chosen per request by prompt length.
# Cache reads cost 0.1x the input rate and 5-minute cache writes 1.25x (shared/prompt-caching.md).
CLEANUP_MODEL = "claude-haiku-5-5"
HAIKU_INPUT_USD_PER_MTOK = 0.10
HAIKU_OUTPUT_USD_PER_MTOK = 0.50
HAIKU_LONG_PROMPT_TOKENS = 100_000  # prompts longer than this are billed on the long card
HAIKU_LONG_INPUT_USD_PER_MTOK = 0.50
HAIKU_LONG_OUTPUT_USD_PER_MTOK = 2.50
CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER = 1.25  # 5-minute TTL

# About 4 characters per token on older Claude tokenizers; Haiku 5.5's tokenizer produces about
# 30% more tokens for the same text (model-migration.md), so 4 / 1.3 ≈ 3.1. Rounded to 3.2.
CHARS_PER_TOKEN = 3.2
REQUEST_OVERHEAD_TOKENS = 200  # message framing + the structured-output format instructions
THINKING_TOKENS_PER_REQUEST = 150  # adaptive thinking at effort "low": often none, sometimes a little
JSON_OVERHEAD_CHARS = 32  # '{"id": "b0042", "speech": ""}, ' around each answer
EQUATION_READING_CHARS = 200  # an equation's reading replaces the filter's "Equation 4."

# ---------------------------------------------------------------- narration length
# Measured: Kokoro af_heart at 1x voiced the test paper (arXiv 2601.17428, 3,935 words,
# 26,177 characters) in 30.4 minutes including the pipeline's pauses -> ~130 wpm, ~860 chars/min.
# ElevenLabs quotes "about 1,000 characters per minute", so this errs on the long side.
WORDS_PER_MINUTE = 130
CHARS_PER_MINUTE = 860

# ---------------------------------------------------------------- TTS engines
# ElevenLabs API prices (USD per 1,000 characters) and per-request character limits, verified
# against elevenlabs.io/pricing/api and the models page on 2026-10-31.
ELEVENLABS_MODELS: dict[str, dict] = {
    "eleven_flash_v2_5": {"label": "ElevenLabs Flash v2.5", "usd_per_1k_chars": 0.04, "max_chars": 40_000},
    "eleven_v4": {"label": "ElevenLabs v4", "usd_per_1k_chars": 0.08, "max_chars": 10_000},
    "eleven_v3": {"label": "ElevenLabs v3", "usd_per_1k_chars": 0.08, "max_chars": 5_000},
    "eleven_multilingual_v2": {"label": "ElevenLabs Multilingual v2", "usd_per_1k_chars": 0.08, "max_chars": 10_000},
}

# Kokoro-82M runs locally for free. Processing time = audio minutes / realtime factor + model load.
# GPU: measured on this machine at ~120x realtime warm (RTF ≈ 0.0085) plus ~9 s of one-time model
# load; 100x keeps a margin. CPU: not measured, a conservative assumption of 3x realtime.
KOKORO = {
    "label": "Kokoro-82M (local)",
    "usd_per_1k_chars": 0.0,
    "realtime_factor": {"cuda": 100.0, "cpu": 3.0},
    "load_minutes": 0.15,
}

# Qwen3-TTS runs locally for free. Speed depends on batching: sentences from many paragraphs are
# generated 32 at a time (tts/qwen_engine.py). Measured on this machine (RTX 5070 Ti Laptop, bf16):
# 0.6B 21x and 1.7B 18.7x realtime at batch 32 on 64-96 paper sentences; a whole paper has some
# partial batches, so the factors below keep a margin. CPU figures are unmeasured guesses.
# QWEN3_LOAD_MINUTES covers loading the weights and, on a voice's first use, designing its reference clip.
QWEN3 = {
    "0.6B": {"label": "Qwen3-TTS 0.6B (local, designed voice)", "realtime_factor": {"cuda": 15.0, "cpu": 0.1}},
    "1.7B": {"label": "Qwen3-TTS 1.7B (local, designed voice)", "realtime_factor": {"cuda": 13.0, "cpu": 0.05}},
    "1.7B-design": {"label": "Qwen3-TTS 1.7B VoiceDesign (local)", "realtime_factor": {"cuda": 13.0, "cpu": 0.05}},
}
QWEN3_LOAD_MINUTES = 0.5


def cleanup_cost(input_tokens: int, output_tokens: int, cache_read: int = 0, cache_write: int = 0,
                 *, long_prompt: bool = False) -> float:
    """USD for Haiku 5.5 usage. Sums of several requests are fine as long as every one of them
    is on the same rate card; long_prompt=True selects the card for prompts over 100K tokens."""
    if long_prompt:
        input_rate, output_rate = HAIKU_LONG_INPUT_USD_PER_MTOK, HAIKU_LONG_OUTPUT_USD_PER_MTOK
    else:
        input_rate, output_rate = HAIKU_INPUT_USD_PER_MTOK, HAIKU_OUTPUT_USD_PER_MTOK
    input_equivalent = (input_tokens + cache_read * CACHE_READ_MULTIPLIER
                        + cache_write * CACHE_WRITE_MULTIPLIER)
    return (input_equivalent * input_rate + output_tokens * output_rate) / 1_000_000


def estimate_cleanup_tokens(paper: Paper) -> tuple[int, int]:
    """(input_tokens, output_tokens) the clean-up of this paper will use, without a network call.

    input_tokens is full-price equivalent: the system prompt once at the cache-write rate, then a
    cache read per further request, plus every request's own message and framing."""
    from .cleanup import SYSTEM_PROMPT, plan_requests  # cleanup imports this module

    if any(b.speak and b.speech is None for b in paper.iter_blocks()):
        paper = paper.model_copy(deep=True)
        apply_filter(paper)
    requests = plan_requests(paper)
    if not requests:
        return 0, 0
    system = len(SYSTEM_PROMPT) / CHARS_PER_TOKEN
    cached_system = system * (CACHE_WRITE_MULTIPLIER + CACHE_READ_MULTIPLIER * (len(requests) - 1))
    messages = sum(len(r.user_message) for r in requests) / CHARS_PER_TOKEN
    input_tokens = cached_system + messages + REQUEST_OVERHEAD_TOKENS * len(requests)

    answer_chars = sum(
        len(block.speech or "") + JSON_OVERHEAD_CHARS + (EQUATION_READING_CHARS if block.type == "equation" else 0)
        for request in requests for block in request.blocks
    )
    output_tokens = answer_chars / CHARS_PER_TOKEN + THINKING_TOKENS_PER_REQUEST * len(requests)
    return round(input_tokens), round(output_tokens)


def kokoro_processing_minutes(audio_minutes: float, device: str | None) -> float:
    factor = KOKORO["realtime_factor"]["cuda" if device == "cuda" else "cpu"]
    return audio_minutes / factor + KOKORO["load_minutes"]


def _cleanup_part(paper: Paper) -> tuple[str | None, float, bool]:
    """(model, cost, is_actual) for the LLM clean-up."""
    stats = paper.cleanup
    if stats is None:
        input_tokens, output_tokens = estimate_cleanup_tokens(paper)
        return CLEANUP_MODEL, cleanup_cost(input_tokens, output_tokens), False
    used = stats.input_tokens + stats.output_tokens + stats.cache_read_input_tokens + stats.cache_creation_input_tokens
    if used or stats.skipped_reason is None:
        return stats.model or CLEANUP_MODEL, stats.cost_usd, True
    return None, 0.0, True  # skipped: nothing was spent and nothing will be


def build_estimate(paper: Paper, engines: list[dict]) -> Estimate:
    """Characters, minutes and cost of narrating this paper with each engine.

    engines: [{"name": "kokoro", "available": True, "reason": None, "device": "cuda"},
              {"name": "elevenlabs", "available": False, "reason": "no ELEVENLABS_API_KEY"}]"""
    units = speech_units(paper)
    characters = sum(len(u.text) for u in units)
    words = sum(len(u.text.split()) for u in units)
    audio_minutes = round(words / WORDS_PER_MINUTE, 2)
    cleanup_model, cleanup_usd, cleanup_is_actual = _cleanup_part(paper)

    tts: list[EngineEstimate] = []
    for engine in engines:
        available, note = bool(engine.get("available")), engine.get("reason")
        if engine.get("name") == "kokoro":
            tts.append(EngineEstimate(
                engine="kokoro", label=KOKORO["label"], characters=characters,
                audio_minutes=audio_minutes, cost_usd=0.0,
                est_processing_minutes=round(kokoro_processing_minutes(audio_minutes, engine.get("device")), 2),
                available=available, note=note,
            ))
        elif engine.get("name") == "qwen3":
            device = "cuda" if str(engine.get("device") or "cuda").startswith("cuda") else "cpu"
            for model_id, meta in QWEN3.items():
                tts.append(EngineEstimate(
                    engine="qwen3", model=model_id, label=meta["label"], characters=characters,
                    audio_minutes=audio_minutes, cost_usd=0.0,
                    est_processing_minutes=round(audio_minutes / meta["realtime_factor"][device]
                                                 + QWEN3_LOAD_MINUTES, 2),
                    available=available, note=note,
                ))
        elif engine.get("name") == "elevenlabs":
            for model_id, meta in ELEVENLABS_MODELS.items():
                tts.append(EngineEstimate(
                    engine="elevenlabs", model=model_id, label=meta["label"], characters=characters,
                    audio_minutes=audio_minutes,
                    cost_usd=round(characters / 1000 * meta["usd_per_1k_chars"], 4),
                    available=available, note=note,
                ))
    return Estimate(
        paper_id=paper.paper_id, characters=characters, words=words, audio_minutes=audio_minutes,
        cleanup_model=cleanup_model, cleanup_cost_usd=round(cleanup_usd, 6),
        cleanup_cost_is_actual=cleanup_is_actual, tts=tts,
    )
