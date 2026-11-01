"""Command line: the same pipeline as the web UI.

    paper-reader serve [--host 127.0.0.1] [--port 8000]
    paper-reader ingest URL [--appendix] [--llm auto|haiku|codex|none] [--llm-model M]
    paper-reader cleanup PAPER_ID [--llm auto|haiku|codex|none] [--equations symbols|meaning]   (re-run the clean-up)
    paper-reader estimate PAPER_ID
    paper-reader delete PAPER_ID [--yes]
    paper-reader cleanup-export PAPER_ID [--parts 4]   (LLM clean-up by an external runner)
    paper-reader cleanup-import PAPER_ID answers.json...
    paper-reader voices [--engine kokoro]
    paper-reader samples [--paper PAPER_ID [--abstract]] [--voices a,b,c|all] [--text ...] [--speed 1.0]
    paper-reader tts PAPER_ID [--engine E] [--voice V] [--model M] [--speed 1.0] [--yes]   (default: the selected voice)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import pipeline

DEFAULT_SAMPLE_VOICES = "af_heart,af_bella,am_michael,am_fenrir,bf_emma,bm_george"


def _progress(fraction: float, message: str) -> None:
    print(f"\r[{fraction * 100:5.1f}%] {message:<60}", end="", file=sys.stderr, flush=True)


def _done() -> None:
    print(file=sys.stderr)


def _llm_choice(value: str) -> str | None:
    """--llm auto|haiku|codex|none -> pipeline runner (auto = the first available one)."""
    return {"auto": pipeline.default_cleanup_runner(), "none": None}.get(value, value)


def _print_cleanup(paper) -> None:
    c = paper.cleanup
    if c and c.skipped_reason:
        print(f"  LLM clean-up skipped: {c.skipped_reason} (rule-based text is used)")
    elif c:
        tokens = f", {c.input_tokens:,} in / {c.output_tokens:,} out tokens" if c.input_tokens else ""
        print(f"  LLM clean-up ({c.model}): {c.blocks_llm} blocks, {c.blocks_rejected} rejected by the guard, "
              f"${c.cost_usd:.4f}{tokens}")
    for w in paper.warnings:
        print(f"  warning: {w}")


def cmd_ingest(args) -> None:
    paper = pipeline.ingest(args.url, include_appendix=args.appendix, llm=_llm_choice(args.llm),
                            model=args.llm_model, equations=args.equations, inline=args.inline, progress=_progress)
    _done()
    blocks = list(paper.iter_blocks())
    print(f"{paper.paper_id}: {paper.title}")
    print(f"  {len(paper.sections)} sections, {len(blocks)} blocks, {len(paper.figures)} figures/tables")
    _print_cleanup(paper)
    print(f"  reader: http://127.0.0.1:8000/reader/{paper.paper_id}  (after `paper-reader serve`)")


def cmd_cleanup(args) -> None:
    paper = pipeline.run_cleanup(pipeline.load_paper(args.paper_id), _llm_choice(args.llm), model=args.llm_model,
                                 equations=args.equations, inline=args.inline, progress=_progress)
    _done()
    print(f"{paper.paper_id}: {paper.title}")
    _print_cleanup(paper)


def _print_estimate(paper_id: str):
    est = pipeline.estimate(paper_id)
    print(f"{paper_id}: {est.characters:,} characters, {est.words:,} words, ~{est.audio_minutes:.0f} min of audio")
    if est.cleanup_model:
        kind = "actual" if est.cleanup_cost_is_actual else "estimated"
        print(f"  clean-up ({est.cleanup_model}): ${est.cleanup_cost_usd:.4f} ({kind})")
    else:
        print("  clean-up: LLM pass skipped, rule-based text only")
    for t in est.tts:
        status = "" if t.available else f"  [unavailable: {t.note}]"
        proc = f", ~{t.est_processing_minutes:.1f} min to render" if t.est_processing_minutes else ""
        print(f"  {t.label:<42} ${t.cost_usd:>7.2f}{proc}{status}")
    return est


def cmd_estimate(args) -> None:
    _print_estimate(args.paper_id)


def cmd_voices(args) -> None:
    eng = pipeline.engine(args.engine)
    ok, reason = eng.available()
    if not ok:
        sys.exit(f"{args.engine} unavailable: {reason}")
    for v in eng.voices():
        print(f"{v.id:<16} {v.label:<32} {v.language:<6} {v.gender or '':<7} {v.grade or ''}")


def cmd_samples(args) -> None:
    voices = [v.strip() for v in args.voices.split(",") if v.strip()]
    text, label = args.text, None
    if args.abstract:
        if not args.paper:
            sys.exit("--abstract needs --paper PAPER_ID")
        text, label = pipeline.abstract_text(args.paper), "abstract"
    models = [m.strip() for m in args.model.split(",")] if args.model else None
    samples = pipeline.make_samples(args.engine, voices, models=models, speed=args.speed, paper_id=args.paper,
                                    text=text, label=label, progress=_progress)
    _done()
    for s in samples:
        rtf = f"  {1 / s['rtf']:6.1f}x realtime" if s.get("rtf") else ""
        print(f"{s['model'] or '':<12} {s['voice']:<14} {s['seconds']:6.1f}s{rtf}  {s['path']}")
    index = pipeline.SAMPLES_DIR / (args.paper or "default") / (f"{args.engine}-{label}" if label else args.engine) / "index.html"
    print(f"compare: {index}  (served at /samples/{index.relative_to(pipeline.SAMPLES_DIR.resolve()).as_posix()})")


def cmd_tts(args) -> None:
    args.engine = args.engine or pipeline.selected_voice()["engine"]
    args.voice, args.model = pipeline.resolve_voice(args.engine, args.voice, args.model)
    print(f"voice: {args.engine} {args.model or ''} {args.voice or '(engine default)'}")
    est = _print_estimate(args.paper_id)
    cost = pipeline.engine_cost(est, args.engine, args.model)
    if not args.yes:
        answer = input(f"Accept ${cost:.2f} and generate audio with {args.engine}? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            sys.exit("Not accepted; nothing was generated.")
    manifest = pipeline.run_tts(args.paper_id, engine_name=args.engine, voice=args.voice, speed=args.speed,
                                model=args.model, accepted_cost_usd=cost, progress=_progress)
    _done()
    print(f"{manifest.run_id}: {manifest.duration / 60:.1f} min, {len(manifest.segments)} segments, "
          f"${manifest.cost_usd:.2f} -> data/papers/{args.paper_id}/{manifest.audio_file}")


def cmd_cleanup_export(args) -> None:
    from .cleanup import export_requests, system_prompt

    paper = pipeline.load_paper(args.paper_id)
    out = Path(args.out) if args.out else pipeline.paper_dir(args.paper_id) / "llm"
    out.mkdir(parents=True, exist_ok=True)
    prompt = system_prompt(args.equations, args.inline)
    (out / "system_prompt.md").write_text(prompt, encoding="utf-8")
    for k, group in enumerate(export_requests(paper, args.parts, args.equations, args.inline), 1):
        path = out / f"part_{k}.json"
        path.write_text(json.dumps(group, ensure_ascii=False, indent=1), encoding="utf-8")
        blocks = sum(len(r["block_ids"]) for r in group)
        print(f"{path}: {len(group)} requests, {blocks} blocks")
    print(f"{out / 'system_prompt.md'}: the instructions + worked examples ({len(prompt):,} chars, "
          f"equations read by {args.equations}, symbols in the text by {args.inline})")


def cmd_cleanup_import(args) -> None:
    from .cleanup import import_answers

    paper = pipeline.load_paper(args.paper_id)
    answers = []
    for name in args.files:
        data = json.loads(Path(name).read_text(encoding="utf-8"))
        answers += data["blocks"] if isinstance(data, dict) else data
    stats = import_answers(paper, answers, model=args.model, mode=args.equations, inline=args.inline)
    pipeline.save_paper(paper)
    pipeline.estimate(paper.paper_id, paper=paper)
    print(f"{paper.paper_id}: {stats.blocks_llm} blocks from {args.model}, "
          f"{stats.blocks_rejected} rejected by the guard (filter text kept)")
    for w in paper.warnings:
        if w.startswith("clean-up:"):
            print(f"  {w}")


def cmd_references(args) -> None:
    ids = args.paper_ids or [p["paper_id"] for p in pipeline.list_papers() if p.get("source", "arxiv-html") == "arxiv-html"]
    for paper_id in ids:
        try:
            result = pipeline.add_references(paper_id)
        except (ValueError, OSError) as exc:
            print(f"{paper_id}: skipped ({exc})")
            continue
        print(f"{paper_id}: {result['references']} references, {result['linked']} blocks/captions linked"
              + (f", {result['skipped']} changed since (left as they are)" if result["skipped"] else ""))


def cmd_delete(args) -> None:
    paper = pipeline.load_paper(args.paper_id)
    if not args.yes:
        answer = input(f"Delete {paper.paper_id} ({paper.title}) with its figures and narrations? [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            sys.exit("Not deleted.")
    result = pipeline.delete_paper(args.paper_id)
    print(f"Deleted {', '.join(result['deleted'])} ({result['freed_bytes'] / 1e6:.1f} MB)")


def cmd_serve(args) -> None:
    from .server import serve

    serve(args.host, args.port, ssl_certfile=args.ssl_certfile, ssl_keyfile=args.ssl_keyfile)


def _add_llm_args(p) -> None:
    p.add_argument("--llm", default="auto", choices=["auto", *pipeline.CLEANUP_RUNNERS, "none"],
                   help="LLM clean-up after the rule filter: haiku (Anthropic API), codex (Codex CLI, "
                        "gpt-6-luna), none; auto = the first available")
    p.add_argument("--llm-model", help="override the runner's model, e.g. gpt-5.6-luna for codex")
    _add_equations_arg(p)


def _add_equations_arg(p) -> None:
    p.add_argument("--equations", default="meaning", choices=["symbols", "meaning"],
                   help='how the LLM reads maths: "meaning" (default: the reward at time step t equals the '
                        'imitation weight ..., from the paper\'s definitions) or "symbols" (r t equals omega I ...)')
    p.add_argument("--inline", default="meaning", choices=["symbols", "meaning"],
                   help='how symbols inside sentences are read: "meaning" (default: "the center-of-mass reward '
                        '$r^c_t$" -> "the center-of-mass reward") or "symbols" ("... reward r t to the c")')


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="paper-reader", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("serve", help="run the web app")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--ssl-certfile", help="HTTPS certificate (e.g. from mkcert); offline saving needs HTTPS")
    p.add_argument("--ssl-keyfile", help="HTTPS private key")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("ingest", help="fetch + parse + clean up an arXiv paper")
    p.add_argument("url")
    p.add_argument("--appendix", action="store_true", help="also read the appendix (off by default)")
    _add_llm_args(p)
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("cleanup", help="re-run the clean-up of an ingested paper (e.g. with another LLM)")
    p.add_argument("paper_id")
    _add_llm_args(p)
    p.set_defaults(func=cmd_cleanup)

    p = sub.add_parser("cleanup-export", help="write the LLM clean-up requests for an external runner")
    p.add_argument("paper_id")
    p.add_argument("--out", help="directory (default: data/papers/<id>/llm)")
    p.add_argument("--parts", type=int, default=4, help="split into this many balanced parts")
    _add_equations_arg(p)
    p.set_defaults(func=cmd_cleanup_export)

    p = sub.add_parser("cleanup-import", help="apply external clean-up answers through the guard")
    p.add_argument("paper_id")
    p.add_argument("files", nargs="+", help='JSON files: {"blocks": [{"id", "speech"}]}')
    p.add_argument("--model", default="claude-haiku (Claude Code subagent)")
    _add_equations_arg(p)
    p.set_defaults(func=cmd_cleanup_import)

    p = sub.add_parser("references", help="add the bibliography to papers ingested before it was kept "
                                           "(citation titles in the reader); clean-up and narration stay")
    p.add_argument("paper_ids", nargs="*", help="default: every paper read from arXiv HTML")
    p.set_defaults(func=cmd_references)

    p = sub.add_parser("delete", help="remove a paper and everything stored for it")
    p.add_argument("paper_id")
    p.add_argument("--yes", action="store_true", help="don't ask for confirmation")
    p.set_defaults(func=cmd_delete)

    p = sub.add_parser("estimate", help="price of clean-up and every TTS option")
    p.add_argument("paper_id")
    p.set_defaults(func=cmd_estimate)

    p = sub.add_parser("voices", help="list an engine's voices")
    p.add_argument("--engine", default="kokoro", choices=pipeline.ENGINE_NAMES)
    p.set_defaults(func=cmd_voices)

    p = sub.add_parser("samples", help="render the same passage in several voices")
    p.add_argument("--engine", default="kokoro", choices=pipeline.ENGINE_NAMES)
    p.add_argument("--voices", default=DEFAULT_SAMPLE_VOICES, help='comma-separated ids, or "all"')
    p.add_argument("--paper", help="use the start of this paper's abstract as the sample text")
    p.add_argument("--abstract", action="store_true", help="speak the paper's whole abstract (needs --paper)")
    p.add_argument("--model", help="engine model(s), comma-separated, e.g. qwen3: 0.6B,1.7B,1.7B-design")
    p.add_argument("--text")
    p.add_argument("--speed", type=float, default=1.0)
    p.set_defaults(func=cmd_samples)

    p = sub.add_parser("tts", help="voice a whole paper (asks you to accept the estimate)")
    p.add_argument("paper_id")
    p.add_argument("--engine", choices=pipeline.ENGINE_NAMES, help="default: the voice selected in the voice lab")
    p.add_argument("--voice")
    p.add_argument("--model", help="qwen3: 0.6B, 1.7B or 1.7B-design; elevenlabs: e.g. eleven_flash_v2_5")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--yes", action="store_true", help="accept the estimate without asking")
    p.set_defaults(func=cmd_tts)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
