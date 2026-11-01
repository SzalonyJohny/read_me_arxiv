# Paper Reader — v1 plan

31 Oct 2026 · Jan · trimmed from `plan-original.md` (the full phone/PWA design stays there for later)

## What v1 does

```
arXiv link ──► fetch arXiv HTML ──► paper.json ──► text filter (rules) ──► LLM clean-up (Haiku 5.5)
                                                                              │
             reader page (sections, figures, equations, clean text) ◄─────────┘
                                                                              │
             price estimate ──► you accept ──► TTS (Kokoro local / ElevenLabs) ──► audio.mp3 + timed manifest
                                                                              │
             reader page: text auto-scrolls on the left, latest mentioned figure on the right
```

1. **Input:** an arXiv link in any form (`/abs/`, `/pdf/`, `/html/`, with or without `vN`). Test paper: <https://arxiv.org/pdf/2601.17428>.
2. **Parse:** arXiv's HTML (LaTeXML) version → sections, paragraphs, display equations (LaTeX), figures and tables (images + captions). **Appendix is skipped by default** (`include_appendix` toggles it). References and acknowledgements are always dropped.
3. **Clean-up, two passes:**
   - **Text filter (rules, free, always on):** drop citation markers, expand `Fig./Tab./Eq./Sec.`, `e.g./i.e./w.r.t.`, speak units and simple inline maths, remove URLs/footnote marks.
   - **LLM pass (Claude Haiku 5.5, `claude-haiku-5-5`):** one call per section, with worked examples in the prompt. It only repairs what the rules broke and puts maths and display equations into words. A guard rejects outputs that rewrite too much and keeps the rule-based text. It is skipped automatically when no `ANTHROPIC_API_KEY` is set.
4. **Reader page (HTML):** a two-pane layout. On the left, the article with numbered sections, rendered equations (KaTeX) and a toggle for citations. On the right, the most recently mentioned figure or table.
5. **Price estimate → accept:** characters to speak, minutes of audio, clean-up cost, and the cost per TTS engine. Nothing paid runs until you click **Accept**. The server re-checks that the accepted price covers the estimate.
6. **TTS:**
   - **Default narration voice: Qwen3-TTS 0.6B, `technical_male`** ("Clear and precise and technical, male middle age man"), chosen in the voice lab. A selection made in the lab overrides it (`data/voices/selected.json`), and the built-in fallback is `DEFAULT_VOICE_SELECTION` in `pipeline.py`.
   - **Kokoro-82M** (local, GPU, $0): the fastest engine, good for quick tries. A **voice-sample** button renders the same passage in several Kokoro voices so you can compare them.
   - **Qwen3-TTS** (local, GPU, $0) speaks in a voice **designed from a text description**. Only the 1.7B VoiceDesign model can design a voice, so it renders a ~10 s reference clip from the description once (cached in `data/cache/qwen_voices/`). A Base model (0.6B or 1.7B) then clones that clip for every sentence, so a whole paper keeps one voice. Model `1.7B-design` uses the description directly for each sentence instead; that follows the description most closely but may drift. **Batched generation:** sentences from many paragraphs and sections are pooled, sorted by length and generated 32 at a time, and decoding runs 4 at a time to keep VRAM around 5–8 GB. That is about 13× faster than one sentence at a time: the whole test paper (29 min of audio) renders in about 2 minutes with 0.6B. Voices are presets in `tts/qwen_engine.py` (`technical_male` = "Clear and precise and technical, male middle age man").
   - **ElevenLabs** (paid, behind the accept gate) for papers you will actually listen to. It only appears when `ELEVENLABS_API_KEY` is set.
7. **Playback sync:** the audio is one MP3 plus a manifest of timed segments (sentence level). The page highlights the paragraph being read and auto-scrolls to it. The right pane switches to each figure or table when the narration mentions it; for unreferenced floats, it switches when the narration passes them. Clicking a paragraph seeks the audio to it.

## Clean-up rules (filter + LLM)

| In the paper | You hear |
| --- | --- |
| “…is robust [3, 7–9].” | “…is robust.” |
| “as shown in [12]” | “as shown in earlier work” (LLM repair) |
| “Smith et al. [7] showed” | “Smith and colleagues showed” |
| “Fig. 3(b)”, “Tab. II”, “Eq. (4)”, “Sec. 3.2” | “Figure 3 b”, “Table 2”, “Equation 4”, “Section 3.2” |
| “e.g.”, “i.e.”, “w.r.t.”, “SOTA” | “for example”, “that is”, “with respect to”, “state of the art” |
| “3.2 ms”, “1e-4”, “±0.3”, “∼40” | “3.2 milliseconds”, “ten to the minus four”, “plus or minus 0.3”, “about 40” |
| inline `$\alpha = 0.5$` | “alpha equals 0.5” |
| display equation (4) | “Equation 4: the total loss is the reconstruction loss plus beta times the KL term.” (LLM), or “Equation 4.” (rules only) |
| URLs, footnote marks, references, acknowledgements, appendix (default) | — |

The LLM must not summarise or paraphrase. It reads the paper, it doesn't retell it.

### Reading equations by meaning (option)

With `equations="meaning"` (the default: the "Read equations by meaning" checkbox, `--equations meaning`), the LLM reads defined symbols by their meaning: "the reward at time step t equals the imitation weight times the imitation objective …" instead of "r t equals omega to the I times r t to the I …". This only works with an LLM runner, and the meaning is never guessed:

1. **Definitions** (`definitions.py`). For each equation, and each paragraph containing maths, the paper's sentences that define its symbols are found deterministically and sent with it:
   - definition cues ("where", "denotes", "represents", "is the", "weights", …) and appositions ("the center-of-mass reward $r^{c}_{t}$");
   - symbols compared as (base, superscript, subscript), with rare symbols weighted above common ones (s_t, a_t);
   - coverage first, so every symbol of a long equation gets its own definition.
2. **Prompt.** A second, byte-stable prompt variant with six worked examples (weighted sum, expectation, fraction, an undefined symbol that keeps its name, indices, inline maths). The examples use their own symbols (λ₁, λ₂), so a paper's `w^e` can't inherit a meaning from an example.
3. **Grounding check.** Every content word the model adds must occur in the paper, in ordinary maths language, or among the spoken units. On DeepMimic this check caught "energy weight" for `w^e`, which that paper defines as the end-effector weight.
4. **Consistency check.** The model reports the name it gave each symbol, and the most-used name becomes canonical.
   - A block fails if it reported a different name, or if a display equation's reading doesn't contain the name's words (in any order).
   - Failing blocks are asked once more with the canonical names. What still disagrees is listed as a warning.
   - The glossary of symbol names appears in the reader's "About this text" box.

On DeepMimic (1804.02717) this gave 52 symbols, 11 blocks re-asked and 0 inconsistencies left, at about 60 s with Codex.

**Symbols inside sentences** are a separate option (`inline`, the second checkbox, `--inline`), so there are four prompt variants: equations × sentences, each by symbols or by meaning. When the words right before a symbol already name it ("the slip penalty $\kappa_t$"), the symbol is dropped. A symbol standing alone is read by its meaning. In the sentence that defines a symbol ("$u_t$ denotes …", "with $u_t$ being …") the letters stay; otherwise the definition becomes "X denotes X". On DeepMimic, "Finally, $r_t^c$ penalizes deviations in the character's center-of-mass $p_t^c$" now reads "Finally, the center-of-mass reward penalizes deviations in the character's center-of-mass".

## Price model (per paper, ~40k spoken characters ≈ 40 min)

| Step | Cost |
| --- | --- |
| arXiv HTML parse | $0 |
| Haiku 5.5 clean-up (~15k tokens in, ~15k out; $0.10 / $0.50 per MTok) | ≈ $0.01 |
| Kokoro / Qwen3-TTS (local GPU) | $0 |
| ElevenLabs Flash v2.5 / v4 Turbo ($0.04 per 1k chars) | ≈ $1.60 |
| ElevenLabs v4 / v3 / Multilingual v2 ($0.08 per 1k chars) | ≈ $3.20 |

The prices live in `paper_reader/pricing.py`. Check them before relying on them.

## Architecture

Python backend (FastAPI + a background job thread), static HTML/JS front end, and files on disk.

```
paper_reader/
  models.py        data contract: Paper/Section/Block/Figure, Estimate, AudioManifest
  arxiv.py         link → arXiv id/version, fetch HTML + figure images
  parse_html.py    LaTeXML HTML → Paper (appendix/bibliography handling)
  textfilter.py    rule-based speech text (+ LaTeX → words for simple maths)
  cleanup.py       Haiku 5.5 pass with few-shot examples + rewrite guard
  pricing.py       estimates for clean-up and every TTS engine
  tts/base.py      engine interface (Chunk, SynthResult, TTSEngine)
  tts/kokoro_engine.py, tts/qwen_engine.py, tts/elevenlabs_engine.py, tts/samples.py
  pipeline.py      ingest → filter → clean-up → estimate; TTS run → mp3 + manifest (with clip cache)
  server.py        FastAPI: API + static web + data files
  cli.py           same pipeline from the terminal
web/               index.html (link input + library), reader.html/.js/.css
data/papers/<paper_id>/  paper.json, figures/, estimate.json, audio/<run_id>/{audio.mp3,manifest.json}
data/cache/tts/    per-unit clip cache keyed by hash(engine, voice, speed, text)
data/samples/      voice samples
```

### HTTP API

| Method | Path | Body / result |
| --- | --- | --- |
| GET | `/` | link input + list of processed papers |
| GET | `/reader/{paper_id}` | reader page |
| GET | `/papers/{paper_id}/…` | files: `paper.json`, `figures/*`, `audio/<run>/*` |
| GET | `/api/config` | `{cleanup_runners: [{id, label, available, reason}], default_cleanup}` |
| GET | `/api/papers` | `[{paper_id, arxiv_id, title, audio_runs: [run_id]}]` |
| POST | `/api/papers/{paper_id}/cleanup` | `{llm: haiku/codex/none, equations: symbols/meaning, inline: symbols/meaning}` → `{job_id}`; re-runs the clean-up of an ingested paper |
| POST | `/api/papers` | `{url, include_appendix=false, llm: haiku/codex/none (default: the first available)}` → `{job_id, arxiv_id}`; the job result has `paper_id` (the version is only known after the fetch) |
| GET | `/api/jobs/{job_id}` | `{job_id, kind, status: queued/running/done/error, progress 0–1, message, result, error}` |
| GET | `/api/papers/{paper_id}/estimate` | `Estimate` |
| GET | `/api/papers/{paper_id}/audio` | `[AudioManifest]` (all runs) |
| GET | `/api/tts/engines` | `[{name, available, reason, default_voice, voices: [VoiceInfo], models: [..]}]` |
| POST | `/api/tts/samples` | `{engine, voices: [..], speed=1.0, paper_id?, text?}` → `{job_id}`; result `{samples: [{voice, url, seconds}]}` |
| POST | `/api/papers/{paper_id}/tts` | `{engine, voice, speed=1.0, model?, accepted_cost_usd}` → `{job_id}`; **409** if the estimate is above `accepted_cost_usd`; result `{manifest_url, run_id}` |
| POST | `/api/pdf` | multipart `file` → `PdfAnalysis` (pages, text/scanned pages, maths hint, arXiv id, `estimates` per method, `cleanup_seconds`) |
| POST | `/api/pdf/{upload_id}/ingest` | `{method: arxiv/text/ocr, include_appendix, llm}` → `{job_id}`; result `{paper_id, title, cleanup}` |
| GET | `/voices?paper=…` | voice lab page |
| GET | `/api/voicelab/{paper_id}` | lab text (first 2 abstract sentences), selected voice, Kokoro voices + clips, Qwen3 voices (with reference clip) + clips per model |
| POST | `/api/voicelab/{paper_id}/render` | `{engine: kokoro/qwen3, voices: ["all"] or ids, models?}` → `{job_id}` |
| POST | `/api/voicelab/{paper_id}/design` | `{description, models?, redesign_of?}` → `{job_id}`; result `{voice_id}` (a new voice, or a new take) |
| GET/POST | `/api/voices/selected` | `{engine, voice, model}`: the voice narration uses when a request names none |
| DELETE | `/api/voices/qwen/{voice_id}` | delete a designed voice (built-in presets can't be deleted) |

## PDF uploads

Drop a PDF on the library page. It is analyzed in well under a second: pages, text layer versus scanned pages, maths density, and an arXiv id from the side stamp. The page then offers three ways in, each with a time estimate, the recommended one preselected, a total that includes the LLM clean-up, and a **Continue** button.

| Method | How | Speed (8-page test paper) | Best for |
| --- | --- | --- | --- |
| arXiv HTML | the PDF is an arXiv paper, so it takes the arXiv path above | ~8 s | any arXiv paper: exact text, LaTeX maths, figures |
| Fast: text layer | PyMuPDF: column order, headers and footers dropped, de-hyphenation, headings, captions, figure crops | ~1 s | born-digital PDFs; equations stay as plain text |
| Accurate: OCR | DeepSeek-OCR 2 (`unsloth/DeepSeek-OCR-2`, local GPU, ~9.4 GB VRAM), run as a subprocess in its own env `ocr_worker/` | ~95 s | maths-heavy or scanned PDFs: equations as LaTeX |

All three produce the same `Paper` through `pdf/assemble.py`, so clean-up, TTS, the reader and the voice lab work unchanged. The code lives in `paper_reader/pdf/`: `analyze.py`, `text_layer.py`, `ocr.py`, `assemble.py`, and `ingest.py` (upload storage, estimates, ingest). Uploads are stored in `data/uploads/<sha>/`, and papers in `data/papers/pdf-<sha>/`, with `source.pdf` alongside. One-time OCR setup: `cd ocr_worker && uv sync`; the weights download from Hugging Face.

## Offline on the phone (Android Chrome)

Each paper in the reader has a **Save offline** button. It stores the paper's text, figures, maths, and the narration selected in the player (about 19 MB for the test paper) on the phone, with a progress bar. After that the paper works with no connection: auto-scrolling text, figures, playback with seeking, and lock-screen controls. A service worker (`web/sw.js`) does this, together with `web/offline.js`. KaTeX and the fonts are bundled in `web/vendor/`, so nothing loads from a CDN.

Setup:
1. Start the server on the laptop with `uv run paper-reader serve --host 0.0.0.0`.
2. Offline saving needs a secure origin. On the phone, open `chrome://flags/#unsafely-treat-insecure-origin-as-secure`, add `http://<laptop-ip>:8000`, enable the flag and relaunch Chrome. Tailscale HTTPS works too, as does mkcert with `serve --ssl-certfile/--ssl-keyfile`.
3. Open a paper whose narration is ready, tap **Save offline** and wait for "Saved ✓". Then use **Add to Home screen**, which makes it less likely that Chrome clears the stored data.

After editing files in `web/`, bump `SHELL_CACHE` in `web/sw.js` so phones pick up the new version.

## Voice lab (`/voices`)

The lab renders the first two sentences of the abstract in every voice, so voices can be compared on the paper's own words.
- **Kokoro:** render all 28 voices in one click (about 10 s), then listen and select one.
- **Qwen3-TTS:** write a description and design a voice. The 1.7B VoiceDesign model renders the reference clip, then all three models render the sentences. "Redesign" samples a new take of the same description, and the description can be edited to try variations.
- **Storage:** designed voices are kept in `data/voices/qwen/` (`voices.json` and one reference clip each). The selected voice goes in `data/voices/selected.json`, and the reader's Listen panel and `paper-reader tts` use it by default.

## Run

```bash
uv sync                                   # Python 3.12 env, CUDA torch, Kokoro
export ANTHROPIC_API_KEY=...              # optional: enables the Haiku 5.5 clean-up
export ELEVENLABS_API_KEY=...             # optional: enables ElevenLabs
uv run paper-reader serve                 # http://127.0.0.1:8000
# or headless:
uv run paper-reader ingest https://arxiv.org/pdf/2601.17428
uv run paper-reader estimate 2601.17428v1
uv run paper-reader samples --paper 2601.17428v1 --voices af_heart,am_michael,bf_emma,bm_george
uv run paper-reader samples --paper 2601.17428v1 --abstract --voices all     # every Kokoro voice + a comparison page
uv run paper-reader samples --engine qwen3 --model 0.6B,1.7B,1.7B-design --voices technical_male --paper 2601.17428v1 --abstract
uv run paper-reader tts 2601.17428v1 --engine qwen3 --model 1.7B --voice technical_male
uv run paper-reader tts 2601.17428v1 --engine kokoro --voice af_heart     # asks to accept the estimate
```

**LLM clean-up runners.** Two runners work with the same requests, prompt, examples and guard. `haiku` calls Claude Haiku 5.5 through the Anthropic API. `codex` runs the Codex CLI non-interactively (`codex exec`, model `gpt-6-luna`, low effort, read-only sandbox, `--output-schema` for the JSON), one request per section with 4 in parallel; it uses your ChatGPT plan, not API billing. On the test paper Codex took about 30 s, with about 315K input tokens, mostly Codex's own agent prompt repeated per request. `auto` picks the first available runner. Choose one in the library page's dropdown, or run `paper-reader ingest URL --llm codex` or `paper-reader cleanup PAPER_ID --llm codex [--llm-model gpt-5.6-luna]`.

**LLM clean-up via Claude Code subagents.** Claude Code subagents on Haiku can stand in for the API pass. `cleanup-export` writes the exact per-section requests and the system prompt with its worked examples. Each subagent answers one part. `cleanup-import` applies the answers through the same rewrite guard; it resets to the rule-based text first, so re-imports are safe.

```bash
uv run paper-reader cleanup-export 2601.17428v1 --parts 4      # -> data/papers/<id>/llm/{system_prompt.md, part_N.json}
# one Haiku subagent per part writes llm/answers_part_N.json = {"blocks": [{"id", "speech"}]}
uv run paper-reader cleanup-import 2601.17428v1 data/papers/2601.17428v1/llm/answers_part_*.json
uv run paper-reader tts 2601.17428v1 --engine kokoro --voice af_heart     # re-voices only the changed blocks
```

## Deferred (see `plan-original.md`)

Mistral OCR for non-arXiv PDFs, figure/table/equation cards with symbol legends, Look/No-look mixes, phone PWA and R2 hosting, bookmarks, Zotero, podcast RSS.
