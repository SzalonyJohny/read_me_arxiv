# Paper Reader

Turn a scientific paper into a clean reader with synchronized narration. Paste an arXiv link or upload a PDF. You get the paper split into sections, with rendered equations and its figures, the text cleaned up for listening, and an audio narration made with a local text-to-speech model. While the audio plays, the text scrolls along on the left and the figure being discussed appears on the right. A paper can also be saved to your phone and read and played offline.

Everything runs locally on your machine. Paid services are optional, and nothing paid runs until you accept a price estimate.

## Features

- **Input**
  - arXiv links in any form: `/abs/`, `/pdf/`, `/html/`, with or without a version.
  - PDF uploads, with three extraction methods; an analysis of the PDF takes about a second and shows a time estimate for each:
    - **arXiv HTML**, when the PDF is an arXiv paper. Exact text, LaTeX maths and figures.
    - **Text layer** (PyMuPDF). About a second; equations stay as plain text.
  - **DeepSeek-OCR 2**, running on your GPU. Equations come out as LaTeX, figures are cut out of the page, and scanned PDFs work.
  - The appendix is skipped by default. References and acknowledgements are always dropped.
- **Clean-up for listening**
  - First a rule-based filter removes citation markers and expands "Fig."/"Eq."/"e.g.". It also reads units and simple maths aloud.
  - Then an optional LLM pass reads equations aloud and repairs sentences that the citation removal broke. It can use Claude Haiku 5.5 (Anthropic API) or Codex (`gpt-6-luna`, through your ChatGPT plan).
  - Read by symbols, every display equation is read in full, symbol by symbol. Labels such as `V^{target}` or `\pi_{old}` are said as words ("V target", "pi old").
  - A guard checks every answer. It rejects paraphrases, equations described instead of read, and anything that lost a symbol, and the rejected block is asked once more with the reason. If it still fails, the rule-based text is kept.
  - **Read equations by meaning** (checkbox, on by default): with an LLM clean-up, maths can be read by what the paper defines the symbols as. For example, "r t equals omega I r t I plus omega G r t G" becomes "the reward at time step t equals the imitation weight times the imitation objective plus the task weight times the task objective".
    - The meanings come from the paper's own definition sentences, which are found automatically and sent with each equation.
    - Symbols used in a standard role of the field get the field's terms when the paper uses those words: π is "the policy", V "the value function", A "the advantage function", γ "the discount factor".
    - A grounding check rejects any word the paper doesn't use, and a consistency check makes sure a symbol gets the same name in every equation.
  - **Read symbols in the text by meaning** (second checkbox, on by default): the same for maths inside sentences. For example, "the center-of-mass reward r t to the c penalizes…" becomes "the center-of-mass reward penalizes…", and a symbol on its own is read by its meaning. The sentence that defines a symbol keeps its letters, so the definition still says something.
- **Narration**
  - **Kokoro-82M**: local, about 100× realtime on a GPU.
  - **Qwen3-TTS**: local. Voices are designed from a text description, such as "Clear and precise and technical, male middle age man". Sentences are batched, so a 30-minute paper renders in about 2 minutes.
  - **ElevenLabs**: paid, used only after you accept the estimate.
  - Rendered clips are cached, so a re-run only voices the paragraphs that changed.
- **Reader**
  - Two panes, with KaTeX maths and citations you can show or hide. Tap a citation such as [14] to see the referenced title; tap it again to search for it on Google in a new tab. A tap on or near a citation never moves the narration.
  - The left pane follows the narration, with the sentence being read softly highlighted, and the right pane shows the most recently mentioned figure.
  - Click a paragraph to jump there in the audio. Click an equation to see how it is read aloud. *About this text* can show the spoken text under every paragraph.
  - Playback speed from 0.5× to 2.5× in steps of 0.1 (the speed button next to *Auto-follow*). The position slider marks where each section starts.
  - On a phone the layout is compact: the title on one small line, the settings on one line below it, and play with −15/+15 at the bottom of the screen. Tap the time to open the position slider. Citations are always shown.
  - On a phone, the button at the bottom right goes full screen, which hides Chrome's address bar. Chrome then shows its own notice on how to leave, which a page can't change. Opened from a home-screen icon, the app has no address bar and no notice.
- **Voice lab** (`/voices`)
  - Hear the start of the abstract in every Kokoro voice.
  - Design Qwen3 voices from a description, and redesign them for a new take.
  - Pick the default narration voice.
- **Offline on the phone**: a *Save offline* button stores a paper's text, figures and audio on the phone (Android Chrome) for reading and listening without a connection.

## Requirements

- Linux with Python 3.12 and [uv](https://docs.astral.sh/uv/). ffmpeg must be on your PATH.
- An NVIDIA GPU for local narration and OCR. The project is set up for CUDA 12.8 builds of PyTorch, and was tested on an RTX 5070 Ti Laptop GPU with 12 GB.
  - Kokoro also runs on a CPU, slowly.
  - DeepSeek-OCR 2 needs about 9.4 GB of GPU memory.
- Disk space:
  - about 8 GB for the Python environment;
  - models, downloaded on first use: Kokoro ~0.3 GB, Qwen3-TTS 0.6B ~2.4 GB, 1.7B ~4.3 GB each for the Base and VoiceDesign models, tokenizer ~0.7 GB, DeepSeek-OCR 2 ~6.8 GB.

## Quick start

```bash
uv sync                       # Python 3.12 env with CUDA PyTorch, Kokoro, Qwen3-TTS
uv run paper-reader serve     # http://127.0.0.1:8000
```

1. Open http://127.0.0.1:8000 and paste an arXiv link (e.g. `https://arxiv.org/abs/2601.17428`), or drop a PDF on the page.
2. The reader opens with the cleaned-up text. Click **Listen**, check the estimate, pick a voice, and click **Accept** to generate the narration.

### Optional services

| What | How to enable |
| --- | --- |
| LLM clean-up with Claude Haiku 5.5 | `export ANTHROPIC_API_KEY=…` (≈ $0.01 per paper) |
| LLM clean-up with Codex (`gpt-6-luna`) | install the [Codex CLI](https://github.com/openai/codex) and run `codex login` (uses your ChatGPT plan) |
| ElevenLabs narration | `export ELEVENLABS_API_KEY=…` (≈ $0.04–0.08 per 1,000 characters) |
| PDF OCR (DeepSeek-OCR 2) | `cd ocr_worker && uv sync`, then download the weights once with `ocr_worker/.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('unsloth/DeepSeek-OCR-2')"` |

When none of the LLM options is configured, the rule-based clean-up is used on its own. The upload page shows why the OCR option is unavailable, if it is.

## Listening on your phone (offline)

Offline saving needs a secure origin. On Android Chrome:

1. On the computer, run `uv run paper-reader serve --host 0.0.0.0`, and allow port 8000 through the firewall.
2. On the phone, open `chrome://flags/#unsafely-treat-insecure-origin-as-secure`, add `http://<computer-ip>:8000`, enable the flag and relaunch Chrome. Serving over HTTPS also works: use Tailscale (`tailscale serve`), or [mkcert](https://github.com/FiloSottile/mkcert) with `serve --ssl-certfile … --ssl-keyfile …`.
3. Open a paper whose narration is ready, tap **Save offline** and wait for "Saved ✓". Then add the app to the home screen, which makes Chrome less likely to clear the saved data.

Saved papers keep working without a connection: auto-scrolling text, figures, playback with seeking, and lock-screen controls.

## Command line

The web app and the CLI run the same pipeline:

```bash
uv run paper-reader ingest https://arxiv.org/pdf/2601.17428 [--appendix] [--llm auto|haiku|codex|none]
uv run paper-reader cleanup 2601.17428v1 --llm codex          # re-run the clean-up
uv run paper-reader cleanup 1804.02717v3 --llm codex --equations symbols --inline symbols   # read maths symbol by symbol instead
uv run paper-reader estimate 2601.17428v1                     # characters, minutes, cost per engine
uv run paper-reader references                               # add the bibliography (citation titles) to papers ingested before it was kept
uv run paper-reader voices --engine kokoro
uv run paper-reader samples --paper 2601.17428v1 --abstract --voices all          # compare every Kokoro voice
uv run paper-reader samples --engine qwen3 --model 0.6B,1.7B --voices technical_male --paper 2601.17428v1 --abstract
uv run paper-reader tts 2601.17428v1                          # asks you to accept the estimate; uses the selected voice
uv run paper-reader serve [--host 0.0.0.0] [--port 8000] [--ssl-certfile F --ssl-keyfile F]
```

`cleanup-export` and `cleanup-import` run the LLM clean-up through any external runner, for example coding-agent subagents. The runner gets the same prompt and examples, and its answers pass through the same guard.

## How it works

```
arXiv link ─► arXiv HTML parser ─┐
PDF ─► analyze ─► text layer │ DeepSeek-OCR 2 ─► assemble ─┤
                                                           ▼
                        Paper (sections, blocks, equations, figures)
                                                           ▼
             rule-based filter ─► LLM clean-up (Haiku / Codex, guarded)
                                                           ▼
                 reader page ◄──────┐   price estimate ─► accept ─► TTS
                                    └── audio.mp3 + timed manifest (sentence-level)
```

Every stage writes files the next one reads, all under `data/` (git-ignored):
- `data/papers/<id>/`: `paper.json`, `figures/`, `audio/<run>/{audio.mp3, manifest.json}`.
- `data/cache/tts/`: the per-paragraph clip cache.
- `data/voices/`: designed Qwen voices and the selected voice.
- `data/uploads/`: uploaded PDFs.

## Project layout

```
paper_reader/
  models.py            data contract: Paper / Section / Block / Figure, Estimate, AudioManifest
  arxiv.py, parse_html.py      arXiv link → LaTeXML HTML → Paper
  pdf/                 analyze.py, text_layer.py, ocr.py (DeepSeek-OCR 2 worker), assemble.py, ingest.py
  textfilter.py        rule-based speech text (LaTeX → words, citations, abbreviations, units)
  definitions.py       finds where a paper defines its symbols (for reading equations by meaning)
  cleanup.py           Claude Haiku 5.5 pass with few-shot examples and the rewrite guard
  cleanup_codex.py     the same pass through `codex exec`
  pricing.py           estimates for clean-up and every TTS option
  tts/                 kokoro_engine.py, qwen_engine.py, elevenlabs_engine.py, samples.py, audio_utils.py
  pipeline.py          ingest → clean-up → estimate; batched TTS runs with clip cache and figure cues
  voicelab.py          voice lab backend
  server.py, cli.py    FastAPI app and command line
ocr_worker/            isolated uv env + worker script for DeepSeek-OCR 2 (run as a subprocess)
web/                   static front end (vanilla JS): library, reader, voice lab, service worker
tests/                 pytest suite
plan.md                design notes, API reference, measurements
```

## Tests

```bash
uv run pytest
```

The test paper (arXiv 2601.17428) is not stored in the repository. Its PDF and HTML are downloaded from arXiv on the first run. The OCR tests need a locally generated fixture, so they are skipped until it exists; `tests/conftest.py` has the command to create it.

## Notes and limitations

- **arXiv only for links.** The arXiv path needs arXiv's HTML version of the paper. For papers without one, upload the PDF instead.
- **Equations from the text layer** are kept as plain text. Use the OCR method for maths-heavy or scanned PDFs.
- **Designed voices are random.** A Qwen voice is made from its description once, and its reference clip is stored in `data/voices/qwen/`. A new checkout therefore designs a new, different take of the same description.
- **Estimates are machine-specific.** Speeds, prices and time estimates were measured on one machine; `pricing.py` and `plan.md` say where each number comes from.
- **Third-party components:**
  - KaTeX (MIT) and Source Serif 4 (SIL OFL) are bundled in `web/vendor/` together with their licences.
  - Models are downloaded from Hugging Face under their own licences.
