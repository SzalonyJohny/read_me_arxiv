"""Shared data contract for the whole pipeline.

Every stage reads/writes these models as JSON under data/papers/<paper_id>/:

    paper.json            Paper          (parser -> textfilter -> cleanup fill in `speech`)
    figures/...           image files    (downloaded by the parser; paths are relative to the paper dir)
    estimate.json         Estimate       (pricing)
    audio/<run>/audio.mp3 + manifest.json   AudioManifest (TTS)

Display vs speech
-----------------
* `Block.html` is what the reader page shows (faithful to the paper, sanitized).
  Inline maths is written as  <span class="math" data-latex="...">\\(...\\)</span>
  so KaTeX auto-render (delimiters \\( \\) and \\[ \\]) can render it.
  Citations are kept as <span class="cite">[12]</span> so the UI can hide/show them.
  Cross-references to figures/tables are <a class="ref" data-ref="FIG_ID" href="#FIG_ID">Figure 3</a>.
* `Block.text` is the source for speech: plain text, inline maths as $LaTeX$,
  citations replaced by the marker [CITE] (numeric) or kept as author names
  for textual citations (\\citet: "Smith et al. [CITE]"), figure refs as "Figure 3".
* `Block.speech` is what the TTS engine reads (rule filter first, then optional LLM).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

BlockType = Literal["paragraph", "equation", "list_item", "figure", "heading_note"]
SectionKind = Literal["abstract", "body", "appendix"]
SpeechSource = Literal["filter", "llm", "llm_rejected"]


class Figure(BaseModel):
    """A figure or a table. Shown in the right-hand panel when it is mentioned."""

    id: str  # LaTeXML element id, e.g. "S3.F2" / "S4.T1"; also the HTML anchor
    kind: Literal["figure", "table"]
    number: str | None = None  # "2"
    label: str  # "Figure 2" / "Table 1"
    caption_html: str = ""  # sanitized HTML, maths as in Block.html
    caption_text: str = ""  # plain text, maths as $LaTeX$
    images: list[str] = Field(default_factory=list)  # paths relative to the paper dir, e.g. "figures/x1.png"
    table_html: str | None = None  # sanitized <table> for tables that are not images
    section_id: str | None = None


class Block(BaseModel):
    id: str  # "b0001", sequential in reading order across the whole paper
    type: BlockType
    html: str = ""  # display HTML (see module docstring)
    text: str = ""  # speech source (see module docstring)
    latex: str | None = None  # equation blocks: the LaTeX of the display equation
    eq_number: str | None = None  # equation blocks: "4" if numbered
    figure_id: str | None = None  # figure blocks: id into Paper.figures (placed where the float sits)
    fig_refs: list[str] = Field(default_factory=list)  # figure/table ids this block mentions, first-mention order
    speak: bool = True  # figure blocks are False (not read aloud in v1)
    speech: str | None = None  # final text for TTS
    speech_source: SpeechSource | None = None


class Section(BaseModel):
    """Sections are FLATTENED in reading order; nesting is expressed by `level`."""

    id: str  # LaTeXML id ("S3", "S3.SS1", "A1") or "abstract"
    number: str | None = None  # "3.1"
    title: str  # without the number
    level: int = 1  # 1 = section, 2 = subsection, 3 = subsubsection
    kind: SectionKind = "body"
    blocks: list[Block] = Field(default_factory=list)
    speech_title: str | None = None  # e.g. "Section 3.1. Policy learning." (filled by textfilter)


class CleanupStats(BaseModel):
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cost_usd: float = 0.0
    blocks_llm: int = 0
    blocks_rejected: int = 0  # LLM output rejected by the rewrite guard -> filter text kept
    skipped_reason: str | None = None  # e.g. "no ANTHROPIC_API_KEY"


class Paper(BaseModel):
    paper_id: str  # filesystem-safe id, e.g. "2601.17428v1"
    arxiv_id: str  # "2601.17428"
    version: str | None = None  # "v1"
    title: str
    authors: list[str] = Field(default_factory=list)
    source_url: str  # what the user pasted
    html_url: str | None = None  # the arXiv HTML page actually parsed
    source: Literal["arxiv-html", "pdf-text", "pdf-ocr"] = "arxiv-html"
    include_appendix: bool = False
    sections: list[Section] = Field(default_factory=list)
    figures: dict[str, Figure] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    cleanup: CleanupStats | None = None

    def iter_blocks(self):
        for s in self.sections:
            yield from s.blocks


class EngineEstimate(BaseModel):
    engine: str  # "kokoro" | "elevenlabs"
    model: str | None = None  # e.g. "eleven_flash_v2_5"
    label: str  # human label, e.g. "ElevenLabs Flash v2.5"
    characters: int
    audio_minutes: float
    cost_usd: float
    est_processing_minutes: float | None = None
    available: bool = True  # False if e.g. no API key / not installed
    note: str | None = None


class Estimate(BaseModel):
    paper_id: str
    characters: int  # total characters that will be spoken (section titles + blocks)
    words: int
    audio_minutes: float  # at ~150 words per minute
    cleanup_model: str | None = None
    cleanup_cost_usd: float  # actual (if already run) or estimated LLM clean-up cost
    cleanup_cost_is_actual: bool = False
    tts: list[EngineEstimate] = Field(default_factory=list)


class AudioSegment(BaseModel):
    """One timed piece of the narration (a sentence or a TTS chunk)."""

    unit_id: str  # Block.id, or Section.id for spoken section titles
    kind: Literal["section_title", "block"]
    start: float  # seconds from the beginning of audio.mp3
    end: float
    text: str  # the spoken text of this piece
    figure_ids: list[str] = Field(default_factory=list)  # figures/tables mentioned in THIS piece


class AudioManifest(BaseModel):
    paper_id: str
    run_id: str  # "<engine>-<voice>-<speed>", also the directory name under audio/
    engine: str
    voice: str
    model: str | None = None
    speed: float = 1.0
    audio_file: str  # relative to the paper dir, e.g. "audio/kokoro-af_heart-1.0/audio.mp3"
    duration: float
    cost_usd: float = 0.0
    segments: list[AudioSegment] = Field(default_factory=list)
