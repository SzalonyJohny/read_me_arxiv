"""DeepSeek-OCR 2 worker: PDF -> raw grounded markdown per page (run in this folder's own venv).

    ocr_worker/.venv/bin/python -I ocr_worker/worker.py --pdf paper.pdf --out result.json [--mode gundam|base]

Runs as a subprocess of paper_reader/pdf/ocr.py, so the GPU memory is freed when it exits (the
server keeps the TTS models on the same 12 GB GPU). It lives in its own venv because the model's
remote code needs torchvision/easydict, which the main env doesn't have.

stdout carries JSON lines only (everything else, including the model code's prints, goes to stderr):
    {"event": "start", "pages": 8}
    {"event": "loaded", "seconds": 9.1}
    {"event": "page", "page": 2, "done": 3, "pages": 8, "tokens": 2210, "eos": true}
    {"event": "done", "result": "/path/result.json", "seconds": 61.0, "peak_vram_gb": 8.4}
    {"event": "error", "message": "..."}               (then exit code 1)

result.json: {"model", "mode", "dpi", "prompt", "pages": [{"page", "width", "height", "image_size": [w, h],
              "raw", "tokens", "eos"}], "load_seconds", "ocr_seconds", "peak_vram_gb"}
The raw text keeps the grounding tags; paper_reader.pdf.ocr parses it (in the main env).
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import time
from pathlib import Path

MODEL_ID = "unsloth/DeepSeek-OCR-2"  # = deepseek-ai/DeepSeek-OCR-2 weights (same sha256), code patched for transformers>=4.5x
PROMPT = "<image>\n<|grounding|>Convert the document to markdown."  # the model card's "document" prompt
IMAGE_TOKEN_ID = 128815  # <image>
TD_TOKEN_IDS = {128821, 128822}  # <td>, </td>: legitimately repeat in tables (whitelisted like the official vLLM script)
MODES = {
    # name: (base_size, image_size, crop_mode). Visual tokens per letter page: gundam 6x144 + 256 + 1, base 257.
    "gundam": (1024, 768, True),  # the model card's default dynamic resolution: (0-6)x768 tiles + 1x1024 global view
    "base": (1024, 1024, False),  # the 1024x1024 global view only
}
MAX_POSITIONS = 8192  # the language model's max_position_embeddings


class Events:
    """JSON lines on the original stdout; fd 1 is redirected to stderr so library prints can't corrupt it."""

    def __init__(self):
        self._out = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
        os.dup2(2, 1)
        sys.stdout = sys.stderr

    def __call__(self, event: str, **data) -> None:
        self._out.write(json.dumps({"event": event, **data}) + "\n")
        self._out.flush()


def render_pages(pdf: Path, dpi: int, pages: list[int] | None):
    import pymupdf
    from PIL import Image

    out = []
    with pymupdf.open(pdf) as doc:
        for index in pages if pages is not None else range(doc.page_count):
            page = doc[index]
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            out.append({"page": index, "width": page.rect.width, "height": page.rect.height, "image": image})
    return out


def load_model(device: str):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    # A local path: transformers 4.57's tokenizer loader otherwise queries the Hub even when offline.
    path = snapshot_download(MODEL_ID, allow_patterns=["*.py", "*.json", "*.safetensors"])
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModel.from_pretrained(path, trust_remote_code=True, use_safetensors=True,
                                      _attn_implementation="sdpa", dtype=torch.bfloat16)
    model = model.eval().to(device)
    patch_vision_mask(model)
    patch_moe()
    patch_last_logits(model)
    return tokenizer, model


def patch_last_logits(model) -> None:
    """The remote forward projects every prompt position onto the 129k vocabulary in float32
    (8 pages x 1.1k tokens = 4.4 GB at prefill). Generation only reads the last position."""
    from transformers.modeling_outputs import CausalLMOutputWithPast

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, labels=None, use_cache=None, output_attentions=None,
                output_hidden_states=None, images=None, images_seq_mask=None, images_spatial_crop=None,
                return_dict=None, **kwargs):
        outputs = self.model(input_ids=input_ids, past_key_values=past_key_values, attention_mask=attention_mask,
                             position_ids=position_ids, inputs_embeds=inputs_embeds, use_cache=use_cache,
                             images=images, images_seq_mask=images_seq_mask,
                             images_spatial_crop=images_spatial_crop, return_dict=True)
        logits = self.lm_head(outputs.last_hidden_state[:, -1:, :]).float()
        return CausalLMOutputWithPast(logits=logits, past_key_values=outputs.past_key_values)

    type(model).forward = forward


def patch_vision_mask(model) -> None:
    """DeepEncoder V2's Qwen2 encoder needs a custom mask: image tokens attend to each other both ways,
    the learned query tokens attend to all image tokens and causally to earlier queries. The remote
    code installs it by overriding Qwen2Model._update_causal_mask, which transformers >= 4.5x no longer
    calls, so it silently runs plain causal attention (layout collapses into one block, then loops).
    A ready 4D mask is passed through untouched by transformers' mask builder, so build it here."""
    import torch

    inner = model.model.qwen2_model.model.model  # Qwen2Decoder2Encoder -> CustomQwen2Decoder -> CustomQwen2ModelInner
    parent_forward = super(type(inner), inner).forward.__func__  # transformers' Qwen2Model.forward

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, token_type_ids=None, use_cache=None, output_attentions=None,
                output_hidden_states=None, return_dict=None, cache_position=None):
        image = token_type_ids == 0  # [B, L]; the rest (== 1) are the causal queries
        query = ~image
        pos = torch.arange(image.shape[1], device=image.device)
        causal = pos[:, None] >= pos[None, :]
        allowed = (image[:, :, None] & image[:, None, :]) | (
            query[:, :, None] & (image[:, None, :] | (query[:, None, :] & causal)))
        if attention_mask is not None and attention_mask.dim() == 2:
            allowed = allowed & attention_mask[:, None, :].bool()
        return parent_forward(self, input_ids=input_ids, attention_mask=allowed[:, None].to(inputs_embeds.device),
                              position_ids=position_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds,
                              use_cache=use_cache, output_attentions=output_attentions,
                              output_hidden_states=output_hidden_states, return_dict=return_dict,
                              cache_position=cache_position)

    type(inner).forward = forward


def patch_moe() -> None:
    """transformers' DeepseekV2MoE.moe loops over all 64 experts and tests `num_tokens == 0` on a GPU
    tensor each time: 64 host syncs per layer per decoded token. Same maths, one sync per layer."""
    import torch
    from transformers.models.deepseek_v2 import modeling_deepseek_v2 as m

    def moe(self, hidden_states, topk_ids, topk_weight):
        counts = torch.bincount(topk_ids.view(-1), minlength=len(self.experts)).tolist()
        indices = topk_ids.view(-1).argsort()
        sorted_tokens = hidden_states[indices // topk_ids.shape[1]]
        outputs, start = [], 0
        for expert_id, n in enumerate(counts):
            if n:
                outputs.append(self.experts[expert_id](sorted_tokens[start:start + n]))
                start += n
        outs = torch.cat(outputs, dim=0) if outputs else sorted_tokens.new_empty(0)
        new_x = torch.empty_like(outs)
        new_x[indices] = outs
        return (new_x.view(*topk_ids.shape, -1).type(topk_weight.dtype).mul_(topk_weight.unsqueeze(dim=-1))
                .sum(dim=1).type(new_x.dtype))

    m.DeepseekV2MoE.moe = moe


def build_inputs(model, tokenizer, image, mode: str) -> dict:
    """One page -> token ids, image mask and pixel tensors, exactly as the model's own infer() builds them."""
    import torch
    from PIL import ImageOps

    remote = importlib.import_module(type(model).__module__)
    base_size, image_size, crop_mode = MODES[mode]
    patch_size, downsample_ratio = 16, 4
    conversation = [{"role": "<|User|>", "content": PROMPT}, {"role": "<|Assistant|>", "content": ""}]
    prompt = remote.format_messages(conversations=conversation, sft_format="plain", system_prompt="")
    transform = remote.BasicImageTransform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True)
    pad_color = tuple(int(x * 255) for x in transform.mean)
    before, after = prompt.split("<image>")
    ids = remote.text_encode(tokenizer, before, bos=False, eos=False)
    mask = [False] * len(ids)
    crops = []
    if crop_mode:
        if image.size[0] <= 768 and image.size[1] <= 768:
            crop_ratio = (1, 1)
        else:
            raw_crops, crop_ratio = remote.dynamic_preprocess(image)
        global_view = ImageOps.pad(image, (base_size, base_size), color=pad_color)
        width_crops, height_crops = crop_ratio
        if width_crops > 1 or height_crops > 1:
            crops = [transform(c).to(torch.bfloat16) for c in raw_crops]
        num_queries = math.ceil((image_size // patch_size) / downsample_ratio)
        num_queries_base = math.ceil((base_size // patch_size) / downsample_ratio)
        image_ids = [IMAGE_TOKEN_ID] * (num_queries_base * num_queries_base) + [IMAGE_TOKEN_ID]
        if width_crops > 1 or height_crops > 1:
            image_ids += [IMAGE_TOKEN_ID] * ((num_queries * width_crops) * (num_queries * height_crops))
    else:
        if image_size <= 768:
            image = image.resize((image_size, image_size))
        global_view = ImageOps.pad(image, (image_size, image_size), color=pad_color)
        width_crops = height_crops = 1
        num_queries = math.ceil((image_size // patch_size) / downsample_ratio)
        image_ids = [IMAGE_TOKEN_ID] * (num_queries * num_queries) + [IMAGE_TOKEN_ID]
    ids += image_ids
    mask += [True] * len(image_ids)
    tail = remote.text_encode(tokenizer, after, bos=False, eos=False)
    ids = [0] + ids + tail  # bos
    mask = [False] + mask + [False] * len(tail)
    return {
        "ids": ids,
        "mask": mask,
        "ori": transform(global_view).to(torch.bfloat16).unsqueeze(0),
        "crops": torch.stack(crops) if crops else torch.zeros((1, 3, base_size, base_size), dtype=torch.bfloat16),
        "spatial": [width_crops, height_crops],
    }


def make_processors(batch_rows, eos_id: int, on_finish, ngram: int, window: int):
    """The official vLLM script's guard (no repeated 20-gram inside a sliding window, <td>/</td> exempt)
    plus a stopping-criteria hook that reports each page as its sequence emits EOS."""
    import torch
    from transformers import LogitsProcessor, StoppingCriteria

    class NoRepeatNGramWindow(LogitsProcessor):
        def __call__(self, input_ids, scores):
            for row, seq in enumerate(input_ids[:, -window:].tolist()):
                if len(seq) < ngram:
                    continue
                prefix = seq[-(ngram - 1):]
                banned = {seq[i + ngram - 1] for i in range(len(seq) - ngram + 1)
                          if seq[i + ngram - 2] == prefix[-1] and seq[i:i + ngram - 1] == prefix}
                banned -= TD_TOKEN_IDS
                if banned:
                    scores[row, list(banned)] = -float("inf")
            return scores

    class ReportFinished(StoppingCriteria):
        def __init__(self):
            self.done = set()

        def __call__(self, input_ids, scores, **kwargs):
            last = input_ids[:, -1].tolist()
            for row, token in enumerate(last):
                if token == eos_id and row not in self.done:
                    self.done.add(row)
                    on_finish(batch_rows[row])
            return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

    return NoRepeatNGramWindow(), ReportFinished()


def ocr_batch(model, tokenizer, pages: list[dict], mode: str, max_new_tokens: int, ngram: int, window: int,
              on_finish) -> list[tuple[str, int, bool]]:
    """Generate several pages at once (left padded). Returns (raw text, tokens, ended_with_eos) per page."""
    import torch
    from transformers import LogitsProcessorList, StoppingCriteriaList

    inputs = [build_inputs(model, tokenizer, p["image"], mode) for p in pages]
    width = max(len(x["ids"]) for x in inputs)
    pad_id = tokenizer.pad_token_id
    ids = torch.tensor([[pad_id] * (width - len(x["ids"])) + x["ids"] for x in inputs])
    attention = torch.tensor([[0] * (width - len(x["ids"])) + [1] * len(x["ids"]) for x in inputs])
    seq_mask = torch.tensor([[False] * (width - len(x["mask"])) + x["mask"] for x in inputs])
    device = model.device
    images = [(x["crops"].to(device), x["ori"].to(device)) for x in inputs]
    spatial = torch.tensor([x["spatial"] for x in inputs], dtype=torch.long)
    eos_id = tokenizer.eos_token_id
    guard, report = make_processors(pages, eos_id, on_finish, ngram, window)
    budget = min(max_new_tokens, MAX_POSITIONS - width - 8)
    with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
        out = model.generate(
            ids.to(device), attention_mask=attention.to(device), images=images,
            images_seq_mask=seq_mask.to(device), images_spatial_crop=spatial,
            do_sample=False, temperature=None, top_p=None, top_k=None,
            eos_token_id=eos_id, pad_token_id=pad_id, max_new_tokens=budget, use_cache=True,
            logits_processor=LogitsProcessorList([guard]), stopping_criteria=StoppingCriteriaList([report]),
        )
    results = []
    for row, page in enumerate(pages):
        new = out[row, width:].tolist()
        eos = eos_id in new
        if eos:
            new = new[:new.index(eos_id)]
        while new and new[-1] == pad_id:
            new.pop()
        if not eos:
            on_finish(page)  # hit the token budget (usually a repetition loop): report it anyway
        results.append((tokenizer.decode(new, skip_special_tokens=False).strip(), len(new), eos))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="result JSON path")
    parser.add_argument("--mode", choices=sorted(MODES), default="gundam")
    parser.add_argument("--dpi", type=int, default=144, help="page render resolution (the official PDF script uses 144)")
    parser.add_argument("--batch", type=int, default=8, help="pages generated together (halved on CUDA OOM)")
    parser.add_argument("--pages", help="comma-separated 0-based pages (default: all)")
    parser.add_argument("--max-new-tokens", type=int, default=6000)
    parser.add_argument("--ngram", type=int, default=20)
    parser.add_argument("--window", type=int, default=90)
    args = parser.parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    emit = Events()
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA GPU available")
        started = time.time()
        pages = render_pages(args.pdf, args.dpi, [int(p) for p in args.pages.split(",")] if args.pages else None)
        emit("start", pages=len(pages))
        tokenizer, model = load_model("cuda")
        torch.cuda.synchronize()
        loaded = time.time()
        emit("loaded", seconds=round(loaded - started, 2), vram_gb=round(torch.cuda.memory_allocated() / 2**30, 2))
        done: list[int] = []
        tokens_done = {}

        def finished(page: dict) -> None:
            if page["page"] not in done:
                done.append(page["page"])
                emit("page", page=page["page"], done=len(done), pages=len(pages))

        results = {}
        batch, start = max(1, args.batch), 0
        while start < len(pages):
            chunk = pages[start:start + batch]
            try:
                outputs = ocr_batch(model, tokenizer, chunk, args.mode, args.max_new_tokens, args.ngram, args.window,
                                    finished)
            except torch.OutOfMemoryError:
                if batch == 1:
                    raise
                batch = max(1, batch // 2)  # e.g. the TTS models hold part of the GPU: retry smaller
                torch.cuda.empty_cache()
                print(f"CUDA out of memory: retrying with {batch} pages per batch", file=sys.stderr)
                continue
            for page, (raw, n, eos) in zip(chunk, outputs):
                results[page["page"]] = (raw, n, eos)
                tokens_done[page["page"]] = n
            start += len(chunk)
        torch.cuda.synchronize()
        ended = time.time()
        result = {
            "model": MODEL_ID, "mode": args.mode, "dpi": args.dpi, "prompt": PROMPT,
            "pages": [{"page": p["page"], "width": p["width"], "height": p["height"], "image_size": list(p["image"].size),
                       "raw": results[p["page"]][0], "tokens": results[p["page"]][1], "eos": results[p["page"]][2]}
                      for p in pages],
            "load_seconds": round(loaded - started, 2), "ocr_seconds": round(ended - loaded, 2),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "versions": {"torch": torch.__version__, "transformers": importlib.import_module("transformers").__version__},
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.out.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        tmp.replace(args.out)
        emit("done", result=str(args.out), seconds=round(ended - started, 2), ocr_seconds=result["ocr_seconds"],
             peak_vram_gb=result["peak_vram_gb"], tokens=sum(tokens_done.values()))
        return 0
    except Exception as exc:  # reported to the parent as an event; the traceback goes to stderr
        import traceback

        traceback.print_exc()
        emit("error", message=f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
