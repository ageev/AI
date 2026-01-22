#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local MRI JPEG analyzer using google/medgemma-1.5-4b-it.

⚠️ IMPORTANT SAFETY NOTICE
This tool is for research and education only. It is NOT a medical device and
must NOT be used for diagnosis, treatment, or any clinical decision-making.
Outputs may be inaccurate or misleading. Always consult a qualified physician
or radiologist for medical advice.

Assumptions:
- Dependencies (torch, transformers, pillow, tqdm) are already installed.
- The model at MODEL_DIR is a local copy of google/medgemma-1.5-4b-it compatible
  with Hugging Face Transformers' AutoProcessor + AutoModelForCausalLM.
- MRI JPEG files follow the naming pattern: IMG-<DATASET:4 digits>-<SLICE:5 digits>.jpg

Outputs:
- JSONL at OUT_PATH with per-slice results and per-dataset summaries.
"""
import os
import re
import json
import math
import time
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any

import torch
from PIL import Image, ImageOps
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers import LogitsProcessorList, TemperatureLogitsWarper, TopPLogitsWarper


# ------------------------ Hardcoded Paths (USE YOURS!!) ---------------------
MODEL_DIR = "/home/user/medgemma-env/models/medgemma-1.5-4b-it" #UPDATE ME!
MRI_ROOT  = "/home/user/medgemma-env/data/mri/JPEG/"
OUT_PATH  = "/home/user/medgemma-env/data/output/local_medgemma_results.jsonl"
# ----------------------------------------------------------------------------

# Regex for filenames like IMG-0009-00001.jpg
FNAME_RX = re.compile(r"^IMG-(?P<dataset>\d{4})-(?P<slice>\d{5})\.jpe?g$", re.IGNORECASE)

# ------------------------------- Logging setup ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("medgemma_mri")

# ------------------------------ Data structures -----------------------------
@dataclass
class SliceResult:
    kind: str  # "slice"
    dataset_id: str
    slice_id: str
    image_path: str
    raw_text: str
    parsed: Optional[Dict[str, Any]]
    prompt_name: str = "slice_prompt_v1"

@dataclass
class DatasetSummary:
    kind: str  # "dataset_summary"
    dataset_id: str
    image_count: int
    raw_text: str
    parsed: Optional[Dict[str, Any]]
    prompt_name: str = "series_prompt_v1"

@dataclass
class GlobalSummary:
    kind: str  # "global_summary"
    dataset_count: int
    total_images: int
    datasets: List[str]
    generated_at: float

# --------------------------- Utility functions ------------------------------
def load_image_rgb(path: str) -> Image.Image:
    """Load an image and normalize orientation; convert to RGB."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        return im.convert("RGB")

def list_datasets(mri_root: str) -> Dict[str, List[Tuple[str, str]]]:
    """Scan MRI_ROOT for JPEGs and group by dataset.

    Returns: dict mapping dataset_id -> list of (slice_id, full_path) sorted by slice_id.
    """
    groups: Dict[str, List[Tuple[str, str]]] = {}
    for entry in os.scandir(mri_root):
        if not entry.is_file():
            continue
        fname = entry.name
        m = FNAME_RX.match(fname)
        if not m:
            continue
        ds = m.group("dataset")
        sl = m.group("slice")
        groups.setdefault(ds, []).append((sl, entry.path))
    # sort slices numerically
    for ds, lst in groups.items():
        lst.sort(key=lambda x: int(x[0]))
    return groups

def try_parse_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Attempt to extract the outermost JSON object from a generated text."""
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            return json.loads(candidate)
    except Exception:
        return None
    return None

# ---------------------------- Prompt templates ------------------------------

# Ask a model to generate you a proper PROMT here. Best to adjust this everytime for specific MRI sets
SLICE_PROMPT = (
    "You are a professor of radiology analysing a single MRI slice.\n"
)

SERIES_PROMPT = (
    "You are a professor of radiology analysing a SERIES of cervical‑spine MRI slices.\n"
)

# --------------------------- Model wrapper class ----------------------------

class MedGemmaLocal:
    def __init__(self, model_dir: str, dtype: Optional[torch.dtype] = None, device: Optional[str] = None):
        self.model_dir = model_dir
        if dtype is not None:
            self.dtype = dtype
        else:
            if torch.cuda.is_available():
                # Prefer bfloat16 for numerical stability on modern GPUs
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.float32

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        logger.info(f"Loading processor from {model_dir}")
        self.processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        logger.info(f"Loading model from {model_dir} (device={self.device}, dtype={self.dtype})")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self.model.to(self.device)
        self.model.eval()

        # Cache tokenizer refs
        self.tok = getattr(self.processor, "tokenizer", None)

        # Ensure special tokens are set on config if available
        if self.tok is not None:
            if getattr(self.model.config, "eos_token_id", None) is None and getattr(self.tok, "eos_token_id", None) is not None:
                self.model.config.eos_token_id = self.tok.eos_token_id
            if getattr(self.model.config, "pad_token_id", None) is None:
                pad_id = getattr(self.tok, "pad_token_id", None)
                if pad_id is None:
                    pad_id = getattr(self.tok, "eos_token_id", None)
                if pad_id is not None:
                    self.model.config.pad_token_id = pad_id

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        images: List[Image.Image],
        max_new_tokens: int = 384,
        temperature: float = 0.0,   # default: greedy (safer)
        top_p: float = 0.95,
    ) -> str:
        imgs = images if isinstance(images, list) else [images]

        # Build chat text with the correct number of image tokens
        if hasattr(self.processor, "apply_chat_template"):
            content = [{"type": "image"} for _ in range(len(imgs))]
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]
            chat_text = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            inputs = self.processor(text=chat_text, images=imgs, return_tensors="pt", padding=True).to(self.device)
        else:
            image_tok = "<image>"
            chat_text = (image_tok + "\n") * len(imgs) + prompt
            inputs = self.processor(text=chat_text, images=imgs, return_tensors="pt", padding=True).to(self.device)

        # Resolve pad/eos ids
        tok = self.tok
        pad_id = getattr(self.model.config, "pad_token_id", None)
        eos_id = getattr(self.model.config, "eos_token_id", None)
        if pad_id is None and tok is not None:
            pad_id = getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", None)
        if eos_id is None and tok is not None:
            eos_id = getattr(tok, "eos_token_id", None)
        if pad_id is None and eos_id is not None:
            pad_id = eos_id

        # Build logits processors when sampling
        do_sample = temperature is not None and temperature > 0.0
        logits_processor = None
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
        )

        if do_sample:
            safe_temp = max(temperature, 0.7)  # floor to avoid extreme scaling
            safe_top_p = min(max(top_p, 1e-3), 1.0)
            logits_processor = LogitsProcessorList([
                TemperatureLogitsWarper(safe_temp),
                TopPLogitsWarper(safe_top_p),
            ])
            gen_kwargs.update(
                do_sample=True,
                logits_processor=logits_processor,
            )
        else:
            gen_kwargs.update(do_sample=False)

        gen = self.model.generate(**inputs, **gen_kwargs)

        if tok is not None:
            out = tok.decode(gen[0], skip_special_tokens=True)
        elif hasattr(self.processor, "batch_decode"):
            out = self.processor.batch_decode(gen, skip_special_tokens=True)[0]
        else:
            out = gen[0].tolist().__repr__()
        return out

# ------------------------------- Main logic ---------------------------------
def analyze_dataset(model: MedGemmaLocal, ds_id: str, slices: List[Tuple[str, str]],
                    out_fh, per_slice: bool = True) -> DatasetSummary:
    slice_results: List[SliceResult] = []

    if per_slice:
        for sl_id, path in tqdm(slices, desc=f"Dataset {ds_id} slices", leave=False):
            try:
                img = load_image_rgb(path)
                text = model.generate(SLICE_PROMPT, [img])
                parsed = try_parse_json_from_text(text)
                sr = SliceResult(
                    kind="slice",
                    dataset_id=ds_id,
                    slice_id=sl_id,
                    image_path=path,
                    raw_text=text,
                    parsed=parsed,
                )
                out_fh.write(json.dumps(asdict(sr), ensure_ascii=False) + "\n")
                out_fh.flush()
                slice_results.append(sr)
            except Exception as e:
                logger.exception(f"Slice {ds_id}-{sl_id} failed: {e}")
                sr = SliceResult(
                    kind="slice",
                    dataset_id=ds_id,
                    slice_id=sl_id,
                    image_path=path,
                    raw_text=f"ERROR: {e}",
                    parsed=None,
                )
                out_fh.write(json.dumps(asdict(sr), ensure_ascii=False) + "\n")
                out_fh.flush()
                slice_results.append(sr)

    # Series-level summary: try to feed a subset (avoid OOM by sampling up to N images)
    MAX_SERIES_IMAGES = 24  # limit to a reasonable number to avoid GPU OOM
    sample_paths = [p for _, p in slices]
    if len(sample_paths) > MAX_SERIES_IMAGES:
        # Uniform sampling across the series
        idxs = [math.floor(i * (len(sample_paths) - 1) / (MAX_SERIES_IMAGES - 1)) for i in range(MAX_SERIES_IMAGES)]
        sample_paths = [sample_paths[i] for i in idxs]

    images = [load_image_rgb(p) for p in sample_paths]
    series_text = model.generate(SERIES_PROMPT, images, max_new_tokens=640, temperature=0.2)
    series_parsed = try_parse_json_from_text(series_text)

    ds_summary = DatasetSummary(
        kind="dataset_summary",
        dataset_id=ds_id,
        image_count=len(slices),
        raw_text=series_text,
        parsed=series_parsed,
    )
    out_fh.write(json.dumps(asdict(ds_summary), ensure_ascii=False) + "\n")
    out_fh.flush()

    return ds_summary

def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    start = time.time()
    logger.info("Scanning MRI root for datasets...")
    groups = list_datasets(MRI_ROOT)
    if not groups:
        logger.warning(f"No datasets found under {MRI_ROOT} matching pattern {FNAME_RX.pattern}")
        return

    logger.info(f"Found {len(groups)} dataset(s). Loading model…")
    model = MedGemmaLocal(MODEL_DIR)

    dataset_ids = sorted(groups.keys(), key=lambda x: int(x))
    global_total_images = sum(len(v) for v in groups.values())
    dataset_summaries: List[DatasetSummary] = []

    with open(OUT_PATH, "w", encoding="utf-8") as out_fh:
        # Write a header comment line for clarity (JSONL comment style with leading '#')
        header = {
            "kind": "metadata",
            "note": "Research-use only; NOT for diagnosis or treatment.",
            "model_dir": MODEL_DIR,
            "mri_root": MRI_ROOT,
            "generated_at": time.time(),
            "generator": "medgemma_mri_analysis.py",
        }
        out_fh.write(json.dumps(header, ensure_ascii=False) + "\n")
        out_fh.flush()

        for ds_id in tqdm(dataset_ids, desc="Datasets"):
            slices = groups[ds_id]
            ds_summary = analyze_dataset(model, ds_id, slices, out_fh, per_slice=True)
            dataset_summaries.append(ds_summary)

        # Global summary line (simple aggregate metadata)
        gsum = GlobalSummary(
            kind="global_summary",
            dataset_count=len(dataset_ids),
            total_images=global_total_images,
            datasets=dataset_ids,
            generated_at=time.time(),
        )
        out_fh.write(json.dumps(asdict(gsum), ensure_ascii=False) + "\n")
        out_fh.flush()

    elapsed = time.time() - start
    logger.info(f"Done. Processed {len(dataset_ids)} dataset(s), {global_total_images} images in {elapsed/60.0:.1f} min.")
    logger.info(f"Results saved to: {OUT_PATH}")

if __name__ == "__main__":
    main()
