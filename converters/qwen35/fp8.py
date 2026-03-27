"""
Conservative FP8 quantization for Qwen3.5-35B-A3B (VLM),
matching the Qwen team's approach for modules_to_not_convert.

Uses AutoModelForCausalLM (proven with llmcompressor), then post-processes
to rename weight keys and fix config for vLLM's ConditionalGeneration loader.

Requires:  pip install "llmcompressor[qwen]>=0.10" accelerate torch safetensors
GPU:       Any Ampere+ (A100/H100/L40S/4090/DGX Spark etc.)
"""

from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor import oneshot
from safetensors.torch import load_file, save_file
import json, shutil, glob
from pathlib import Path
from huggingface_hub import hf_hub_download

MODEL_ID = "huihui-ai/Huihui-Qwen3.5-35B-A3B-Claude-4.6-Opus-abliterated"
OUTPUT_DIR = Path("./Huihui-Qwen3.5-35B-A3B-Claude-4.6-Opus-abliterated-FP8")

# ---------------------------------------------------------------------------
# Conservative ignore list — match Qwen/Qwen3.5-35B-A3B-FP8 strategy.
# AutoModelForCausalLM uses model.layers.X naming (no language_model prefix),
# so regex patterns match accordingly.
# ---------------------------------------------------------------------------
IGNORE_PATTERNS = [
    "lm_head",
    "re:.*embed_tokens$",
    "re:.*linear_attn\\.conv1d$",
    "re:.*linear_attn\\.in_proj_a$",
    "re:.*linear_attn\\.in_proj_b$",
    "re:.*mlp\\.gate$",
    "re:.*mlp\\.shared_expert_gate$",
    # Vision encoder won't exist in CausalLM, but included for safety
    "re:.*visual\\..*",
    "re:^mtp\\..*",
]

# ---------------------------------------------------------------------------
# Load model as CausalLM (proven to work with llmcompressor).
# This strips the vision encoder and flattens the config, but we fix
# both in post-processing.
# ---------------------------------------------------------------------------
print(f"Loading {MODEL_ID} ...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

try:
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
except Exception:
    processor = None

# ---------------------------------------------------------------------------
# Quantize (FP8_DYNAMIC is data-free — no calibration needed)
# ---------------------------------------------------------------------------
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=IGNORE_PATTERNS,
)

print("Running FP8 quantization ...")
oneshot(
    model=model,
    recipe=recipe,
    output_dir=str(OUTPUT_DIR),
    tokenizer=tokenizer,
)

if processor is not None:
    processor.save_pretrained(str(OUTPUT_DIR))

# ---------------------------------------------------------------------------
# Post-processing Step 1: Rename weight keys
#
# CausalLM saves:      model.layers.X.mlp.experts.0.weight
# vLLM expects:        model.language_model.layers.X.mlp.experts.0.weight
#
# We rename all model.layers.X → model.language_model.layers.X
# and model.embed_tokens → model.language_model.embed_tokens
# to match Qwen's official ConditionalGeneration format.
# ---------------------------------------------------------------------------
print("Post-processing: renaming weight keys ...")

for sf_path in sorted(OUTPUT_DIR.glob("model*.safetensors")):
    print(f"  Processing {sf_path.name} ...")
    tensors = load_file(str(sf_path))
    renamed = {}
    for key, tensor in tensors.items():
        if key.startswith("model.layers."):
            new_key = "model.language_model." + key[len("model."):]
            renamed[new_key] = tensor
        elif key.startswith("model.embed_tokens"):
            new_key = "model.language_model." + key[len("model."):]
            renamed[new_key] = tensor
        elif key.startswith("model.norm"):
            new_key = "model.language_model." + key[len("model."):]
            renamed[new_key] = tensor
        else:
            renamed[key] = tensor
    tmp_path = sf_path.with_suffix(".tmp")
    save_file(renamed, str(tmp_path))
    tmp_path.rename(sf_path)

# Fix safetensors index if it exists (sharded models)
index_path = OUTPUT_DIR / "model.safetensors.index.json"
if index_path.exists():
    idx = json.loads(index_path.read_text())
    new_wm = {}
    for key, fname in idx.get("weight_map", {}).items():
        if key.startswith("model.layers."):
            new_key = "model.language_model." + key[len("model."):]
            new_wm[new_key] = fname
        elif key.startswith("model.embed_tokens"):
            new_key = "model.language_model." + key[len("model."):]
            new_wm[new_key] = fname
        elif key.startswith("model.norm"):
            new_key = "model.language_model." + key[len("model."):]
            new_wm[new_key] = fname
        else:
            new_wm[key] = fname
    idx["weight_map"] = new_wm
    index_path.write_text(json.dumps(idx, indent=2) + "\n")

print("  Weight keys renamed to model.language_model.layers.X format")

# ---------------------------------------------------------------------------
# Post-processing Step 2: Copy visual encoder weights from source model.
# The CausalLM loader strips the vision encoder, so we copy those tensors
# from the original BF16 model (they stay in BF16 — not quantized).
# ---------------------------------------------------------------------------
print("Post-processing: copying visual encoder weights from source ...")
from huggingface_hub import snapshot_download

src_dir = Path(snapshot_download(MODEL_ID))
visual_tensors = {}
for sf_file in sorted(src_dir.glob("model*.safetensors")):
    src_tensors = load_file(str(sf_file))
    for key, tensor in src_tensors.items():
        if key.startswith("model.visual.") or key.startswith("mtp."):
            visual_tensors[key] = tensor

if visual_tensors:
    visual_sf_path = OUTPUT_DIR / "model_visual.safetensors"
    save_file(visual_tensors, str(visual_sf_path))
    print(f"  Saved {len(visual_tensors)} visual/mtp tensors to model_visual.safetensors")

    # Update or create index
    if index_path.exists():
        idx = json.loads(index_path.read_text())
    else:
        # Single-file model: create index
        main_sf = OUTPUT_DIR / "model.safetensors"
        main_tensors = load_file(str(main_sf), device="cpu")
        idx = {
            "metadata": {"total_size": 0},
            "weight_map": {k: "model.safetensors" for k in main_tensors.keys()}
        }

    for key in visual_tensors:
        idx["weight_map"][key] = "model_visual.safetensors"
    index_path.write_text(json.dumps(idx, indent=2) + "\n")
    print("  Updated safetensors index")
else:
    print("  No visual/mtp tensors found in source (unexpected)")

# ---------------------------------------------------------------------------
# Post-processing Step 3: Fix configs
# ---------------------------------------------------------------------------
print("Post-processing: fixing configs ...")

# Copy preprocessor configs from source
for fname in ("preprocessor_config.json", "video_preprocessor_config.json"):
    dst = OUTPUT_DIR / fname
    try:
        src = hf_hub_download(MODEL_ID, fname)
        shutil.copy2(src, dst)
        print(f"  Copied {fname}")
    except Exception as e:
        print(f"  WARNING: could not copy {fname}: {e}")

# Fix tokenizer_config.json
tok_cfg_path = OUTPUT_DIR / "tokenizer_config.json"
if tok_cfg_path.exists():
    tok_cfg = json.loads(tok_cfg_path.read_text())
    changed = False
    if tok_cfg.get("tokenizer_class") != "Qwen2TokenizerFast":
        tok_cfg["tokenizer_class"] = "Qwen2TokenizerFast"
        changed = True
    if "chat_template" not in tok_cfg:
        try:
            src_path = hf_hub_download(MODEL_ID, "tokenizer_config.json")
            src_tok = json.loads(Path(src_path).read_text())
            if "chat_template" in src_tok:
                tok_cfg["chat_template"] = src_tok["chat_template"]
                changed = True
                print("  Copied chat_template into tokenizer_config.json")
        except Exception:
            pass
    if changed:
        tok_cfg_path.write_text(json.dumps(tok_cfg, indent=2, ensure_ascii=False) + "\n")
        print("  Fixed tokenizer_config.json")

# Fix config.json — restructure from CausalLM flat → ConditionalGeneration nested
cfg_path = OUTPUT_DIR / "config.json"
cfg = json.loads(cfg_path.read_text())
src_cfg_path = hf_hub_download(MODEL_ID, "config.json")
src_cfg = json.loads(Path(src_cfg_path).read_text())

# Extract quantization_config before rebuilding
qcfg = cfg.get("quantization_config", {})

# Clean up quantization_config nulls
if "config_groups" in qcfg:
    for group in qcfg["config_groups"].values():
        for section_key in ("weights", "input_activations"):
            section = group.get(section_key, {})
            for k in list(section.keys()):
                if section[k] is None:
                    del section[k]
            if section.get("observer_kwargs") == {}:
                del section["observer_kwargs"]
        if group.get("output_activations") is None:
            group.pop("output_activations", None)
for k in list(qcfg.keys()):
    if qcfg[k] is None or (isinstance(qcfg[k], dict) and not qcfg[k]):
        del qcfg[k]

# Fix ignore list: rename model.layers.X → model.language_model.layers.X
if "ignore" in qcfg:
    new_ignore = []
    for entry in qcfg["ignore"]:
        if entry.startswith("model.layers."):
            new_ignore.append("model.language_model." + entry[len("model."):])
        elif entry.startswith("model.embed_tokens"):
            new_ignore.append("model.language_model." + entry[len("model."):])
        else:
            new_ignore.append(entry)
    # Add visual entries that were excluded (not in CausalLM but should be listed)
    for i in range(27):
        for suffix in ("attn.qkv", "attn.proj", "mlp.linear_fc1", "mlp.linear_fc2"):
            visual_entry = f"model.visual.blocks.{i}.{suffix}"
            if visual_entry not in new_ignore:
                new_ignore.append(visual_entry)
    for extra in ("model.visual.merger.linear_fc1", "model.visual.merger.linear_fc2",
                   "model.visual.patch_embed.proj", "model.visual.pos_embed",
                   "mtp.fc", "mtp.layers.0.mlp.gate", "mtp.layers.0.mlp.shared_expert_gate"):
        if extra not in new_ignore:
            new_ignore.append(extra)
    qcfg["ignore"] = new_ignore

# Rebuild config from source structure + quantization
new_cfg = dict(src_cfg)  # Start from source (has proper nested structure)
new_cfg["quantization_config"] = qcfg
# Remove dev transformers_version
new_cfg.pop("transformers_version", None)

cfg_path.write_text(json.dumps(new_cfg, indent=2, ensure_ascii=False) + "\n")
print("  Fixed config.json (ConditionalGeneration + nested text_config)")

print(f"\nDone — quantized model saved to {OUTPUT_DIR}")
