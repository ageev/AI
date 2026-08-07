EN | [RU](README.ru.md)

# About
My current AI setup:
- Asus GB10 box as the main LLM carrier
- Chinese Intel N95 tiny PC where the Hermes agent lives
- Synology NAS as a docker station and storage
- Macbook as a daily driver

This setup allows me to disable internet for the LLM host (Spark). Just in case.

# TL;DR
1. Claude Code makes local AI shine:
  - sets up and manages the environment: software installs, issue debugging etc.
  - **secrets never leave the LAN**: local Vaultwarden has a dedicated Organization where the bots store their secrets (and Claude never reads them into its context!)
  - writes skills for the Hermes agent and helps him when he struggles.
  - Claude has strict guardrails set to never read anything private/confidential and to immediately report incidents (we had a few minor ones - see [PRIVACY.md](PRIVACY.md)).
2. Local AI *almost* all the time loses to frontier cloud models. Often even the best local LLMs fail basic tasks. BUT this scheme, where Claude is the untrusted brain and the local LLM is a (dumb) trusted worker, is surprisingly effective!
3. Speculative Decoding (MTP, DFlash, etc.) sucks. It boosts TPS (tokens-per-second) and works fine for plain text generation and coding, but completely fails you on many real-life agentic tasks - see the [failure log](spark/README.md#speculative-decoding-on-gb10-the-failure-log).
4. The Hermes agent works great via Telegram and lets all family members feel the power of local (trusted) AI.

## Most used LLM usecases:
- **food calorie intake monitoring** - I send a photo of my lunch to Hermes, he tracks proteins and calories; I see the current counters on my Android widget. Result: minus 5 kilos in 2 months! Sanitized sources: the skill [`skills/food-log/`](skills/food-log/) and the widget pipeline [`skills/food-widget/`](skills/food-widget/).
- **sport activities monitoring** - I collect 120+ health/sleep/sport parameters via Garmin watches into my Home Assistant. After an activity is done, I get a full report in my Telegram chat. This has already helped me improve my running significantly! Feels like a private coach.
- **private doctor** - Hermes has access to my blood tests, sleep data and DNA.


# Repo map

- [`spark/`](spark/) - the box itself: hardware, headless setup, swap vs OOM, specdec failure log, KVM/HDMI fix, launch recipes
- [`skills/`](skills/) - sanitized real skills of the local agent: [`food-log`](skills/food-log/) + the Android [`food-widget`](skills/food-widget/) pipeline
- [`benchmarks/`](benchmarks/) - model bake-offs run on the Spark
- [`converters/`](converters/) - quantization helpers
- [`usecases/`](usecases/) - end-to-end examples (MRI analysis)
- [`PRIVACY.md`](PRIVACY.md) - the data boundary between the cloud copilot and the local agent

# Hardware

Asus Aspire GX10 (DGX Spark / GB10 clone), running mostly headless now - details, OOM/swap notes and fixes in [`spark/`](spark/).

# LLMs

## Current pick (since 2026-07-18): Step-3.7-Flash

[`stepfun-ai/Step-3.7-Flash`](https://huggingface.co/stepfun-ai/Step-3.7-Flash-GGUF) - 196B MoE, 11B active, native vision. Served as GGUF IQ4_XS (~98 GiB) via llama.cpp in docker: ctx 262144, 2 parallel slots, KV cache q4_0, ~28 tok/s warm decode on the GB10.

Why it won:

- **Vision layers.** The daily agent workload is multimodal; the killer task is food-photo calorie/macro logging, and Step-3.7 reads plates and portions noticeably better than any Qwen3.6-27B variant tried here.
- **MoE fits the box.** 11B active params on a memory-bandwidth-bound machine decode ~3x faster than a 27B dense, while the 196B total keeps answer quality up.
- **Benchmarks lied, prod did not.** The bench winner among the 27B candidates failed a 2.5-hour live trial (repeated tasks, mislogged meals). Step-3.7 won on real agent work. Full story in [`benchmarks/`](benchmarks/qwen3.6-27b-spark-eval/).

Tip: keep the served-model-name stable across swaps - proxies and agent configs do not change when the backend model does.

## Plan B: Qwopus

[`Jackrong/Qwopus3.6-27B-v2-FP8`](https://huggingface.co/Jackrong/Qwopus3.6-27B-v2-FP8) (Qwen3.6-27B Opus-trace distill) via [vLLM](https://github.com/eugr/spark-vllm-docker) - the previous production model and the rollback path. Use the author's fixed chat template + `--tool-call-parser hermes`, `--gpu-memory-utilization 0.60`, no specdec (see the [failure log](spark/README.md#speculative-decoding-on-gb10-the-failure-log)).

## Earlier picks

- Qwen3.6 27b NVFP4 + DFlash
- Abliterated GPT-OSS-120b [1](https://huggingface.co/batsclamp/Huihui-gpt-oss-120b-mxfp4-abliterated), [2](https://huggingface.co/justinjja/gpt-oss-120b-Derestricted-MXFP4) - fast on vLLM with full context (131k), detailed GPT4-like answers

## Interesting models

- [Dark Desires](https://huggingface.co/ReadyArt/Dark-Desires-12B-v1.0-GGUF)
- [CWC](https://huggingface.co/CWClabs/CWC-Mistral-Nemo-12B-V2-q4_k_m)
   - interesting medical model which provides *alternative view* on the pharma industry (e.g., a lot of argumented critics for different Big Pharma products.

# Privacy boundaries

Two AIs run this setup: a cloud copilot that designs, debugs and teaches, and a local agent that touches the actual data. The rule is asymmetric on purpose: the cloud one never reads secrets, chats, health data or agent memory - it works from structure, counts and statuses only. Full contract and the tooling that enforces it: [`PRIVACY.md`](PRIVACY.md).

# Links
- Nvidia forum https://forums.developer.nvidia.com/c/accelerated-computing/dgx-spark-gb10/dgx-spark-gb10/721
- Reddit's localLLM community https://www.reddit.com/r/LocalLLM/ https://www.reddit.com/r/LocalLLaMA/
- Nice youtube videos https://www.youtube.com/channel/UCajiMK_CY9icRhLepS8_3ug
- Spark models benchmarks https://spark-arena.com/leaderboard
