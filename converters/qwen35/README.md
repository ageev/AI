#convertion strategy

1. preserve vision layers
2. use conservative strategy from Qwen/Qwen3.5-35B-A3B-FP8 (e.g., do not touch important layers)
3. keep speed inference speed high is possible while keeping the quality as close as possible to model's original format

# how to use

HW: Asus GX10 (Nvidia DGX Spark clode), 128GB VRAM.

There was some dependency struggle initially with the enviroment (check the env script), so I switched to TF5.
