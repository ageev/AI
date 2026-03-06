# About
I am going to put my notes and interesting things related to self-hosting AI solutions. 

# Hardware
My current hardware is Asus Aspire GX10 (almost full clone of Nvidia DGX Spark GB-10). It is 1k cheaper than Spark, has no vapor chamber and 1TB drive only.
I may update to 4TB later when Samsung PM9E1 will be generally available. 

I use Gl.Inet [Comet PoE](https://www.gl-inet.com/products/gl-rm1pe/?utm_source=website&utm_medium=menubar) as remote KVM. 

# Tips&Tricks
```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
echo "printf '\e[?2004l'" >> ~/.bashrc  #fix arrows in the terminal
source ~/.bashrc
```
## mount NFS folder on NAS drive
    echo '10.0.0.2:/volume2/media	/mnt/nas/media	nfs	rw,_netdev,vers=3,noatime,x-systemd.automount 0 0' | sudo tee -a /etc/fstab

## Packets I cant live without
    sudo apt install nvtop btop tree ncdu duf uv mc
    
## FW
Reboot into BIOS

    sudo systemctl reboot --firmware-setup

Get FW updates

    sudo fwupdmgr get-devices
    sudo fwupdmgr refresh
    sudo fwupdmgr get-updates
    sudo fwupdmgr update # this updates the firmware
    sudo fwupdmgr get-history

# LLMs
Best models so far:

1. *[vLLM](https://github.com/eugr/spark-vllm-docker)*: Abliterated GPT-OSS-120b [1](https://huggingface.co/batsclamp/Huihui-gpt-oss-120b-mxfp4-abliterated), [2](https://huggingface.co/justinjja/gpt-oss-120b-Derestricted-MXFP4)
   - abliterated
   - fast on vLLM with full context (131k)
   - gives detailed answers, similar to GPT4

2. *LM Studio*: [qwen3-vl](https://huggingface.co/huihui-ai/Huihui-Qwen3-VL-30B-A3B-Instruct-abliterated)
   - fast
   - not chatty -> good for MCPs and agentic use
   - works with pictures

Interesting models:

- [Dark Desires](https://huggingface.co/ReadyArt/Dark-Desires-12B-v1.0-GGUF)
- [CWC](https://huggingface.co/CWClabs/CWC-Mistral-Nemo-12B-V2-q4_k_m)
   - interesting medical model which provides *alternative view* on the pharma industry (e.g., a lot of argumented critics for different Big Pharma products. 

# Links
- Nvidia forum https://forums.developer.nvidia.com/c/accelerated-computing/dgx-spark-gb10/dgx-spark-gb10/721
- Reddit's localLLM community https://www.reddit.com/r/LocalLLM/ https://www.reddit.com/r/LocalLLaMA/
- Nice youtube videos https://www.youtube.com/channel/UCajiMK_CY9icRhLepS8_3ug
- Spark models benchmarks https://spark-arena.com/leaderboard
