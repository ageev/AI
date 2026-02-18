# About
I am going to put my notes and interesting things related to self-hosting AI solutions. 

# Hardware
My current hardware is Asus Aspire GX10 (almost full clone of Nvidia DGX Spark GB-10)

# Tips&Tricks
```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
echo "printf '\e[?2004l'" >> ~/.bashrc  #fix arrows in the terminal
source ~/.bashrc
```
## mount NFS folder on NAS drive
    echo '10.0.0.2:/volume2/media	/mnt/nas/media	nfs	rw,_netdev,vers=3,noatime,x-systemd.automount 0 0' | sudo tee -a /etc/fstab

## Packets I cant live without
    sudo apt install nvtop btop tree ncdu 
    
## flash FW
Reboot into BIOS

    sudo systemctl reboot --firmware-setup


## LLMs
Best models so far:

1. *[vLLM](https://github.com/eugr/spark-vllm-docker)*: Abliterated GPT-OSS-120b (1)[https://huggingface.co/batsclamp/Huihui-gpt-oss-120b-mxfp4-abliterated], (2)[https://huggingface.co/justinjja/gpt-oss-120b-Derestricted-MXFP4]
   - abliterated
   - fast on vLLM with full context (131k)
   - gives detailed answers, similar to GPT4

2. *LM Studio*: (qwen3-vl)[https://huggingface.co/huihui-ai/Huihui-Qwen3-VL-30B-A3B-Instruct-abliterated]
   - fast
   - not chatty -> good for MCPs and agentic use
   - works with pictures
