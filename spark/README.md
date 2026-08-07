EN | [RU](README.ru.md)

# Spark box (GB10)

Everything about the hardware lives here. Model choices and use cases stay in the repo root.

## Hardware

Asus Aspire GX10 - almost a full clone of the NVIDIA DGX Spark (same GB10, 128 GB unified memory). 1k cheaper than the Spark, no vapor chamber, 1 TB drive only. May upgrade to 4 TB when the Samsung PM9E1 is generally available.

## Headless operation (current mode)

The box now runs headless most of the time: no desktop session, everything over SSH + docker compose. LM Studio is out of autostart, GUI is off.

- The desktop and stray apps eat unified memory that the models need.
- The whole DPMS/HDMI-sleep bug class below disappears when nothing drives the display.
- Serving needs no screen: clients talk to the OpenAI-compatible endpoint on the LAN.

The Comet KVM stays for BIOS/firmware sessions and for recovery when SSH is gone.

## Swap saves you from OOM crashes

GB10 has one unified memory pool for CPU and GPU. When it runs out you do not lose a process - you lose the box: reclaim livelock, dead SSH, power cycle (this is where the KVM earns its keep). Hit for real: vLLM at `--gpu-memory-utilization 0.85` took the whole machine down; a second model JIT-loading next to prod can do the same.

The fix that turned crashes into non-events - a big swapfile:

```bash
sudo fallocate -l 48G /swap.img
sudo chmod 600 /swap.img
sudo mkswap /swap.img
sudo swapon /swap.img
echo '/swap.img none swap sw 0 0' | sudo tee -a /etc/fstab
```

Why it works: steady-state inference stays in RAM (weights and KV cache are hot), so swap sits idle. It absorbs the transient peaks - model load and tensor conversion, a ~98 GiB GGUF warming up, compile/autotune caches, leftovers of a desktop session. Instead of an instantly dead box you get a visible slowdown in `btop`/`nvtop` and time to react.

Swap is the airbag, not the seatbelt:

- keep vLLM `--gpu-memory-utilization` at 0.50-0.60 on this box, never above 0.65;
- one model on the pool at a time (watch autostarts - see below);
- `earlyoom` is a reasonable extra belt; with 48 GB of swap and conservative util the Spark has not needed it.

## Speculative decoding on GB10: the failure log

Specdec is seductive on a memory-bound box: native MTP k=3 took a 27B dense from ~8 to 17-19 tok/s (~74% draft acceptance). Every attempt still ended badly:

- **MTP + images = crash.** With native MTP on, EngineCore died on every image request (`_merge_multimodal_embeddings` -> `cudaErrorNotPermitted`; vLLM [#41743](https://github.com/vllm-project/vllm/issues/41743), [#21797](https://github.com/vllm-project/vllm/issues/21797), NVIDIA forum thread 366660). The nasty part: text-only smoke tests pass, so it reached prod and users saw "model provider failed" on photos. Container RestartCount is the tell.
- **Specdec + structured output + thinking = HTTP 500.** xgrammar rejects the speculated `</think>` token (vLLM [#34650](https://github.com/vllm-project/vllm/issues/34650)); any `json_schema` or forced `tool_choice` with thinking on returns 500. Upstream fix stayed unmerged for months; the only workaround is disabling thinking on grammar paths client-side.
- **Generic DFlash drafter on a finetune that ships its own MTP head = pointless.** ~31% acceptance and 1.35x at best, vs 2.1x from the model's native head. Never pair an external drafter with a model that carries trained MTP tensors.

Since 2026-07-11 all specdec is off in prod. Rule: before calling specdec safe, test vision and forced `json_schema` - not just plain text decode.

## Comet KVM + DGX Spark: HDMI "No Signal" fix

Mostly moot in headless mode, kept for desktop use. I use a GL.iNet [Comet PoE](https://www.gl-inet.com/products/gl-rm1pe/) as remote KVM. The DGX Spark has a known bug where the NVIDIA GH100 display engine enters a bad state after DPMS puts the display to sleep. GPU registers start returning `0xbadf5600` errors, Xorg loses the display, HDMI output dies, and the Comet correctly reports "No HDMI signal detected". Same story if your screen does not want to wake up with the Spark.

**Diagnosis (on Comet via SSH):**
```bash
ssh root@<comet-ip>  # password = admin password from web UI
dmesg | grep 6911
# "check chipid ok" = hardware is fine
# "0xD211 is 0" = no HDMI signal from source
# "unsupported resolution" = source outputs a resolution the LT6911C chip rejects
```

**Diagnosis (on Spark via SSH):**
```bash
sudo dmesg | tail -30
# Look for: NVRM: gpuHandleSanityCheckRegReadError_GH100: Possible bad register read: regvalue: 0xbadf5600
# This confirms the GPU display engine is in a bad state
```

**Fix - disable ALL display power management layers on the Spark:**

1. GNOME GUI: Settings -> Power -> Screen Blank -> "Never"; Settings -> Privacy & Security -> Screen Lock -> disable
2. gsettings:
```bash
gsettings set org.gnome.desktop.session idle-delay 0
gsettings set org.gnome.settings-daemon.plugins.power idle-dim false
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-timeout 0
```
3. Disable DPMS at Xorg level:
```bash
sudo mkdir -p /etc/X11/xorg.conf.d
sudo tee /etc/X11/xorg.conf.d/90-disable-dpms.conf << EOF
Section "Extensions"
    Option "DPMS" "Disable"
EndSection

Section "ServerFlags"
    Option "StandbyTime" "0"
    Option "SuspendTime" "0"
    Option "OffTime" "0"
    Option "BlankTime" "0"
EndSection
EOF
```
4. Persist xset at login:
```bash
mkdir -p ~/.config/autostart
cat > ~/.config/autostart/disable-dpms.desktop << EOF
[Desktop Entry]
Type=Application
Name=Disable DPMS
Exec=bash -c "xset s off -dpms && xset dpms 0 0 0"
X-GNOME-Autostart-enabled=true
EOF
```
5. systemd-logind - edit `/etc/systemd/logind.conf`:
```
IdleAction=ignore
IdleActionSec=infinity
```
6. ```sudo reboot```

**Verify:**
```bash
export DISPLAY=:0
export XAUTHORITY=/run/user/1000/gdm/Xauthority
xset q | grep -A 5 "DPMS"
# Should show: "DPMS is Disabled"
```

## Tips & tricks

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
echo "printf '\e[?2004l'" >> ~/.bashrc  #fix arrows in the terminal
source ~/.bashrc
```

### Mount NFS folder on NAS drive

    echo '10.0.0.2:/volume2/media	/mnt/nas/media	nfs	rw,_netdev,vers=3,noatime,x-systemd.automount 0 0' | sudo tee -a /etc/fstab

### Packets I cant live without

    sudo apt install nvtop btop tree ncdu duf uv mc

### Autostart hygiene

Two mechanisms can autostart a model and they will happily double-load the same port: docker compose `restart: unless-stopped` and GNOME `~/.config/autostart/*.desktop` launchers. Keep exactly one active (compose; it also restarts on crash and does not depend on a desktop login).

### FW

Reboot into BIOS

    sudo systemctl reboot --firmware-setup

Get FW updates

    sudo fwupdmgr get-devices
    sudo fwupdmgr refresh
    sudo fwupdmgr get-updates
    sudo fwupdmgr update # this updates the firmware
    sudo fwupdmgr get-history

## Launch recipes

[`recipes/`](recipes/) - vLLM launch recipes for this box.
