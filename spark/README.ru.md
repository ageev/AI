[EN](README.md) | RU

# Коробка Spark (GB10)

Всё про железо живёт здесь. Выбор моделей и юзкейсы - в корне репозитория.

## Железо

Asus Aspire GX10 - почти полный клон NVIDIA DGX Spark (тот же GB10, 128 GB unified memory). На 1k дешевле Spark, без vapor chamber и только с 1 TB диском. Возможен апгрейд до 4 TB, когда Samsung PM9E1 станет общедоступен.

## Headless-режим (текущий)

Коробка теперь большую часть времени работает headless: без десктоп-сессии, всё по SSH + docker compose. LM Studio убран из автостарта, GUI выключен.

- Десктоп и случайные приложения съедают unified memory, которая нужна моделям.
- Весь класс багов DPMS/HDMI-сна (ниже) исчезает, когда дисплей никто не дёргает.
- Сервингу экран не нужен: клиенты ходят на OpenAI-совместимый эндпоинт по LAN.

Comet KVM остаётся для сессий BIOS/firmware и для спасения, когда SSH умер.

## Swap спасает от OOM-крашей

У GB10 один общий пул памяти на CPU и GPU. Когда он кончается, теряется не процесс - теряется коробка: reclaim-livelock, мёртвый SSH, ребут по питанию (тут-то KVM и отрабатывает свою цену). Проверено на себе: vLLM с `--gpu-memory-utilization 0.85` уложил всю машину; вторая модель, JIT-загружающаяся рядом с продом, делает то же самое.

Фикс, превративший краши в не-события, - большой swap-файл:

```bash
sudo fallocate -l 48G /swap.img
sudo chmod 600 /swap.img
sudo mkswap /swap.img
sudo swapon /swap.img
echo '/swap.img none swap sw 0 0' | sudo tee -a /etc/fstab
```

Почему работает: в устоявшемся режиме инференс сидит в RAM (веса и KV-кэш горячие), swap простаивает. Он ловит переходные пики - загрузку модели и конверсию тензоров, прогрев ~98 GiB GGUF, compile/autotune-кэши, остатки десктоп-сессии. Вместо мгновенно мёртвой коробки получаешь видимое в `btop`/`nvtop` замедление и время среагировать.

Swap - подушка безопасности, а не ремень:

- держите vLLM `--gpu-memory-utilization` на 0.50-0.60 на этой коробке, никогда выше 0.65;
- одна модель на пуле за раз (следите за автостартами - см. ниже);
- `earlyoom` - разумный дополнительный ремень; с 48 GB swap и консервативным util на Spark он пока не понадобился.

## Speculative decoding на GB10: журнал провалов

Specdec на memory-bound коробке соблазнителен: нативный MTP k=3 разгонял плотную 27B с ~8 до 17-19 tok/s (~74% принятых драфтов). Каждая попытка всё равно кончалась плохо:

- **MTP + картинки = краш.** С включённым нативным MTP EngineCore умирал на КАЖДОМ image-запросе (`_merge_multimodal_embeddings` -> `cudaErrorNotPermitted`; vLLM [#41743](https://github.com/vllm-project/vllm/issues/41743), [#21797](https://github.com/vllm-project/vllm/issues/21797), тред 366660 на форуме NVIDIA). Подлость в том, что text-only смоуки проходят - поэтому оно доехало до прода, и юзеры ловили «model provider failed» на фото. Палево - растущий RestartCount контейнера.
- **Specdec + structured output + thinking = HTTP 500.** xgrammar отвергает speculated-токен `</think>` (vLLM [#34650](https://github.com/vllm-project/vllm/issues/34650)); любой `json_schema` или форсированный `tool_choice` при включённом thinking возвращает 500. Апстрим-фикс месяцами не смержен; единственный обход - выключать thinking на grammar-путях на стороне клиента.
- **Универсальный DFlash-драфтер на файнтюне со своей MTP-головой = бессмысленно.** ~31% принятия и максимум 1.35x против 2.1x от нативной головы модели. Никогда не цепляйте внешний драфтер к модели, которая везёт обученные MTP-тензоры.

С 2026-07-11 весь specdec в проде выключен. Правило: прежде чем объявить specdec безопасным, протестируйте vision и форсированный `json_schema`, а не только плоский текстовый декод.

## Comet KVM + DGX Spark: фикс HDMI «No Signal»

В headless-режиме почти неактуально, оставлено для десктоп-использования. В роли удалённого KVM у меня GL.iNet [Comet PoE](https://www.gl-inet.com/products/gl-rm1pe/). У DGX Spark известный баг: display engine NVIDIA GH100 входит в битое состояние после того, как DPMS усыпляет дисплей. Регистры GPU начинают возвращать ошибки `0xbadf5600`, Xorg теряет дисплей, HDMI-выход умирает, и Comet честно сообщает «No HDMI signal detected». Та же история, если ваш экран не хочет просыпаться со Spark.

**Диагностика (на Comet по SSH):**
```bash
ssh root@<comet-ip>  # пароль = admin-пароль из web UI
dmesg | grep 6911
# "check chipid ok" = железо в порядке
# "0xD211 is 0" = нет HDMI-сигнала от источника
# "unsupported resolution" = источник выдаёт разрешение, которое чип LT6911C не принимает
```

**Диагностика (на Spark по SSH):**
```bash
sudo dmesg | tail -30
# Ищите: NVRM: gpuHandleSanityCheckRegReadError_GH100: Possible bad register read: regvalue: 0xbadf5600
# Это подтверждает битое состояние display engine
```

**Фикс - выключить ВСЕ слои display power management на Spark:**

1. GNOME GUI: Settings -> Power -> Screen Blank -> "Never"; Settings -> Privacy & Security -> Screen Lock -> выключить
2. gsettings:
```bash
gsettings set org.gnome.desktop.session idle-delay 0
gsettings set org.gnome.settings-daemon.plugins.power idle-dim false
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-timeout 0
```
3. Выключить DPMS на уровне Xorg:
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
4. Закрепить xset на логине:
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
5. systemd-logind - правим `/etc/systemd/logind.conf`:
```
IdleAction=ignore
IdleActionSec=infinity
```
6. ```sudo reboot```

**Проверка:**
```bash
export DISPLAY=:0
export XAUTHORITY=/run/user/1000/gdm/Xauthority
xset q | grep -A 5 "DPMS"
# Должно показать: "DPMS is Disabled"
```

## Tips & tricks

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
echo "printf '\e[?2004l'" >> ~/.bashrc  # фикс стрелок в терминале
source ~/.bashrc
```

### Смонтировать NFS-папку с NAS

    echo '10.0.0.2:/volume2/media	/mnt/nas/media	nfs	rw,_netdev,vers=3,noatime,x-systemd.automount 0 0' | sudo tee -a /etc/fstab

### Пакеты, без которых не живу

    sudo apt install nvtop btop tree ncdu duf uv mc

### Гигиена автостартов

Модель могут автостартовать два механизма, и они с радостью загрузят её дважды на один порт: docker compose `restart: unless-stopped` и GNOME-лончеры `~/.config/autostart/*.desktop`. Держите активным ровно один (compose: он ещё и рестартует после краха и не зависит от десктоп-логина).

### FW

Ребут в BIOS

    sudo systemctl reboot --firmware-setup

Обновления прошивок

    sudo fwupdmgr get-devices
    sudo fwupdmgr refresh
    sudo fwupdmgr get-updates
    sudo fwupdmgr update # это и обновляет прошивку
    sudo fwupdmgr get-history

## Рецепты запуска

[`recipes/`](recipes/) - vLLM-рецепты запуска для этой коробки.
