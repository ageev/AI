# About
I am going to put my notes and interesting things related to self-hosting AI solutions. 

# Hardware
My current hardware is Asus Aspire GX10 (almost full clone of Nvidia DGX Spark GB-10)

# Tips&Tricks
## allow copy-paste with no garbage
```bash
echo "printf '\e[?2004l'" >> ~/.bashrc
source ~/.bashrc
```
## mount NFS folder on NAS drive
    echo '10.0.0.2:/volume2/media	/mnt/nas/media	nfs	rw,_netdev,vers=3,noatime,x-systemd.automount 0 0' | sudo tee -a /etc/fstab

## Favorite packets
    sudo apt install nvtop # shows CPU/GPU resources
