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
## Favorite packets
<pre><code class="bash">
sudo apt install nvtop # shows CPU/GPU resources
</code></pre>
