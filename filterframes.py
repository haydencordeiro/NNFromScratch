import os
import subprocess
from PIL import Image

# Adjust this to your actual resolution
width, height = 1920, 1080  

# List and sort only existing images
frames = sorted(f for f in os.listdir("frames") if f.endswith(".png"))

# Launch FFmpeg
ffmpeg = subprocess.Popen([
    "ffmpeg", "-y",
    "-f", "image2pipe",
    "-framerate", "10",
    "-vcodec", "png",
    "-i", "-",  # Read from stdin
    "-vf", f"scale={width}:{height}",
    "-pix_fmt", "yuv420p",
    "output5.mp4"
], stdin=subprocess.PIPE)

for fname in frames:
    path = os.path.join("frames", fname)
    with Image.open(path) as img:
        img.save(ffmpeg.stdin, format="PNG")

ffmpeg.stdin.close()
ffmpeg.wait()
