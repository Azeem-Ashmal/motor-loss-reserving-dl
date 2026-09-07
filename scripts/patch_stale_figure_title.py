"""
fig5_6_windscreen_heatmap.png predates this repo's figure-generation scripts
(no source script for it exists here), and its title had a stale hardcoded
figure number ("Figure 5.6") left over from an earlier draft's numbering,
inconsistent with its actual position in the dissertation. Reconstructing the
chart from scratch was attempted (a Mack one-step-ahead residual computation
over the calibration window) but produced a numerically unstable result at
high development ages where too few calibration cohorts exist to estimate a
reliable cross-sectional standard deviation -- worse and less trustworthy
than the original. This script instead patches only the title text on the
original image, leaving the verified heatmap content untouched.
"""
import sys
import glob

from PIL import Image, ImageDraw, ImageFont


def main(image_path: str, new_title: str) -> None:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, img.width, 55], fill="white")

    fonts = glob.glob("/usr/share/fonts/**/DejaVuSans.ttf", recursive=True)
    font = ImageFont.truetype(fonts[0], 26) if fonts else ImageFont.load_default()

    bbox = draw.textbbox((0, 0), new_title, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((img.width - tw) // 2, 12), new_title, fill="black", font=font)
    img.save(image_path)
    print("patched:", image_path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
