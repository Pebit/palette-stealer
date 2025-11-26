# Palette Stealer

A Python tool that extracts color palettes from images and remaps other images using those palettes.

<img width="65%" alt="image" src="https://github.com/user-attachments/assets/8a34c6df-f8c8-47d4-b732-3ab6949fff04" />

1,088 x 1,088 px image (1,183,744 total px) <br>
15,125 different color samples <br>
5.18 seconds 

## Features

- Extracts all unique colors from an image to build a palette
- Uniformly downsamples palettes to a desired size
- Palettizes target images using perceptual Lab color distance
- Displays results side-by-side with Matplotlib
- Handles chunked processing for large images efficiently

## Installation

Clone this repository:

```bash
git clone git@github.com:Pebit/palette-stealer.git
cd palette-stealer
```
Install dependencies:

```bash
pip install numpy opencv-python matplotlib
```

## Usage

Place your images inside:

- `images_to_edit/` - the target images
- `palette_images/` - the images to extract palettes from

Then run:

```bash
python run_program.py
```

After you run the program you're prompted to:

- choose the images by their index inside the folder
- choose the maximum size of the image (choose a number equal or bigger than the image's maximum dimension, hight or width, if you want to keep the same resolution)
- choose the maximum colors inside the palette (choose a number bigger than hight*width of the palette image if you want to use all the colors found)
- you're done, wait as the image is remapped chunk by chunk

<img width="524" height="177" alt="image" src="https://github.com/user-attachments/assets/063b69d4-cdd0-4688-9002-8964f67f8c88" />


The in-program usage might change but there will be very explicit prints at runtime to guide the user.

## Example Usage

<img src="https://github.com/user-attachments/assets/94de9471-4b21-43df-8820-8e215d901521"
       alt="discord_example"
       width="65%">
       
matching your profile picture with your banner
<img src="https://github.com/user-attachments/assets/0cacd761-c190-4c08-aac5-b60958b833b4"
       alt="EXAMPLE1"
       width="100%">
       
matching your app icons to your background image 

*the other image processing I mentioned was done on another image editing software! 

The app <b>DOES NOT</b> have the following features: desaturation control, contrast control, inverting colors. 

## Future Plans !!!
- Learn and understand more about KDTrees for huge code optimisation (100x faster for 1000px palettes - complexity from O(n) to O(log2n))
- impement dithering for better looking images
