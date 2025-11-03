# Palette Stealer

A Python tool that extracts color palettes from images and remaps other images using those palettes.

<img width="50%" alt="image" src="https://github.com/user-attachments/assets/d059d546-78a5-4468-b632-d4103e095732" />

600x600 px image with 9999 different color samples

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


The in-program usage is probably going to change but there will be very explicit prints at runtime.

## Example Usage

<img src="https://github.com/user-attachments/assets/2db2427c-6871-49ed-9f76-e358889c1193"
       alt="discord_example"
       width="90%">
       
matching your profile picture with your banner

<img src="https://github.com/user-attachments/assets/d4ed51f7-3166-4ae3-adda-d702216468bc"
       alt="EXAMPLE1"
       width="90%">
       
matching your apps to your background image

