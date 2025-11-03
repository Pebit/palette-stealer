import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
from helper_functions import *
from datetime import datetime

original_dir_path = "images_to_edit/"
pallette_dir_path = "palette_images/"
output_dir = "result_images"

FULL_DIMENSIONS = 99999999999999
choose_dimensions = True

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

while True:
    # choosing images
    original_image_path, palette_image_path = choose_images(original_dir_path, pallette_dir_path)
    # printing chosen images
    print(original_image_path, palette_image_path)

    # reading the image as np.array types
    original_image = read_image_rgb(original_image_path)
    palette_image = read_image_rgb(palette_image_path)

    # tessting the images were read properly
    # plt_show_two_images(original_image, palette_image, original_image)

    # resizing images to fit chosen max dimensions
    if choose_dimensions:
        max_dim = choose_max_dimensions("the original image")
        original_image = fitted_image(original_image, max_dim)
        palette_image = fitted_image(palette_image, FULL_DIMENSIONS)

    max_dim = choose_max_dimensions("the palette")
    palette = image_to_palette(palette_image, max_dim)

    palletized_image = palletized(original_image, palette, 2_000)
    plt_show_four_images_square(original_image, palette_image, palette_square_padded(palette), palletized_image)

    # save image in output folder
    os.makedirs(output_dir, exist_ok=True)
    cv.imwrite(os.path.join(output_dir, f"{timestamp}_palletized.png"), cv.cvtColor(palletized_image, cv.COLOR_RGB2BGR))

    continue_ = input(f"q to quit")
    if continue_.lower() == "q" or continue_.lower() == "quit":
        break
