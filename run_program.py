import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
from helper_functions import *

original_dir_path = "images_to_edit/"
pallette_dir_path = "palette_images/"
output_dir = "result_images"

FULL_DIMENSIONS = 100000000000000
choose_dimensions = True



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
        palette_image = fitted_image(palette_image, FULL_DIMENSIONS, cv.INTER_CUBIC)

    max_dim = choose_max_dimensions("the palette")
    palette = image_to_palette(palette_image, max_dim)
    palletized_4 = palletized_colorspace_fast(original_image, palette, cv.COLOR_RGB2LAB, 1, 0.25, 0.25)
    # save image in output folder
    save_image(palletized_4, output_dir)
    plt_show_four_images_square(original_image, palette_square_padded(palette), palette_image, palletized_4)
    continue_ = input(f"q to quit")
    if continue_.lower() == "q" or continue_.lower() == "quit":
        break
