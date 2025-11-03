import math
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime

def is_supported_image(filename: str) -> bool:
    valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.gif', '.ppm', '.pgm', '.pbm', '.webp']
    ext = "." + filename.split(".")[-1].lower()
    return ext in valid_extensions


def choose_image(dir_path: str, image_list: [str]) -> str:
    for i in range(1, len(image_list) + 1):
        print(f"{i}. {image_list[i - 1]}")
    while True:
        image_index = input("> choose an image index from the list above\n")
        try:
            image_index = int(image_index) - 1
            if 0 <= image_index < len(image_list):
                break
            else:
                print("> index out of bounds !!")
        except ValueError:
            print("> not a valid number !!")
    return dir_path + image_list[image_index]


def choose_images(original_images_dir: str, palette_images_dir: str) -> (str, str):
    images = {"original": [], "palette": []}
    for img_name in os.listdir(original_images_dir):
        if is_supported_image(img_name):
            images["original"].append(img_name)
    for img_name in os.listdir(palette_images_dir):
        if is_supported_image(img_name):
            images["palette"].append(img_name)

    # choosing the image_to_edit (original image)
    print("[original images folder]")
    original_images_path = choose_image(original_images_dir, images["original"])

    # choosing the palette_image (image we want to color match)
    print("[palette images folder]")
    palette_image_path = choose_image(palette_images_dir, images["palette"])

    return original_images_path, palette_image_path


def choose_max_dimensions(name: str = "the image") -> int:
    while True:
        max_dim = input(f"> choose max dimension for {name}\n")
        try:
            max_dim = int(max_dim)
            if max_dim > 0:
                return max_dim
            else:
                print("> dimensions must equal or greater than 1 !!")
        except ValueError:
            print("> not a valid number !!")


def fitted_image(image: np.ndarray, max_dimension: int, interpolation_method=cv.INTER_NEAREST) -> np.ndarray:
    height, width = image.shape[:2]
    scale = max(height, width) / float(max_dimension)
    if scale > 1.0:
        new_w = int(round(width / scale))
        new_h = int(round(height / scale))
        image = cv.resize(image, (new_w, new_h), interpolation=interpolation_method)
    return image


def read_image_rgb(image_path: str, background: tuple[int, int, int] | str | None = None) -> np.ndarray:
    img = cv.imread(image_path, cv.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Grayscale -> RGB
    if img.ndim == 2:
        rgb = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        return rgb
    if img.shape[2] == 1:
        rgb = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        return rgb

    # BGR -> RGB
    if img.shape[2] == 3:
        rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        return rgb

    # BGRA -> RGB
    if img.shape[2] == 4:
        bgr = img[:, :, :3]
        a = img[:, :, 3].astype(np.float32) / 255.0
        rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)

        if background is None:
            return rgb

        if isinstance(background, str):
            background = (255, 255, 255) if background.lower() == "white" else (0, 0, 0)
        bg = np.array(background, dtype=np.float32).reshape(1, 1, 3)
        out = rgb.astype(np.float32) * a[..., None] + bg * (1.0 - a[..., None])
        return np.clip(out, 0, 255).astype(np.uint8)

    raise ValueError(f"unsupported channel layout: shape={img.shape}")


def plt_show_image(image: np.array) -> None:
    plt.imshow(image)
    plt.show()


def plt_show_two_images(image1: np.array, image2: np.array) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    axs[0].imshow(image1)
    axs[1].imshow(image2)
    plt.tight_layout()
    plt.show()


def plt_show_three_images_horisontal(image1: np.array, image2: np.array, image3: np.array) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(20, 10))
    axs[0].imshow(image1)
    axs[1].imshow(image2)
    axs[2].imshow(image3)
    plt.tight_layout()
    plt.show()


def plt_show_three_images_vertical(image1: np.array, image2: np.array, image3: np.array) -> None:
    fig, axs = plt.subplots(3, 1, figsize=(20, 10))
    axs[0].imshow(image1)
    axs[1].imshow(image2)
    axs[2].imshow(image3)
    plt.tight_layout()
    plt.show()


def plt_show_four_images_square(image1: np.array, image2: np.array, image3: np.array, image4: np.array) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(20, 10))
    axs[0, 0].imshow(image1)
    axs[0, 1].imshow(image2)
    axs[1, 0].imshow(image3)
    axs[1, 1].imshow(image4)
    plt.tight_layout()
    plt.show()


def palette_square_padded(palette: np.ndarray, fill: (int, int, int) = (255, 255, 255)) -> np.ndarray:
    n = len(palette)
    s = int(math.ceil(math.sqrt(n)))            # minimal square side
    total = s * s
    out = np.full((total, 3), fill, dtype=palette.dtype)
    out[:n] = palette
    img = out.reshape(s, s, 3)
    return img


def image_to_palette(image_rgb: np.array, max_dim) -> np.ndarray:
    print(image_rgb.shape)
    pixels = image_rgb.reshape(-1, 3)  # flattens the image to a 2d array where each row is one pixel [R, G, B]
    palette = np.unique(pixels, axis=0)

    if len(palette) <= max_dim:
        return palette

    indices = np.linspace(0, len(palette) - 1, max_dim, dtype=int)
    reduced_palette = palette[indices]
    return reduced_palette

def seconds_to_minutes(seconds: float | int) -> float | int:
    return math.floor(seconds/60)

def minutes_to_hours(minutes: float | int) -> float | int:
    return math.floor(minutes/60)

def hours_to_days(hours: float | int) -> float | int:
    return math.floor(hours/24)

def estimated_time_dhms(seconds:float) -> (int, int, int, float):
    total_minutes = seconds_to_minutes(seconds)
    seconds = seconds - total_minutes * 60
    total_hours = minutes_to_hours(total_minutes)
    minutes = int(total_minutes - total_hours * 60)
    days = int(hours_to_days(total_hours))
    hours = int(total_hours - days * 24)
    return days, hours, minutes, seconds
def palletized(image_rgb: np.ndarray, palette_rgb: np.ndarray, chunk: int = 1_000) -> np.ndarray:

    image_lab_flat = cv.cvtColor(image_rgb.astype(np.uint8), cv.COLOR_RGB2LAB).reshape(-1, 3)
    palette_lab = cv.cvtColor(palette_rgb[np.newaxis, :, :].astype(np.uint8), cv.COLOR_RGB2LAB)[0]

    # we cast to uint16 to avoid uint8 wraparound on subtraction
    palette_lab_16 = palette_lab.astype(np.int16)
    image_flat_length = image_lab_flat.shape[0]
    out_index = np.empty(image_flat_length, dtype=np.int32)  # int32 for square int16 values (avoiding wraparound)

    number_of_chunks = int(image_flat_length / chunk) if image_flat_length % chunk == 0 else int(image_flat_length / chunk) + 1
    for chunk_start in range(0, image_flat_length, chunk):
        start = time.time()
        chunk_end = min(chunk_start + chunk, image_flat_length)
        print(f"chunk {int(chunk_start / chunk) + 1}/{number_of_chunks} (pixels {chunk_start}:{chunk_end})")
        block = image_lab_flat[chunk_start:chunk_end].astype(np.int16)
        diffs = block[:, None, :] - palette_lab_16[None, :, :]
        dist2 = (diffs.astype(np.int32) * diffs.astype(np.int32)).sum(axis=2)
        out_index[chunk_start:chunk_end] = dist2.argmin(axis=1)
        end = time.time()
        days_remaining, hours_remaining, minutes_remaining, seconds_remaining = estimated_time_dhms((end - start) * (number_of_chunks - int(chunk_start / chunk) + 1))
        print(f"ESTIMATED TIME REMAINING:{days_remaining}d, {hours_remaining}h, {minutes_remaining}m, {round(seconds_remaining, 2)}s" )

    out = palette_rgb[out_index].reshape(image_rgb.shape).astype(np.uint8)
    return out


def save_image(img: np.ndarray, output_dir: str) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    cv.imwrite(os.path.join(output_dir, f"{timestamp}_palletized.png"), cv.cvtColor(img, cv.COLOR_RGB2BGR))
    print("> image saved sucessfuly")

# img = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_NEAREST)
