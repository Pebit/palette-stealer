import math
import cv2 as cv
import numpy as np
import os
import time
from datetime import datetime
from scipy.spatial import cKDTree


VIDEO_EXTENSIONS = {
    '.mp4', '.avi', '.mov', '.mkv', '.webm', '.wmv',
    '.flv', '.m4v', '.mpg', '.mpeg', '.3gp', '.ts'
}

IMAGE_EXTENSIONS = {
    '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif',
    '.gif', '.ppm', '.pgm', '.pbm', '.webp'
}


def is_supported_image(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in IMAGE_EXTENSIONS


def is_video_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in VIDEO_EXTENSIONS


def is_gif_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext == '.gif'


def compute_clamped_dimensions(width: int, height: int, max_dimension: int) -> tuple[int, int]:
    """
    Clamps (width, height) down so that max(new_w, new_h) <= max_dimension.
    If max(width, height) <= max_dimension (or max_dimension <= 0), keeps (width, height).
    """
    if max_dimension <= 0:
        return max(1, int(width)), max(1, int(height))
    biggest = max(width, height)
    if biggest <= max_dimension:
        return max(1, int(width)), max(1, int(height))
    scale = biggest / float(max_dimension)
    new_w = max(1, int(round(width / scale)))
    new_h = max(1, int(round(height / scale)))
    return new_w, new_h


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
                print(">> index out of bounds !!")
        except ValueError:
            print(">> not a valid number !!")
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
                print(">> dimensions must equal or greater than 1 !!")
        except ValueError:
            print(">> not a valid number !!")


def fitted_image(image: np.ndarray, max_dimension: int, interpolation_method=cv.INTER_NEAREST) -> np.ndarray:
    height, width = image.shape[:2]
    new_w, new_h = compute_clamped_dimensions(width, height, max_dimension)
    if (new_w, new_h) != (width, height):
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

def palette_square_padded(
    palette: np.ndarray,
    fill: tuple[int, int, int] | None = None
) -> np.ndarray:
    n = len(palette)
    if n == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    if fill is None:
        # Find the darkest pixel in the palette using perceived luminance
        luminance = (
            0.299 * palette[:, 0].astype(np.float32)
            + 0.587 * palette[:, 1].astype(np.float32)
            + 0.114 * palette[:, 2].astype(np.float32)
        )
        fill = palette[int(np.argmin(luminance))]

    s = int(math.ceil(math.sqrt(n)))  # minimal square side
    total = s * s
    out = np.full((total, 3), fill, dtype=palette.dtype)
    out[:n] = palette
    img = out.reshape(s, s, 3)
    img.sort(axis=1)
    return img


def image_to_palette_deprecated(
    image_rgb: np.ndarray,
    max_dim: int,
    progress_callback=None,
    cancel_check=None,
) -> np.ndarray | None:
    pixels = image_rgb.reshape(-1, 3)
    total_pixels = pixels.shape[0]

    # Process in chunks so large images report smooth progress & allow instant cancel
    chunk_size = 100000
    num_chunks = int(np.ceil(total_pixels / chunk_size))
    accumulated_uniques = []

    start_time = time.time()
    last_ui_time = 0.0

    for chunk_start in range(0, total_pixels, chunk_size):
        if cancel_check and cancel_check():
            return None

        chunk_end = min(chunk_start + chunk_size, total_pixels)
        chunk_unique = np.unique(pixels[chunk_start:chunk_end], axis=0)
        accumulated_uniques.append(chunk_unique)

        # Periodically compact accumulated chunks to keep memory low
        if len(accumulated_uniques) >= 8:
            if cancel_check and cancel_check():
                return None
            accumulated_uniques = [np.unique(np.vstack(accumulated_uniques), axis=0)]

        now = time.time()
        if progress_callback and (now - last_ui_time >= 0.08 or chunk_end == total_pixels):
            last_ui_time = now
            fraction_done = chunk_end / total_pixels
            elapsed = now - start_time
            eta_seconds = (elapsed / fraction_done) * (1.0 - fraction_done) if fraction_done > 0 else 0.0
            pct = fraction_done * 90.0  # Reserve final 10% for final merge
            progress_callback(
                pct,
                f"Scanning pixels... (ETA: {format_eta(eta_seconds)})"
            )

    if cancel_check and cancel_check():
        return None

    if progress_callback:
        progress_callback(95.0, "Merging unique colors...")

    if len(accumulated_uniques) == 1:
        palette = accumulated_uniques[0]
    else:
        palette = np.unique(np.vstack(accumulated_uniques), axis=0)

    if cancel_check and cancel_check():
        return None

    if len(palette) > max_dim:
        indices = np.linspace(0, len(palette) - 1, max_dim, dtype=int)
        palette = palette[indices]

    if progress_callback:
        progress_callback(100.0, f"Extracted {len(palette)} colors!")

    return palette

def image_to_palette(
    image_rgb: np.ndarray,
    max_dim: int,
    progress_callback=None,
    cancel_check=None,
) -> np.ndarray | None:
    pixels = image_rgb.reshape(-1, 3)
    total_pixels = pixels.shape[0]

    if total_pixels == 0:
        return np.zeros((0, 3), dtype=np.uint8)

    chunk_size = 100000
    accumulated_codes = []
    accumulated_counts = []

    start_time = time.time()
    last_ui_time = 0.0

    for chunk_start in range(0, total_pixels, chunk_size):
        if cancel_check and cancel_check():
            return None

        chunk_end = min(chunk_start + chunk_size, total_pixels)
        chunk = pixels[chunk_start:chunk_end]

        # Pack [R, G, B] into a single 24-bit integer for fast unique counting
        packed = (
            (chunk[:, 0].astype(np.uint32) << 16)
            | (chunk[:, 1].astype(np.uint32) << 8)
            | chunk[:, 2].astype(np.uint32)
        )
        u_codes, u_counts = np.unique(packed, return_counts=True)
        accumulated_codes.append(u_codes)
        accumulated_counts.append(u_counts.astype(np.int64))

        # Periodically compact accumulated chunks to keep RAM usage low
        if len(accumulated_codes) >= 8:
            if cancel_check and cancel_check():
                return None
            all_c = np.concatenate(accumulated_codes)
            all_cnt = np.concatenate(accumulated_counts)
            merged_c, inv = np.unique(all_c, return_inverse=True)
            merged_cnt = np.bincount(inv, weights=all_cnt).astype(np.int64)
            accumulated_codes = [merged_c]
            accumulated_counts = [merged_cnt]

        now = time.time()
        if progress_callback and (now - last_ui_time >= 0.08 or chunk_end == total_pixels):
            last_ui_time = now
            fraction_done = chunk_end / total_pixels
            elapsed = now - start_time
            eta_seconds = (elapsed / fraction_done) * (1.0 - fraction_done) if fraction_done > 0 else 0.0
            pct = fraction_done * 90.0
            progress_callback(
                pct,
                f"Counting colors... (ETA: {format_eta(eta_seconds)})"
            )

    if cancel_check and cancel_check():
        return None

    if progress_callback:
        progress_callback(95.0, "Merging & weighting color frequencies...")

    # Final merge of unique colors and their total pixel counts
    if len(accumulated_codes) == 1:
        unique_codes = accumulated_codes[0]
        counts = accumulated_counts[0]
    else:
        all_c = np.concatenate(accumulated_codes)
        all_cnt = np.concatenate(accumulated_counts)
        unique_codes, inv = np.unique(all_c, return_inverse=True)
        counts = np.bincount(inv, weights=all_cnt).astype(np.int64)

    if cancel_check and cancel_check():
        return None

    # Unpack 24-bit integers back into [R, G, B] uint8 rows
    palette = np.column_stack([
        (unique_codes >> 16) & 0xFF,
        (unique_codes >> 8) & 0xFF,
        unique_codes & 0xFF,
    ]).astype(np.uint8)

    # Cull palette weighted by pixel frequency if larger than max_dim
    if len(palette) > max_dim:
        # Cap any single color's interval span to 1/max_dim of the total weight
        # so one giant background color doesn't swallow multiple palette slots
        weights = counts.astype(np.float64)
        max_slot_weight = weights.sum() / max_dim
        capped_weights = np.minimum(weights, max_slot_weight)
        cum_weights = np.cumsum(capped_weights)

        # Divide the cumulative weight distribution into max_dim equal-weight buckets
        edges = np.linspace(0.0, cum_weights[-1], max_dim + 1)
        bounds = np.searchsorted(cum_weights, edges, side="left")
        bounds[0] = 0
        bounds[-1] = len(palette)

        selected_mask = np.zeros(len(palette), dtype=bool)
        selected_indices = []

        # Pick the highest-frequency color inside each weighted bucket
        for i in range(max_dim):
            start_idx = bounds[i]
            end_idx = max(start_idx + 1, bounds[i + 1])
            end_idx = min(end_idx, len(palette))

            bucket_counts = counts[start_idx:end_idx]
            best_local = int(np.argmax(bucket_counts))
            best_idx = start_idx + best_local

            if not selected_mask[best_idx]:
                selected_mask[best_idx] = True
                selected_indices.append(best_idx)

        # If any buckets collided, fill remaining slots with the most frequent unselected colors
        if len(selected_indices) < max_dim:
            needed = max_dim - len(selected_indices)
            unselected_idx = np.flatnonzero(~selected_mask)
            top_unselected = unselected_idx[
                np.argpartition(-counts[unselected_idx], min(needed, len(unselected_idx) - 1))[:needed]
            ]
            selected_indices.extend(top_unselected.tolist())

        selected_indices = np.sort(np.array(selected_indices[:max_dim], dtype=np.intp))
        palette = palette[selected_indices]

    if progress_callback:
        progress_callback(100.0, f"Extracted {len(palette)} colors!")

    return palette


def format_eta(total_seconds: float) -> str:
    """Formats seconds into '1d 14h 32m 3.4 s', showing d/h/m only when non-zero."""
    total_seconds = max(0.0, float(total_seconds))
    days, rem = divmod(total_seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)

    parts = []
    if int(days) > 0:
        parts.append(f"{int(days)}d")
    if int(hours) > 0:
        parts.append(f"{int(hours)}h")
    if int(minutes) > 0:
        parts.append(f"{int(minutes)}m")
    parts.append(f"{seconds:.1f} s")
    return " ".join(parts)

def rgb_to_colorspace(image_rgb, colorspace: int | None = None):
    if colorspace:
        image_space = cv.cvtColor(image_rgb.astype(np.uint8), colorspace)
    else:
        image_space = image_rgb
    return image_space


def palletized_colorspace(image_rgb: np.ndarray, palette_rgb: np.ndarray, chunk: int = 1_000,
                          colorspace: int | None = None, w0: float = 1, w1: float = 1, w2: float = 1) -> np.ndarray:
    image_space = rgb_to_colorspace(image_rgb, colorspace)
    palette_space = rgb_to_colorspace(palette_rgb[np.newaxis, :, :].astype(np.uint8), colorspace)[0]

    image_space_flat = image_space.reshape(-1, 3)
    # we cast to uint16 to avoid uint8 wraparound on subtraction
    palette_space_16 = palette_space.astype(np.int16)
    image_flat_length = image_space_flat.shape[0]
    out_index = np.empty(image_flat_length, dtype=np.int32)  # int32 for square int16 values (avoiding wraparound)

    number_of_chunks = int(image_flat_length / chunk) if image_flat_length % chunk == 0 else int(image_flat_length / chunk) + 1
    for chunk_start in range(0, image_flat_length, chunk):
        start = time.time()
        chunk_end = min(chunk_start + chunk, image_flat_length)
        print(f"chunk {int(chunk_start / chunk) + 1}/{number_of_chunks} (pixels {chunk_start}:{chunk_end})")
        block = image_space_flat[chunk_start:chunk_end].astype(np.int16)
        diffs = block[:, None, :] - palette_space_16[None, :, :]
        diffs = diffs.astype(np.int32)
        dist2 = ((diffs[:, :, 0] * w0) ** 2
                 + (diffs[:, :, 1] * w1) ** 2
                 + (diffs[:, :, 2] * w2) ** 2)
        out_index[chunk_start:chunk_end] = dist2.argmin(axis=1)
        end = time.time()
        print(f"ESTIMATED TIME REMAINING:{format_eta((end - start) * (number_of_chunks - int(chunk_start / chunk) + 1))}" )

    out = palette_rgb[out_index].reshape(image_rgb.shape).astype(np.uint8)
    return out


def save_image(img: np.ndarray, output_dir: str) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    cv.imwrite(os.path.join(output_dir, f"{timestamp}_palletized.png"), cv.cvtColor(img, cv.COLOR_RGB2BGR))
    print("> image saved sucessfuly")


# def palletized_colorspace_fast(image_rgb: np.ndarray, palette_rgb: np.ndarray, colorspace: int | None = None,
#                                w0: float = 1, w1: float = 1, w2: float = 1):
#     image_space = rgb_to_colorspace(image_rgb, colorspace).astype(np.float32)
#     palette_space = rgb_to_colorspace(palette_rgb[np.newaxis, :, :].astype(np.uint8), colorspace)[0].astype(np.float32)
#
#     weights = np.array([w0, w1, w2], dtype=np.float32)
#     image_weighted = image_space * weights
#     palette_weighted = palette_space * weights
#
#     h, w, c = image_weighted.shape
#     image_flat = image_weighted.reshape(-1, 3)
#
#     palette_tree = cKDTree(palette_weighted)
#     _, indices = palette_tree.query(image_flat, k=1)
#
#     out = palette_rgb[indices].reshape(h, w, 3).astype(np.uint8)
#     return out

def _get_bayer_matrix(n: int) -> np.ndarray:
    if n == 2:
        m = np.array([[0, 2], [3, 1]], dtype=np.float32)
    else:
        smaller = _get_bayer_matrix_raw(n // 2)
        m = np.block([
            [4 * smaller,     4 * smaller + 2],
            [4 * smaller + 3, 4 * smaller + 1]
        ])
    return (m + 0.5) / (n * n) - 0.5


def _get_bayer_matrix_raw(n: int) -> np.ndarray:
    if n == 2:
        return np.array([[0, 2], [3, 1]], dtype=np.float32)
    smaller = _get_bayer_matrix_raw(n // 2)
    return np.block([
        [4 * smaller,     4 * smaller + 2],
        [4 * smaller + 3, 4 * smaller + 1]
    ])


def palletized_colorspace_fast(
    image_rgb: np.ndarray,
    palette_rgb: np.ndarray,
    chunk: int = 25000,
    colorspace: int | None = None,
    w0: float = 1.0,
    w1: float = 1.0,
    w2: float = 1.0,
    dither_type: str = "None",
    dither_strength: float = 16.0,
    progress_callback=None,
    cancel_check=None,
):
    if progress_callback:
        progress_callback(2.0, "Preparing palette & building KD-Tree...")

    if cancel_check and cancel_check():
        return None

    unique_palette_rgb = np.unique(palette_rgb.reshape(-1, 3), axis=0)

    image_space = rgb_to_colorspace(image_rgb, colorspace).astype(np.float32)
    palette_space = rgb_to_colorspace(
        unique_palette_rgb[np.newaxis, :, :].astype(np.uint8), colorspace
    )[0].astype(np.float32)

    if cancel_check and cancel_check():
        return None

    weights = np.array([w0, w1, w2], dtype=np.float32)
    palette_weighted = palette_space * weights

    h, w, c = image_space.shape
    palette_tree = cKDTree(palette_weighted)

    if cancel_check and cancel_check():
        return None

    # ---------------------------------------------------------
    # A. ERROR DIFFUSION DITHERING (3D LUT + Vectorized Wavefronts)
    # ---------------------------------------------------------
    if dither_type in ("Floyd-Steinberg", "Atkinson"):
        start = time.time()
        last_ui_time = 0.0

        # Step 1: Precompute a 64x64x64 3D Lookup Table (262,144 points total)
        # using fast multi-core KD-Tree batch queries
        lut_size = 64
        grid_vals = np.linspace(0.0, 255.0, lut_size, dtype=np.float32)
        g0, g1, g2 = np.meshgrid(grid_vals, grid_vals, grid_vals, indexing="ij")
        lut_points_weighted = np.stack([g0, g1, g2], axis=-1).reshape(-1, 3) * weights

        total_lut_pts = lut_points_weighted.shape[0]
        lut_flat = np.empty(total_lut_pts, dtype=np.int32)
        lut_chunk = 32768

        for c_start in range(0, total_lut_pts, lut_chunk):
            if cancel_check and cancel_check():
                return None
            c_end = min(c_start + lut_chunk, total_lut_pts)
            _, indices = palette_tree.query(
                lut_points_weighted[c_start:c_end], k=1, workers=-1
            )
            lut_flat[c_start:c_end] = indices

            now = time.time()
            if progress_callback and (now - last_ui_time >= 0.025):
                last_ui_time = now
                pct = (c_end / total_lut_pts) * 35.0
                progress_callback(pct, "Building 3D Color Lookup Table...")

        lut_3d = lut_flat.reshape((lut_size, lut_size, lut_size))

        # Step 2: Sweep across independent anti-diagonals (k = x + 2*y)
        # Pad work_img by (1 left, 2 right, 2 bottom) to avoid boundary branches
        work_img = np.zeros((h + 2, w + 3, 3), dtype=np.float32)
        work_img[:h, 1:w + 1] = image_space
        out_index = np.empty((h, w), dtype=np.int32)

        # Clamp strength_scale to [0.0, 1.25] so error never explodes infinitely
        strength_scale = np.clip(dither_strength / 16.0, 0.0, 1.25)
        scale_to_lut = (lut_size - 1) / 255.0

        total_waves = (w - 1) + 2 * (h - 1) + 1
        wave_start_time = time.time()

        for k in range(total_waves):
            if (k & 31) == 0 and cancel_check and cancel_check():
                return None

            y_min = max(0, (k - w + 2) // 2)
            y_max = min(h - 1, k // 2)
            ys = np.arange(y_min, y_max + 1, dtype=np.intp)
            xs_unpadded = k - 2 * ys
            xs = xs_unpadded + 1  # Offset for 1-pixel left padding

            # Clamp pixel values to valid [0, 255] range before lookup
            old_pixels = np.clip(work_img[ys, xs], 0.0, 255.0)
            q_idx = (old_pixels * scale_to_lut + 0.5).astype(np.intp)

            matched_idx = lut_3d[q_idx[:, 0], q_idx[:, 1], q_idx[:, 2]]
            out_index[ys, xs_unpadded] = matched_idx

            new_pixels = palette_space[matched_idx]
            quant_error = (old_pixels - new_pixels) * strength_scale

            if dither_type == "Floyd-Steinberg":
                work_img[ys, xs + 1] += quant_error * (7.0 / 16.0)
                work_img[ys + 1, xs - 1] += quant_error * (3.0 / 16.0)
                work_img[ys + 1, xs] += quant_error * (5.0 / 16.0)
                work_img[ys + 1, xs + 1] += quant_error * (1.0 / 16.0)
            else:  # Atkinson
                err_part = quant_error * (1.0 / 8.0)
                work_img[ys, xs + 1] += err_part
                work_img[ys, xs + 2] += err_part
                work_img[ys + 1, xs - 1] += err_part
                work_img[ys + 1, xs] += err_part
                work_img[ys + 1, xs + 1] += err_part
                work_img[ys + 2, xs] += err_part

            now = time.time()
            if progress_callback and (now - last_ui_time >= 0.025 or k == total_waves - 1):
                last_ui_time = now
                frac = (k + 1) / total_waves
                pct = 35.0 + frac * 65.0
                elapsed = now - wave_start_time
                eta = (elapsed / frac) * (1.0 - frac) if frac > 0 else 0.0
                progress_callback(
                    pct,
                    f"{dither_type}: {int(frac * 100)}% (ETA: {format_eta(eta)})"
                )

        if progress_callback:
            progress_callback(100.0, "Done!")
        return unique_palette_rgb[out_index].reshape(h, w, 3).astype(np.uint8)

    # ---------------------------------------------------------
    # B. FAST VECTORIZED DITHERING (Bayer & White Noise)
    # ---------------------------------------------------------
    image_weighted = image_space * weights

    if dither_type in ("Bayer 2x2", "Bayer 4x4", "Bayer 8x8", "White Noise"):
        if progress_callback:
            progress_callback(8.0, f"Applying {dither_type} pattern...")
        if dither_type.startswith("Bayer"):
            size = int(dither_type.split()[-1][0])
            bayer = _get_bayer_matrix(size)
            tiled_bayer = np.tile(bayer, (int(np.ceil(h / size)), int(np.ceil(w / size))))[:h, :w]
            offset = tiled_bayer[:, :, np.newaxis] * (dither_strength * weights)
            image_weighted = image_weighted + offset
        elif dither_type == "White Noise":
            noise = np.random.uniform(-0.5, 0.5, size=(h, w, 1)).astype(np.float32)
            image_weighted = image_weighted + noise * (dither_strength * weights)

    if cancel_check and cancel_check():
        return None

    # ---------------------------------------------------------
    # C. STANDARD CHUNKED KD-TREE QUERY
    # ---------------------------------------------------------
    image_flat = image_weighted.reshape(-1, 3)
    image_flat_length = image_flat.shape[0]
    out_index = np.empty(image_flat_length, dtype=np.int32)

    effective_chunk = min(chunk, 25000)
    number_of_chunks = int(np.ceil(image_flat_length / effective_chunk))

    overall_start = time.time()
    last_ui_time = 0.0

    for chunk_start in range(0, image_flat_length, effective_chunk):
        if cancel_check and cancel_check():
            return None

        chunk_end = min(chunk_start + effective_chunk, image_flat_length)
        current_chunk = (chunk_start // effective_chunk) + 1

        _, indices = palette_tree.query(
            image_flat[chunk_start:chunk_end], k=1, workers=-1
        )
        out_index[chunk_start:chunk_end] = indices

        if cancel_check and cancel_check():
            return None

        now = time.time()
        if progress_callback and (now - last_ui_time >= 0.025 or chunk_end == image_flat_length):
            last_ui_time = now
            elapsed = now - overall_start
            fraction_done = chunk_end / image_flat_length
            eta_seconds = (elapsed / fraction_done) * (1.0 - fraction_done) if fraction_done > 0 else 0.0

            pct = fraction_done * 100.0
            status = (
                f"Chunk {current_chunk}/{number_of_chunks} "
                f"(ETA: {format_eta(eta_seconds)})"
            )
            progress_callback(pct, status)

    if progress_callback:
        progress_callback(100.0, "Done!")

    out = unique_palette_rgb[out_index].reshape(h, w, 3).astype(np.uint8)
    return out