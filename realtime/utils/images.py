import base64
import io

import numpy as np
from PIL import Image


def convert_image_to_url(image: Image.Image, format: str = "jpeg") -> str:
    """Encode a pillow image object to a data URL with the specified format."""
    global i
    with io.BytesIO() as buffer:
        image.save(buffer, format=format, quality=95)
        return f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode()}"


def rgb_to_grayscale(rgb_array: np.ndarray) -> np.ndarray:
    """Convert an RGB image to a grayscale image using numpy."""
    # Use the common formula for converting RGB to grayscale
    gray_array = np.dot(rgb_array[..., :3], [0.2989, 0.5870, 0.1140])
    return gray_array


def image_euclidean_distance(img1: Image.Image, img2: Image.Image):
    # Convert images to numpy arrays
    img1_np = np.array(img1)
    img2_np = np.array(img2)

    if img1_np.shape != img2_np.shape:
        return 1
    # Convert images to grayscale
    gray1 = rgb_to_grayscale(img1_np)
    gray2 = rgb_to_grayscale(img2_np)

    # Calculate the Euclidean distance
    diff = gray1 - gray2
    diffMag = np.linalg.norm(diff) / ((np.linalg.norm(gray1) + np.linalg.norm(gray2)) / 2.0)
    return diffMag


def image_hamming_distance(img1: Image.Image, img2: Image.Image):
    # Convert images to numpy arrays
    img1_np = np.array(img1)
    img2_np = np.array(img2)

    if img1_np.shape != img2_np.shape:
        return 1
    # Convert images to grayscale
    gray1 = rgb_to_grayscale(img1_np)
    gray2 = rgb_to_grayscale(img2_np)

    # Calculate the Euclidean distance
    diff = gray1 - gray2
    diffMag = np.count_nonzero(diff) / np.count_nonzero(gray1)
    return diffMag


def convert_yuv420_to_pil(frame):
    data = frame.to_ndarray(format="yuv420p")
    w, h = data.shape[1], round(data.shape[0] * 2 / 3)
    data1d = data.reshape(-1)

    # Extract Y, U, and V channels
    Y_end = w * h
    Y = data1d[:Y_end].reshape(h, w)
    U_end = Y_end + (w // 2) * (h // 2)
    U = data1d[Y_end:U_end].reshape((h // 2, w // 2))
    V = data1d[U_end:].reshape((h // 2, w // 2))

    # Upsample U and V channels
    U_upsampled = U.repeat(2, axis=0).repeat(2, axis=1)
    V_upsampled = V.repeat(2, axis=0).repeat(2, axis=1)

    # Stack the channels
    ycbcr_image = np.stack([Y, U_upsampled, V_upsampled], axis=-1)
    return Image.fromarray(ycbcr_image, "YCbCr")
