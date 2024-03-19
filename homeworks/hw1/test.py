import numpy as np
from PIL import Image


def load_image(filename):
    img = np.asarray(Image.open(filename))
    img = img.astype("float32") / 255.0
    return img


arr = np.arange(25).reshape(5, 5)


def linear_filter(image, k, output):
    height, width = image.shape
    mid = k // 2
    for i in range(height - k + 1):
        for j in range(width - k + 1):
            output[i : k + i, j : k + j][mid, mid] = image[i : k + i, j : k + j].mean()


def convolve_image(image_main, filter_matrix, k=3):
    """Convolve a 2D image using the filter matrix.
    Args:
        image: a 2D numpy array.
        filter_matrix: a 2D numpy array.
        k: window size
    Returns:
        the convolved image, which is a 2D numpy array same size as the input image.

    TODO: Implement the convolve_image function here.
    """
    final_image = np.zeros_like(image_main)
    kernel_dim = filter_matrix.shape[0]
    pad = (kernel_dim - 1) // 2
    image_main = np.pad(image_main, [(pad, pad)], mode="constant")
    height, width = image_main.shape
    filter_matrix = np.fliplr(np.flipud(filter_matrix))
    for i in range(height - k + 1):
        for j in range(width - k + 1):
            final_image[i, j] = (
                image_main[i : k + i, j : k + j].ravel().dot(filter_matrix.ravel())
            )

    return final_image


mean_filt = np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]])
arr = np.array(
    [
        [25, 100, 75, 49, 130],
        [50, 80, 0, 70, 100],
        [5, 10, 20, 30, 0],
        [60, 50, 12, 24, 32],
        [37, 53, 55, 21, 90],
        [140, 17, 0, 23, 222],
    ]
)
input_image = np.array(
    [
        [1, 0, 1, 2, 2],
        [1, 1, 2, 2, 3],
        [1, 2, 2, 6, 3],
        [1, 1, 2, 2, 3],
    ]
)
kernel = np.array(
    [
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0],
    ]
)
convolve_image(input_image, kernel, 3)


def circular_mask(height, width, center=None, radius=None):
    if center is None:
        center = (height // 2, width // 2)
    if radius is None:
        radius = height // 30
    Y, X = np.ogrid[:height, :width]
    xnorm = (X - center[0]) ** 2
    ynorm = (Y - center[1]) ** 2
    dist_from_center = np.sqrt(xnorm + ynorm)
    mask = dist_from_center <= radius
    return mask


path_to_dog_image = "/Users/rajmani/Documents/research/Home/python/computer-vision/homeworks/hw1/dog.jpg"
path_to_cat_image = "/Users/rajmani/Documents/research/Home/python/computer-vision/homeworks/hw1/cat.jpg"
dog = load_image(path_to_dog_image).mean(axis=-1)[:, 25:-24]
cat = load_image(path_to_cat_image).mean(axis=-1)[:, 25:-24]

cat_fft = np.fft.fftshift(np.fft.fft2(cat))
dog_fft = np.fft.fftshift(np.fft.fft2(dog))

print(cat_fft)
test_cat = cat_fft[:6, :6]
circular_mask(*cat_fft.shape)