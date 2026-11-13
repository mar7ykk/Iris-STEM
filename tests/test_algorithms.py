import numpy as np
from algorithms.histogram import histogram_equalization
from algorithms.otsu import otsu_threshold
from algorithms.sobel import sobel_filter
from algorithms.lbp import lbp_texture
from algorithms.glcm import glcm_texture


def test_histogram():
    image = np.array([
        [0, 0],
        [100, 255]
    ], dtype=np.uint8)

    result = histogram_equalization(image)

    expected = np.array([
        [0, 0],
        [127, 255]
    ], dtype=np.uint8)

    assert np.array_equal(result, expected)


def test_otsu():
    image = np.array([
        [20, 20],
        [200, 200]
    ], dtype=np.uint8)

    result, threshold = otsu_threshold(image)

    expected = np.array([
        [255, 255],
        [0, 0]
    ], dtype=np.uint8)

    assert threshold == 20
    assert np.array_equal(result, expected)


def test_sobel():
    image = np.full((3, 3), 50, dtype=np.uint8)

    result = sobel_filter(image)

    assert np.array_equal(result, np.zeros((3, 3)))


def test_lbp():
    image = np.array([
        [20, 20, 20],
        [20, 10, 20],
        [20, 20, 20]
    ], dtype=np.uint8)

    result = lbp_texture(image)

    assert result[1, 1] == 255
    assert np.sum(result) == 255


def test_glcm():
    image = np.array([
        [0, 255],
        [0, 255]
    ], dtype=np.uint8)

    matrix, contrast, energy, homogeneity = glcm_texture(image)

    assert matrix[0, 15] == 1
    assert np.isclose(np.sum(matrix), 1)
    assert np.isclose(contrast, 225)
    assert np.isclose(energy, 1)
    assert np.isclose(homogeneity, 1 / 226)

def test_lbp_mask():
    image = np.full((5, 5), 100, dtype=np.uint8)
    mask = np.zeros((5, 5), dtype=bool)
    mask[1:4, 1:4] = True

    result = lbp_texture(image, mask)

    assert result[2, 2] == 255
    assert result[1, 1] == 0