import pytest
import numpy as np
from numpy.testing import assert_almost_equal

from ..utils import pad_image


def test_pad_image():
    raw_ct_image = np.arange(12, dtype=np.int16).reshape(3, 2, 2)
    assert_almost_equal(pad_image(raw_ct_image, 0, 1), np.array([[[0, 0],
                                                                  [0, 0]],

                                                                 [[0, 1],
                                                                  [2, 3]],

                                                                 [[4, 5],
                                                                  [6, 7]]]))

    assert_almost_equal(pad_image(raw_ct_image, 1, 1), np.array([[[0, 1],
                                                                  [2, 3]],

                                                                 [[4, 5],
                                                                  [6, 7]],

                                                                 [[8, 9],
                                                                  [10, 11]]]))

    assert_almost_equal(pad_image(raw_ct_image, 2, 1), np.array([[[4, 5],
                                                                  [6, 7]],

                                                                 [[8, 9],
                                                                  [10, 11]],

                                                                 [[0, 0],
                                                                  [0, 0]]]))

    assert_almost_equal(pad_image(raw_ct_image, 2, 2), np.array([[[0, 1],
                                                                  [2, 3]],

                                                                 [[4, 5],
                                                                  [6, 7]],

                                                                 [[8, 9],
                                                                  [10, 11]],

                                                                 [[0, 0],
                                                                  [0, 0]],

                                                                 [[0, 0],
                                                                  [0, 0]]]))


def test_pad_image2():
    raw_ct_image = np.arange(16, dtype=np.int16).reshape(4, 2, 2)
    assert_almost_equal(pad_image(raw_ct_image, 0, 2), np.array([[[0, 0],
                                                                  [0, 0]],

                                                                 [[0, 0],
                                                                  [0, 0]],

                                                                 [[0, 1],
                                                                  [2, 3]],

                                                                 [[4, 5],
                                                                  [6, 7]],

                                                                 [[8, 9],
                                                                  [10, 11]]]))

    assert_almost_equal(pad_image(raw_ct_image, 1, 2), np.array([[[0, 0],
                                                                  [0, 0]],

                                                                 [[0, 1],
                                                                  [2, 3]],

                                                                 [[4, 5],
                                                                  [6, 7]],

                                                                 [[8, 9],
                                                                  [10, 11]],

                                                                 [[12, 13],
                                                                  [14, 15]]]))

    assert_almost_equal(pad_image(raw_ct_image, 2, 2), np.array([[[0, 1],
                                                                  [2, 3]],

                                                                 [[4, 5],
                                                                  [6, 7]],

                                                                 [[8, 9],
                                                                  [10, 11]],

                                                                 [[12, 13],
                                                                  [14, 15]],

                                                                 [[0, 0],
                                                                  [0, 0]]]))

    assert_almost_equal(pad_image(raw_ct_image, 3, 2), np.array([[[4, 5],
                                                                  [6, 7]],

                                                                 [[8, 9],
                                                                  [10, 11]],

                                                                 [[12, 13],
                                                                  [14, 15]],

                                                                 [[0, 0],
                                                                  [0, 0]],

                                                                 [[0, 0],
                                                                  [0, 0]]]))

    with pytest.raises(IndexError):
        _ = pad_image(raw_ct_image, 4, 2)
