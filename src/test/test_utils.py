from collections import OrderedDict

import pytest
from numpy.testing import assert_almost_equal
import numpy as np

from ..dataset import Meta
from ..utils import merge_slices_of_patient, pad_image


def test_merge_slices_of_patient1():
    lesions_on_slices = OrderedDict()
    lesions_on_slices[77] = np.array(
        [["1", "278.6", "199.5", "443.1", "401.4", "0.0962"], ["2", "214.2", "162.9", "259.8", "204.0", "0.0504"]]
    )
    meta = Meta(
        {
            "ObjectType": "Image",
            "NDims": "3",
            "BinaryData": "True",
            "BinaryDataByteOrderMSB": "False",
            "CompressedData": "False",
            "TransformMatrix": "1 0 0 0 1 0 0 0 1",
            "Offset": "-173.19999694824219 -180 -295.5",
            "CenterOfRotation": "0 0 0",
            "AnatomicalOrientation": "RAI",
            "ElementSpacing": "0.703125 0.703125 3.75",
            "seriesuid": "000",
            "DimSize": "512 512 78",
            "ElementType": "MET_SHORT",
            "ElementDataFile": "322782.raw",
        }
    )
    merged = merge_slices_of_patient("322782", lesions_on_slices, meta)
    assert merged.keys() == ["322782"]
    assert merged["322782"].keys() == [77]
    np.testing.assert_almost_equal(
        merged["322782"][77],
        np.array(
            [["1", "278.6", "199.5", "443.1", "401.4", "0.0962"], ["2", "214.2", "162.9", "259.8", "204.0", "0.0504"]]
        ),
    )


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
