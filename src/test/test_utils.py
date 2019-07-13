from collections import OrderedDict, namedtuple

import pytest
from numpy.testing import assert_almost_equal
import numpy as np

from ..dataset import Meta
from ..utils import (
    merge_slices_of_patient,
    pad_image,
    x_y_w_h_2_xmn_ymn_xmx_ymx,
    xmn_ymn_xmx_ymx_2_x_y_w_h,
    get_coord_z_and_diameter_z,
    get_expanded_slice_idx,
    cuboid_pix_2_real,
)


MetaLite = namedtuple("MetaLite", "offset element_spacing")
Cuboid = namedtuple("Cuboid", "coord_x coord_y coord_z diameter_x diameter_y diameter_z")


@pytest.fixture
def meta1():
    return Meta(
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


# def test_merge_slices_of_patient1(meta1):
#     lesions_on_slices = OrderedDict()
#     lesions_on_slices[77] = np.array(
#         [["1", "278.6", "199.5", "443.1", "401.4", "0.0962"], ["2", "214.2", "162.9", "259.8", "204.0", "0.0504"]]
#     )
#     merged = merge_slices_of_patient("322782", lesions_on_slices, meta1)
#     assert merged.keys() == ["322782"]
#     assert merged["322782"].keys() == [77]
#     np.testing.assert_almost_equal(
#         merged["322782"][77],
#         np.array(
#             [["1", "278.6", "199.5", "443.1", "401.4", "0.0962"],
#              ["2", "214.2", "162.9", "259.8", "204.0", "0.0504"]]
#         ),
#     )


def test_merge_slices_of_patient2(meta1):
    slice_idx = [33, 34, 35, 36]
    classes = [2, 2, 2, 2]
    confidences = [0.1799, 0.2455, 0.2774, 0.0504]
    lesion_bboxes = np.array([[326.1, 215.5, 347.5, 235.1],
                              [326.5, 215.4, 348.9, 235.8],
                              [326.4, 215.6, 349.2, 236.8],
                              [326.4, 215.4, 347.6, 235.4]])
    merged = merge_slices_of_patient(slice_idx, lesion_bboxes, confidences, meta1)
    assert_almost_equal(merged, np.array([63.98164368,  -21.35742188, -166.125, 16.13671875, 14.9765625, 15.0, 0.1883]))


def test_xmn_ymn_xmx_ymx_2_x_y_w_h():
    assert_almost_equal(xmn_ymn_xmx_ymx_2_x_y_w_h([215., 227., 256., 264.]), np.array([235.5, 245.5, 42.0, 38.0]))


def test_x_y_w_h_2_xmn_ymn_xmx_ymx():
    assert_almost_equal(x_y_w_h_2_xmn_ymn_xmx_ymx(235.4999999999936, 245.49999999931285, 42.0, 38.0),
                        np.array([215, 227, 256, 264]))


def test_pix_to_real_3d_testcase1():
    meta = MetaLite([-157.85000610351562, -25.399999618530273, -385.41000366210938], [0.68359375, 0.68359375, 1])
    pix_cuboid = [
        Cuboid(235.4999999999936,245.49999999931285,41.000000000109424,42.0,38.0,13.0),
        Cuboid(268.00000000002285,283.9999999993129,118.00000000010937,11.0,7.0,1.0),
        Cuboid(291.00000000002285,304.4999999993128,170.00000000010937,27.0,28.0,5.0),
        Cuboid(193.50000000002288,168.00000000004428,176.00000000010937,32.0,25.0,3.0),
        Cuboid(297.50000000002285,209.49999999931285,223.00000000010937,26.0,34.0,9.0),
        Cuboid(391.49999999929145,321.9999999993128,221.00000000010937,14.0,11.0,3.0),
        Cuboid(293.50000000002285,230.99999999931285,236.00000000010937,14.0,7.0,1.0),
        Cuboid(297.00000000002285,190.49999999931285,259.50000000010937,21.0,26.0,10.0),
        Cuboid(280.50000000002285,256.4999999993129,269.00000000010937,14.0,24.0,5.0),
    ]
    real_gt = [
        (3.13632202148,142.422266006,-344.41000366199995,28.7109375,25.9765625,13.0),
        (25.3531188965,168.740625381,-267.410003662,7.51953125,4.78515625,1.0),
        (41.0757751465,182.75429725599997,-215.410003662,18.45703125,19.140625,5.0),
        (-25.5746154785,89.4437503815,-209.410003662,21.875,17.08984375,3.0),
        (45.51913452149999,117.812891006,-162.410003662,17.7734375,23.2421875,9.0),
        (109.776947021,194.71718788099997,-164.410003662,9.5703125,7.51953125,3.0),
        (42.7847595215,132.510156631,-149.410003662,9.5703125,4.78515625,1.0),
        (45.17733764649999,104.824609756,-125.91000366200001,14.35546875,17.7734375,10.0),
        (33.8980407715,149.941797256,-116.41000366200001,9.5703125,16.40625,5.0),
    ]
    for pc, r in zip(pix_cuboid, real_gt):
        assert_almost_equal(cuboid_pix_2_real(*pc, meta), np.array(r))


def test_pix_to_real_3d_testcase2():
    meta = MetaLite([-143.30000305175781, -153.5, -712.70001220703125], [0.599609375, 0.599609375, 5])
    pix_cuboid = [
        Cuboid(348.4999999999297,358.0,17.000000000006253,24.0,29.0,3.0),
        Cuboid(443.50000000043,183.5,23.000000000006253,32.0,42.0,3.0),
        Cuboid(204.49999999992963,244.5,38.00000000000625,22.0,20.0,1.0),
        Cuboid(309.9999999999297,303.5,44.50000000000625,13.0,16.0,2.0),
        Cuboid(349.4999999999297,274.0,56.00000000000625,14.0,15.0,1.0),
    ]
    real_gt = [
        (65.6638641357,61.16015625,-627.700012207,14.390625,17.388671875,15.0),
        (122.626754761,-43.4716796875,-597.700012207,19.1875,25.18359375,15.0),
        (-20.6798858643,-6.8955078125,-522.700012207,13.19140625,11.9921875,5.0),
        (42.5789031982,28.4814453125,-490.200012207,7.794921875,9.59375,10.0),
        (66.2634735107,10.79296875,-432.700012207,8.39453125,8.994140625,5.0),
    ]
    for pc, r in zip(pix_cuboid, real_gt):
        assert_almost_equal(cuboid_pix_2_real(*pc, meta), np.array(r))


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


def test_get_coord_z_and_diameter_z():
    coord_z, diameter_z = get_coord_z_and_diameter_z([6, 7, 8])
    assert coord_z == 7
    assert diameter_z == 3


def test_coord_z_diameter_z_and_slice_idx_conversion():
    coord_z = np.random.randint(20, 40, 100)
    diameter_z = np.random.randint(1, 8, 100)
    for c, d in zip(coord_z, diameter_z):
        if d % 2 == 0:
            c += 0.5
        slice_idx = get_expanded_slice_idx(c, d)
        _c, _d = get_coord_z_and_diameter_z(slice_idx)
        _slice_idx = get_expanded_slice_idx(_c, _d)
        assert _c == c
        assert _d == d
        assert _slice_idx == slice_idx
