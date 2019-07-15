import math
from collections import Iterable
import operator

import numpy as np


class _Coord(list):
    def __init__(self, coords):
        super(_Coord, self).__init__(coords)

    def __repr__(self):
        ele_fmt = '{:.5f}' if any([isinstance(c, float) for c in self]) else '{:d}'
        return '({})'.format(', '.join([ele_fmt.format(c) for c in self]))

    def _adaptive_op(self, other, op):
        """this div function adapts to both scalar and iterable with the same length"""
        if isinstance(other, int) or isinstance(other, float):
            return _Coord([op(c, other) for c in self])

        if isinstance(other, Iterable) and len(other) == len(self):
            return _Coord([op(s, o) for s, o in zip(self, other)])

    def __mul__(self, other):
        return self._adaptive_op(other, operator.mul)

    def __sub__(self, other):
        return self._adaptive_op(other, operator.sub)

    def __add__(self, other):
        return self._adaptive_op(other, operator.add)

    def __truediv__(self, other):
        return self._adaptive_op(other, operator.truediv)

    def __floordiv__(self, other):
        return self._adaptive_op(other, operator.floordiv)


class Coord2D(_Coord):
    def __init__(self, coords):
        assert len(coords) == 2
        super(Coord2D, self).__init__(coords)


class Coord4D(_Coord):
    def __init__(self, coords):
        assert len(coords) == 4
        super(Coord4D, self).__init__(coords)


class BBox:
    def __init__(self, left, top, right, bottom):
        self.coord = Coord4D([float(c) for c in [left, top, right, bottom]])

    @classmethod
    def from_xml_element(cls, root):
        return BBox(root.find('xmin').text, root.find('ymin').text, root.find('xmax').text, root.find('ymax').text)

    @classmethod
    def from_json(cls, data):
        return BBox(data['xmin'], data['ymin'], data['xmax'], data['ymax'])

    @classmethod
    def from_xywh(cls, x, y, w, h):
        half_width, half_height = w / 2, h / 2
        return BBox(x - half_width, y - half_height, x + half_width, y + half_height)

    @property
    def left(self):
        return self.coord[0]

    @property
    def top(self):
        return self.coord[1]

    @property
    def right(self):
        return self.coord[2]

    @property
    def bottom(self):
        return self.coord[3]

    @property
    def width(self):
        return max(0, self.right - self.left + 1)

    @property
    def height(self):
        return max(0, self.bottom - self.top + 1)

    @property
    def size(self):
        return Coord2D([self.width, self.height])

    @property
    def center(self):
        return Coord2D((self.left + self.right, self.top + self.bottom)) / 2

    def hscale(self, scalar, in_place=False):
        """scale horizontally"""
        if not in_place:
            return BBox(self.coord[0] * scalar, self.coord[1], self.coord[2] * scalar, self.coord[3])

        self.coord[0] *= scalar
        self.coord[2] *= scalar
        return self

    def vscale(self, scalar, in_place=False):
        """scale vertically"""
        if not in_place:
            return BBox(self.coord[0], self.coord[1] * scalar, self.coord[2], self.coord[3] * scalar)

        self.coord[1] *= scalar
        self.coord[3] *= scalar
        return self

    def gscale(self, scalars, in_place=False):
        """global scale"""
        assert len(scalars) == 2

        if not in_place:
            return BBox(self.coord[0] * scalars[0], self.coord[1] * scalars[1],
                        self.coord[2] * scalars[0], self.coord[3] * scalars[1])

        return self.hscale(scalars[0]).vscale(scalars[1])

    def clamp(self, x_max, y_max, x_min=0, y_min=0, in_place=False):
        assert x_max > x_min and y_max > y_min

        if not in_place:
            return BBox(max(x_min, min(self.left, x_max)),
                        max(y_min, min(self.top, y_max)),
                        max(x_min, min(self.right, x_max)),
                        max(y_min, min(self.bottom, y_max)))

        self.coord[0] = max(x_min, min(self.left, x_max))
        self.coord[1] = max(y_min, min(self.top, y_max))
        self.coord[2] = max(x_min, min(self.right, x_max))
        self.coord[3] = max(y_min, min(self.bottom, y_max))
        return self

    def area(self):
        return self.width * self.height

    def is_empty(self):
        return self.left == self.right or self.top == self.bottom

    def __repr__(self):
        return ', '.join(['{:.5f}'.format(c) for c in self.coord])

    def __eq__(self, other):
        return math.fabs(self.coord[0] - other.coord[0]) < 1e-7 and \
               math.fabs(self.coord[1] - other.coord[1]) < 1e-7 and \
               math.fabs(self.coord[2] - other.coord[2]) < 1e-7 and \
               math.fabs(self.coord[3] - other.coord[3]) < 1e-7

    def __sub__(self, other):
        """Intersection between self and the other BBox. This operation should obey commutative law."""
        intsec_area_left = max(self.left, other.left)
        intsec_area_right = min(self.right, other.right)
        intsec_area_top = max(self.top, other.top)
        intsec_area_bottom = min(self.bottom, other.bottom)
        return BBox(intsec_area_left, intsec_area_top, intsec_area_right, intsec_area_bottom)

    def enclosed_by(self, other):
        return self.left >= other.left and self.right <= other.right and \
               self.top >= other.top and self.bottom <= other.bottom


def get_indices(ind_path, root_dir, suffix):
    _sfx = suffix.strip(".")
    if ind_path:
        with open(ind_path, "r") as f:
            indices = [l.strip() for l in f.readlines()]
    else:
        indices = sorted([str(p.relative_to(root_dir).with_suffix("")) for p in root_dir.rglob(f"*.{_sfx}")])

    return indices


def x_y_w_h_2_xmn_ymn_xmx_ymx(coord_x, coord_y, diameter_x, diameter_y):
    half_diameter_x, half_diameter_y = (diameter_x - 1) / 2, (diameter_y - 1) / 2
    return np.array(
        [coord_x - half_diameter_x, coord_y - half_diameter_y, coord_x + half_diameter_x, coord_y + half_diameter_y],
        dtype=np.float32,
    )


def xmn_ymn_xmx_ymx_2_x_y_w_h(bbox):
    """bbox's 4 values are: xmin, ymin, xmax, ymax"""
    w, h = bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1
    center_x, center_y = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2,
    return [center_x, center_y, w, h]


def assert_int(number, name):
    assert abs(number - round(number)) < 1e-6, f"{name} is expected to be a integer."


def get_expanded_slice_idx(coord_z, diameter_z):
    assert_int(diameter_z, "diameter_z")
    start = coord_z - (diameter_z - 1) / 2
    assert_int(start, "start")
    return list(range(int(round(start)), int(round(start + diameter_z))))


def get_coord_z_and_diameter_z(slice_idx):
    """Return values are represented in terms of pixels"""
    diameter_z = len(slice_idx)
    coord_z = np.mean(slice_idx)  # center of lesion on z-axis
    return coord_z, diameter_z


def pad_image(raw_ct_img, slice_idx, padding):
    padding_slices = np.zeros((padding, *raw_ct_img.shape[1:]), dtype=raw_ct_img.dtype)
    padded_raw_ct_img = np.concatenate((padding_slices, raw_ct_img, padding_slices), axis=0)
    if slice_idx >= len(raw_ct_img):
        raise IndexError(f"Index {slice_idx} larger than the max index ({len(raw_ct_img)}) of raw_ct_img")
    return padded_raw_ct_img[slice_idx : slice_idx + padding * 2 + 1]


def merge_slices_of_patient(slice_idx, lesion_bboxes, confidences, meta):
    bbox_mean = lesion_bboxes.mean(axis=0)
    conf_mean = np.mean(confidences)
    coord_z, diameter_z = get_coord_z_and_diameter_z(slice_idx)
    coord_x, coord_y, diameter_x, diameter_y = xmn_ymn_xmx_ymx_2_x_y_w_h(bbox_mean)
    coord_x, coord_y, coord_z, diameter_x, diameter_y, diameter_z = \
        cuboid_pix_2_real(coord_x, coord_y, coord_z, diameter_x, diameter_y, diameter_z, meta)
    return coord_x, coord_y, coord_z, diameter_x, diameter_y, diameter_z, conf_mean


def cuboid_pix_2_real(coord_x, coord_y, coord_z, diameter_x, diameter_y, diameter_z, meta):
    return coord_x * meta.element_spacing[0] + meta.offset[0], \
        coord_y * meta.element_spacing[1] + meta.offset[1], \
        coord_z * meta.element_spacing[2] + meta.offset[2], \
        diameter_x * meta.element_spacing[0], \
        diameter_y * meta.element_spacing[1], \
        diameter_z * meta.element_spacing[2]


def iou(a: BBox, b: BBox):
    i = (a - b).area()
    u = a.area() + b.area() - (a - b).area()
    return i / (u + 1e-7)


def calc_overlaps(dets_a, dets_b):
    overlaps = np.zeros((len(dets_a), len(dets_b)))
    for i, a in enumerate(dets_a):
        for j, b in enumerate(dets_b):
            overlaps[i, j] = (a[1] == b[1]) * iou(BBox(*a[2:6]), BBox(*b[2:6]))
    return overlaps
