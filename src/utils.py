import numpy as np


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
