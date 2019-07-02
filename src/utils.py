import numpy as np


def get_indices(ind_path, root_dir, suffix):
    _sfx = suffix.strip(".")
    if ind_path:
        with open(ind_path, "r") as f:
            indices = [l.strip() for l in f.readlines()]
    else:
        indices = sorted([str(p.relative_to(root_dir).with_suffix("")) for p in root_dir.rglob(f"*.{_sfx}")])

    return indices


def derive_bbox(coord_x, coord_y, diameter_x, diameter_y):
    return np.array([coord_x - diameter_x, coord_y - diameter_y, coord_x + diameter_x, coord_y + diameter_y],
                    dtype=np.int32)


def assert_int(number, name):
    assert abs(number - round(number)) < 1e-6, f'{name} is expected to be a integer.'


def get_expanded_slice_idx(coord_z, diameter_z):
    assert_int(diameter_z, 'diameter_z')
    start = coord_z - (diameter_z - 1) / 2
    assert_int(start, 'start')
    return list(range(int(round(start)), int(round(start + diameter_z))))


def pad_image(raw_ct_img, slice_idx, padding):
    padding = np.zeros((padding, *raw_ct_img.shape[1:]), dtype=raw_ct_img.dtype)
    padded_raw_ct_img = np.concatenate((padding, raw_ct_img, padding), axis=0)
    return padded_raw_ct_img[slice_idx : slice_idx + padding * 2 + 1]
