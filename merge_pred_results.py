import argparse
from collections import OrderedDict
from pathlib import Path

import numpy as np

from src.dataset import Meta
from src.utils import merge_slices_of_patient


parser = argparse.ArgumentParser()
parser.add_argument("--objinfo-path", type=Path, required=True)
parser.add_argument("--meta-dir", type=Path, required=True)
parser.add_argument("--output-file-path", type=Path, required=True)
args = parser.parse_args()


def read_pred_lesions():
    """Only returns detected lesions (foreground) """
    with open(args.objinfo_path, 'r') as f:
        lines = [l.strip().split(' ') for l in f]
    sorted_lines = sorted(lines, key=lambda x: x[0])
    single_slice_lesions = OrderedDict()
    for l in sorted_lines:
        if int(l[1]) > 0:
            patient_id, slice_id = l[0].split('_')
            single_slice_lesions[patient_id] = OrderedDict()
            single_slice_lesions[patient_id][int(slice_id)] = np.array(l[2:]).reshape(int(l[1]), 6)
    return single_slice_lesions


def merge_pred_results():
    pred_lesions = read_pred_lesions()
    for patient_id, pred in pred_lesions.items():
        merge_slices_of_patient(patient_id, pred, Meta.from_path(args.meta_dir / f"{patient_id}.mhd"))


if __name__ == '__main__':
    merge_pred_results()
