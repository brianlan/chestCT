import argparse
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.dataset import Meta
from src.utils import merge_slices_of_patient, calc_overlaps


np.set_printoptions(suppress=True)
parser = argparse.ArgumentParser()
parser.add_argument("--objinfo-path", type=Path, required=True)
parser.add_argument("--meta-dir", type=Path, required=True)
parser.add_argument("--output-file-path", type=Path, required=True)
parser.add_argument("--match-iou-thresh", type=float, default=0.6)
args = parser.parse_args()


def read_predictions():
    """Only returns detected lesions (foreground) """
    with open(args.objinfo_path, 'r') as f:
        lines = [l.strip().split(' ') for l in f]
    sorted_lines = sorted(lines, key=lambda x: x[0])
    single_slice_lesions = defaultdict(OrderedDict)
    for l in sorted_lines:
        if int(l[1]) > 0:
            patient_id, slice_id = l[0].split('_')
            single_slice_lesions[patient_id][int(slice_id)] = np.array(l[2:], dtype=np.float32).reshape(int(l[1]), 6)
    return single_slice_lesions


def match_detections_into_groups(slice_partition):
    # prepend slice_idx to each det
    for s_id in slice_partition:
        nrow = slice_partition[s_id].shape[0]
        slice_partition[s_id] = np.concatenate((np.ones((nrow, 1)) * int(s_id), slice_partition[s_id]), axis=1)

    # initialize groups with the first slice detections
    slice_iter = iter(slice_partition.keys())
    first_slice_id = next(slice_iter)
    groups = [det[None, ...] for det in slice_partition[first_slice_id]]

    # go through the next slice to find close dets and put them into the same group
    for slice_id in slice_iter:
        prev_idx, prev_dets = zip(*[(i, g[-1]) for i, g in enumerate(groups) if g[-1].sum() != 0.0])
        overlaps = calc_overlaps(slice_partition[slice_id], prev_dets)
        assignment = linear_sum_assignment(1 - overlaps)

        for s_id, g_id in zip(assignment[0], assignment[1]):
            r_g_id = prev_idx[g_id]  # real g_id
            cur_det = slice_partition[slice_id][None, s_id]
            if overlaps[s_id, g_id] > args.match_iou_thresh:
                groups[r_g_id] = np.concatenate((groups[r_g_id], cur_det), axis=0)
            else:
                groups[r_g_id] = np.concatenate((groups[r_g_id], np.zeros((1, 7))), axis=0)
                groups.append(cur_det)

        # append unmatched cur dets to groups
        for i in set(range(len(slice_partition[slice_id]))) - set(assignment[0]):
            groups.append(slice_partition[slice_id][None, i])

        # concat 0-placeholder to unmatched prev_dets
        for i in set(range(len(prev_dets))) - set(assignment[1]):
            r_g_id = prev_idx[i]  # real g_id
            groups[r_g_id] = np.concatenate((groups[r_g_id], np.zeros((1, 7))), axis=0)

    # remove 0-placeholder
    for i, g in enumerate(groups):
        groups[i] = g[g.sum(axis=1) != 0.0]

    return groups


def process_slice_partition(slice_partition, meta):
    groups = match_detections_into_groups(slice_partition)
    a = 19


def process_patient(patient_id, pred, meta):
    """the lesions of a patient will be parted into several partitions with consecutive slice_id,
    then, each detected bboxes at different slices with high IOU will be considered as a group.
    Finally the bboxes in a group will be merged as a lesion output.
    """
    prev = None
    partition = OrderedDict()
    for slice_id, dets in pred.items():
        if prev is not None and prev + 1 != slice_id:
            process_slice_partition(partition, meta)
            partition = OrderedDict({slice_id: dets})
        else:
            partition[slice_id] = dets
        prev = slice_id


def merge_pred_results():
    pred_lesions = read_predictions()
    for patient_id, pred in pred_lesions.items():
        process_patient(patient_id, pred, Meta.from_path(args.meta_dir / f"{patient_id}.mhd"))


if __name__ == '__main__':
    merge_pred_results()
