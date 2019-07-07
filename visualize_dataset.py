import argparse
from pathlib import Path
from collections import defaultdict
import multiprocessing

from tqdm import tqdm
import cv2
import numpy as np

from src.dataset import read_im, Label
from src.utils import get_indices, assert_int, x_y_w_h_2_xmn_ymn_xmx_ymx, get_expanded_slice_idx


parser = argparse.ArgumentParser()
parser.add_argument("--dataset-dir", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--label-path", type=Path, required=True)
parser.add_argument("--output-normal-slices", action="store_true", default=False)
parser.add_argument("--indices", type=Path)
parser.add_argument("--num-processes", type=int, default=1)
args = parser.parse_args()
label = Label(str(args.label_path))

lesionid2color = {
    1: (0, 255, 255),
    5: (0, 0, 255),
    31: (255, 0, 255),
    32: (0, 255, 0),
}


def draw_bbox(im, bbox, color, linewidth=1):
    return cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, linewidth)


def ctimg2rgb(ct_im):
    _im = ((ct_im - ct_im.min()) / (ct_im.max() - ct_im.min()) * 255).astype(np.uint8)
    return _im[..., None].repeat(3, axis=2)


def draw_lesions(im, lesions, save_dir):
    _save_dir = Path(save_dir)
    _save_dir.mkdir(parents=True, exist_ok=True)

    # due to multiple lesions can be on the same slice,
    # we need to store all of them.
    boxes_to_draw = defaultdict(list)
    for _, les in lesions.iterrows():
        slice_idx = get_expanded_slice_idx(les.coordZ, les.diameterZ)
        for idx in slice_idx:
            _bbox = x_y_w_h_2_xmn_ymn_xmx_ymx(les.coordX, les.coordY, les.diameterX, les.diameterY)
            boxes_to_draw[idx].append((_bbox, les.label))

    if args.output_normal_slices:
        for i in range(im.shape[0]):
            if i not in boxes_to_draw.keys():
                im_save_path = _save_dir / f'{_save_dir.stem}_{i:03}.png'
                cv2.imwrite(str(im_save_path), ctimg2rgb(im[i]))

    # Visualize lesions on each image.
    for idx, boxes_labels in boxes_to_draw.items():
        im_save_path = _save_dir / f'{_save_dir.stem}_{idx:03}.png'
        _im = ctimg2rgb(im[idx])
        for _bbox, _label in boxes_labels:
            _im = draw_bbox(_im, _bbox, lesionid2color[_label])
        cv2.imwrite(str(im_save_path), _im)


def process_patient(patient_id):
    ct_img, meta = read_im(patient_id, args.dataset_dir)
    if int(patient_id) in label.all_patient_ids:
        lesions = label.get(patient_id, meta=meta)
        draw_lesions(ct_img, lesions, args.output_dir / patient_id)


def main():
    indices = get_indices(args.indices, args.dataset_dir, ".mhd")
    with multiprocessing.Pool(processes=args.num_processes) as pool:
        _ = list(tqdm(pool.imap_unordered(process_patient, indices), total=len(indices)))


if __name__ == '__main__':
    main()
