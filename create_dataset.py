import argparse
from pathlib import Path
import multiprocessing

from tqdm import tqdm
import numpy as np

from src.dataset import read_im, Label
from src.utils import get_indices, pad_image


parser = argparse.ArgumentParser()
parser.add_argument("--dataset-dir", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--num-slices-to-pad", type=int, default=0)
parser.add_argument("--indices", type=Path)
parser.add_argument("--num-processes", type=int, default=1)
args = parser.parse_args()


def save_images(im, save_dir):
    _save_dir = Path(save_dir)
    _save_dir.mkdir(parents=True, exist_ok=True)

    for i in range(im.shape[0]):
        im_save_path = _save_dir / f'{_save_dir.stem}_{i:03}.npy'
        np.save(str(im_save_path), pad_image(im, i, args.num_slices_to_pad))


def process_patient(patient_id):
    ct_img, meta = read_im(patient_id, args.dataset_dir)
    save_images(ct_img, args.output_dir / patient_id)


def main():
    indices = get_indices(args.indices, args.dataset_dir, ".mhd")
    for ind in indices:
        process_patient(ind)
    # with multiprocessing.Pool(processes=args.num_processes) as pool:
    #     _ = list(tqdm(pool.imap_unordered(process_patient, indices), total=len(indices)))


if __name__ == '__main__':
    main()
