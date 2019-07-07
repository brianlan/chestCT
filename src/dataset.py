from pathlib import Path
import pandas as pd
import numpy as np


class Meta:
    def __init__(self, meta_dict):
        self.meta_dict = meta_dict
        self.derive_other_fields()

    def derive_other_fields(self):
        setattr(self, "dim_size", self._convert(self.meta_dict['DimSize'], dtype=int))
        setattr(self, "offset", self._convert(self.meta_dict['Offset'], dtype=float))
        setattr(self, "element_spacing", self._convert(self.meta_dict['ElementSpacing'], dtype=float))

    @classmethod
    def from_path(cls, path):
        meta_dict = Meta.read_meta(path)
        return Meta(meta_dict)

    @staticmethod
    def _convert(str_list, dtype=float):
        return [dtype(i) for i in str_list.split(' ')]

    @staticmethod
    def read_meta(path):
        meta = {}
        with open(path, 'r') as f:
            for line in f:
                k, v = line.strip().split('=')
                meta[k.strip()] = v.strip()
        return meta

    def __repr__(self):
        return f"Meta(DimSize: {self.dim_size}, offset_z: {self.offset[2]}, z_intvl: {self.element_spacing[2]})"


class Label:
    def __init__(self, path):
        self._df = pd.read_csv(path)

    def __getitem__(self, _id):
        return self._df[self._df.seriesuid == int(_id)]

    def get(self, _id, meta=None):
        if meta is None:
            return self[_id]
        restored_label = self[_id]
        restored_label.loc[:, 'coordX'] = (restored_label.loc[:, 'coordX'] - meta.offset[0]) / meta.element_spacing[0]
        restored_label.loc[:, 'coordY'] = (restored_label.loc[:, 'coordY'] - meta.offset[1]) / meta.element_spacing[1]
        restored_label.loc[:, 'coordZ'] = (restored_label.loc[:, 'coordZ'] - meta.offset[2]) / meta.element_spacing[2]
        restored_label.loc[:, 'diameterX'] = restored_label.loc[:, 'diameterX'] / meta.element_spacing[0]
        restored_label.loc[:, 'diameterY'] = restored_label.loc[:, 'diameterY'] / meta.element_spacing[1]
        restored_label.loc[:, 'diameterZ'] = restored_label.loc[:, 'diameterZ'] / meta.element_spacing[2]
        return restored_label

    @property
    def all_patient_ids(self):
        return self._df.seriesuid.unique().tolist()


def read_im(_id, data_dir):
    with open(data_dir / f"{_id}.raw", 'rb') as f:
        data = f.read()
    meta = Meta.from_path(data_dir / f"{_id}.mhd")
    im = np.frombuffer(data, dtype=np.short).reshape(meta.dim_size[2], *meta.dim_size[:2])
    return im, meta
