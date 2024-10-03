import numpy as np
# import pandas as pd
import torch
from torch.utils.data import Dataset
# from tqdm import tqdm
from pathlib import Path
import SimpleITK as sitk
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from constants import INPUT_SIZE, CHANNELS, STATE


class OCTDataset(Dataset):
    def __init__(
        self,
        data_dir: Path = Path('./data'),
        validation: bool = False,
    ):
        self.data_dir = data_dir
        self.validation = validation
        self.files = [file for file in list((self.data_dir / "test" / "labels").glob('./*')) if file != Path('data/test/labels/.DS_Store')]
        self.size = len(self.files)
        
        indices = np.arange(self.size)
        self.train_files, self.val_files = train_test_split(self.files, test_size=0.2, random_state=STATE)
        
        if self.validation:
            # self.dataframe = pd.DataFrame(val_indices, columns=["indices"])
            self.load_images(self.val_files)
        else:
            self.load_images(self.train_files)


    def load_images(self, files):
        #TODO: make it work for tr and ts
        size = len(files)
        self.raw_images = np.zeros((size,) + (INPUT_SIZE[0], INPUT_SIZE[1], CHANNELS), dtype=np.float32)
        self.raw_segmentation_labels = np.zeros((size,) + (INPUT_SIZE[0], INPUT_SIZE[1]), dtype=np.uint8)

        for i, file in enumerate(files):
            file_base = str(file).split("/")[-1].split(".")[0]

            image = np.zeros((INPUT_SIZE[0], INPUT_SIZE[1], CHANNELS))
            for channel in range(CHANNELS):
                img = sitk.ReadImage(
                        self.data_dir / "train" / "images" / f"{file_base}_00{channel+12}.nii.gz" #TODO: remove the +12, this is just for testing one image
                    )
                color_channel = sitk.GetArrayFromImage(img).squeeze() / 255.0
                image[:,:,channel] = color_channel
            
            # plt.imsave(f"test_{file_base}.jpg", image)

            segmentation = sitk.ReadImage(
                    self.data_dir / "train" / "labels" / f"{file_base}.nii.gz"
                )
            
            # plt.imsave(f"test_{file_base}_labels.jpg", sitk.GetArrayFromImage(segmentation).squeeze())
            
            self.raw_images[i] = image
            self.raw_segmentation_labels[i] = sitk.GetArrayFromImage(segmentation).squeeze()

    
    def __len__(self):
        return len(self.raw_images)
        
    def __getitem__(self, index: int) -> dict:

        sample = {
            "image": self.raw_images[index],
            "labels": self.raw_segmentation_labels[index],
            # "malignancy_label": self._malignancy_labels[index],
            # "noduletype_label": self._noduletype_labels[index],
            # "origin": torch.from_numpy(metadata["origin"].copy()),
            # "spacing": torch.from_numpy(metadata["spacing"].copy()),
            # "transform": torch.from_numpy(metadata["transform"].copy()),
            # "shape": torch.from_numpy(metadata["shape"].copy()),
            # "noduleid": dataframe_row.noduleid,
        }

        return sample

