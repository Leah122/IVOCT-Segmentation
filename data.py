import numpy as np
# import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
# from tqdm import tqdm
from pathlib import Path
import SimpleITK as sitk
import matplotlib.pyplot as plt

from constants import INPUT_SIZE, CHANNELS, STATE

class OCTDataset(Dataset):
    def __init__(
        self,
        data_dir: Path = Path('./data'),
        validation: bool = False,
        test: bool = False,
    ):
        self.validation = validation
        self.test = test

        # set data directory depending on data split
        if self.test:
            self.data_dir = data_dir / "test" 
        elif self.validation:
            self.data_dir = data_dir / "val" 
        else:
            self.data_dir = data_dir / "train"

        # get label files from the data directory
        self.files = [file for file in list((self.data_dir / "labels").glob('./*')) if file != (self.data_dir / "labels" / ".DS_Store")]
        self.size = len(self.files)
        
        # print(self.files)

        self.load_images(self.files)


    def load_images(self, files):
        size = len(files)
        self.raw_images = np.zeros((size,) + (INPUT_SIZE[0], INPUT_SIZE[1], CHANNELS), dtype=np.float32)
        self.raw_segmentation_labels = np.zeros((size,) + (INPUT_SIZE[0], INPUT_SIZE[1]), dtype=np.uint8)
        self.metadata = []

        for i, file in enumerate(files):
            file_base = str(file).split("/")[-1].split(".")[0]
            # print(file_base)

            image = np.zeros((INPUT_SIZE[0], INPUT_SIZE[1], CHANNELS))
            for channel in range(CHANNELS):
                img = sitk.ReadImage(
                        self.data_dir / "images" / f"{file_base}_00{channel+12}.nii.gz" #TODO: remove the +12, this is just for testing one image
                    )
                color_channel = sitk.GetArrayFromImage(img).squeeze() / 255.0
                image[:,:,channel] = color_channel
            
            # plt.imsave(f"train_{file_base}.jpg", image)

            segmentation = sitk.ReadImage(
                    self.data_dir / "labels" / f"{file_base}.nii.gz"
                )
            
            # plt.imsave(f"train_{file_base}_labels.jpg", sitk.GetArrayFromImage(segmentation).squeeze())
            
            self.raw_images[i] = image
            self.raw_segmentation_labels[i] = sitk.GetArrayFromImage(segmentation).squeeze()
            self.metadata.append({
                "file_path": str(file),
                # "origin": np.flip(image.GetOrigin()),
                # "spacing": np.flip(image.GetSpacing()),
                # "transform": np.array(np.flip(image.GetDirection())).reshape(3, 3),
                # "shape": np.flip(image.GetSize()),
            })



    def __len__(self):
        return len(self.raw_images)
        
    def __getitem__(self, index: int) -> dict:

        sample = {
            "image": self.raw_images[index],
            "labels": self.raw_segmentation_labels[index],
            "metadata": self.metadata[index]
        }

        return sample


# dataset_train = OCTDataset(validation=True)

# dataloader_train = DataLoader(
#     dataset=dataset_train,
#     batch_size=1,
# )