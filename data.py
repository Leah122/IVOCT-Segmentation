import numpy as np
# import pandas as pd
import torch
from torch.utils.data import Dataset
# from tqdm import tqdm
from pathlib import Path
import SimpleITK as sitk
import torchvision.transforms.functional as tf
import torchvision
torch.manual_seed(17)
import random
from tqdm import tqdm

from constants import INPUT_SIZE, CHANNELS, NEW_CLASSES, UNCERTAIN

class OCTDataset(Dataset):
    def __init__(
        self,
        data_dir: Path = Path('./data'),
        validation: bool = False,
        test: bool = False,
        polish: bool = False,
        aug_rotations: int = 0,
        aug_flip_chance: float = 0,
        aug_color_chance: float = 0,
        aug_gaussian_sigma: float = 0,
        class_weight_power: float = 2.0,
        debugging: bool = False,
        uncertain: bool = False,
    ): 
        
        self.validation = validation
        self.test = test
        self.aug_rotations = aug_rotations
        self.aug_flip_chance = aug_flip_chance
        self.aug_color_chance = aug_color_chance
        self.aug_gaussian_sigma = aug_gaussian_sigma
        self.class_weight_power = class_weight_power
        self.debugging = debugging
        self.polish = polish
        self.uncertain = uncertain

        # set data directory depending on data split
        if self.test:
            self.data_dir = data_dir / "test" 
        elif self.validation:
            self.data_dir = data_dir / "val" 
        elif self.polish:
            self.data_dir = data_dir / "polish" 
        else:
            self.data_dir = data_dir / "train"

        # get label files from the data directory
        self.files = [file for file in list((self.data_dir / "labels").glob('./*')) if file != (self.data_dir / "labels" / ".DS_Store")]
        
        if self.polish: # polish data has no labels, so we get the file names directly from the images
            self.files = [file for file in list((self.data_dir / "images").glob('./*')) if file != (self.data_dir / "images" / ".DS_Store")]
            self.files = ["_".join(str(file).split("_")[:-1])+".nii.gz" for file in self.files]
            self.files = list(set(self.files))


        if self.debugging:
            self.files = self.files[:50]
        if self.uncertain:
            self.files = [file for file in self.files if any(frame in str(file) for frame in UNCERTAIN)]
            print(self.files)

        self.size = len(self.files)
        
        self.load_images()
        self.compute_class_weights()

    def preprocess_labels(self, labels):
        labels[labels == 9] = 1
        labels[labels == 10] = 1
        labels[labels == 12] = 1
        labels[labels == 11] = 3
        labels[labels == 14] = 3
        labels[labels == 13] = 9
        return labels


    def augment(self, index):
        image = torch.Tensor(self.raw_images[index]).permute(2,0,1)
        labels = torch.Tensor(self.raw_segmentation_labels[index].reshape((1,704,704)))
        
        hflip = False
        vflip = False

        # Random flipping
        if random.random() < self.aug_flip_chance:
            image = tf.hflip(image)
            labels = tf.hflip(labels)
            hflip = True

        if random.random() < self.aug_flip_chance:
            image = tf.vflip(image)
            labels = tf.vflip(labels)
            vflip = True

        # rotations 
        rotation = random.randint(-self.aug_rotations, self.aug_rotations)
        image = tf.rotate(image, rotation)
        labels = tf.rotate(labels, rotation, fill=0)

        # color changes
        if random.random() < self.aug_color_chance:
            colorJitter = torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)
            image = colorJitter(image)

        # random gaussian noise
        sigma = random.random()
        if sigma < self.aug_gaussian_sigma:
            image = tf.gaussian_blur(image, kernel_size=(5,5), sigma=sigma)

        self.metadata[index].update({
            "hflip": hflip,
            "vflip": vflip,
            "rotation": rotation,
        })

        return image, labels


    def load_images(self):
        size = len(self.files)
        self.raw_images = np.zeros((size,) + (INPUT_SIZE[0], INPUT_SIZE[1], CHANNELS), dtype=np.float32)
        self.raw_segmentation_labels = np.zeros((size,) + (INPUT_SIZE[0], INPUT_SIZE[1]), dtype=np.uint8)
        self.metadata = []

        for i, file in tqdm(enumerate(self.files), total=size, desc="Loading data"):
            file_base = str(file).split("/")[-1].split(".")[0]

            image = np.zeros((INPUT_SIZE[0], INPUT_SIZE[1], CHANNELS))
            for channel in range(CHANNELS):
                img = sitk.ReadImage(
                        self.data_dir / "images" / f"{file_base}_000{channel}.nii.gz"
                    )
                color_channel = sitk.GetArrayFromImage(img).squeeze() / 255.0
                image[:,:,channel] = color_channel
            
            if not self.polish: # polish data does not have segmentations
                segmentation = sitk.ReadImage(
                        self.data_dir / "labels" / f"{file_base}.nii.gz"
                    )
            
                labels = sitk.GetArrayFromImage(segmentation).squeeze()
                labels = self.preprocess_labels(labels)
                self.raw_segmentation_labels[i] = labels

            self.raw_images[i] = image
            self.metadata.append({
                "file_name": str(file_base),
                # "origin": np.flip(image.GetOrigin()),
                # "spacing": np.flip(image.GetSpacing()),
                # "transform": np.array(np.flip(image.GetDirection())).reshape(3, 3),
                # "shape": np.flip(image.GetSize()),
            })


    def compute_class_weights(self):
        self.class_weights = []
        for c in range(NEW_CLASSES):
            self.class_weights.append(np.size(self.raw_segmentation_labels) / np.sum(self.raw_segmentation_labels == c))
        
        self.class_weights = [np.power(weight/np.sum(self.class_weights), 1/self.class_weight_power) for weight in self.class_weights]
        

    def __len__(self):
        return len(self.raw_images)
        

    def __getitem__(self, index: int) -> dict:
        image, labels = self.augment(index)

        sample = {
            "image": image,
            "labels": labels,
            "metadata": self.metadata[index]
        }

        return sample


# dataset_train = OCTDataset(validation=True)

# dataloader_train = DataLoader(
#     dataset=dataset_train,
#     batch_size=1,
# )