import numpy as np
from pathlib import Path
# import SimpleITK as sitk
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib

from constants import STATE, CLASS_DICT, CLASS_COLORS

def make_train_val_split(data_dir):
    ''' splits up train folder into train and val'''

    #get all files and file names, find unique patients
    files = [file for file in list((data_dir / "train" / "labels").glob('./*')) if file != Path('data/train/labels/.DS_Store')]
    file_names = [str(file).split("/")[-1] for file in files]
    patients = ["_".join(file.split("_")[:3]) for file in file_names] #TODO: temp :3 for testing purposes
    patients = list(set(patients)) #converting to set and back to list gets unique items

    # split up files
    train_patients, val_patients = train_test_split(patients, test_size=0.2, random_state=STATE)

    for patient in val_patients:
        # get file name and file itself for the labels of a patient
        label_file_name = str(list((data_dir / "train" / "labels").glob(f"{patient}*"))[0]).split("/")[-1]
        label_file = Path(data_dir / "train" / "labels" / label_file_name)
        label_file.rename(data_dir / "val" / "labels" / label_file_name)

        # get image files of the same patient
        patient_images = list((data_dir / "train" / "images").glob(f"{patient}*"))

        # move all image files to validation
        for image_file in patient_images:
            image_file_name = str(image_file).split("/")[-1]
            image_file.rename(data_dir / "val" / "images" / image_file_name)


def hex_to_rgb(hex: str):
    assert len(hex) == 6
    rgb = tuple(int(hex[i:i + 2], 16) for i in (0, 2, 4))
    return tuple(channel / 255 for channel in rgb) 


def create_color_map():
    cvals = list(range(15))
    norm=plt.Normalize(min(cvals),max(cvals))
    colors = [tuple(channel / 255 for channel in color) for color in list(CLASS_COLORS.values())]
    tuples = list(zip(map(norm,cvals), colors))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)

    # get new norm to discretize the values
    bounds = np.linspace(-0.5, 14.5, 16)
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    return norm, cmap


def plot_uncertainty_per_class(prediction, file_name="uncertainties_per_class"):
    pred_proba = np.mean(prediction, axis=0)

    fig, ax = plt.subplots(nrows=3, ncols=5, figsize=(15,9))
    for i, image in enumerate(pred_proba):
        ax[int(i/5), i%5].imshow(image, cmap='viridis')
        ax[int(i/5), i%5].set_xticks([])
        ax[int(i/5), i%5].set_yticks([])
        ax[int(i/5), i%5].set_title(f"{CLASS_DICT[i]}")
    plt.title("Uncertainty maps per class")
    plt.savefig(file_name + ".png")


def plot_image(prediction, file_name="output_per_class"):
    fig, ax = plt.subplots(nrows=3, ncols=5, figsize=(15,9))
    for i in range(prediction.shape[0]):
        ax[int(i/5), i%5].imshow(prediction[i], cmap='viridis')
        ax[int(i/5), i%5].set_xticks([])
        ax[int(i/5), i%5].set_yticks([])
        ax[int(i/5), i%5].set_title(f"{CLASS_DICT[i]}")
    plt.savefig(file_name + ".png")


def plot_labels(labels, file_name = "labels"):
    norm, mycmap = create_color_map()
    fig = plt.figure(figsize=(6,6))

    plt.imshow(labels.reshape((704,704,1)), cmap=mycmap, norm=norm)

    plt.axis("off")
    plt.title("labels")
    plt.colorbar()
    plt.savefig(file_name + ".png", dpi=800) #high dpi to prevent blending of colors between classes


def plot_image_overlay_labels(image, labels, file_name = "image_overlay_labels", alpha=0.5):
    norm, mycmap = create_color_map()
    fig = plt.figure(figsize=(6,6))
    # print(image.shape)

    if image.shape[0] == 3:
        image = image.transpose(1,2,0)


    plt.imshow(image)
    plt.imshow(labels.reshape((704,704,1)), cmap=mycmap, norm=norm, alpha=alpha)

    plt.axis("off")
    plt.title("image overlayed with label")
    plt.savefig(file_name + ".png", dpi=800) #high dpi to prevent blending of colors between classes
