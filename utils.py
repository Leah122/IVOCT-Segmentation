import numpy as np
from pathlib import Path
# import SimpleITK as sitk
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.nn.functional as F

from constants import STATE, NEW_CLASS_DICT, NEW_CLASS_COLORS, NEW_CLASSES

def dice_score(input, target): #TODO: is this multiclass??
    eps = 1e-6

    iflat = input.reshape((704**2))
    tflat = target.reshape((704**2))
    
    dice_per_class = np.zeros(NEW_CLASSES)

    for c in range(0, NEW_CLASSES):
        iflat_ = iflat==c
        tflat_ = tflat==c
        intersection = (iflat_ * tflat_).sum()
        union = iflat_.sum() + tflat_.sum() # removed dim=1
        if union == 0:
            dice_per_class[c] = np.nan
        else: 
            d = ((2.0 * intersection + eps) / (union + eps)).mean()
            dice_per_class[c] = d

    return dice_per_class

def soft_dice_loss(output, target, epsilon=1e-6, dims=(-2, -1)):
    numerator = 2. * torch.sum(output * target, dim=dims)
    denominator = torch.sum(output + target, dim=dims)
    return (numerator) / (denominator + epsilon)


def soft_dice(output, target, reduction='mean', dims=(-2, -1)):
    # main source: https://github.com/Nacriema/Loss-Functions-For-Semantic-Segmentation/blob/master/loss/__init__.py
    # target shape = (N, X, Y), output shape = (N, C, X, Y)
    num_classes = output.shape[1]
    if dims == -1:
        one_hot_target = F.one_hot(target.to(torch.int64), num_classes=num_classes).permute((0, 2, 1)).to(torch.float)
    else: 
        one_hot_target = F.one_hot(target.to(torch.int64), num_classes=num_classes).permute((0, 3, 1, 2)).to(torch.float)
    assert output.shape == one_hot_target.shape

    dice_per_class_batch = soft_dice_loss(output, one_hot_target, dims=dims)

    # set dice to 1e-6 if it is predicted but is not present in the image
    for i in range(dice_per_class_batch.shape[0]):
            for j in range(dice_per_class_batch.shape[1]):
                if (output.argmax(dim=1)[i] == j).sum() > 0:
                    if dice_per_class_batch[i][j] == 0:
                        dice_per_class_batch[i][j] += 1e-6

    mask = dice_per_class_batch!=0
    dice_per_class = (dice_per_class_batch*mask).sum(dim=0)/mask.sum(dim=0)
    if reduction == 'dice_per_class': # gives dice per class (so not dice loss!!)
        return dice_per_class
    elif reduction == 'mean': # gives dice loss 
        return 1.0 - torch.mean((dice_per_class_batch[dice_per_class_batch != 0])[1:])
    elif reduction == 'none':
        return dice_per_class_batch
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")


def make_train_val_split(data_dir):
    ''' splits up train folder into train and val'''

    #get all files and file names, find unique patients
    files = [file for file in list((data_dir / "train" / "labels").glob('./*')) if file != Path('data/train/labels/.DS_Store')]
    file_names = [str(file).split("/")[-1] for file in files]
    patients = ["_".join(file.split("_")[:2]) for file in file_names] #TODO: temp :3 for testing purposes
    patients = list(set(patients)) #converting to set and back to list gets unique items
    print(f"INFO: patients found: {patients}")
    print(f"INFO: total of {len(patients)} patients found")

    # split up files
    train_patients, val_patients = train_test_split(patients, test_size=0.2, random_state=STATE)
    print(f"INFO: moving {len(val_patients)} patients to val")

    for patient in val_patients:
        # get file name and file itself for the labels of a patient
        label_file_names = [str(file).split("/")[-1] for file in list((data_dir / "train" / "labels").glob(f"./*{patient}*"))]
        print(f"INFO: {len(label_file_names)} label files: ", label_file_names)
        for label_file_name in label_file_names:
            label_file = Path(data_dir / "train" / "labels" / label_file_name)
            label_file.rename(data_dir / "val" / "labels" / label_file_name)
            # print(f"INFO: moved labels for patient {label_file_name} to val")

        # get image files of the same patient
        patient_images = list((data_dir / "train" / "images").glob(f"{patient}*"))
        print(f"INFO: {len(patient_images)} image files: ", patient_images)
        # move all image files to validation
        for image_file in patient_images:
            image_file_name = str(image_file).split("/")[-1]
            image_file.rename(data_dir / "val" / "images" / image_file_name)
            # print(f"INFO: moved image for patient {image_file_name} to val")


def hex_to_rgb(hex: str):
    assert len(hex) == 6
    rgb = tuple(int(hex[i:i + 2], 16) for i in (0, 2, 4))
    return tuple(channel / 255 for channel in rgb) 


def create_color_map():
    cvals = list(range(NEW_CLASSES))
    norm=plt.Normalize(min(cvals),max(cvals))
    colors = [tuple(channel / 255 for channel in color) for color in list(NEW_CLASS_COLORS.values())]
    tuples = list(zip(map(norm,cvals), colors))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)
    return norm, cmap


def plot_softmax_labels_per_class(prediction, file_path = ".", file_name="uncertainties_per_class"):
    pred_proba = np.mean(prediction, axis=0)

    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(6,6))
    for i, image in enumerate(pred_proba[1:]):
        # create color map
        color = [tuple((0,0,0)), tuple(channel / 255 for channel in list(NEW_CLASS_COLORS.values())[i+1])] # +1 because skipping background
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", color)

        ax[int(i/3), i%3].imshow(image, vmin=0, vmax=1)
        ax[int(i/3), i%3].set_xticks([])
        ax[int(i/3), i%3].set_yticks([])
        ax[int(i/3), i%3].set_title(f"{NEW_CLASS_DICT[i+1]}")
    fig.suptitle(f"average softmax over MC samples")
    plt.savefig(file_path + "/" + file_name + ".png", dpi=800)
    plt.close()

def plot_uncertainty_per_class(uncertainty_map, file_path = ".", file_name="uncertainties_per_class", metric=""):
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(6,6))
    for i, image in enumerate(uncertainty_map[1:]):
        # create color map
        color = [tuple((0,0,0)), tuple(channel / 255 for channel in list(NEW_CLASS_COLORS.values())[i+1])] # +1 because skipping background
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", color)

        cb = ax[int(i/3), i%3].imshow(image, vmin=0, vmax=1)
        ax[int(i/3), i%3].set_xticks([])
        ax[int(i/3), i%3].set_yticks([])
        ax[int(i/3), i%3].set_title(f"{NEW_CLASS_DICT[i+1]}")
        # fig.colorbar(cb, ax=ax)
    fig.suptitle(f"uncertainty map per class: {metric}")
    # plt.colorbar()
    plt.savefig(file_path + "/" + file_name + ".png", dpi=800)
    plt.close()


def plot_image(prediction, file_path = ".", file_name="output_per_class"):
    fig, ax = plt.subplots(nrows=3, ncols=5, figsize=(15,9))
    for i in range(prediction.shape[0]):
        ax[int(i/5), i%5].imshow(prediction[i], cmap='viridis')
        ax[int(i/5), i%5].set_xticks([])
        ax[int(i/5), i%5].set_yticks([])
        ax[int(i/5), i%5].set_title(f"{NEW_CLASS_DICT[i]}")
    plt.savefig(file_path + "/" + file_name + ".png")
    plt.close()


def plot_labels(labels, file_path = ".", file_name = "labels"):
    norm, mycmap = create_color_map()
    fig = plt.figure(figsize=(6,6))

    plt.imshow(labels.reshape((704,704,1)), cmap=mycmap, norm=norm)

    plt.axis("off")
    plt.title("labels")
    plt.colorbar()
    plt.savefig(file_path + "/" + file_name + ".png", dpi=800) #high dpi to prevent blending of colors between classes
    plt.close()


def plot_image_overlay_labels(image, labels, file_path = ".", file_name = "image_overlay_labels", alpha=0.5):
    norm, mycmap = create_color_map()
    fig = plt.figure(figsize=(6,6))

    if image.shape[0] == 3:
        image = image.transpose(1,2,0)

    plt.imshow(image)
    plt.imshow(labels.reshape((704,704,1)), cmap=mycmap, norm=norm, alpha=alpha)

    plt.axis("off")
    plt.title("image overlayed with label/prediction")
    plt.savefig(file_path + "/" + file_name + ".png", dpi=800) #high dpi to prevent blending of colors between classes
    plt.close()


def plot_uncertainty(uncertainty_map, file_path = ".", file_name = "image_overlay_labels", metric = ""):
    fig = plt.figure(figsize=(6,6))
    plt.imshow(uncertainty_map.reshape((704,704,1)))

    plt.axis("off")
    plt.title(f"uncertainty map {metric}")
    plt.colorbar()
    plt.savefig(file_path + "/" + file_name + ".png", dpi=800) #high dpi to prevent blending of colors between classes
    plt.close()


def plot_metrics(metrics, metrics_to_plot, file_path = ".", file_name = "image_overlay_labels"):
    fig = plt.figure(figsize=(6,6))
    for metric in metrics_to_plot:
        metric_train = [epoch[metric] for epoch in metrics["train"]]
        metric_test = [epoch[metric] for epoch in metrics["valid"]]
        plt.plot(metric_train, label=f"train_{metric}")
        plt.plot(metric_test, label=f"valid_{metric}")

    plt.legend()
    plt.savefig(file_path + "/" + file_name + ".png", dpi=800)


def plot_roc_curve(fpr, tpr, file_path = ".", file_name = "ROC_curve"):
    fig = plt.figure(figsize=(6,6))

    plt.plot(fpr, tpr)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    plt.title("ROC")
    plt.savefig(file_path + "/" + file_name + ".png") #high dpi to prevent blending of colors between classes
    plt.close()


def plot_image_prediction_certain(image, predictions, uncertainty_map, file_path = ".", file_name = "image_overlay_labels", alpha=0.5, percentile=95, p=None):
    if p == None:
        p = np.percentile(uncertainty_map.flatten(), percentile)

    certain = np.where(uncertainty_map < p, uncertainty_map, 2)
    predictions = predictions.mean(axis=0).argmax(axis=0).squeeze()
    predictions_certain = np.where(certain != 2, predictions, 0)

    norm, mycmap = create_color_map()
    fig = plt.figure(figsize=(6,6))

    if image.shape[0] == 3:
        image = image.transpose(1,2,0)

    plt.imshow(image)
    plt.imshow(predictions_certain.reshape((704,704,1)), cmap=mycmap, norm=norm, alpha=alpha)

    plt.axis("off")
    plt.title(f"image overlayed with certain predictions with {percentile}th percentile")
    plt.savefig(file_path + "/" + file_name + ".png", dpi=800) #high dpi to prevent blending of colors between classes
    plt.close()



def plot_image_prediction_certain_structures(image, predictions, uncertainty_per_class, file_path = ".", file_name = "image_overlay_labels", alpha=0.5, percentile=95, p=None):
    p = []
    for i in range(NEW_CLASSES):
        p.append(np.percentile(uncertainty_per_class.mean(axis=(-1, -2)).flatten(), percentile))

    certain = np.where(uncertainty_per_class < p, uncertainty_per_class, 2)
    predictions = predictions.mean(axis=0).argmax(axis=0).squeeze()
    predictions_certain = np.where(certain != 2, predictions, 0)

    norm, mycmap = create_color_map()
    fig = plt.figure(figsize=(6,6))

    if image.shape[0] == 3:
        image = image.transpose(1,2,0)

    plt.imshow(image)
    plt.imshow(predictions_certain.reshape((704,704,1)), cmap=mycmap, norm=norm, alpha=alpha)

    plt.axis("off")
    plt.title(f"image overlayed with certain predictions with {percentile}th percentile")
    plt.savefig(file_path + "/" + file_name + ".png", dpi=800) #high dpi to prevent blending of colors between classes
    plt.close()

    
