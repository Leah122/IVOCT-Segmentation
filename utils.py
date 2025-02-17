import numpy as np
from pathlib import Path
# import SimpleITK as sitk
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.nn.functional as F
from scipy.ndimage import convolve
from tqdm import tqdm
import einops
import torch.nn as nn
from torch.nn.functional import grid_sample
import pandas as pd
import regex as re
from matplotlib.patches import Patch
import torchvision.transforms.functional as tf
import seaborn as sns
from scipy.ndimage.interpolation import geometric_transform
import cv2
from scipy.stats import mannwhitneyu
from sklearn.metrics import confusion_matrix

from model2 import U_Net
from constants import STATE, NEW_CLASS_DICT, NEW_CLASS_COLORS, NEW_CLASSES, NEW_CLASS_DICT_CAPITAL
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

plt.rcParams.update({'font.size': 14})


# loading models

def load_model_mc(save_dir, model_id, dropout=0.2, device=device):
    model = U_Net(dropout=dropout, softmax=True).to(device)

    checkpoint = torch.load(save_dir / f"last_model_{model_id}.pth", weights_only=True)#, map_location=torch.device('cpu')) #TODO: added map location temporarily
    model.load_state_dict(checkpoint)
    model.eval()
    if dropout > 0:
        for m in model.modules(): # turn dropout back on
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
    return model


def load_model_tta(save_dir, model_id):
    model = U_Net(dropout=0.0, softmax=True).to(device)

    checkpoint = torch.load(save_dir / f"last_model_{model_id}.pth", weights_only=True, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.eval()

    return model


def load_model_ensemble(save_dir, model_id, nr_models):
    models = []

    for i in range(nr_models):
        model = U_Net(dropout=0.0, softmax=True).to(device)
        checkpoint = torch.load(save_dir / f"model_{i}" / f"last_model_{model_id}.pth", weights_only=True, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)
        model.eval()

        models.append(model)
    return models



# generating samples and augmentation 

def augment(image, samples = 10):
        flips = [[0,0], [0,1], [1,0], [1,1]]
        rotations = [0, 60, 120, 240, 300]

        augmentations = []
        for i in range(len(flips)):
            for j in range(len(rotations)):
                augmentations.append(flips[i] + [rotations[j]])

        images_aug = []
        augmentations = augmentations[:samples]

        for aug in augmentations:
            image_aug = tf.rotate(image, aug[2], fill=0)
            
            if aug[0] == 1:
                image_aug = tf.hflip(image_aug)

            if aug[1] == 1:
                image_aug = tf.vflip(image_aug)

            images_aug.append(image_aug)
            del image_aug

        return images_aug, augmentations


def reverse_augment(images, augmentations):
    outputs = []

    for aug, image in zip(augmentations, images):
    
        if aug[1] == 1:
            image = tf.vflip(image)

        if aug[0] == 1:
            image = tf.hflip(image)

        image = tf.rotate(image, 360-aug[2], fill=[1,0,0,0,0,0,0,0,0,0])
        
        outputs.append(image.detach().cpu().numpy().squeeze())

    return outputs


def make_predictions_mc(sample, model, nr_samples):
    image = sample["image"].to(device)
    labels = sample["labels"].detach().cpu().numpy()

    with torch.no_grad():
        outputs = np.zeros((nr_samples, NEW_CLASSES, 704, 704))
        for mc in range(nr_samples):
            outputs[mc] = model(image).detach().cpu().numpy().squeeze()

    image = image.detach().cpu().numpy()

    return outputs, image, labels


def make_predictions_ensemble(sample, model, nr_samples):
    image = sample["image"].to(device)
    labels = sample["labels"].detach().cpu().numpy()

    with torch.no_grad():
        outputs = np.zeros((nr_samples, NEW_CLASSES, 704, 704))
        for i, model_i in enumerate(model):
            outputs[i] = model_i(image).detach().cpu().numpy().squeeze()

    image = image.detach().cpu().numpy()

    return outputs, image, labels


def make_predictions_tta(sample, model, nr_samples):
    image = sample["image"].to(device)
    labels = sample["labels"].detach().cpu().numpy()

    images, augmentations = augment(image, nr_samples)

    with torch.no_grad():
        outputs = [] 
        for image_i in images:
            outputs.append(model(image_i).squeeze())
    
    outputs = reverse_augment(outputs, augmentations)

    image = image.detach().cpu().numpy()

    return np.array(outputs), image, labels



# training functions

def dice_score(input, target): 
    eps = 1e-6

    iflat = input.reshape(-1)
    tflat = target.reshape(-1)
    
    dice_per_class = np.zeros(NEW_CLASSES)

    for c in range(0, NEW_CLASSES):
        iflat_ = iflat==c
        tflat_ = tflat==c
        intersection = (iflat_ * tflat_).sum()
        union = iflat_.sum() + tflat_.sum()
        if union == 0 and intersection == 0:
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
    dice_per_class = (dice_per_class_batch*mask).sum(dim=0)/mask.sum(dim=0) # masks 0's, so true negatives are not counted.
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



# plotting

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


def plot_legend_colorbar(file_path = "."): #TODO: test
    # plot a seperate legend for classes and color bar for entropies.
    # https://matplotlib.org/stable/gallery/text_labels_and_annotations/custom_legends.html
    colors = [tuple(channel / 255 for channel in color) for color in list(NEW_CLASS_COLORS.values())]
    # norm, cmap = create_color_map()

    legend_elements = []
    for c in range(NEW_CLASSES):
        legend_elements.append(Patch(facecolor=colors[c], edgecolor=colors[c], label=NEW_CLASS_DICT_CAPITAL[c]))

    fig, ax = plt.subplots()
    ax.legend(handles=legend_elements, loc='center')
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0, 1), cmap='viridis'), ax=ax, orientation='vertical')

    plt.savefig(file_path + "/legend_colorbar.png", dpi=600, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots()
    ax.legend(handles=legend_elements, loc='upper center', ncol=10)
    plt.savefig(file_path + "/legend_horizontal.png", dpi=600, bbox_inches='tight')
    plt.close()


def plot_softmax_labels_per_class(prediction, file_path = ".", file_name="uncertainties_per_class", rows = 2, cols = 5):
    pred_proba = np.mean(prediction, axis=0)

    fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(cols*2,rows*2))
    for i, image in enumerate(pred_proba[1:]):
        # create color map
        color = [tuple((0,0,0)), tuple(channel / 255 for channel in list(NEW_CLASS_COLORS.values())[i+1])] # +1 because skipping background
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", color)

        ax[int(i/cols), int(i%cols)].imshow(image, vmin=0, vmax=1)
        ax[int(i/cols), int(i%cols)].set_xticks([])
        ax[int(i/cols), int(i%cols)].set_yticks([])
        ax[int(i/cols), int(i%cols)].set_title(f"{NEW_CLASS_DICT_CAPITAL[i+1]}")
    fig.suptitle(f"average softmax over MC samples")
    plt.savefig(file_path + "/" + file_name + ".png", dpi=600, bbox_inches='tight')
    plt.close()


def plot_uncertainty_per_class(uncertainty_map, file_path = ".", file_name="uncertainties_per_class", metric="", rows = 2, cols = 5):
    fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(cols*3+1,rows*3))
    for i, image in enumerate(uncertainty_map):
        im = ax[int(i/cols), int(i%cols)].imshow(image)
        ax[int(i/cols), int(i%cols)].set_xticks([])
        ax[int(i/cols), int(i%cols)].set_yticks([])
        ax[int(i/cols), int(i%cols)].set_title(f"{NEW_CLASS_DICT_CAPITAL[i]}", fontsize=14)
    fig.colorbar(im, ax=ax.ravel().tolist(), fraction=0.046)
    # plt.colorbar(shrink=0.8)
    plt.savefig(file_path + "/" + file_name + ".png", dpi=600, bbox_inches='tight')
    plt.close()


def plot_labels(labels, file_path = ".", file_name = "labels"):
    norm, mycmap = create_color_map()
    fig = plt.figure(figsize=(5,5))

    plt.imshow(labels.reshape((704,704,1)), cmap=mycmap, norm=norm)

    plt.axis("off")
    plt.colorbar(shrink=0.7)
    plt.savefig(file_path + "/" + file_name + ".png", dpi=600, bbox_inches='tight') #high dpi to prevent blending of colors between classes
    plt.close()


def plot_image_overlay_labels(image, labels, file_path = ".", file_name = "image_overlay_labels", alpha=0.5, title=None):
    norm, mycmap = create_color_map()
    fig = plt.figure(figsize=(4,4))

    if image.shape[0] == 3:
        image = image.transpose(1,2,0)

    plt.imshow(image)
    plt.imshow(labels.reshape((704,704,1)), cmap=mycmap, norm=norm, alpha=alpha)

    if title: plt.title(title, fontsize=21)#, fontweight="bold")
    plt.axis("off")
    plt.savefig(file_path + "/" + file_name + ".png", dpi=600, bbox_inches='tight') #high dpi to prevent blending of colors between classes
    plt.close()


def plot_image_overlay_labels_variable_alpha(image, labels, uncertainty_map, file_path = ".", file_name = "image_overlay_labels", title=None):
    norm, mycmap = create_color_map()
    fig = plt.figure(figsize=(4,4))

    labels = labels.reshape((704,704,1))
    
    normalised_uncertainty_map = (uncertainty_map - np.min(uncertainty_map))/(np.max(uncertainty_map) - np.min(uncertainty_map))

    # colors = list(set([mycmap(a) for a in np.linspace(0,1,NEW_CLASSES)]))
    labels_alpha = np.zeros((704,704,4))
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            color = NEW_CLASS_COLORS[labels[i,j,0]]
            # color = colors[labels[i,j,0]]
            labels_alpha[i,j,:] = [color[0], color[1], color[2], (1-normalised_uncertainty_map[i,j])]

    plt.imshow(labels_alpha)
    plt.gcf().set_facecolor("black")

    if title: plt.title(title, fontsize=21)#, fontweight="bold")
    plt.axis("off")
    plt.savefig(file_path + "/" + file_name + ".png", dpi=600, bbox_inches='tight') #high dpi to prevent blending of colors between classes
    plt.close()


def plot_image_overlay_labels_separated(image, labels, file_path = ".", file_name = "image_overlay_labels", alpha=0.5, title=None):
    norm, mycmap = create_color_map()
    classes = [2, 7, 1, 8, 3, 6, 4, 5, 9]

    for i in range(len(classes)+1):
        fig = plt.figure(figsize=(4,4))

        if image.shape[0] == 3:
            image = image.transpose(1,2,0)

        plt.imshow(image)

        mask = np.in1d(labels, classes[i:]+[0]).reshape(704, 704)
        print("mask shape: ", mask.shape)

        labels_separated = np.ma.masked_where(mask, labels.squeeze())
        plt.imshow(labels_separated.reshape((704,704,1)), cmap=mycmap, norm=norm, alpha=alpha)

        if title: plt.title(title, fontsize=21)#, fontweight="bold")
        plt.axis("off")
        plt.savefig(file_path + "/" + file_name + f"_{i}.png", dpi=1000, bbox_inches='tight') #high dpi to prevent blending of colors between classes
        plt.close()


def plot_uncertainty(uncertainty_map, file_path = ".", file_name = "image_overlay_labels", metric = "",):
    fig = plt.figure(figsize=(5,5))
    plt.imshow(uncertainty_map.reshape((704,704,1)))

    plt.axis("off")
    plt.colorbar(shrink=0.7)
    plt.savefig(file_path + "/" + file_name + ".png", dpi=600, bbox_inches='tight') #high dpi to prevent blending of colors between classes
    plt.close()


def plot_metrics(metrics, metrics_to_plot, file_path = ".", file_name = "training_metrics"):
    fig = plt.figure(figsize=(5,5))
    for metric in metrics_to_plot:
        metric_train = [epoch[metric] for epoch in metrics["train"]]
        metric_test = [epoch[metric] for epoch in metrics["valid"]]
        plt.plot(metric_train, label=f"train_{metric}")
        plt.plot(metric_test, label=f"valid_{metric}")

    plt.legend()
    plt.savefig(file_path + "/" + file_name + ".png", bbox_inches='tight')


def plot_roc_curve(fpr, tpr, file_path = ".", file_name = "ROC_curve"):
    fig = plt.figure(figsize=(4,4))

    plt.plot(fpr, tpr)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    plt.savefig(file_path + "/" + file_name + ".png", bbox_inches='tight') 
    plt.close()


def plot_image_prediction_certain(image, predictions, uncertainty_map, file_path = ".", file_name = "certain_pixels", alpha=0.5, percentile=95, p=None, title=None):
    if p == None:
        p = np.percentile(uncertainty_map.flatten(), percentile)

    # certain = np.where(uncertainty_map < p, uncertainty_map, -1)
    predictions = predictions.mean(axis=0).argmax(axis=0).squeeze()
    # predictions_certain = np.ma.masked_where(certain == -1, predictions)
    predictions_certain = np.where(uncertainty_map < p, predictions, 0)

    norm, mycmap = create_color_map()
    fig = plt.figure(figsize=(4,4))

    if image.shape[0] == 3:
        image = image.transpose(1,2,0)

    plt.imshow(image)
    plt.imshow(predictions_certain.reshape((704,704,1)), cmap=mycmap, norm=norm, alpha=alpha)

    if title: plt.title(title, fontsize=21)#, fontweight="bold")
    plt.axis("off")
    plt.savefig(file_path + "/" + file_name + ".png", dpi=800, bbox_inches='tight')
    plt.close()


def plot_image_prediction_wrong(image, predictions, labels, file_path, file_name):
    wrong_mask = np.ma.masked_where(predictions == labels, predictions)

    plot_image_overlay_labels(image, wrong_mask, file_path = file_path, file_name = file_name, alpha=0.6)


def plot_image_prediction_one_class(uncertainty_map, predictions, file_path, file_name, clas):
    class_mask = np.where(predictions == clas, 1, 0) # if target class put 1, otherwise 0
    class_mask = np.ma.masked_where(class_mask == 1, class_mask)

    norm, mycmap = create_color_map()

    fig = plt.figure(figsize=(5,5))

    plt.imshow(uncertainty_map.reshape((704,704,1)))
    plt.imshow(class_mask.reshape((704,704,1)), cmap=mycmap, norm=norm, alpha=0.5)

    plt.axis("off")
    plt.savefig(file_path + "/" + file_name + ".png", dpi=600, bbox_inches='tight') #high dpi to prevent blending of colors between classes
    plt.close()



# def normalise_uncertainty_per_class(uncertainty, prediction, clas):
#     class_fraction = len(prediction[prediction == clas]) / len(prediction.flatten())
#     uncertainty_score = uncertainty * class_fraction
#     return uncertainty_score


# def normalise_uncertainty_lipid_calcium(uncertainty_map_per_class, prediction, clas="calcium"):
#     if clas == "lipid":
#         class_value = 4
#     else:
#         class_value = 5

#     fraction = len(prediction[prediction == class_value]) / len(prediction.flatten())
#     normalised_uncertainty_map = uncertainty_map_per_class[class_value] * fraction

#     return normalised_uncertainty_map


def plot_uncertainty_vs_vessel_fraction(uncertainty_scores_en, uncertainty_scores_mi, vessel_fractions, save_dir):
    plt.figure(figsize=(5,5))
    plt.scatter(uncertainty_scores_en, vessel_fractions, label="Entropy", color="cornflowerblue")
    plt.scatter(uncertainty_scores_mi, vessel_fractions, label="Mutual Information", color="mediumpurple")

    uncertainty_scores_en, vessel_fractions_en = zip(*sorted(zip(uncertainty_scores_en, vessel_fractions)))
    z = np.polyfit(uncertainty_scores_en, vessel_fractions_en, 1)
    p_en = np.poly1d(z)
    plt.plot(uncertainty_scores_en, p_en(uncertainty_scores_en),"--", color="cornflowerblue")

    uncertainty_scores_mi, vessel_fractions_mi = zip(*sorted(zip(uncertainty_scores_mi, vessel_fractions)))
    z = np.polyfit(uncertainty_scores_mi, vessel_fractions_mi, 1)
    p_mi = np.poly1d(z)
    plt.plot(uncertainty_scores_mi, p_mi(uncertainty_scores_mi),"--", color="mediumpurple")

    plt.xlabel("uncertainty scores")
    plt.ylabel("vessel fraction")
    plt.legend()
    plt.savefig(save_dir + "/uncertainty_vs_vessel_fraction.png", bbox_inches='tight')

    plt.figure(figsize=(5,5))
    plt.scatter(uncertainty_scores_en, vessel_fractions_en, label="Entropy", color="cornflowerblue")
    plt.plot(uncertainty_scores_en, p_en(uncertainty_scores_en),"--", color="cornflowerblue")

    plt.xlabel("uncertainty scores")
    plt.ylabel("vessel fraction")
    plt.legend()
    plt.savefig(save_dir + "/uncertainty_vs_vessel_fraction_en.png", bbox_inches='tight')

    plt.figure(figsize=(5,5))
    plt.scatter(uncertainty_scores_mi, vessel_fractions_mi, label="mutual information", color="mediumpurple")
    plt.plot(uncertainty_scores_mi, p_mi(uncertainty_scores_mi),"--", color="mediumpurple")

    plt.xlabel("uncertainty scores")
    plt.ylabel("vessel fraction")
    plt.legend()
    plt.savefig(save_dir + "/uncertainty_vs_vessel_fraction_mi.png", bbox_inches='tight')


# def neighbouring(image, clas): 
#     neighbourhood = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
#     for i in range(1, 703):
#         for j in range(1, 703):
#             if image[i,j] == clas:
#                 neighbours = [image[i-1,j], image[i+1,j], image[i,j-1], image[i,j+1], image[i-1,j-1], image[i-1,j+1], image[i+1,j-1], image[i+1,j+1]] # 8 surrounding
#                 for k in neighbours:
#                     if k != clas:
#                         neighbourhood[k] += 1
#     return neighbourhood
                
# def find_closest_dist(centroid, image, clas):
#     closest_dist = 1000
#     for i in range(704):
#         for j in range(704):
#             if image[i,j] == clas:
#                 dist = np.linalg.norm(centroid-(i,j))
#                 if dist < closest_dist:
#                     closest_dist = dist
                
#     return closest_dist


# def find_farthest_dist(centroid, image, clas):
#     farthest_dist = 0
#     for i in range(704):
#         for j in range(704):
#             if image[i,j] == clas:
#                 dist = np.linalg.norm(centroid-(i,j))
#                 if dist > farthest_dist:
#                     farthest_dist = dist
                
#     return farthest_dist


# def mispredictions(outputs, labels, clas):
#     if clas not in labels:
#         return np.nan
    
#     mispredictions = np.zeros(NEW_CLASSES)

#     for i in range(704):
#         for j in range(704):
#             if outputs[i,j] == clas and labels[i,j] != clas:
#                 mispredictions[int(labels[i,j])] += 1

#     mispredictions = [pixels/mispredictions.sum() for pixels in mispredictions]
#     return mispredictions


def plot_metric_vs_uncertainty_polish(x, y, uncertainty_metric, metric, save_dir, pearson):
    # plot an annotated feature vs a combination of method and metric with pearson coefficient
    plt.figure(figsize=(5,5))
    plt.scatter(x, y, label=uncertainty_metric, color="cornflowerblue")
    plt.xlabel(uncertainty_metric)
    plt.ylabel(metric)
    plt.title(f"pearson coefficient: {round(pearson.statistic, 5)}, p-value: {round(pearson.pvalue, 5)}")
    plt.savefig(str(save_dir) + f"/plot_{metric}_vs_{uncertainty_metric}.png", bbox_inches='tight')
    plt.close()


# def plot_distances_per_image(predictions, uncertainty_maps, distances, save_dir, file_name, bins=10):
#     # plot seperate figure for each class, bin the pixels per image, since there are too many
#     for c in range(NEW_CLASSES):
#         middle_points = []
#         bins = []
#         for i in range(predictions.shape[0]):
#             distances_class = distances[i][predictions[i] == c]
#             uncertainty_class = uncertainty_maps[i][predictions[i] == c]

#             sums, edges = np.histogram(distances_class, bins=10, weights=uncertainty_class)
#             counts, _ = np.histogram(distances_class, bins=10)
#             binned = sums / counts
            
#             middle_point = np.array([(edges[i] + edges[i+1])/2 for i in range(edges.shape[0]-1)])
#             bins.append(binned)
#             middle_points.append(middle_point)

#         fig = plt.figure(figsize=(5,5))
#         plt.scatter(middle_points, bins)
#         plt.xlabel(f"Distance to {file_name}")
#         plt.ylabel("uncertainty value")
#         plt.savefig(save_dir + f"/plot_{file_name}_{NEW_CLASS_DICT_CAPITAL[c]}.png", bbox_inches='tight')
#         plt.close()


def plot_distances(predictions, uncertainty_maps, distances, save_dir, file_name, bins=20):
    # plot seperate figure for each class, bin the pixels, since there are too many
    # for c in range(NEW_CLASSES):
    #     distances_class = distances[predictions == c]
    #     uncertainty_class = uncertainty_maps[predictions == c]

    #     sums, edges = np.histogram(distances_class, bins=bins, weights=uncertainty_class)
    #     counts, _ = np.histogram(distances_class, bins=bins)
    #     binned = sums / counts
    #     middle_points = np.array([(edges[i] + edges[i+1])/2 for i in range(edges.shape[0]-1)])

    #     fig = plt.figure(figsize=(5,5))
    #     plt.plot(middle_points, binned)
    #     plt.xlabel(f"Distances to {file_name}")
    #     plt.ylabel("Entropy")
    #     plt.savefig(save_dir + f"/plot_distances_{file_name}_{NEW_CLASS_DICT_CAPITAL[c]}.png", bbox_inches='tight')
    #     plt.close()
        
    for c in range(1, NEW_CLASSES):
        fig = plt.figure(figsize=(4,4))
        color = "tab:blue"
        edge_color = "#103e5d"
        distances_class = distances[(predictions == c) & (distances > 0)]
        uncertainty_class = uncertainty_maps[(predictions == c) & (distances > 0)]
        if len(distances_class) <= 0:
            continue

        print("test: ", distances_class.shape, c)

        if len(distances_class) > 100000: # take subset if there is too much data
            indices = np.random.choice(len(distances_class), size=100000, replace=False)
            distances_class = distances_class[indices]
            uncertainty_class = uncertainty_class[indices]

        distances_class, uncertainty_class = zip(*sorted(zip(distances_class, uncertainty_class)))
        x = []
        y = []
        bin_size = int(len(distances_class) / 200)
        for i in range(199):
            x.append(np.mean(distances_class[i*bin_size:(i+1)*bin_size]))
            y.append(np.mean(uncertainty_class[i*bin_size:(i+1)*bin_size]))
    
        plt.scatter(x, y, label=NEW_CLASS_DICT_CAPITAL[c], c=color, edgecolors=edge_color)

        # Adding the correlation line
        if file_name == "border":
            sns.regplot(x=x, y=y, scatter=False, color=edge_color, line_kws={"linewidth": 2}, logx=True)
        else:
            sns.regplot(x=x, y=y, scatter=False, color=edge_color, line_kws={"linewidth": 2})

        ax = plt.gca()
        ax.tick_params(axis='y', which='both', left=False)
        ax.tick_params(axis='x', which='both', bottom=False)
        ax.yaxis.set_tick_params(labelsize='large')
        ax.xaxis.set_tick_params(labelsize='x-large')

        name_dict = {"catheter": "catheter", "border": "closest border"}
        # plt.title(f"Distance to {name_dict[file_name]}")
        plt.xlabel(f"Euclidean distance", fontsize=14)
        plt.ylabel("Entropy", fontsize=14)
        # if file_name == "border": plt.xscale('log')
        # plt.legend()
        plt.savefig(save_dir + f"/plot_distances_{file_name}_{NEW_CLASS_DICT[c]}.png", bbox_inches='tight')
        plt.close()


def plot_distances_lipid_calcium(predictions, uncertainty_maps, distances, save_dir, file_name):
    fig = plt.figure(figsize=(4,4))
    colors = ["tab:blue", "tab:green"]
    edge_colors = ["#103e5d", "#1c641c"]
    for c in [4,5]:
        distances_class = distances[(predictions == c) & (distances > 0)]
        uncertainty_class = uncertainty_maps[(predictions == c) & (distances > 0)]
        print("test_lipid_calcium: ", distances_class.shape, c)

        if len(distances_class) > 100000: # take subset if there is too much data
            indices = np.random.choice(len(distances_class), size=100000, replace=False)
            distances_class = distances_class[indices]
            uncertainty_class = uncertainty_class[indices]

        distances_class, uncertainty_class = zip(*sorted(zip(distances_class, uncertainty_class)))
        x = []
        y = []
        bin_size = int(len(distances_class) / 300)
        for i in range(299):
            x.append(np.mean(distances_class[i*bin_size:(i+1)*bin_size]))
            y.append(np.mean(uncertainty_class[i*bin_size:(i+1)*bin_size]))
        
        plt.scatter(x, y, label=NEW_CLASS_DICT_CAPITAL[c], c=colors[c-4], edgecolors=edge_colors[c-4])

        # Adding the correlation line
        if file_name == "border":
            sns.regplot(x=x, y=y, scatter=False, color=edge_colors[c-4], line_kws={"linewidth": 2}, logx=True)
        else:
            sns.regplot(x=x, y=y, scatter=False, color=edge_colors[c-4], line_kws={"linewidth": 2})

    ax = plt.gca()
    ax.tick_params(axis='y', which='both', left=False)
    ax.tick_params(axis='x', which='both', bottom=False)
    ax.yaxis.set_tick_params(labelsize='large')
    ax.xaxis.set_tick_params(labelsize='x-large')

    name_dict = {"catheter": "catheter", "border": "closest border"}
    plt.title(f"Distance to {name_dict[file_name]}")
    plt.xlabel(f"Euclidean distance", fontsize=14)
    plt.ylabel("Entropy", fontsize=14)
    if file_name == "border": plt.xscale('log')
    plt.legend()
    plt.savefig(save_dir + f"/plot_distances_{file_name}_lipid_calcium.png", bbox_inches='tight')
    plt.close()


def plot_vicinity(predictions, uncertainty_maps, nr_pixels_vicinity, save_dir, bins=20):
    # plot seperate figure for each class, bin the pixels, since there are too many
    for c in range(NEW_CLASSES):
        # vicinity_class = nr_pixels_vicinity[predictions == c]
        # uncertainty_class = uncertainty_maps[predictions == c]

        # max_classes = int(vicinity_class.max())
        # boxplots = []
        # for i in range(max_classes):
        #     boxplots.append(uncertainty_class[vicinity_class == i])

        # fig = plt.figure(figsize=(5,5))
        # plt.boxplot(boxplots, tick_labels=list(range(max_classes)))
        # plt.ylabel("Entropy")
        # plt.savefig(save_dir + f"/plot_vicinity_{NEW_CLASS_DICT[c]}.png", bbox_inches='tight')
        # plt.close()

        fig = plt.figure(figsize=(10,4))
        df_dict = {"Uncertainty": [], "Vicinity": []}
        vicinity_class = nr_pixels_vicinity[predictions == c]
        uncertainty_class = uncertainty_maps[predictions == c]

        if len(vicinity_class) > 1000000: # take subset if there is too much data
            indices = np.random.choice(len(vicinity_class), size=100000, replace=False)
            vicinity_class = vicinity_class[indices]
            uncertainty_class = uncertainty_class[indices]

        for i in range(2, 9):
            if len(uncertainty_class[vicinity_class == i]) > 0:
                df_dict["Uncertainty"].extend(uncertainty_class[vicinity_class == i])
                df_dict["Vicinity"].extend([i] * len(uncertainty_class[vicinity_class == i]))

        # palette = ['#2b94db', '#2286ca', '#1f77b4', '#1b699e', '#175b88', '#144c72']
        df = pd.DataFrame(df_dict)
        # print("test: ", c)
        print("Counts: ", df["Vicinity"].value_counts(normalize=True))

        sns.violinplot(data=df, x="Vicinity", y="Uncertainty", color='#2b94db', cut=0)#, width=2)
        plt.ylabel("Entropy", fontsize=16)

        ax = plt.gca()
        ax.tick_params(axis='y', which='both', left=False)
        ax.tick_params(axis='x', which='both', bottom=False)
        ax.yaxis.set_tick_params(labelsize='large')
        ax.xaxis.set_tick_params(labelsize='x-large')

        # plt.title("Nr of classes in the vicinity")
        plt.savefig(save_dir + f"/plot_vicinity_{NEW_CLASS_DICT[c]}.png", bbox_inches='tight')
        plt.close()


def plot_vicinity_violin(predictions, uncertainty_maps, nr_pixels_vicinity, save_dir):
    # plot seperate figure for each class, bin the pixels, since there are too many
    fig = plt.figure(figsize=(10,4))
    df_dict = {"Uncertainty": [], "Class": [], "Vicinity": []}
    for c in [4,5]:
        vicinity_class = nr_pixels_vicinity[predictions == c]
        uncertainty_class = uncertainty_maps[predictions == c]

        for i in range(2, 9):
            if len(uncertainty_class[vicinity_class == i]) > 0:
                df_dict["Uncertainty"].extend(uncertainty_class[vicinity_class == i])
                df_dict["Class"].extend([NEW_CLASS_DICT_CAPITAL[c]] * len(uncertainty_class[vicinity_class == i]))
                df_dict["Vicinity"].extend([i] * len(uncertainty_class[vicinity_class == i]))

    palette = ['#2b94db', '#1b699e'] # '#2286ca', '#1f77b4', '#1b699e', '#175b88', '#144c72', #103e5d
    df = pd.DataFrame(df_dict)

    sns.violinplot(data=df, x="Vicinity", y="Uncertainty", hue="Class", palette=palette, cut=0, split=True, inner=None, legend=False)#, width=1)
    plt.ylabel("Entropy", fontsize=16)

    ax = plt.gca()
    ax.tick_params(axis='y', which='both', left=False)
    ax.tick_params(axis='x', which='both', bottom=False)
    ax.yaxis.set_tick_params(labelsize='large')
    ax.xaxis.set_tick_params(labelsize='x-large')

    plt.title("Nr of classes in the vicinity")
    plt.savefig(save_dir + f"/plot_vicinity_lipid_calcium.png", bbox_inches='tight')
    plt.close()


def plot_artefact_uncertainty_correlation(predictions, artefact_maps, uncertainty_maps, save_dir):
    for c in range(1, NEW_CLASSES):
        fig = plt.figure(figsize=(10,4))
        df_dict = {"Uncertainty": [], "Class": [], "Artefact score": []}

        artefact_class = artefact_maps[predictions == c]
        uncertainty_class = uncertainty_maps[predictions == c]

        for i in range(3): 
            df_dict["Uncertainty"].extend(uncertainty_class[artefact_class == i])
            df_dict["Class"].extend([NEW_CLASS_DICT_CAPITAL[c]] * len(uncertainty_class[artefact_class == i]))
            df_dict["Artefact score"].extend([i] * len(uncertainty_class[artefact_class == i]))

        palette = ['#2b94db', '#1f77b4', '#1b699e'] # '#2286ca', '#1f77b4', '#1b699e', '#175b88', '#144c72', #103e5d
        df = pd.DataFrame(df_dict)
        print("Counts: ", df["Artefact score"].value_counts(normalize=True))
        # print(df)
        sns.violinplot(data=df, x="Artefact score", y="Uncertainty", hue="Artefact score", palette=palette, cut=5, legend=False)
        plt.ylabel("Entropy", fontsize=16)
        ax = plt.gca()
        ax.tick_params(axis='y', which='both', left=False)
        ax.tick_params(axis='x', which='both', bottom=False)
        ax.yaxis.set_tick_params(labelsize='large')
        ax.xaxis.set_tick_params(labelsize='x-large')
        plt.title("Artefact severity")
        plt.savefig(save_dir + f"/plot_artefacs_{NEW_CLASS_DICT[c]}.png", bbox_inches='tight')
        plt.close()



def plot_artefact_uncertainty_correlation_classes_combined(predictions, artefact_maps, uncertainty_maps, save_dir, classes, name_classes):
    fig = plt.figure(figsize=(10,4))
    df_dict = {"Uncertainty": [], "Artefact score": []}

    if len(classes) == 1:
        artefact_class = artefact_maps[(predictions == classes[0])]
        uncertainty_class = uncertainty_maps[(predictions == classes[0])]
    if len(classes) == 2:
        artefact_class = artefact_maps[(predictions == classes[0]) | (predictions == classes[1]) ]
        uncertainty_class = uncertainty_maps[(predictions == classes[0]) | (predictions == classes[1])]
    if len(classes) == 3:
        artefact_class = artefact_maps[(predictions == classes[0]) | (predictions == classes[1]) | (predictions == classes[2])]
        uncertainty_class = uncertainty_maps[(predictions == classes[0]) | (predictions == classes[1]) | (predictions == classes[2])]

    for i in range(3): 
        df_dict["Uncertainty"].extend(uncertainty_class[artefact_class == i])
        df_dict["Artefact score"].extend([i] * len(uncertainty_class[artefact_class == i]))

    palette = ['#2b94db', '#1f77b4', '#1b699e'] # '#2286ca', '#1f77b4', '#1b699e', '#175b88', '#144c72', #103e5d
    df = pd.DataFrame(df_dict)

    for i, j in [(0,1), (1,2)]:
        stat, p_value = mannwhitneyu(df[df["Artefact score"] == i]["Uncertainty"].values, df[df["Artefact score"] == j]["Uncertainty"].values)
        significance = p_value <= 0.05
            
        if significance:
            print(i, j, stat, p_value)
            y_pos = max(df[(df['Artefact score'] == i) | (df['Artefact score'] == j)]['Uncertainty']) * (1 + 0.02)
            plt.plot([i, j], [y_pos, y_pos], color='black', lw=1)
            if 0.05 > p_value > 0.01:
                p_symbol = '*'
            elif 0.01 > p_value > 0.001:
                p_symbol = '**'
            elif p_value < 0.001:
                p_symbol = '***'
            plt.text((i+j)/2, y_pos, p_symbol, fontsize=12, ha='center', va='bottom', color='black')
        
    print("Counts: ", df["Artefact score"].value_counts(normalize=True))
    # print(df)
    sns.violinplot(data=df, x="Artefact score", y="Uncertainty", hue="Artefact score", palette=palette, cut=0, legend=False)
    plt.ylabel("Entropy", fontsize=16)
    ax = plt.gca()
    ax.tick_params(axis='y', which='both', left=False)
    ax.tick_params(axis='x', which='both', bottom=False)
    ax.yaxis.set_tick_params(labelsize='large')
    ax.xaxis.set_tick_params(labelsize='x-large')
    plt.title("Artefact severity")
    plt.savefig(save_dir + f"/plot_artefacs_{name_classes}.png", bbox_inches='tight')
    plt.close()



def plot_artefact_uncertainty_correlation_violin(predictions, artefact_maps, uncertainty_maps, save_dir):
    # for each class create boxplots for the 3 severity levels.
    fig = plt.figure(figsize=(10,4))
    df_dict = {"Uncertainty": [], "Class": [], "Artefact score": []}
    for c in [4,5]:
        artefact_class = artefact_maps[predictions == c]
        uncertainty_class = uncertainty_maps[predictions == c]

        for i in range(3):
            df_dict["Uncertainty"].extend(uncertainty_class[artefact_class == i])
            df_dict["Class"].extend([NEW_CLASS_DICT_CAPITAL[c]] * len(uncertainty_class[artefact_class == i]))
            df_dict["Artefact score"].extend([i] * len(uncertainty_class[artefact_class == i]))

    palette = ['#2b94db', '#1b699e'] # '#2286ca', '#1f77b4', '#1b699e', '#175b88', '#144c72', #103e5d
    df = pd.DataFrame(df_dict)
    # print(df)
    ax = sns.violinplot(data=df, x="Artefact score", y="Uncertainty", hue="Class", palette=palette, cut=0, split=True, inner=None)#, fill=False)
    sns.move_legend(ax, "upper left")

    plt.ylabel("Entropy", fontsize=16)
    ax = plt.gca()
    ax.tick_params(axis='y', which='both', left=False)
    ax.tick_params(axis='x', which='both', bottom=False)
    ax.yaxis.set_tick_params(labelsize='large')
    ax.xaxis.set_tick_params(labelsize='x-large')
    plt.title("Artefact severity")
    plt.savefig(save_dir + f"/plot_artefacs_lipid_calcium.png", bbox_inches='tight')
    plt.close()




# quantification of uncertainty

def dist_uncertain_pixels_to_catheter(image, uncertainty_map, centroid, percentile=90, clas=4):
    mean_predictions = image.mean(axis=0).argmax(axis=0)

    if clas not in mean_predictions:
        return np.nan, np.nan, np.nan, np.nan
    p = np.percentile(uncertainty_map.flatten(), percentile)
    certain = np.where(uncertainty_map < p, uncertainty_map, -1)
    # certain_group_predictions = np.where(certain > -1, mean_predictions, -1)
    # uncertain_group_predictions = np.where(certain == -1, mean_predictions, -1)

    distances_certain = []
    distances_uncertain = []
    for i in range(704):
        for j in range(704):
            if mean_predictions[i,j] == clas:
                dist = np.linalg.norm(centroid-(i,j))
                if certain[i,j] > -1:
                    distances_certain.append(dist)
                else:
                    distances_uncertain.append(dist)

    return np.mean(distances_certain), np.mean(distances_uncertain), len(distances_certain), len(distances_uncertain)


def find_edge_pixels(predictions, clas):
    # create mask of class and convolve with an edge detection filter
    mask = np.where(predictions == clas, 1, 0)
    kernel = [[-1,-1,-1], [-1, 8,-1], [-1,-1,-1]]
    edges = np.where(convolve(mask, kernel, mode='constant') > 1)

    return np.array(edges)


def find_closest_edge(pixel, edges):
    closest_dist = 1000
    for i, j in zip(edges[0], edges[1]):
        dist = np.linalg.norm(pixel-(i, j))
        if dist < closest_dist:
            closest_dist = dist
    return closest_dist


def dist_pixels_to_closest_border(predictions):
    # find all edge pixels for each class
    edges_per_class = []
    for clas in range(NEW_CLASSES):
        edges_per_class.append(find_edge_pixels(predictions, clas))

    # distances = np.zeros((704, 704))
    distances = np.full((704, 704), fill_value = -1)

    # for each pixel (with gaps of 2 for efficiency), find the closest edge pixel 
    for i in tqdm(range(0, 704, 4)):
        for j in range(0, 704, 4):
            pixel_class = int(predictions[i,j])
            if pixel_class != 0 and pixel_class != 2 and pixel_class != 7:
                distances[i, j] = find_closest_edge(np.array([i,j]), edges_per_class[pixel_class])

    return distances


def dist_pixels_to_catheter():
    catheter = (704/2, 704/2) # cathether is middle of image

    # for each pixel, calculate the distance
    distances = np.zeros((704, 704))
    for i in range(704):
        for j in range(704):
            distances[i,j] = np.linalg.norm(np.array([i,j])-catheter)
    return distances


def search_neighbourhood(predictions, i, j, radius):
    # for pixel (i, j), find the number of different classes in the neighbourhood
    # classes = [0] * NEW_CLASSES
    neighbourhood = predictions[i-radius:i+radius, j-radius:j+radius]
    return len(np.unique(neighbourhood))

    # for ii in range(i-radius, i+radius):
    #     for jj in range(j-radius, j+radius):
    #         if ii >= 0 and ii < 704 and jj >= 0 and jj < 704:
    #             classes[int(predictions[ii, jj])] = 1

    # return sum(classes)


def nr_classes_in_vicinity(predictions, radius = 10):
    nr_classes = np.zeros((704, 704))
    # TODO: make this circular
    # for each pixel, calculate the number of different classes in a square around it 
    for i in range(radius, 704-radius):
        for j in range(radius, 704-radius):
            nr_classes[i,j] = search_neighbourhood(predictions, i, j, radius)

    return nr_classes


# group analysis

def group_analysis_percentiles(predictions, uncertainty_maps, ground_truths):
    start_range = 100
    end_range = 50

    dices = []
    standard_deviations = []
    percentiles = []
    images_used = []

    for percentile in tqdm(range(start_range, end_range, -1), desc="recalculating over percentiles"):
        # calculate the p-value for a percentile and set all predictions over p to (0 or correct)
        p = np.percentile(uncertainty_maps.flatten(), percentile)
        certain_predictions = np.where(uncertainty_maps < p, predictions, 0)
        # certain_predictions = np.where(uncertainty_maps < p, predictions, -1)

        mean_dices = []
        total_dices = []
        classes_present = [0] * (NEW_CLASSES+1)

        # calculate dices for each image seperately
        for i in range(predictions.shape[0]):
            dice_per_class_certain = dice_score(certain_predictions[i], ground_truths[i]) #[certain_predictions[i] != -1]
            dice = np.nanmean(dice_per_class_certain[1:])
            mean_dices.append(dice_per_class_certain)
            total_dices.append(dice)
            
            # keep count of when certain classes are present or not
            for c in range(NEW_CLASSES):
                if dice_per_class_certain[c] > 0:
                    classes_present[c] += 1

        # mean dices of all classes with total dice at the end
        std_dices = np.append(np.nanstd(np.array(mean_dices), axis=0), np.nanstd(np.array(total_dices)))
        mean_dices = np.append(np.nanmean(np.array(mean_dices), axis=0), np.nanmean(np.array(total_dices)))

        percentiles.append(percentile)
        dices.append(mean_dices)
        standard_deviations.append(std_dices)
        images_used.append(classes_present)

    standard_deviations = np.array(standard_deviations)
    dices = np.array(dices)
    images_used = np.array(images_used)

    return dices, percentiles, images_used, standard_deviations


def group_analysis_confusion_matrices(predictions, uncertainty_maps, labels, save_dir):
    start_range = 100
    end_range = 90

    # Compute misclassification map
    
    # print(misclassification_map)

    # conf_matrix = confusion_matrix(misclassification_map.ravel(), uncertainty_maps.ravel())
    # print(conf_matrix)

    for c in range(NEW_CLASSES):

        labels_class = np.ma.masked_where(labels == c, labels)
        predictions_class = np.ma.masked_where(predictions == c, predictions)

        misclassification_map = (labels_class != predictions_class)
        classification_map = (labels_class == predictions_class)

        for percentile in tqdm(range(start_range, end_range, -1), desc="confusion matrices"):
            p = np.percentile(uncertainty_maps.flatten(), percentile)

            tp = np.sum(np.where(uncertainty_maps >= p, True, False) & misclassification_map) # TP: filtered when it was wrong
            fp = np.sum(np.where(uncertainty_maps >= p, True, False) & classification_map) # FP: filtered when it was correct
            tn = np.sum(np.where(uncertainty_maps < p, True, False) & classification_map) # TN: not filtered when it was correct
            fn = np.sum(np.where(uncertainty_maps < p, True, False) & misclassification_map) # FN: not filtered when it was wrong
            # conf_matrix = np.array([[tp, fp],[fn, tn]]) / len(labels_class.flatten())
            precision = round(tp / (tp + fp), 4)
            recall = round(tp / (tp + fn), 4)
            # print(conf_matrix)
            masked_elements = len(labels.flatten()) - np.ma.count_masked(labels_class)
            # print(len(labels.flatten()), masked_elements)
            print(f"Class {c}, percentile {percentile}: ", precision, recall, f"({round((tp + fp) / len(labels.flatten()), 3)}, {round((tp + fp) / masked_elements, 3)})")

            # plt.figure(figsize=(4,4))
            # sns.heatmap(conf_matrix, annot=True, cmap="crest", vmin=0, vmax=1, fmt=".3f")
            # plt.savefig(save_dir + f"/conf_matrix_{percentile}.png", bbox_inches='tight')
        
    return




# artefact loading

class CartesianTransformTrivial(nn.Module): # source: piereandrea
    def __init__(
        self, row_bins: int, col_bins: int, interp="bilinear", device="cpu", dtype=torch.double
    ):
        super().__init__()
        self.row_bins = row_bins
        self.col_bins = col_bins
        self.device = device
        self.dtype = dtype
        self.interp = interp
        self.gridsample_kws = {"padding_mode": "zeros", "align_corners": True, "mode": self.interp}
        self.radius_normalized = 1
        self.register_buffer("warped_grid", self.make_grid())

    def make_grid(self):
        # create the meshgrid
        col_axis = torch.linspace(1, -1, self.col_bins, dtype=self.dtype)
        row_axis = torch.linspace(-1, 1, self.row_bins, dtype=self.dtype)
        rows, cols = torch.meshgrid(row_axis, col_axis, indexing="ij")
        grid = torch.stack([rows, cols], dim=0)
        grid = einops.rearrange(grid, "RC H W -> H W RC")
        # warp the grid
        warped_grid = torch.zeros_like(grid)
        # apply formula to infer the angle in (-pi, pi]
        theta = torch.atan2(grid[..., 0], grid[..., 1])
        # normalize to -1,1 for grid_sample input
        warped_grid[..., 0] = theta / torch.pi
        # apply magnitude formula
        magnitude = torch.sqrt(einops.reduce(grid**2, "H W RC -> H W", "sum"))
        # normalize to -1,1 for grid_sample input
        warped_grid[..., 1] = -(magnitude / self.radius_normalized * 2 - 1)
        return warped_grid

    def forward(self, image) -> torch.Tensor:
        B = einops.parse_shape(image, "B N H W")["B"]
        return grid_sample(
            image, einops.repeat(self.warped_grid, f"H W ϱϑ -> {B} H W ϱϑ"), **self.gridsample_kws
        )


def preds_to_segmasks(labels, device="cuda") -> torch.Tensor: # source: piereandrea
    """Transform a label tensor (output of the artifact model) into cartesian coordinates.
 
    Args:
        labels (torch.Tensor | np.ndarray): `labels` have shape Frames×Angular_dimension.
        mode (str, optional): nearest for integer labels, bilinear for real images. Defaults to "nearest".
        device (str, optional): whether to use the cpu or the gpu for the transform. Defaults to "cuda".
 
    Returns:
        torch.Tensor: _description_
    """
    if isinstance(labels, np.ndarray):
        if labels.dtype != np.float64:
            labels = labels.astype(np.float64)
        labels = torch.from_numpy(labels)
    if labels.dtype != torch.float64:
        labels = labels.to(torch.float64)
 
    repeated = einops.repeat(labels, f"F ϑ -> F 1 704 ϑ")
    c = CartesianTransformTrivial(row_bins=704, col_bins=704, interp="nearest").to(device)
 
    ys = [
        c(repeated[i : i + 25].to(device)).to("cpu").to(torch.uint8)
        for i in range(0, repeated.shape[0], 25)
    ]
    ys = einops.pack(ys, "* C H W")[0]
    ys = einops.rearrange(ys, "F 1 H W -> F H W")
    return ys



def load_artefacts(dataset_files):

    files_dict = {}

    # load files for the dataset and artefacts
    dataset_info = pd.read_excel("/data/diag/rubenvdw/Info_files_Dataset_split/15_classes_dataset_newsplit_29102024.xlsx")
    dir = Path("/data/diag/thijsLuttikholt/full_pred_pipeline/all_pectus/artefact_predictions")
    artefact_files = [str(file) for file in list((dir).glob('./*')) if file != (dir / ".DS_Store")]
    
    # make dictionary to go from patient+pullback id to the associated artefact file
    for file in artefact_files:
        pullback = dataset_info.loc[dataset_info["Pullback"] == file.split("/")[-1].split(".")[0]]
        if len(pullback) > 0:
            dataset_file_name = "".join(pullback["Patient"].values[0].split("-")) + "_" + str(pullback["Nº pullback"].values[0])
            files_dict[dataset_file_name] = file.split("/")[-1]

    # get patient id in addition to the frame numbers, and group by patient id
    dataset_files = [(str(file).split("/")[-1].split(".")[0].split("_")[0] + "_" + str(file).split("/")[-1].split(".")[0].split("_")[1], numbers(str(file).split("/")[-1].split(".")[0].split("_")[2])[0]) for file in dataset_files] # gets (patient+pullback id, frame number)
    dataset_files = group_by(dataset_files) # group frame numbers by patient

    artefact_dict = {}
    artefact_count = 0

    for patient, frames in tqdm(dataset_files.items(), desc="Loading artefacts"):
        try:
            artefacts = np.load(str(dir) + "/" + files_dict[patient]) # load artefacts from file
            artefacts = artefacts[frames] # take only frames in dataset
            
            # transform polar a-lines to cartesian coordinates
            artefact_maps = preds_to_segmasks(artefacts)
            artefact_maps = artefact_maps.numpy()

            for frame in range(artefact_maps.shape[0]):
                artefact_dict[patient + "_frame" + str(frames[frame])] = artefact_maps[frame]
        except: # when artefacts are not computed for a patient, assume that there are no artefacts
            print(f"artefact map not available for patient: {patient}")
            for frame in range(len(frames)):
                artefact_dict[patient + "_frame" + str(frames[frame])] = np.zeros((704,704))

                # temp code for printing the maps
                # if 2 in frames[frame]:
                #     artefact_count += 1
                #     fig = plt.figure(figsize=(5,5))
                #     plt.imshow(frame.reshape((704,704,1)))

                #     plt.axis("off")
                #     plt.colorbar(shrink=0.7)
                #     plt.savefig(f"/data/diag/leahheil/IVOCT-Segmentation/saved/images_percentiles/artefact_example_{patient}_{frame}.png", dpi=600, bbox_inches='tight')
                #     plt.close()

    print(artefact_count)
    return artefact_dict


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


def topolar(img, order=5):
    # max_radius = 0.5*np.linalg.norm(img.shape)

    # def transform(coords):
    #     theta = 2.0*np.pi*coords[1] / (img.shape[1] - 1.)
    #     radius = max_radius * coords[0] / img.shape[0]
    #     i = 0.5*img.shape[0] - radius*np.sin(theta)
    #     j = radius*np.cos(theta) + 0.5*img.shape[1]
    #     return i,j
 
    # polar = geometric_transform(img, transform, order=order,mode='nearest',prefilter=True)

    
    img64_float = img.astype(np.float64)
    
    Mvalue = np.sqrt(((img64_float.shape[0]/2.0)**2.0)+((img64_float.shape[1]/2.0)**2.0))
    
    polar = cv2.linearPolar(img64_float,(img64_float.shape[0]/2, img64_float.shape[1]/2),Mvalue,cv2.WARP_FILL_OUTLIERS)

    return polar


def plot_polar_image(image, file_path, file_name):
    if image.shape[0] == 3:
        image = image.transpose(1,2,0)
    
    polar_image = topolar(image)
    fig = plt.figure(figsize=(4,4))

    plt.imshow(polar_image)
    plt.axis("off")
    plt.savefig(file_path + "/" + file_name + ".png", dpi=600, bbox_inches='tight') #high dpi to prevent blending of colors between classes
    plt.close()


# misc

def fraction_uncertainty(uncertainty_map, prediction):
    vessel_fraction = len(prediction[prediction != 0]) / len(prediction.flatten())

    return uncertainty_map.mean(), vessel_fraction


def normalise_uncertainty(uncertainty, prediction):
    vessel_fraction = len(prediction[prediction != 0]) / len(prediction.flatten())
    uncertainty_score = uncertainty * vessel_fraction
    return uncertainty_score


def print_corr(corr):
    return f"{round(corr.statistic,3)} (p={round(corr.pvalue,5)})"


def numbers(string):
    # find all numbers in string
    return re.findall(r'\d+', string)


def group_by(my_list):
    # group list of key values by key and turn into dict
    result = {}
    for k, v in my_list:
        if k not in result:
            result[k] = [int(v)]
        else:
            result[k].append(int(v))
    return result 