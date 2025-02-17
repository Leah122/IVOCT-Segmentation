from pathlib import Path

import numpy as np
import pandas as pd
# import scipy.ndimage as ndi
# import SimpleITK as sitk
import torch
from tqdm import tqdm
import argparse
import gc
from torch.utils.data import DataLoader
import torchvision.transforms.functional as tf
import shutil
import matplotlib.pyplot as plt

from data import OCTDataset
from model2 import U_Net
from constants import NEW_CLASSES, NEW_CLASS_DICT
from metrics import AUCPR, AUCROC, MI, entropy, entropy_per_class, MI_per_class, group_analysis, filter_uncertain_images, filter_uncertain_images_test
from utils import plot_uncertainty_per_class, plot_image_overlay_labels, plot_uncertainty, plot_softmax_labels_per_class, plot_image_prediction_certain, dice_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

def load_model_mc(dropout, save_dir, model_id):
    model = U_Net(dropout=dropout, softmax=True).to(device)

    checkpoint = torch.load(save_dir / f"last_model_{model_id}.pth", weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()
    if dropout > 0:
        for m in model.modules(): # turn dropout back on
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
    return model

def load_model_tta(save_dir, model_id):
    model = U_Net(dropout=0.0, softmax=True).to(device)

    checkpoint = torch.load(save_dir / f"last_model_{model_id}.pth", weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()

    return model

def load_model_ensemble(save_dir, model_id, nr_models):
    models = []

    for i in range(nr_models):
        model = U_Net(dropout=0.0, softmax=True).to(device)
        checkpoint = torch.load(save_dir / f"model_{i}" / f"last_model_{model_id}.pth", weights_only=True)
        model.load_state_dict(checkpoint)
        model.eval()

        models.append(model)
    return models

@torch.no_grad()
def make_predictions_mc(sample, model, nr_samples):
    image = sample["image"].to(device)
    labels = sample["labels"].detach().cpu().numpy()

    with torch.no_grad():
        outputs = np.zeros((nr_samples, NEW_CLASSES, 704, 704))
        for mc in range(nr_samples):
            outputs[mc] = model(image).detach().cpu().numpy().squeeze()

    image = image.detach().cpu().numpy()

    return outputs, image, labels

@torch.no_grad()
def make_predictions_ensemble(sample, model, nr_samples):
    image = sample["image"].to(device)
    labels = sample["labels"].detach().cpu().numpy()

    with torch.no_grad():
        outputs = np.zeros((nr_samples, NEW_CLASSES, 704, 704))
        for i, model_i in enumerate(model):
            outputs[i] = model_i(image).detach().cpu().numpy().squeeze()

    image = image.detach().cpu().numpy()

    return outputs, image, labels

@torch.no_grad()
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

@torch.no_grad()
def inference(
    data_dir: Path, 
    save_dir: Path, 
    dropout: float,
    validation: bool = True,
    mc_samples: int = 0,
    model_id: str = "1",
    nr_ensembles: int = 0,
    tta_samples: int = 0,
    ):

    nr_samples = 0

    if nr_ensembles > 0:
        model = load_model_ensemble(save_dir, model_id, nr_ensembles)
        print(f"nr of ensemble models: {nr_ensembles}")
        nr_samples = nr_ensembles
    elif tta_samples > 0:
        model = load_model_tta(save_dir, model_id)
        print(f"nr of TTA samples: {tta_samples}")
        nr_samples = tta_samples
    else:
        model = load_model_mc(dropout, save_dir, model_id)
        print(f"nr of MC samples: {mc_samples}, with dropout: {dropout}")
        nr_samples = mc_samples
    print("model id: ", model_id)

    debugging = False

    print("INFO: loading data")
    if validation:
        dataset_test = OCTDataset(data_dir, validation=True, debugging = debugging)
    else:
        dataset_test = OCTDataset(data_dir, test=True, debugging = debugging)

    dataloader_test = DataLoader(
        dataset=dataset_test,
        batch_size=1,
    )

    
    if tta_samples > 0:
        save_dir_experiment = str(save_dir) + "/nr_samples_experiment/tta"
    elif mc_samples > 0:
        save_dir_experiment = str(save_dir) + "/nr_samples_experiment/mc"
    else:
        save_dir_experiment = str(save_dir) + "/nr_samples_experiment/ens"

    for iter in range(10, nr_samples, 2):

        print(f"starting inference for {iter} samples")

        metrics = pd.DataFrame(columns=['file_name', 'dice', 'background', 'lumen', 'guidewire', 'intima', 'lipid', 'calcium', 'media', 'catheter', 'sidebranch', 'healedplaque', 'AUC_MI', 'AUC_EN'])

        for sample in tqdm(dataloader_test): 
            file_name = str(sample["metadata"]["file_name"][0])
            if tta_samples > 0: 
                outputs, image, labels = make_predictions_tta(sample, model, iter)
            elif mc_samples > 0: 
                outputs, image, labels = make_predictions_mc(sample, model, iter)
            else: 
                outputs, image, labels = make_predictions_ensemble(sample, model, iter)

            dice_per_class = dice_score(outputs.mean(axis=0).argmax(axis=0), labels.squeeze())
            dice = np.nanmean(dice_per_class[1:])

            metrics_list = [file_name, dice]
            for item in dice_per_class:
                metrics_list.append(item)

            mutual_map = MI(outputs)
            entropy_map = entropy(outputs)
            auc_mi = AUCROC(outputs, mutual_map, labels)
            auc_en = AUCROC(outputs, entropy_map, labels)

            metrics_list.append(auc_mi)
            metrics_list.append(auc_en)
            metrics.loc[len(metrics)] = metrics_list

            del outputs
            del sample
            gc.collect()
            torch.cuda.empty_cache()

        for column in metrics.columns:
            if column != "file_name":
                metrics[column] = metrics[column].apply(lambda x: None if x == 1 else x)
                metrics.loc["mean", column] = metrics[column].mean(skipna=True)

        metrics.to_csv(save_dir_experiment + f"/metrics_{iter}.csv", index=False)
        print(metrics.loc['mean'])

    plot_metrics(save_dir, mc_samples, nr_ensembles, tta_samples)


def plot_metrics(save_dir, mc_samples, nr_ensembles, tta_samples):
    if nr_ensembles > 0:
        save_dir_experiment = str(save_dir) + "/nr_samples_experiment/ens"
        nr_samples = nr_ensembles
    elif tta_samples > 0:
        save_dir_experiment = str(save_dir) + "/nr_samples_experiment/tta"
        nr_samples = tta_samples
    else:
        save_dir_experiment = str(save_dir) + "/nr_samples_experiment/mc"
        nr_samples = mc_samples
    
    files = [str(file) for file in list(Path(save_dir_experiment).glob('./*')) if "metrics" in str(file)]
    # samples_list = [int(file.split("/")[-1].split("_")[1].split(".")[0]) for file in files]
    samples_list = [int(file.split("/")[-1].split("_")[1].split(".")[0]) for file in files]
    samples_list = list(zip(files, samples_list))

    samples_list = sorted(samples_list, key=lambda x: x[1], reverse=False)

    print(samples_list)
    metrics_list = {"dice": [], "AUC_MI": [], "AUC_EN": []}
    y = []

    for file, i in samples_list:
        metrics = pd.read_csv(file)
        metrics_list["dice"].append(metrics.loc[len(metrics)-1, "dice"])
        metrics_list["AUC_MI"].append(metrics.loc[len(metrics)-1, "AUC_MI"])
        metrics_list["AUC_EN"].append(metrics.loc[len(metrics)-1, "AUC_EN"])
        y.append(i)

    print(metrics_list)
    
    plt.figure(figsize=(5,5))
    plt.plot(y, metrics_list["AUC_MI"], label="AUC_MI", color="cornflowerblue")
    plt.plot(y, metrics_list["AUC_EN"], label="AUC_EN", color="mediumpurple")
    plt.xlabel("nr of samples")
    plt.ylabel("AUC scores")
    plt.title("effect on AUC with number of samples")
    plt.legend()
    plt.savefig(save_dir_experiment + "/" + "plot_auc.png", dpi=800, bbox_inches='tight')

    plt.figure(figsize=(5,5))
    plt.plot(y, metrics_list["dice"], label="dice", color="cornflowerblue")
    plt.xlabel("nr of samples")
    plt.ylabel("dice score")
    plt.title("effect on dice with number of samples")
    plt.legend()
    plt.savefig(save_dir_experiment + "/" + "plot_dice.png", dpi=800, bbox_inches='tight')


def plot_all(save_dir):
    save_dir_experiment = str(save_dir) + "/nr_samples_experiment"
    methods = ["mc", "tta", "ens"]
    metrics_all = {"mc": {}, "tta": {}, "ens": {}}

    for method in methods:
        files = [str(file) for file in list(Path(save_dir_experiment + "/" + method).glob('./*')) if "metrics" in str(file)]
        samples_list = [int(file.split("/")[-1].split("_")[1].split(".")[0]) for file in files]
        samples_list = list(zip(files, samples_list))

        samples_list = sorted(samples_list, key=lambda x: x[1], reverse=False)

        metrics_list = {"dice": [], "AUC_MI": [], "AUC_EN": [], "y": []}

        # print(samples_list)
        for file, i in samples_list:
            metrics = pd.read_csv(file)
            metrics_list["dice"].append(metrics.loc[len(metrics)-1, "dice"])
            metrics_list["AUC_MI"].append(metrics.loc[len(metrics)-1, "AUC_MI"])
            metrics_list["AUC_EN"].append(metrics.loc[len(metrics)-1, "AUC_EN"])
            metrics_list["y"].append(i)
        # print(metrics_list)

        metrics_all[method] = metrics_list
    print(metrics_all)
    
    plt.figure(figsize=(5,5))

    plt.plot(metrics_all["mc"]["y"], metrics_all["mc"]["AUC_MI"], label="mc: AUC_MI", color="cornflowerblue")
    plt.plot(metrics_all["mc"]["y"], metrics_all["mc"]["AUC_EN"], label="mc: AUC_EN", color="mediumpurple")
    plt.plot(metrics_all["tta"]["y"], metrics_all["tta"]["AUC_MI"], label="tta: AUC_MI", color="deeppink")
    plt.plot(metrics_all["tta"]["y"], metrics_all["tta"]["AUC_EN"], label="tta: AUC_EN", color="darkorange")

    plt.xlabel("nr of samples")
    plt.ylabel("AUC scores")
    plt.legend(loc='lower right')
    plt.savefig(save_dir_experiment + "/" + "plot_auc.png", dpi=800, bbox_inches='tight')

    plt.figure(figsize=(5,5))
    plt.plot(metrics_all["mc"]["y"], metrics_all["mc"]["dice"], label="mc: dice", color="cornflowerblue")
    plt.plot(metrics_all["tta"]["y"], metrics_all["tta"]["dice"], label="tta: dice", color="mediumpurple")
    plt.xlabel("nr of samples")
    plt.ylabel("dice score")
    plt.legend(loc='lower right')
    plt.savefig(save_dir_experiment + "/" + "plot_dice.png", dpi=800, bbox_inches='tight')

    print("auc mi mc: \t", [round((1-(j/i))*100, 4) for i, j in zip(metrics_all["mc"]["AUC_MI"][1:], metrics_all["mc"]["AUC_MI"][:-1])])
    print("auc en mc: \t", [round((1-(j/i))*100, 4) for i, j in zip(metrics_all["mc"]["AUC_EN"][1:], metrics_all["mc"]["AUC_EN"][:-1])])
    print("auc mi tta: \t", [round((1-(j/i))*100, 4) for i, j in zip(metrics_all["tta"]["AUC_MI"][1:], metrics_all["tta"]["AUC_MI"][:-1])])
    print("auc en tta: \t", [round((1-(j/i))*100, 4) for i, j in zip(metrics_all["tta"]["AUC_MI"][1:], metrics_all["tta"]["AUC_MI"][:-1])])
    print("dice mc: \t", [round((1-(j/i))*100, 4) for i, j in zip(metrics_all["mc"]["dice"][1:], metrics_all["mc"]["dice"][:-1])])
    print("dice tta: \t", [round((1-(j/i))*100, 4) for i, j in zip(metrics_all["tta"]["dice"][1:], metrics_all["tta"]["dice"][:-1])])



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/data/diag/leahheil/data", help='path to the directory that contains the data')
    parser.add_argument("--save_dir", type=str, default="/data/diag/leahheil/saved", help='path to the directory that you want to save in')
    parser.add_argument("--mc_samples", type=int, default=0, help='number of mc samples')
    parser.add_argument("--dropout", type=float, default=0.2, help='fraction of dropout to use')
    parser.add_argument("--model_id", type=str, default="1", help='id of the model used for inference')
    parser.add_argument("--ensembles", type=int, default=0, help='number of mc samples')
    parser.add_argument("--tta_samples", type=int, default=0, help='number of mc samples')
    parser.add_argument("--only_plot", default=False, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)

    if args.only_plot:
        # plot_metrics(save_dir, mc_samples=args.mc_samples, nr_ensembles=args.ensembles, tta_samples=args.tta_samples)
        plot_all(save_dir)
    else:
        if args.ensembles > 0:
            save_dir = Path(args.save_dir) / "ensemble"
        inference(data_dir, save_dir, dropout=args.dropout, mc_samples=args.mc_samples, model_id=args.model_id, nr_ensembles=args.ensembles, tta_samples=args.tta_samples)

if __name__ == "__main__":
    main()

