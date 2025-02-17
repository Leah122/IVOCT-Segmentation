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
# import torchvision.transforms.functional as tf
# import shutil

from data import OCTDataset
# from model2 import U_Net
from constants import NEW_CLASSES, NEW_CLASS_DICT
from metrics import AUCPR, AUCROC, MI, entropy, entropy_per_class, MI_per_class, group_analysis, filter_uncertain_images, sensitivity_specificity
from utils import *
# plot_uncertainty_per_class, plot_image_overlay_labels, plot_uncertainty, plot_softmax_labels_per_class, plot_image_prediction_certain, dice_score, normalise_uncertainty, plot_uncertainty_vs_vessel_fraction, fraction_uncertainty, plot_image_prediction_wrong

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# def augment(image, samples = 10):
#         flips = [[0,0], [0,1], [1,0], [1,1]]
#         rotations = [0, 60, 120, 240, 300]

#         augmentations = []
#         for i in range(len(flips)):
#             for j in range(len(rotations)):
#                 augmentations.append(flips[i] + [rotations[j]])

#         images_aug = []
#         augmentations = augmentations[:samples]

#         for aug in augmentations:
#             image_aug = tf.rotate(image, aug[2], fill=0)
            
#             if aug[0] == 1:
#                 image_aug = tf.hflip(image_aug)

#             if aug[1] == 1:
#                 image_aug = tf.vflip(image_aug)

#             images_aug.append(image_aug)
#             del image_aug

#         return images_aug, augmentations

# def reverse_augment(images, augmentations):
#     outputs = []

#     for aug, image in zip(augmentations, images):
    
#         if aug[1] == 1:
#             image = tf.vflip(image)

#         if aug[0] == 1:
#             image = tf.hflip(image)

#         image = tf.rotate(image, 360-aug[2], fill=[1,0,0,0,0,0,0,0,0,0])
        
#         outputs.append(image.detach().cpu().numpy().squeeze())

#     return outputs

# def load_model_mc(dropout, save_dir, model_id):
#     model = U_Net(dropout=dropout, softmax=True).to(device)

#     checkpoint = torch.load(save_dir / f"last_model_{model_id}.pth", weights_only=True)
#     model.load_state_dict(checkpoint)
#     model.eval()
#     if dropout > 0:
#         for m in model.modules(): # turn dropout back on
#             if m.__class__.__name__.startswith('Dropout'):
#                 m.train()
#     return model

# def load_model_tta(save_dir, model_id):
#     model = U_Net(dropout=0.0, softmax=True).to(device)

#     checkpoint = torch.load(save_dir / f"last_model_{model_id}.pth", weights_only=True)
#     model.load_state_dict(checkpoint)
#     model.eval()

#     return model

# def load_model_ensemble(save_dir, model_id, nr_models):
#     models = []

#     for i in range(nr_models):
#         model = U_Net(dropout=0.0, softmax=True).to(device)
#         checkpoint = torch.load(save_dir / f"model_{i}" / f"last_model_{model_id}.pth", weights_only=True)
#         model.load_state_dict(checkpoint)
#         model.eval()

#         models.append(model)
#     return models

# @torch.no_grad()
# def make_predictions_mc(sample, model, nr_samples):
#     image = sample["image"].to(device)
#     labels = sample["labels"].detach().cpu().numpy()

#     with torch.no_grad():
#         outputs = np.zeros((nr_samples, NEW_CLASSES, 704, 704))
#         for mc in range(nr_samples):
#             outputs[mc] = model(image).detach().cpu().numpy().squeeze()

#     image = image.detach().cpu().numpy()

#     return outputs, image, labels

# @torch.no_grad()
# def make_predictions_ensemble(sample, model, nr_samples):
#     image = sample["image"].to(device)
#     labels = sample["labels"].detach().cpu().numpy()

#     with torch.no_grad():
#         outputs = np.zeros((nr_samples, NEW_CLASSES, 704, 704))
#         for i, model_i in enumerate(model):
#             outputs[i] = model_i(image).detach().cpu().numpy().squeeze()

#     image = image.detach().cpu().numpy()

#     return outputs, image, labels

# @torch.no_grad()
# def make_predictions_tta(sample, model, nr_samples):
#     image = sample["image"].to(device)
#     labels = sample["labels"].detach().cpu().numpy()

#     images, augmentations = augment(image, nr_samples)

#     with torch.no_grad():
#         outputs = [] 
#         for image_i in images:
#             outputs.append(model(image_i).squeeze())
    
#     outputs = reverse_augment(outputs, augmentations)

#     image = image.detach().cpu().numpy()

#     return np.array(outputs), image, labels


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
    debug: bool = False,
    polish_data: bool = False,
    ):

    if nr_ensembles > 0:
        model = load_model_ensemble(save_dir, model_id, nr_ensembles)
        print(f"nr of ensemble models: {nr_ensembles}")
    elif tta_samples > 0:
        model = load_model_tta(save_dir, model_id)
        print(f"nr of TTA samples: {tta_samples}")
    else:
        model = load_model_mc(dropout, save_dir, model_id)
        print(f"nr of MC samples: {mc_samples}, with dropout: {dropout}")
    print("model id: ", model_id)

    debugging = debug

    print("INFO: loading data")
    if polish_data:
        dataset_test = OCTDataset(data_dir, polish=True, debugging = debugging)
    elif validation:
        dataset_test = OCTDataset(data_dir, validation=True, debugging = debugging) 
    else:
        dataset_test = OCTDataset(data_dir, test=True, debugging = debugging)

    dataloader_test = DataLoader(
        dataset=dataset_test,
        batch_size=1,
    )

    sens_spec = {4: np.zeros(4), 5: np.zeros(4), 8: np.zeros(4), 9: np.zeros(4)}
    neighbours_lipid = np.zeros(NEW_CLASSES)
    mispredicted_lipid = np.zeros(NEW_CLASSES)
    mispredicted_calcium = np.zeros(NEW_CLASSES)

    info = pd.DataFrame(columns=['file_name', 'dice', 'total_MI', 'total_en', 'nr_structures', 'structures_present', 'lipid_present', 'percentage_lipid', 'dice_lipid', 'calcium_present', 'percentage_calcium', 'dice_calcium'])
    info_lipid = pd.DataFrame(columns=['file_name', 'dice', 'total_MI', 'total_en', 'nr_structures', 'structures_present', 'lipid_present', 'percentage_lipid', 'dice_lipid', 'neighbours', 'guidwire_neighbour', 'dist_center', 'dist_closest', 'dist_farthest', 'mispredicted'])
    info_calcium = pd.DataFrame(columns=['file_name', 'dice', 'total_MI', 'total_en', 'nr_structures', 'structures_present', 'calcium_present', 'percentage_calcium', 'dice_calcium', 'neighbours', 'guidwire_neighbour', 'dist_center', 'dist_closest', 'dist_farthest', 'mispredicted'])

    image_samples = 0

    if tta_samples > 0:
        save_dir_images = str(save_dir) + "/images_tta"
    elif polish_data:
        save_dir_images = str(save_dir) + "/images_polish"
    else:
        save_dir_images = str(save_dir) + "/images"
    

    for sample in tqdm(dataloader_test): 

        # make predictions
        file_name = str(sample["metadata"]["file_name"][0])
        if tta_samples > 0: 
            outputs, image, labels = make_predictions_tta(sample, model, tta_samples)
        elif mc_samples > 0: 
            outputs, image, labels = make_predictions_mc(sample, model, mc_samples)
        else: 
            outputs, image, labels = make_predictions_ensemble(sample, model, nr_ensembles)

        dice_per_class = dice_score(outputs.mean(axis=0).argmax(axis=0), labels.squeeze())
        dice = np.nanmean(dice_per_class[1:])

        mi_map = MI(outputs)
        entropy_map = entropy(outputs)

        # group analysis info
        info_list = [file_name, dice]
        info_list.append(normalise_uncertainty(mi_map.mean(), outputs.mean(axis=0).argmax(axis=0)))
        info_list.append(normalise_uncertainty(entropy_map.mean(), outputs.mean(axis=0).argmax(axis=0)))
        structures = []
        for i in range(NEW_CLASSES):
            if i in labels:
                structures.append(NEW_CLASS_DICT[i])
        info_list.append(len(structures))
        info_list.append(structures)

        info_list_calcium = info_list.copy()

        # percentages and dice for lipid and calcium only if in image
        if np.any(outputs.mean(axis=0).argmax(axis=0) == 4) and np.any(labels == 4):
            info_list.append(int(np.any(labels == 4)))
            info_list.append(len(labels[labels==4])/len(labels.flatten()))
            info_list.append(dice_per_class[4])
        else: 
            info_list.extend([0, np.nan, np.nan])

        info_list_lipid = info_list.copy()

        if np.any(outputs.mean(axis=0).argmax(axis=0) == 5) and np.any(labels == 5):
            info_list.append(int(np.any(labels == 5)))
            info_list.append(len(labels[labels==5])/len(labels.flatten()))
            info_list.append(dice_per_class[5])
            info_list_calcium.append(int(np.any(labels == 5)))
            info_list_calcium.append(len(labels[labels==5])/len(labels.flatten()))
            info_list_calcium.append(dice_per_class[5])
        else: 
            info_list_calcium.extend([0, np.nan, np.nan])
            info_list.extend([0, np.nan, np.nan])

        for i in [4, 5]:
            if i in labels.squeeze():
                neighbourhood = neighbouring(labels.squeeze(), i)
                neighbours = np.array([int(value > 0) for value in neighbourhood.values()])
                neighbours_lipid += neighbours
                guidewire_neighbour = neighbours[2]
                
                catheter_mask = np.where(labels.squeeze() == 7, 1, 0)
                catheter_centroid = np.mean(np.argwhere(catheter_mask),axis=0)
                lipid_mask = np.where(labels.squeeze() == i, 1, 0)
                lipid_centroid = np.mean(np.argwhere(lipid_mask),axis=0)
                # centroid_x, centroid_y = int(centroid[1]), int(centroid[0])
                center_dist = np.linalg.norm(catheter_centroid-lipid_centroid)
                closest_dist = find_closest_dist(catheter_centroid, labels.squeeze(), i)
                farthest_dist = find_farthest_dist(catheter_centroid, labels.squeeze(), i)
                mispredicted = mispredictions(outputs.mean(axis=0).argmax(axis=0), labels.squeeze(), i)
                
                

                if i == 4:
                    info_list_lipid.extend([neighbours, guidewire_neighbour, center_dist, closest_dist, farthest_dist, mispredicted])
                    if mispredicted != np.nan:

                        mispredicted_lipid += np.nan_to_num(mispredicted)
                else: 
                    info_list_calcium.extend([neighbours, guidewire_neighbour, center_dist, closest_dist, farthest_dist, mispredicted])
                    if mispredicted != np.nan:
                        mispredicted_calcium += np.nan_to_num(mispredicted)
            else: 
                if i == 4:
                    info_list_lipid.extend([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
                else: 
                    info_list_calcium.extend([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])


        info.loc[len(info)] = info_list
        info_lipid.loc[len(info_lipid)] = info_list_lipid
        info_calcium.loc[len(info_calcium)] = info_list_calcium

        rates = sensitivity_specificity(outputs, labels)
        indices = [4,5,8,9]
        for i in indices:
            sens_spec[i] += rates[i]
        

        # plot the first few images for inspection
        if image_samples > 0:
            plot_image_overlay_labels(image.squeeze(), outputs.mean(axis=0).argmax(axis=0), file_path=save_dir_images, file_name=file_name + "_pred")
            plot_image_overlay_labels(image.squeeze(), labels, file_path=save_dir_images, file_name=file_name)
            plot_image_overlay_labels(image.squeeze(), labels, file_path=save_dir_images, file_name=file_name + "_image", alpha=0)
            plot_uncertainty(mi_map, file_path=save_dir_images, file_name=file_name + "_uncertainty_MI", metric="MI")
            plot_uncertainty(entropy_map, file_path=save_dir_images, file_name=file_name + "_uncertainty_en", metric="Entropy")
            image_samples -= 1
        
        del outputs
        del sample
        gc.collect()
        torch.cuda.empty_cache()

    if nr_ensembles > 0:
        save_str = str(nr_ensembles)
    elif tta_samples > 0:
        save_str = "tta" + str(tta_samples)
    else:
        save_str = "mc" + str(mc_samples)

    # sensitivity / specificity
    for i in [4,5,8,9]:
        TP, FP, TN, FN = sens_spec[i]
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
        print(NEW_CLASS_DICT[i], "sensitivity / specificity: ", sensitivity, specificity)
    
    uncertain_images = filter_uncertain_images(info["total_en"], percentile=90)
    info["uncertain"] = uncertain_images
    info_lipid["uncertain"] = uncertain_images
    info_calcium["uncertain"] = uncertain_images

    print("uncertain image files: ", info["file_name"].values[uncertain_images])

    for column in info.columns: 
        if column != "file_name" and column != "structures_present" and column != "neighbours":
            info.loc["mean", column] = info[column].mean(skipna=True)
            info.loc["mean_uncertain", column] = info[column][info["uncertain"] == True].mean(skipna=True)
            info.loc["mean_certain", column] = info[column][info["uncertain"] == False].mean(skipna=True)
            info.loc["std", column] = info[column].std(skipna=True)
            info.loc["std_uncertain", column] = info[column][info["uncertain"] == True].std(skipna=True)
            info.loc["std_certain", column] = info[column][info["uncertain"] == False].std(skipna=True)

    for column in info_lipid.columns: 
        if column != "file_name" and column != "structures_present" and column != "neighbours" and column != "mispredicted":
            info_lipid.loc["mean", column] = info_lipid[column].mean(skipna=True)
            info_lipid.loc["mean_uncertain", column] = info_lipid[column][info_lipid["uncertain"] == True].mean(skipna=True)
            info_lipid.loc["mean_certain", column] = info_lipid[column][info_lipid["uncertain"] == False].mean(skipna=True)
            info_lipid.loc["std", column] = info_lipid[column].std(skipna=True)
            info_lipid.loc["std_uncertain", column] = info_lipid[column][info_lipid["uncertain"] == True].std(skipna=True)
            info_lipid.loc["std_certain", column] = info_lipid[column][info_lipid["uncertain"] == False].std(skipna=True)

    for column in info_calcium.columns: 
        if column != "file_name" and column != "structures_present" and column != "neighbours" and column != "mispredicted":
            info_calcium.loc["mean", column] = info_calcium[column].mean(skipna=True)
            info_calcium.loc["mean_uncertain", column] = info_calcium[column][info_calcium["uncertain"] == True].mean(skipna=True)
            info_calcium.loc["mean_certain", column] = info_calcium[column][info_calcium["uncertain"] == False].mean(skipna=True)
            info_calcium.loc["std", column] = info_calcium[column].std(skipna=True)
            info_calcium.loc["std_uncertain", column] = info_calcium[column][info_calcium["uncertain"] == True].std(skipna=True)
            info_calcium.loc["std_certain", column] = info_calcium[column][info_calcium["uncertain"] == False].std(skipna=True)


    info.to_csv(str(save_dir) + f"/metrics_{save_str}_info.csv")
    info_lipid.to_csv(str(save_dir) + f"/metrics_{save_str}_info_lipid.csv")
    info_calcium.to_csv(str(save_dir) + f"/metrics_{save_str}_info_calcium.csv")
    
    print((info_lipid.loc[["mean", "mean_uncertain", "mean_certain"]]).T)
    print((info_calcium.loc[["mean", "mean_uncertain", "mean_certain"]]).T)
    
    print("dice lipid (nanmean): ", np.nanmean(info[info["uncertain"] == True]["dice_lipid"].values), " / ", np.nanmean(info["dice_lipid"].values))
    print("dice lipid (mean): ", np.mean(np.nan_to_num(info[info["uncertain"] == True]["dice_lipid"].values)), " / ", np.mean(np.nan_to_num(info["dice_lipid"].values)))

    print("avg nr of structures: ", np.nanmean(info[info["uncertain"] == True]["nr_structures"].values), " / ", np.nanmean(info["nr_structures"].values))
    print("avg lipid percentage: ", np.nanmean(info[info["uncertain"] == True]["percentage_lipid"].values), " / ", np.nanmean(info["percentage_lipid"].values))
    print("avg calcium percentage: ", np.nanmean(info[info["uncertain"] == True]["percentage_calcium"].values), " / ", np.nanmean(info["percentage_calcium"].values))
    print("fraction of images with lipid: ", np.nanmean(info[info["uncertain"] == True]["lipid_present"].values), " / ", np.nanmean(info["lipid_present"].values))
    print("fraction of images with calcium: ", np.nanmean(info[info["uncertain"] == True]["calcium_present"].values), " / ", np.nanmean(info["calcium_present"].values))

    print(f"neighbours lipid: {neighbours_lipid}")
    print(f"mispredicted lipid: {mispredicted_lipid / mispredicted_lipid.sum()}")
    print(f"mispredicted calcium: {mispredicted_calcium / mispredicted_calcium.sum()}")

    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/data/diag/leahheil/data", help='path to the directory that contains the data')
    parser.add_argument("--save_dir", type=str, default="/data/diag/leahheil/saved", help='path to the directory that you want to save in')
    parser.add_argument("--mc_samples", type=int, default=0, help='number of mc samples')
    parser.add_argument("--dropout", type=float, default=0.2, help='fraction of dropout to use')
    parser.add_argument("--model_id", type=str, default="1", help='id of the model used for inference')
    parser.add_argument("--ensembles", type=int, default=0, help='number of mc samples')
    parser.add_argument("--tta_samples", type=int, default=0, help='number of mc samples')
    parser.add_argument("--debug", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--polish_data", default=False, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    if args.ensembles > 0:
        save_dir = Path(args.save_dir) / "ensemble"
    else:
        save_dir = Path(args.save_dir)

    inference(data_dir, save_dir, dropout=args.dropout, mc_samples=args.mc_samples, model_id=args.model_id, nr_ensembles=args.ensembles, tta_samples=args.tta_samples, debug=args.debug, polish_data=args.polish_data)

if __name__ == "__main__":
    main()

