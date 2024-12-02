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

from data import OCTDataset
from model2 import U_Net
from constants import NEW_CLASSES, NEW_CLASS_DICT
from metrics import AUCPR, AUCROC, MI, entropy, entropy_per_class, MI_per_class, group_analysis, filter_uncertain_images, sensitivity_specificity
from utils import *
# plot_uncertainty_per_class, plot_image_overlay_labels, plot_uncertainty, plot_softmax_labels_per_class, plot_image_prediction_certain, dice_score, normalise_uncertainty, plot_uncertainty_vs_vessel_fraction, fraction_uncertainty, plot_image_prediction_wrong

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def augment(image, samples = 10):
        flips = [[0,0], [0,1], [1,0], [1,1]]
        rotations = [0, 30, 60, 90, 120, 180]

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

        image = tf.rotate(image, 360-aug[2])#, fill=[1,0,0,0,0,0,0,0,0,0])
        
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

    # tprs = []
    # base_fpr = np.linspace(0, 1, 101)
    uncertainty_maps = {"avg_MI": [], "avg_Entropy": []}
    uncertainty_scores_en = []
    uncertainty_scores_mi = []
    vessel_fractions = []
    sens_spec = {4: np.zeros(4), 5: np.zeros(4), 8: np.zeros(4), 9: np.zeros(4)}

    metrics = pd.DataFrame(columns=['file_name', 'dice', 'background', 'lumen', 'guidewire', 'intima', 'lipid', 'calcium', 'media', 'catheter', 'sidebranch', 'healedplaque', 'auc_roc'])
    uncertainty_metrics = pd.DataFrame(columns=['file_name', 'AUC_MI', 'total_MI', 'background', 'lumen', 'guidewire', 'intima', 'lipid', 'calcium', 'media', 'catheter', 'sidebranch', 'healedplaque', 'AUC_Entropy', 'total_Entropy', 'background', 'lumen', 'guidewire', 'intima', 'lipid', 'calcium', 'media', 'catheter', 'sidebranch', 'healedplaque'])
    group_dice_en = pd.DataFrame(columns=['file_name', 'dice', 'dice_certain', 'dice_uncertain', 'background_c', 'background_uc', 'lumen_c', 'lumen_uc', 'guidewire_c', 'guidewire_uc', 'intima_c', 'intima_uc', 'lipid_c', 'lipid_uc', 'calcium_c', 'calcium_uc', 'media_c', 'media_uc', 'catheter_c', 'catheter_uc', 'sidebranch_c', 'sidebranch_uc', 'healedplaque_c', 'healedplaque_uc'])
    info = pd.DataFrame(columns=['file_name', 'dice', 'dice_certain', 'dice_uncertain', 'nr_structures', 'structures_present', 'percentage_lipid', 'percentage_calcium'])

    seperate_performance_ens = {"0": [], "1": [], "2": [], "3": [], "4": [], "5": [], "6": [], "7": [], "8": [], "9": []}
    #pd.DataFrame(columns=['file_name', 'dice', 'background', 'lumen', 'guidewire', 'intima', 'lipid', 'calcium', 'media', 'catheter', 'sidebranch', 'healedplaque'])

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

        if polish_data:
            plot_image_overlay_labels(image.squeeze(), outputs.mean(axis=0).argmax(axis=0), file_path=save_dir_images, file_name=file_name + "_pred")
            plot_image_overlay_labels(image.squeeze(), outputs.mean(axis=0).argmax(axis=0), file_path=save_dir_images, file_name=file_name + "_image", alpha=0)
            
            mi_map = normalise_uncertainty(MI(outputs), outputs.mean(axis=0).argmax(axis=0))
            entropy_map = normalise_uncertainty(entropy(outputs), outputs.mean(axis=0).argmax(axis=0))
            print("MI: ", mi_map.mean())
            print("Entropy: ", entropy_map.mean())

            plot_uncertainty(mi_map, file_path=save_dir_images, file_name=file_name + "_uncertainty_MI", metric="MI", title=f"{mi_map.mean()}")
            plot_uncertainty(entropy_map, file_path=save_dir_images, file_name=file_name + "_uncertainty_en", metric="Entropy", title=f"{entropy_map.mean()}")
            continue

        dice_per_class = dice_score(outputs.mean(axis=0).argmax(axis=0), labels.squeeze())
        dice = np.nanmean(dice_per_class[1:])

        # save seperate performance for each ensemble
        if nr_ensembles > 0:
            for i in range(nr_ensembles):
                dice_per_class_ens = dice_score(outputs[i].argmax(axis=0), labels.squeeze())
                dice_ens = np.nanmean(dice_per_class_ens[1:])

                dice_list = []
                dice_list.append(dice_ens)
                dice_list.extend(dice_per_class_ens)
                
                seperate_performance_ens[str(i)].append(dice_list)

        # dice metrics
        metrics_list = [file_name, dice]
        for item in dice_per_class:
            metrics_list.append(item)

        mi_map = MI(outputs)
        auc_roc = AUCROC(outputs, mi_map, labels)

        metrics_list.append(auc_roc)
        metrics.loc[len(metrics)] = metrics_list

        # uncertainty metrics
        uncertainty_list = [file_name, auc_roc]
        mi_per_class = MI_per_class(outputs)
        uncertainty_list.append(np.mean(mi_per_class))
        for mi in mi_per_class:
            uncertainty_list.append(normalise_uncertainty(mi.mean(), outputs.mean(axis=0).argmax(axis=0)))

        entropy_map = entropy(outputs)
        en_per_class = entropy_per_class(outputs)
        uncertainty_list.append(AUCROC(outputs, entropy_map, labels))
        uncertainty_list.append(np.mean(en_per_class))

        for en in en_per_class:
            uncertainty_list.append(normalise_uncertainty(en.mean(), outputs.mean(axis=0).argmax(axis=0)))
        uncertainty_metrics.loc[len(uncertainty_metrics)] = uncertainty_list


        # vessel fraction vs uncertainty scores
        uncertainty_score_en, vessel_fraction = fraction_uncertainty(entropy_map, outputs.mean(axis=0).argmax(axis=0))
        uncertainty_score_mi, vessel_fraction = fraction_uncertainty(mi_map, outputs.mean(axis=0).argmax(axis=0))
        uncertainty_scores_en.append(uncertainty_score_en)
        uncertainty_scores_mi.append(uncertainty_score_mi)
        vessel_fractions.append(vessel_fraction)


        # fpr, tpr, thresholds = ROC_curve(outputs, mi_map, labels)
        # tpr = np.interp(base_fpr, fpr, tpr)
        # tpr[0] = 0.0
        # tprs.append(tpr)
        # plot_roc_curve(fpr, tpr, file_path=save_dir_images, file_name=file_name + "ROC")


        # group analysis
        dice_per_class_certain_en, dice_per_class_uncertain_en = group_analysis(outputs, entropy_map, labels.squeeze(), percentile=90)
        dice_certain = np.nanmean(dice_per_class_certain_en[1:])
        dice_uncertain = np.nanmean(dice_per_class_uncertain_en[1:])
        group_list_en = [file_name, dice, np.nanmean(dice_per_class_certain_en[1:]), np.nanmean(dice_per_class_uncertain_en[1:])]
        for i in range(NEW_CLASSES):
            group_list_en.append(dice_per_class_certain_en[i])
            group_list_en.append(dice_per_class_uncertain_en[i])
        group_dice_en.loc[len(group_dice_en)] = group_list_en

        # group analysis info
        info_list = [file_name, dice, dice_certain, dice_uncertain]
        structures = []
        for i in range(NEW_CLASSES):
            if i in labels:
                structures.append(NEW_CLASS_DICT[i])
        info_list.append(len(structures))
        info_list.append(structures)

        #percentages for calcium and lipid
        info_list.append(len(labels[labels==4])/len(labels.flatten()))
        info_list.append(len(labels[labels==5])/len(labels.flatten()))
        info.loc[len(info)] = info_list

        # make dataframe for the group analysis (possibly using different percentiles)
        uncertainty_maps["avg_MI"].append(normalise_uncertainty(mi_map.mean(), outputs.mean(axis=0).argmax(axis=0)))
        uncertainty_maps["avg_Entropy"].append(normalise_uncertainty(entropy_map.mean(), outputs.mean(axis=0).argmax(axis=0)))

        rates = sensitivity_specificity(outputs, labels)
        indices = [4,5,8,9]
        for i in indices:
            sens_spec[i] += rates[i]

        # plot the first few images for inspection
        if image_samples > 0:
            plot_image_overlay_labels(image.squeeze(), outputs.mean(axis=0).argmax(axis=0), file_path=save_dir_images, file_name=file_name + "_pred")
            plot_image_overlay_labels(image.squeeze(), labels, file_path=save_dir_images, file_name=file_name)
            plot_image_overlay_labels(image.squeeze(), labels, file_path=save_dir_images, file_name=file_name + "_image", alpha=0)
            # plot_softmax_labels_per_class(outputs, file_path=save_dir_images, file_name=file_name + "_softmax_per_class")
            plot_uncertainty_per_class(mi_per_class, file_path=save_dir_images, file_name=file_name + "_uncertainty_per_class_MI", metric="MI")
            plot_uncertainty_per_class(en_per_class, file_path=save_dir_images, file_name=file_name + "_uncertainty_per_class_en", metric="Entropy")
            plot_uncertainty(mi_map, file_path=save_dir_images, file_name=file_name + "_uncertainty_MI", metric="MI")
            plot_uncertainty(entropy_map, file_path=save_dir_images, file_name=file_name + "_uncertainty_en", metric="Entropy")
            # plot_image_prediction_certain(image.squeeze(), outputs, entropy_map, file_path=save_dir_images, file_name=file_name + "_certain_labels_en")
            # plot_image_prediction_certain(image.squeeze(), outputs, mi_map, file_path=save_dir_images, file_name=file_name + "_certain_labels_MI")
            # plot_image_prediction_wrong(image.squeeze(), outputs.mean(axis=0).argmax(axis=0), labels.squeeze(), file_path=save_dir_images, file_name=file_name + "_wrong")
            image_samples -= 1
        
        del outputs
        del sample
        gc.collect()
        torch.cuda.empty_cache()

    for column in metrics.columns:
        if column != "file_name":
            metrics[column] = metrics[column].apply(lambda x: None if x == 1 else x)
            metrics.loc["mean", column] = metrics[column].mean(skipna=True)

    for column in uncertainty_metrics.columns:
        if column != "file_name":
            uncertainty_metrics.loc["mean", column] = uncertainty_metrics[column].mean(skipna=True)
    
    for column in group_dice_en.columns:
        if column != "file_name":
            group_dice_en.loc["mean", column] = group_dice_en[column].mean(skipna=True)

    plot_uncertainty_vs_vessel_fraction(uncertainty_scores_en, uncertainty_scores_mi, vessel_fractions, str(save_dir))

    if nr_ensembles > 0:
        print("performance for seperate ens models")
        for i in range(nr_ensembles):
            print(i)
            print(np.nanmean(np.array(seperate_performance_ens[str(i)]), axis=0))

    # p_mi = np.percentile(np.array(uncertainty_maps["avg_MI"]).flatten(), 90)
    # p_en = np.percentile(np.array(uncertainty_maps["avg_Entropy"]).flatten(), 90)
    # print("p (MI / Entropy): ", p_mi, p_en)

    if nr_ensembles > 0:
        save_str = str(nr_ensembles)
    elif tta_samples > 0:
        save_str = "tta" + str(tta_samples)
    else:
        save_str = "mc" + str(mc_samples)

    metrics.to_csv(str(save_dir) + f"/metrics_{save_str}.csv")
    uncertainty_metrics.to_csv(str(save_dir) + f"/metrics_{save_str}_uncertainty.csv")
    group_dice_en.to_csv(str(save_dir) + f"/metrics_{save_str}_group_en.csv")
    

    print(metrics.loc['mean'])
    print(uncertainty_metrics.loc['mean'])
    print(group_dice_en.loc['mean'])
    # for column in group_dice_mi.columns:
    #     print(column , ":\t", round(group_dice_en.loc['mean', column], 5), " / ", round(group_dice_mi.loc['mean', column], 5))   

    print(sens_spec)

    for i in [4,5,8,9]:
        # print(i)
        TP, FP, TN, FN = sens_spec[i]
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
        print(NEW_CLASS_DICT[i], "sensitivity / specificity: ", sensitivity, specificity)
    uncertain_images = filter_uncertain_images(uncertainty_maps["avg_Entropy"], percentile=90)

    print("uncertain image files: ", info["file_name"].values[uncertain_images])
    info["mean_en"] = uncertainty_maps["avg_Entropy"]
    info["mean_mi"] = uncertainty_maps["avg_MI"]
    info["uncertain"] = uncertain_images

    info.to_csv(str(save_dir) + f"/metrics_{save_str}_info.csv")

    # copy uncertain images into their own folder
    # uncertain_files = [Path(save_dir_images).glob(f'./{file_name}*') for file_name in info["file_name"].values[uncertain_images_all]]
    # uncertain_files = [Path(j) for sub in uncertain_files for j in sub]
    # move_to = ["/".join(str(file).split("/")[:-1]) + "/uncertain_all/" + str(file).split("/")[-1] for file in uncertain_files]

    # # print(move_to)
    # for i in range(len(uncertain_files)):
    #     shutil.copyfile(uncertain_files[i], move_to[i])

    print("avg dice uncertain images: ", np.nanmean(info[info["uncertain"] == True]["dice"].values), " / ", np.nanmean(info["dice"].values))
    print("avg dice certain images: ", np.nanmean(info[info["uncertain"] == False]["dice"].values), " / ", np.nanmean(info["dice"].values))
    print("avg dice uncertain images (certain pixels): ", np.nanmean(info[info["uncertain"] == True]["dice_certain"].values), " / ", np.nanmean(info["dice_certain"].values))
    print("avg dice uncertain images (uncertain pixels): ", np.nanmean(info[info["uncertain"] == True]["dice_uncertain"].values), " / ", np.nanmean(info["dice_uncertain"].values))
    print("avg nr of structures: ", np.nanmean(info[info["uncertain"] == True]["nr_structures"].values), " / ", np.nanmean(info["nr_structures"].values))
    print("avg lipid percentage: ", np.nanmean(info[info["uncertain"] == True]["percentage_lipid"].values), " / ", np.nanmean(info["percentage_lipid"].values))
    print("avg calcium percentage: ", np.nanmean(info[info["uncertain"] == True]["percentage_calcium"].values), " / ", np.nanmean(info["percentage_calcium"].values))
    print("fraction of images with lipid: ", np.count_nonzero(info[info["uncertain"] == True]["percentage_lipid"].values != 0)/len(info[info["uncertain"] == True]), " / ", np.count_nonzero(info["percentage_lipid"].values != 0)/len(info))
    print("fraction of images with calcium: ", np.count_nonzero(info[info["uncertain"] == True]["percentage_calcium"].values != 0)/len(info[info["uncertain"] == True]), " / ", np.count_nonzero(info["percentage_calcium"].values != 0)/len(info))

    for c in range(NEW_CLASSES):
        print("\ncertainty for class: ", NEW_CLASS_DICT[c])
        clas = "".join(NEW_CLASS_DICT[c].split(" "))
        # print((uncertainty_metrics[info["uncertain"] == True][NEW_CLASS_DICT[c]].values).mean())
        print("avg dice uncertain images: ", np.nanmean(metrics[:-1][info["uncertain"] == True][clas].values), " / ", np.nanmean(metrics[clas][:-1].values))
        print("avg dice certain images: ", np.nanmean(metrics[:-1][info["uncertain"] == False][clas].values), " / ", np.nanmean(metrics[clas][:-1].values))
        # print("avg dice certain images: ", (uncertainty_metrics[info["uncertain"] == False][NEW_CLASS_DICT[c]].values).mean(), " / ", (uncertainty_metrics[NEW_CLASS_DICT[c]].values).mean())
        # print("avg dice uncertain images (certain pixels): ", (uncertainty_metrics[info["uncertain"] == True][NEW_CLASS_DICT[c]].values).mean(), " / ", (uncertainty_metrics[NEW_CLASS_DICT[c]].values).mean())

    

    #roc code from: https://stats.stackexchange.com/questions/186337/average-roc-for-repeated-10-fold-cross-validation-with-probability-estimates
    # tprs = np.array(tprs)
    # mean_tprs = tprs.mean(axis=0)
    # std = tprs.std(axis=0)
    # tprs_upper = np.minimum(mean_tprs + std, 1)
    # tprs_lower = mean_tprs - std

    # plt.plot(base_fpr, mean_tprs, 'b')
    # plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)

    # plt.plot([0, 1], [0, 1],'r--')
    # plt.xlim([-0.01, 1.01])
    # plt.ylim([-0.01, 1.01])
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.savefig(str(save_dir) + "/roc/" + f"roc_curve_{mc_samples}" + ".png")
    # plt.close()

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


# python inference.py --data_dir "./data/val" --save_dir "./data"
# /data/diag/rubenvdw/nnU-net/Codes/dataset-conversion/Carthesian_view/15_classes/segs_conversion_2d