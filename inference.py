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
from metrics import AUCPR, AUCROC, MI, entropy, entropy_per_class, MI_per_class, group_analysis, filter_uncertain_images, filter_uncertain_images_test
from utils import plot_uncertainty_per_class, plot_image_overlay_labels, plot_uncertainty, plot_softmax_labels_per_class, plot_image_prediction_certain, dice_score

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

    return outputs, image.detach().cpu().numpy(), labels

@torch.no_grad()
def make_predictions_ensemble(sample, model, nr_samples):
    image = sample["image"].to(device)
    labels = sample["labels"].detach().cpu().numpy()

    with torch.no_grad():
        outputs = np.zeros((nr_samples, NEW_CLASSES, 704, 704))
        for i, model_i in enumerate(model):
            outputs[i] = model_i(image).detach().cpu().numpy().squeeze()

    return outputs, image.detach().cpu().numpy(), labels

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

    return np.array(outputs), image.detach().cpu().numpy(), labels

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

    debugging = True

    print("INFO: loading data")
    if validation:
        dataset_test = OCTDataset(data_dir, validation=True, debugging = debugging)
    else:
        dataset_test = OCTDataset(data_dir, test=True, debugging = debugging)

    dataloader_test = DataLoader(
        dataset=dataset_test,
        batch_size=1,
    )

    # tprs = []
    # base_fpr = np.linspace(0, 1, 101)
    uncertainty_maps = {"MI": [], "Entropy": [], "MI_per_class": [], "Entropy_per_class": []}
    predictions = np.zeros((dataset_test.size, 704, 704))

    metrics = pd.DataFrame(columns=['file_name', 'dice', 'background', 'lumen', 'guidewire', 'intima', 'lipid', 'calcium', 'media', 'catheter', 'sidebranch', 'healedplaque', 'auc_roc', 'auc_pr'])
    uncertainty_metrics = pd.DataFrame(columns=['file_name', 'AUC_MI', 'total_MI', 'background', 'lumen', 'guidewire', 'intima', 'lipid', 'calcium', 'media', 'catheter', 'sidebranch', 'healedplaque', 'AUC_Entropy', 'total_Entropy', 'background', 'lumen', 'guidewire', 'intima', 'lipid', 'calcium', 'media', 'catheter', 'sidebranch', 'healedplaque'])
    group_dice_en = pd.DataFrame(columns=['file_name', 'dice', 'dice_certain', 'dice_uncertain', 'background_c', 'background_uc', 'lumen_c', 'lumen_uc', 'guidewire_c', 'guidewire_uc', 'intima_c', 'intima_uc', 'lipid_c', 'lipid_uc', 'calcium_c', 'calcium_uc', 'media_c', 'media_uc', 'catheter_c', 'catheter_uc', 'sidebranch_c', 'sidebranch_uc', 'healedplaque_c', 'healedplaque_uc'])
    # group_dice_mi = pd.DataFrame(columns=['file_name', 'dice', 'dice_certain', 'dice_uncertain', 'background_c', 'background_uc', 'lumen_c', 'lumen_uc', 'guidewire_c', 'guidewire_uc', 'intima_c', 'intima_uc', 'lipid_c', 'lipid_uc', 'calcium_c', 'calcium_uc', 'media_c', 'media_uc', 'catheter_c', 'catheter_uc', 'sidebranch_c', 'sidebranch_uc', 'healedplaque_c', 'healedplaque_uc'])
    info = pd.DataFrame(columns=['file_name', 'dice', 'dice_certain', 'dice_uncertain', 'nr_structures', 'structures_present', 'percentage_lipid', 'percentage_calcium'])

    image_samples = 0

    if tta_samples > 0:
        save_dir_images = str(save_dir) + "/images_tta"
    else:
        save_dir_images = str(save_dir) + "/images"

    iteration = 0

    for sample in tqdm(dataloader_test): 

        file_name = str(sample["metadata"]["file_name"][0])
        if tta_samples > 0: 
            outputs, image, labels = make_predictions_tta(sample, model, tta_samples)
        elif mc_samples > 0: 
            outputs, image, labels = make_predictions_mc(sample, model, mc_samples)
        else: 
            outputs, image, labels = make_predictions_ensemble(sample, model, nr_ensembles)

        dice_per_class = dice_score(outputs.mean(axis=0).argmax(axis=0), labels.squeeze())
        dice = np.nanmean(dice_per_class[1:])

        # s_dice_per_class = soft_dice(torch.Tensor(outputs.mean(axis=0).reshape((1, NEW_CLASSES, 704, 704))), labels.reshape((1, 704, 704)), reduction="dice_per_class").detach().cpu().numpy()
        # s_dice = np.nanmean(s_dice_per_class[1:])

        metrics_list = [file_name, dice]
        for item in dice_per_class:
            metrics_list.append(item)

        mutual_information = MI(outputs)
        auc_roc = AUCROC(outputs, mutual_information, labels)
        auc_pr = AUCPR(outputs, mutual_information, labels)

        metrics_list.append(auc_roc)
        metrics_list.append(auc_pr)
        metrics.loc[len(metrics)] = metrics_list

        uncertainty_list = [file_name, auc_roc]
        mi_per_class = MI_per_class(outputs)
        uncertainty_list.append(np.mean(mi_per_class))
        for mi in mi_per_class:
            uncertainty_list.append(mi.mean())

        entropy_map = entropy(outputs)
        en_per_class = entropy_per_class(outputs)
        uncertainty_list.append(AUCROC(outputs, entropy_map, labels))
        uncertainty_list.append(np.mean(en_per_class))

        for en in en_per_class:
            uncertainty_list.append(en.mean())
        uncertainty_metrics.loc[len(uncertainty_metrics)] = uncertainty_list

        # # fpr, tpr, thresholds = ROC_curve(outputs, mutual_information, labels.detach().cpu().numpy())
        # # tpr = np.interp(base_fpr, fpr, tpr)
        # # tpr[0] = 0.0
        # # tprs.append(tpr)
        # # plot_roc_curve(fpr, tpr, file_path=save_dir_images, file_name=file_name + "ROC")

        dice_per_class_certain_en, dice_per_class_uncertain_en = group_analysis(outputs, entropy_map, labels.squeeze(), percentile=95)
        # dice_per_class_certain_mi, dice_per_class_uncertain_mi = group_analysis(outputs, mutual_information, labels.squeeze(), percentile=95)
        dice_certain = np.nanmean(dice_per_class_certain_en[1:])
        dice_uncertain = np.nanmean(dice_per_class_uncertain_en[1:])
        group_list_en = [file_name, dice, np.nanmean(dice_per_class_certain_en[1:]), np.nanmean(dice_per_class_uncertain_en[1:])]
        # group_list_mi = [file_name, dice, np.nanmean(dice_per_class_certain_mi[1:]), np.nanmean(dice_per_class_uncertain_mi[1:])]
        for i in range(NEW_CLASSES):
            group_list_en.append(dice_per_class_certain_en[i])
            group_list_en.append(dice_per_class_uncertain_en[i])
            # group_list_mi.append(dice_per_class_certain_mi[i])
            # group_list_mi.append(dice_per_class_uncertain_mi[i])
        group_dice_en.loc[len(group_dice_en)] = group_list_en
        # group_dice_mi.loc[len(group_dice_mi)] = group_list_mi

        info_list = [file_name, dice, dice_certain, dice_uncertain]
        # unique structures:
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

        #make dataframe for the group analysis (possibly using different percentiles)
        uncertainty_maps["MI"].append(mutual_information)
        uncertainty_maps["Entropy"].append(entropy_map)
        uncertainty_maps["MI_per_class"].append(mi_per_class)
        uncertainty_maps["Entropy_per_class"].append(en_per_class)

        predictions[iteration] = outputs.mean(axis=0).argmax(axis=0)
        iteration += 1

        # plot the first few images for inspection
        if image_samples > 0:
            plot_image_overlay_labels(image.squeeze(), outputs.mean(axis=0).argmax(axis=0), file_path=save_dir_images, file_name=file_name + "_pred")
            plot_image_overlay_labels(image.squeeze(), labels, file_path=save_dir_images, file_name=file_name)
            plot_image_overlay_labels(image.squeeze(), labels, file_path=save_dir_images, file_name=file_name + "_image", alpha=0)
            plot_softmax_labels_per_class(outputs, file_path=save_dir_images, file_name=file_name + "_softmax_per_class")
            plot_uncertainty_per_class(mi_per_class, file_path=save_dir_images, file_name=file_name + "_uncertainty_per_class_MI", metric="MI")
            plot_uncertainty_per_class(en_per_class, file_path=save_dir_images, file_name=file_name + "_uncertainty_per_class_en", metric="Entropy")
            plot_uncertainty(mutual_information, file_path=save_dir_images, file_name=file_name + "_uncertainty_MI", metric="MI")
            plot_uncertainty(entropy_map, file_path=save_dir_images, file_name=file_name + "_uncertainty_en", metric="Entropy")
            plot_image_prediction_certain(image.squeeze(), outputs, entropy_map, file_path=save_dir_images, file_name=file_name + "_certain_labels_en")
            plot_image_prediction_certain(image.squeeze(), outputs, mutual_information, file_path=save_dir_images, file_name=file_name + "_certain_labels_MI")
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

    # for column in group_dice_mi.columns:
    #     if column != "file_name":
    #         group_dice_mi.loc["mean", column] = group_dice_mi[column].mean(skipna=True)


    p_mi = np.percentile(np.array(uncertainty_maps["MI"]).flatten(), 90)
    p_en = np.percentile(np.array(uncertainty_maps["Entropy"]).flatten(), 90)
    print("p (MI / Entropy): ", p_mi, p_en)


    if nr_ensembles > 0:
        save_str = str(nr_ensembles)
    elif tta_samples > 0:
        save_str = "tta" + str(tta_samples)
    else:
        save_str = "mc" + str(mc_samples)

    metrics.to_csv(str(save_dir) + f"/metrics_{save_str}.csv")
    uncertainty_metrics.to_csv(str(save_dir) + f"/metrics_{save_str}_uncertainty.csv")
    group_dice_en.to_csv(str(save_dir) + f"/metrics_{save_str}_group_en.csv")
    # group_dice_mi.to_csv(str(save_dir) + f"/metrics_{save_str}_group_mi.csv")
    info.to_csv(str(save_dir) + f"/metrics_{save_str}_info.csv")

    print(metrics.loc['mean'])
    print(uncertainty_metrics.loc['mean'])
    print(group_dice_en.loc['mean'])
    # for column in group_dice_mi.columns:
    #     print(column , ":\t", round(group_dice_en.loc['mean', column], 5), " / ", round(group_dice_mi.loc['mean', column], 5))   

    # uncertain_images_MI = filter_uncertain_images(uncertainty_metrics["total_Entropy"].values[:-1], percentile=90)
    uncertain_images_lipid = filter_uncertain_images(np.mean(np.array(uncertainty_maps["Entropy_per_class"]), axis=(2,3)), percentile=90, clas='lipid')
    uncertain_images_calcium = filter_uncertain_images(np.mean(np.array(uncertainty_maps["Entropy_per_class"]), axis=(2,3)), percentile=90, clas='calcium')
    uncertain_images_both = filter_uncertain_images(np.mean(np.array(uncertainty_maps["Entropy_per_class"]), axis=(2,3)), percentile=90, clas='both')
    uncertain_images_important_structures = filter_uncertain_images(np.mean(np.array(uncertainty_maps["Entropy_per_class"]), axis=(2,3)), percentile=90, clas='important_structures')
    uncertain_images_all = filter_uncertain_images(np.mean(np.array(uncertainty_maps["Entropy_per_class"]), axis=(2,3)), percentile=90)
    uncertain_images_all_not_per_class = filter_uncertain_images(np.mean(np.array(uncertainty_maps["Entropy"]), axis=(1,2)), percentile=90, clas='not_per_class')

    # uncertain_images_test = filter_uncertain_images_test(np.mean(np.array(uncertainty_maps["Entropy_per_class"]), axis=(2,3)), predictions)

    # print("uncertain image file lipid: ", info["file_name"].values[uncertain_images_lipid])
    # print("uncertain image file calcium: ", info["file_name"].values[uncertain_images_calcium])
    # print("uncertain image file both: ", info["file_name"].values[uncertain_images_both])
    # print("uncertain image file important structures: ", info["file_name"].values[uncertain_images_important_structures])
    print("uncertain image file test: ", info["file_name"].values[uncertain_images_all_not_per_class])
    info["uncertain"] = uncertain_images_all_not_per_class

    # copy uncertain images into their own folder
    # uncertain_files = [Path(save_dir_images).glob(f'./{file_name}*') for file_name in info["file_name"].values[uncertain_images_all]]
    # uncertain_files = [Path(j) for sub in uncertain_files for j in sub]
    # move_to = ["/".join(str(file).split("/")[:-1]) + "/uncertain_all/" + str(file).split("/")[-1] for file in uncertain_files]

    # # print(move_to)
    # for i in range(len(uncertain_files)):
    #     shutil.copyfile(uncertain_files[i], move_to[i])

    print("avg dice uncertain images: ", (info[info["uncertain"] == True]["dice"].values).mean(), " / ", (info["dice"].values).mean())
    print("avg dice uncertain images (certain pixels): ", (info[info["uncertain"] == True]["dice_certain"].values).mean(), " / ", (info["dice_certain"].values).mean())
    print("avg dice uncertain images (uncertain pixels): ", (info[info["uncertain"] == True]["dice_uncertain"].values).mean(), " / ", (info["dice_uncertain"].values).mean())
    print("avg nr of structures: ", (info[info["uncertain"] == True]["nr_structures"].values).mean(), " / ", (info["nr_structures"].values).mean())



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

    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    if args.ensembles > 0:
        save_dir = Path(args.save_dir) / "ensemble"
    else:
        save_dir = Path(args.save_dir)

    inference(data_dir, save_dir, dropout=args.dropout, mc_samples=args.mc_samples, model_id=args.model_id, nr_ensembles=args.ensembles, tta_samples=args.tta_samples)

if __name__ == "__main__":
    main()


# python inference.py --data_dir "./data/val" --save_dir "./data"
# /data/diag/rubenvdw/nnU-net/Codes/dataset-conversion/Carthesian_view/15_classes/segs_conversion_2d