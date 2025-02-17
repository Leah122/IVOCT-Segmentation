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
from scipy.stats import pearsonr, pointbiserialr, spearmanr, shapiro, mannwhitneyu, ttest_ind, wilcoxon

from data import OCTDataset
# from model2 import U_Net
from constants import NEW_CLASSES, NEW_CLASS_DICT
from metrics import AUCPR, AUCROC, MI, entropy, entropy_per_class, MI_per_class, group_analysis, filter_uncertain_images, sensitivity_specificity, entropy_per_class_avg, MI_per_class_avg, brier_score, brier_score_per_class
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

# def load_model_mc(save_dir, model_id, dropout=0.2):
#     model = U_Net(dropout=dropout, softmax=True).to(device)

#     checkpoint = torch.load(save_dir / f"last_model_{model_id}.pth", weights_only=True)
#     model.load_state_dict(checkpoint)
#     model.eval()
#     if dropout > 0:
#         for m in model.modules(): # turn dropout back on
#             if m.__class__.__name__.startswith('Dropout'):
#                 m.train()
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

# def load_model_tta(save_dir, model_id):
#     model = U_Net(dropout=0.0, softmax=True).to(device)

#     checkpoint = torch.load(save_dir / f"last_model_{model_id}.pth", weights_only=True)
#     model.load_state_dict(checkpoint)
#     model.eval()

#     return model

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


def percentual_diff(val1, val2):
    return abs(val1 - val2) / ((val1 + val2) / 2)


def abs_diff(val1, val2):
    return abs(val1 - val2)


def read_uncertainty_excel():
    annotation = pd.read_excel('/data/diag/leahheil/External_validation_dataset.xlsx')

    print(type(annotation['Corelab_lipid_arc'][0]))

    #drop excluded
    annotation = annotation[annotation['Corelab_Exclusion'] != 1]
    annotation = annotation[annotation['Ken_Exclusion'] != 1]

    print("lipid")
    print(len(annotation[(annotation['Corelab_lipid'] == 1) & (annotation['Ken_lipid'] == 1)]))
    print(len(annotation[((annotation['Corelab_lipid'].isnull()) | (annotation['Corelab_lipid'] == 0)) & (annotation['Ken_lipid'] == 1)]))
    print(len(annotation[(annotation['Corelab_lipid'] == 1) & ((annotation['Ken_lipid'].isnull()) | (annotation['Ken_lipid'] == 0))]))
    print(len(annotation[((annotation['Corelab_lipid'].isnull()) | (annotation['Corelab_lipid'] == 0)) & ((annotation['Ken_lipid'].isnull()) | (annotation['Ken_lipid'] == 0))]))

    print("calcium")
    print(len(annotation[(annotation['Corelab_calcium'] == 1) & (annotation['Ken_calcium'] == 1)]))
    print(len(annotation[((annotation['Corelab_calcium'].isnull()) | (annotation['Corelab_calcium'] == 0)) & (annotation['Ken_calcium'] == 1)]))
    print(len(annotation[(annotation['Corelab_calcium'] == 1) & ((annotation['Ken_calcium'].isnull()) | (annotation['Ken_calcium'] == 0))]))
    print(len(annotation[((annotation['Corelab_calcium'].isnull()) | (annotation['Corelab_calcium'] == 0)) & ((annotation['Ken_calcium'].isnull()) | (annotation['Ken_calcium'] == 0))]))


    differences = pd.DataFrame(columns=['file_name', 'lipid_present', 'lipid_agree_presence', 'FC_thickness', 'lipid_arc', 'calcium_present', 'calcium_agree_presence', 'calcium_depth', 'calcium_thickness', 'calcium_arc'])

    for index, row in annotation.iterrows():
        file_name = row['Pullback_ID'] + "_1_frame" + str(row['Frame']-1) + "_" + str(row['Pullback_ID'])[-3:] # frame-1 because the frames are numbered 1 off
        
        diff = [file_name]

        if row['Corelab_lipid'] == 1 and row['Ken_lipid'] == 1: # lipid present
            diff.extend([1, 0]) # present, agree presence = 0, cause no uncertainty since they agree
            diff.append(abs_diff(row['Corelab_FCT'], row['Ken_FCT']))
            diff.append(abs_diff(row['Corelab_lipid_arc'], row['Ken_lipid_arc']))
        else:
            if (row['Corelab_lipid'] == 0 or np.isnan(row['Corelab_lipid'])) and (row['Ken_lipid'] == 0 or np.isnan(row['Ken_lipid'])):
                diff.extend([0, 0, np.nan, np.nan])
            else: 
                diff.extend([0.5, 1, np.nan, np.nan])


        if row['Corelab_calcium'] == 1 and row['Ken_calcium'] == 1: # calcium present
            diff.extend([1, 0])
            diff.append(abs_diff(row['Corelab_calcium_depth'], row['Ken_calcium_depth']))
            diff.append(abs_diff(row['Corelab_calcium_thickness'], row['Ken_calcium_thickness']))
            diff.append(abs_diff(row['Corelab_calcium_arc'], row['Ken_calcium_arc']))
        else:
            if (row['Corelab_calcium'] == 0 or np.isnan(row['Corelab_calcium'])) and (row['Ken_calcium'] == 0 or np.isnan(row['Ken_calcium'])):
                diff.extend([0, 0, np.nan, np.nan, np.nan])
            else: 
                diff.extend([0.5, 1, np.nan, np.nan, np.nan])

        differences.loc[len(differences)] = diff
    
    return differences



@torch.no_grad()
def inference(
    data_dir: Path, 
    save_dir: Path, 
    method: str,
    model_id: str = "2",
    samples: int = 10,
    debug: bool = False,
    load: bool = False,
    ):

    if not load:
        if method == "tta":
            model = load_model_tta(save_dir, model_id)
            print(f"nr of TTA samples: {samples}")
        elif method == "mc":
            model = load_model_mc(save_dir, model_id)
            print(f"nr of MC samples: {samples}")
        elif method == "ens":
            model = load_model_ensemble(Path("/data/diag/leahheil/saved/ensemble"), 1, samples)
            print(f"nr of Ensemble samples: {samples}")

        debugging = debug
        dataset_test = OCTDataset(data_dir, polish=True, debugging = debugging)
        dataloader_test = DataLoader(
            dataset=dataset_test,
            batch_size=1,
        )

    info = pd.DataFrame(columns=['file_name', 'total_MI', 'total_en', 'total_brier', 'total_MI_raw', 'total_en_raw', 'total_brier_raw', 'nr_structures', 'structures_present', 'lipid_present', 'percentage_lipid', 'lipid_en', 'lipid_MI', 'lipid_brier', 'calcium_present', 'percentage_calcium', 'calcium_en', 'calcium_MI', 'calcium_brier'])
    # 'total_brier', 'lipid_brier', 'calcium_brier'
    image_samples = 0
    save_dir_images = str(save_dir) + "/images_polish"
    save_dir = save_dir / "polish"

    differences = read_uncertainty_excel()
    print(differences)

    if not load:
        for i, sample in enumerate(tqdm(dataloader_test)): 
            file_name = str(sample["metadata"]["file_name"][0])
            if file_name not in differences['file_name'].values:
                continue

            if method == "tta":
                outputs, image, labels = make_predictions_tta(sample, model, samples)
            elif method == "mc":
                outputs, image, labels = make_predictions_mc(sample, model, samples)
            elif method == "ens":
                outputs, image, labels = make_predictions_ensemble(sample, model, samples)
            prediction = outputs.mean(axis=0).argmax(axis=0)

            mi_map = MI(outputs)
            entropy_map = entropy(outputs)
            brier_map = brier_score(outputs)

            # entropy_map_per_class_avg = entropy_per_class_avg(outputs)
            entropy_map_per_class = entropy_per_class(outputs)
            # mi_map_per_class = MI_per_class(outputs)
            # brier_map_per_class = brier_score_per_class(outputs)

            info_list = [file_name]
            info_list.append(normalise_uncertainty(mi_map.mean(), prediction))
            info_list.append(normalise_uncertainty(entropy_map.mean(), prediction))
            info_list.append(normalise_uncertainty(brier_map.mean(), prediction))
            info_list.append(mi_map.mean())
            info_list.append(entropy_map.mean())
            info_list.append(brier_map.mean()) 
            structures = []
            for i in range(NEW_CLASSES):
                if i in prediction:
                    structures.append(NEW_CLASS_DICT[i])
            info_list.append(len(structures)) # nr structures
            info_list.append(structures) # structures

            # percentages, dice, and uncertainty scores for lipid and calcium only if in image
            for i in [4, 5]:
                # info_list.append(int(np.any(prediction == i)))
                # info_list.append(len(prediction[prediction==i])/len(prediction.flatten()))

                # if np.any(outputs.mean(axis=0).argmax(axis=0) == i) and np.any(prediction == i):
                #     fraction_class = len(prediction[prediction==i])/len(prediction.flatten())
                # else:
                #     fraction_class = 1

                # entropy_class = np.mean(entropy_map_per_class[i])/fraction_class
                # info_list.append(entropy_class)

                # mi_class = np.mean(mi_map_per_class[i])/fraction_class
                # info_list.append(mi_class)

                # brier_class = np.mean(brier_map_per_class[i])/fraction_class
                # info_list.append(brier_class)

                if np.any(outputs.mean(axis=0).argmax(axis=0) == i) and np.any(prediction == i):
                    info_list.append(int(np.any(prediction == i)))
                    info_list.append(len(prediction[prediction==i])/len(prediction.flatten()))

                    # entropy_class = np.where(prediction == i, entropy_map, np.nan)
                    # info_list.append(np.nanmean(entropy_class))
                    entropy_class = entropy_map_per_class[i]
                    info_list.append(np.nanmean(entropy_class) / len(prediction[prediction==i]))

                    mi_class = np.where(prediction == i, mi_map, np.nan)
                    info_list.append(np.nanmean(mi_class))

                    brier_class = np.where(prediction == i, brier_map, np.nan)
                    info_list.append(np.nanmean(brier_class))
                else: 
                    info_list.extend([0, np.nan, np.nan, np.nan, np.nan])

        
            info.loc[len(info)] = info_list

            # make dataframe for the group analysis (possibly using different percentiles)
            # uncertainty_maps["avg_MI"].append(normalise_uncertainty(mi_map.mean(), outputs.mean(axis=0).argmax(axis=0)))
            # uncertainty_maps["avg_Entropy"].append(normalise_uncertainty(entropy_map.mean(), outputs.mean(axis=0).argmax(axis=0)))


            # plot the first few images for inspection
            # if image_samples > 0:
            # if normalise_uncertainty(entropy_map.mean(), prediction) > 0.015:
            # if info_list[8] > 0.29 and info_list[7] > 0.01:
                #TODO: test plot_image_prediction_one_class
                # if 4 in prediction:
                #     plot_image_prediction_one_class(entropy_map_per_class[4], prediction, file_path=save_dir_images, file_name=file_name + "_lipid_uncertainty", clas=4)
                # plot_image_prediction_wrong(image.squeeze(), outputs.mean(axis=0).argmax(axis=0), labels.squeeze(), file_path=save_dir_images, file_name=file_name + "_wrong")
            
                # print(f"{file_name}, {normalise_uncertainty(entropy_map.mean(), prediction)}")
                # plot_image_overlay_labels(image.squeeze(), outputs.mean(axis=0).argmax(axis=0), file_path=save_dir_images, file_name=file_name + "_image", alpha=0)
                # plot_image_overlay_labels(image.squeeze(), prediction, file_path=save_dir_images, file_name=file_name + "_pred")
                # plot_uncertainty(entropy_map, file_path=save_dir_images, file_name=file_name + "_uncertainty_en", metric="Entropy")
                # plot_uncertainty(mi_map, file_path=save_dir_images, file_name=file_name + "_uncertainty_mi", metric="MI")
                # plot_uncertainty_per_class(brier_map_per_class, file_path=save_dir_images, file_name=file_name + "_uncertainty_per_class_br", metric="Brier")
                # plot_uncertainty_per_class(entropy_map_per_class_avg, file_path=save_dir_images, file_name=file_name + "_uncertainty_per_class_en_avg", metric="Entropy")
                # image_samples -= 1
            
            del outputs
            del sample
            gc.collect()
            torch.cuda.empty_cache()

    if load:
        info = pd.read_csv(str(save_dir) + "/metrics_mc10_info_polish_rel.csv")
        print(info)
    else:
        # uncertain_images = filter_uncertain_images(uncertainty_maps["avg_Entropy"], percentile=90)
        # print("uncertain image files: ", info["file_name"].values[uncertain_images])

        # info["uncertain"] = uncertain_images

        info = pd.merge(info, differences, on=['file_name','file_name'])
        print(info)
        save_str = method + str(samples)
        info.to_csv(str(save_dir) + f"/metrics_{save_str}_info_polish_test.csv")


    info_lipid = info[(info['lipid_present_y'] == 1) & (info['lipid_present_x'] == 1)] 
    info_calcium = info[(info['calcium_present_y'] == 1) & (info['calcium_present_x'] == 1)]
    print("nr of lipid samples:", len(info_lipid))
    print("nr of calcium samples:", len(info_calcium))

    for metric, metric_name in zip(['lipid_en', 'lipid_MI', 'lipid_brier'], ["Entropy", "MI", "Brier"]): #, 'total_en', 'total_MI', 'total_brier', 'total_en_raw', 'total_MI_raw', 'total_brier_raw']:
        print("\n", metric)
 
        # first is for relative difference, second for absolute difference
        # arc_groups = [0, 0.1, 0.25, 0.5, 1] 
        arc_groups = [0, 15, 35, 60, 360]

        arc_groups_dict = {}

        for i in range(len(arc_groups)-1):
            arc_groups_dict[f"{arc_groups[i]} - {arc_groups[i+1]}"] = info_lipid[(info_lipid["FC_thickness"] > arc_groups[i]) & (info_lipid["lipid_arc"] < arc_groups[i+1])][metric].values
        
        df = pd.DataFrame(columns=["Relative difference", metric_name])

        for key, value in arc_groups_dict.items():
            group_df = pd.DataFrame({'Relative difference': [key]*len(value),
                                    metric_name: value})
            print(key, len(group_df))
            df = pd.concat([df, group_df], ignore_index = True)

        plt.figure(figsize=(5 * 1.2, 4))
        sns.violinplot(x='Relative difference', y=metric_name, data=df, color='tab:blue', cut=0,
                    linewidth=1, legend=False)

        for i in range(len(arc_groups_dict)):
            for j in range(len(arc_groups_dict)):
                if i < j:
                    stat, p_value = mannwhitneyu(arc_groups_dict[i], arc_groups_dict[j])
                    significance = p_value <= 0.05
                        
                    if significance:
                        print(i, j, stat, p_value)
                        y_pos = max(df[(df['Relative difference'] == i) | (df['Relative difference'] == j)][metric]) * (1 + 0.06*i)
                        plt.plot([i, j], [y_pos, y_pos], color='black', lw=1)
                        if 0.05 > p_value > 0.01:
                            p_symbol = '*'
                        elif 0.01 > p_value > 0.001:
                            p_symbol = '**'
                        elif p_value < 0.001:
                            p_symbol = '***'
                        plt.text((i+j)/2, y_pos, p_symbol, fontsize=12, ha='center', va='bottom', color='black')
        
        plt.tight_layout()
        plt.savefig(str(save_dir) + f"plot_arc_abs_{metric}", dpi=300, bbox_inches='tight')





    
    # info_lipid = info[(info['lipid_present_y'] == 1) & (info['lipid_present_x'] == 1)] 
    # info_calcium = info[(info['calcium_present_y'] == 1) & (info['calcium_present_x'] == 1)]
    # print("nr of lipid samples:", len(info_lipid))
    # print("nr of calcium samples:", len(info_calcium))

    # lipid_corr = pd.DataFrame(columns=['lipid_arc', 'FCT'])
    # for metric in ['lipid_en', 'lipid_MI', 'lipid_brier']:
    #     pearson_lipid_arc = pearsonr(info_lipid[metric].values, info_lipid['lipid_arc'].values)
    #     pearson_lipid_FCT = pearsonr(info_lipid[metric].values, info_lipid['FC_thickness'].values)

    #     plot_metric_vs_uncertainty_polish(info_lipid[metric].values, info_lipid['lipid_arc'].values, metric, "lipid_arc", str(save_dir), pearson_lipid_arc)
    #     plot_metric_vs_uncertainty_polish(info_lipid[metric].values, info_lipid['FC_thickness'].values, metric, "FC_thickness", str(save_dir), pearson_lipid_FCT)

    #     lipid_corr.loc[metric] = [print_corr(pearson_lipid_arc), print_corr(pearson_lipid_FCT)]

    # for metric in ['total_en', 'total_MI', 'total_brier']:
    #     pointbiserial_lipid_present = pointbiserialr(info[metric].values, info['lipid_agree_presence'].values)

    #     plot_metric_vs_uncertainty_polish(info[metric].values, info['lipid_agree_presence'].values, metric, "lipid_agree_presence", str(save_dir), pointbiserial_lipid_present)
    #     print(metric, print_corr(pointbiserial_lipid_present),)
    
    # print("\n", lipid_corr, "\n")

    # calcium_corr = pd.DataFrame(columns=['calcium_depth', 'calcium_arc', 'calcium_thickness'])
    # for metric in ['calcium_en', 'calcium_MI', 'calcium_brier']:
    #     pearson_calcium_depth = pearsonr(info_calcium[metric].values, info_calcium['calcium_depth'].values)
    #     pearson_calcium_arc = pearsonr(info_calcium[metric].values, info_calcium['calcium_arc'].values)
    #     pearson_calcium_thickness = pearsonr(info_calcium[metric].values, info_calcium['calcium_thickness'].values)

    #     plot_metric_vs_uncertainty_polish(info_calcium[metric].values, info_calcium['calcium_depth'].values, metric, "calcium_depth", str(save_dir), pearson_calcium_depth)
    #     plot_metric_vs_uncertainty_polish(info_calcium[metric].values, info_calcium['calcium_arc'].values, metric, "calcium_arc", str(save_dir), pearson_calcium_arc)
    #     plot_metric_vs_uncertainty_polish(info_calcium[metric].values, info_calcium['calcium_thickness'].values, metric, "calcium_thickness", str(save_dir), pearson_calcium_thickness)

    #     calcium_corr.loc[metric] = [print_corr(pearson_calcium_depth), print_corr(pearson_calcium_arc), print_corr(pearson_calcium_thickness)]
    
    # for metric in ['total_en', 'total_MI', 'total_brier']:
    #     spearmanr_calcium_present = pointbiserialr(info[metric].values, info['calcium_agree_presence'].values)

    #     plot_metric_vs_uncertainty_polish(info[metric].values, info['calcium_agree_presence'].values, metric, "calcium_agree_presence", str(save_dir), spearmanr_calcium_present)
    #     print(metric, print_corr(spearmanr_calcium_present))
    

    # print("\n", calcium_corr, "\n")

    # print("\nSpearman lipid")
    # for metric in ['total_en', 'lipid_en', 'total_MI', 'lipid_MI']:
    #     spearmanr_lipid_present = spearmanr(info[metric].values, info['lipid_present_y'].values)
    #     print(f"{metric} / lipid present: ", spearmanr_lipid_present)
    #     plot_metric_vs_uncertainty_polish(info[metric].values, info['lipid_present_y'].values, metric, "lipid_present_y", str(save_dir), spearmanr_lipid_present)

    # print("\nSpearman calcium")
    # for metric in ['total_en', 'calcium_en', 'total_MI', 'calcium_MI']:
    #     spearmanr_calcium_present = spearmanr(info[metric].values, info['calcium_present_y'].values)
    #     print(f"{metric} / calcium present: ", spearmanr_calcium_present)
    #     plot_metric_vs_uncertainty_polish(info[metric].values, info['calcium_present_y'].values, metric, "calcium_present_y", str(save_dir), spearmanr_calcium_present)



    # plt.figure(figsize=(6,6))
    # plt.scatter(info['total_en'].values, info['lipid_arc'].values, label="entropy", color="cornflowerblue")
    # plt.xlabel("entropy")
    # plt.ylabel("uncertainty score (annotation)")
    # plt.savefig(str(save_dir) + "/uncertainty_vs_entropy.png", bbox_inches='tight')

    # plt.figure(figsize=(6,6))
    # plt.scatter(info['total_MI'].values, info['uncertainty_score'].values, label="MI", color="mediumpurple")
    # plt.xlabel("MI")
    # plt.ylabel("uncertainty score (annotation)")
    # plt.savefig(str(save_dir) + "/uncertainty_vs_MI.png", bbox_inches='tight')


    #file_name', 'lipid_present', 'FC_thickness', 'lipid_arc', 'uncertainty_score_lipid', 'calcium_present', 'calcium_depth', 'calcium_thickness', 'calcium_arc', 'uncertainty_score_calcium', 'uncertainty_score'])
    # info["uncertainty_score"] = uncertainty_scores
    # info["uncertainty_score_lipid"] = uncertainty_scores_lipid
    # info["uncertainty_score_calcium"] = uncertainty_scores_calcium

    # nr_samples = len(info)

    # info = info.loc[info['uncertainty_score'] != None]
    
    # print("nr of samples where both are 1: ", len(info[info['uncertainty_score'] == 3.0]))
    # info = info.loc[info['uncertainty_score'] != 3.0]
    # info = info.loc[info['uncertainty_score_lipid'] != 1.0]
    # info = info.loc[info['uncertainty_score_calcium'] != 1.0]

    # print("nr of samples with score that is not 2: ", len(info), "out of", nr_samples)
    # print(info['uncertainty_score'].values)

    # info.to_csv(str(save_dir) + f"/metrics_{save_str}_info_polish.csv")
    # info.to_csv(str(save_dir) + f"/metrics_{save_str}_differences_polish.csv")

    # print("pearson correlation entropy: \n", info[['total_en', 'uncertainty_score']].corr(method='pearson'))
    # print("spearman correlation entropy: \n", info[['total_en', 'uncertainty_score']].corr(method='spearman'))
    # print("pearson correlation MI: \n", info[['total_MI', 'uncertainty_score']].corr(method='pearson'))
    # print("spearman correlation MI: \n", info[['total_MI', 'uncertainty_score']].corr(method='spearman'))

    # print("pearson correlation entropy: \n", info[['total_en', 'uncertainty_score_lipid']].corr(method='pearson'))
    # print("spearman correlation entropy: \n", info[['total_en', 'uncertainty_score_lipid']].corr(method='spearman'))
    # print("pearson correlation MI: \n", info[['total_MI', 'uncertainty_score_lipid']].corr(method='pearson'))
    # print("spearman correlation MI: \n", info[['total_MI', 'uncertainty_score_lipid']].corr(method='spearman'))

    
    # plt.figure(figsize=(6,6))
    # plt.scatter(info['total_en'].values, info['uncertainty_score_lipid'].values, label="entropy", color="cornflowerblue")
    # plt.xlabel("entropy")
    # plt.ylabel("uncertainty score (annotation)")
    # plt.savefig(str(save_dir) + "/uncertainty_vs_entropy_lipid.png", bbox_inches='tight')

    # plt.figure(figsize=(6,6))
    # plt.scatter(info['total_MI'].values, info['uncertainty_score_lipid'].values, label="MI", color="mediumpurple")
    # plt.xlabel("MI")
    # plt.ylabel("uncertainty score (annotation)")
    # plt.savefig(str(save_dir) + "/uncertainty_vs_MI_lipid.png", bbox_inches='tight')
    

    # print("avg nr of structures: ", np.nanmean(info[info["uncertain"] == True]["nr_structures"].values), " / ", np.nanmean(info["nr_structures"].values))
    # print("avg lipid percentage: ", np.nanmean(info[info["uncertain"] == True]["percentage_lipid"].values), " / ", np.nanmean(info["percentage_lipid"].values))
    # print("avg calcium percentage: ", np.nanmean(info[info["uncertain"] == True]["percentage_calcium"].values), " / ", np.nanmean(info["percentage_calcium"].values))
    # print("fraction of images with lipid: ", np.nanmean(info[info["uncertain"] == True]["lipid_present"].values), " / ", np.nanmean(info["lipid_present"].values))
    # print("fraction of images with calcium: ", np.nanmean(info[info["uncertain"] == True]["calcium_present"].values), " / ", np.nanmean(info["calcium_present"].values))
    # print("uncertainty score (annotated): ", np.nanmean(info[info["uncertain"] == True]["uncertainty_score"].values), " / ", np.nanmean(info["uncertainty_score"].values))
    
    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/data/diag/leahheil/data", help='path to the directory that contains the data')
    parser.add_argument("--save_dir", type=str, default="/data/diag/leahheil/saved", help='path to the directory that you want to save in')
    parser.add_argument("--model_id", type=str, default="2", help='id of the model used for inference')
    parser.add_argument("--samples", type=int, default=10, help='number of samples')
    parser.add_argument("--method", type=str, default="mc", help='method to use to make samples')
    parser.add_argument("--debug", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--load", default=False, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)

    inference(data_dir, save_dir, method=args.method, model_id=args.model_id, samples=args.samples, debug=args.debug, load=args.load)

if __name__ == "__main__":
    main()


# python inference.py --data_dir "./data/val" --save_dir "./data"
# /data/diag/rubenvdw/nnU-net/Codes/dataset-conversion/Carthesian_view/15_classes/segs_conversion_2d