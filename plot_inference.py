from pathlib import Path

import numpy as np
import pandas as pd
# import scipy.ndimage as ndi
# import SimpleITK as sitk
import torch
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader
# import torchvision.transforms.functional as tf
import pickle
# import regex as re

# from polar_transform.layers import CartesianTransformTrivial

# from torchsummary import summary
# plt.rcParams.update({'font.size': 14})

from data import OCTDataset
# from model2 import U_Net
from constants import NEW_CLASSES, NEW_CLASS_DICT
from metrics import AUCPR, AUCROC, MI, entropy, entropy_per_class, MI_per_class, group_analysis, filter_uncertain_images, sensitivity_specificity, brier_score, brier_score_per_class
from utils import *
# plot_uncertainty_per_class, plot_image_overlay_labels, plot_uncertainty, plot_softmax_labels_per_class, plot_image_prediction_certain, dice_score, normalise_uncertainty, plot_uncertainty_vs_vessel_fraction, fraction_uncertainty, plot_image_prediction_wrong

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def dist_analysis(all_predictions, all_uncertainty_maps, save_dir):
    # calculate and plot distances to closest edge and catheter
    # distances_catheter = []
    # distances_closest_border = []

    # print("calculate distances")
    # for predictions in tqdm(all_predictions):
    #     distances_catheter.append(dist_pixels_to_catheter())
    #     distances_closest_border.append(dist_pixels_to_closest_border(predictions))

    # distances_catheter = np.array(distances_catheter)
    # distances_closest_border = np.array(distances_closest_border)

    # np.save(f"{save_dir}/distances_catheter.npy", distances_catheter)
    # np.save(f"{save_dir}/distances_closest_border.npy", distances_closest_border)

    print("Loading distances")
    distances_catheter = np.load(f"{save_dir}/distances_catheter_full_test.npy")
    distances_closest_border = np.load(f"{save_dir}/distances_closest_border_full_test.npy")
    print("shape distances: ", distances_catheter.shape, distances_closest_border.shape)

    plot_distances_lipid_calcium(all_predictions, all_uncertainty_maps, distances_catheter, save_dir, "catheter")
    plot_distances_lipid_calcium(all_predictions, all_uncertainty_maps, distances_closest_border, save_dir, "border")
    plot_distances(all_predictions, all_uncertainty_maps, distances_catheter, save_dir, "catheter")
    plot_distances(all_predictions, all_uncertainty_maps, distances_closest_border, save_dir, "border")
    return


def vicinity_analysis(all_predictions, all_uncertainty_maps, save_dir):
    # nr_pixels_vicinity = []
    # for predictions in tqdm(all_predictions):
    #     nr_pixels_vicinity.append(nr_classes_in_vicinity(predictions))
    # nr_pixels_vicinity = np.array(nr_pixels_vicinity)
    # np.save(f"{save_dir}/nr_pixels_vicinity_full_test.npy", nr_pixels_vicinity)

    print("Loading vicinity")
    nr_pixels_vicinity = np.load(f"{save_dir}/nr_pixels_vicinity_full_test.npy")
    print("shape vicinity: ", nr_pixels_vicinity.shape)

    plot_vicinity_violin(all_predictions, all_uncertainty_maps, nr_pixels_vicinity, save_dir)
    plot_vicinity(all_predictions, all_uncertainty_maps, nr_pixels_vicinity, save_dir)


def amount_of_uncertainty_per_class(all_predictions, uncertainty_maps):
    for c in range(NEW_CLASSES):
        fraction = len(all_predictions[all_predictions == c]) / len(all_predictions.flatten())
        uncertainty = sum(uncertainty_maps[all_predictions == c]) / sum(uncertainty_maps.flatten())
        uncertainty_mean = np.mean(uncertainty_maps[all_predictions == c])
        print(c, fraction, uncertainty, uncertainty_mean)


@torch.no_grad()
def inference(
    data_dir: Path, 
    save_dir: Path, 
    dropout: float,
    validation: bool = True,
    samples: int = 10,
    method: str = None,
    debug: bool = False,
    load: bool = False,
    polish: bool = False,
    ):

    model_tta = load_model_tta(save_dir, 2)
    model_mc = load_model_mc(save_dir, 2)
    model_ens = load_model_ensemble(Path("/data/diag/leahheil/saved/ensemble"), 1, samples)

    # print(summary(model_mc, (3, 704, 704)))

    debugging = debug

    if not load:
        if polish:
            dataset_test = OCTDataset(data_dir, polish=True, debugging = debugging) 
        elif validation:
            dataset_test = OCTDataset(data_dir, validation=True, debugging = debugging) 
        else:
            dataset_test = OCTDataset(data_dir, test=True, debugging = debugging)

        dataloader_test = DataLoader(
            dataset=dataset_test,
            batch_size=1,
        )

        files = dataset_test.files
    
        artefact_dict = load_artefacts(files)

    save_dir_images = str(save_dir) + f"/images_percentiles"

    dices_dict = {}
    dices_std_dict = {}
    images_used_dict = {}
    percentiles = list(range(100, 50, -1))

    if method != None: # override the method and metric to only do one instead of all 9
        metric_override = method.split("_")[0]
        method_override = method.split("_")[1]

    for model, sampling_method in zip([model_tta, model_mc, model_ens], ["tta", "mc", "ens"]): #, model_mc, model_ens] , "MC", "ensemble"
        if load: # when loading dices_dict
            break
        if method is not None and sampling_method != method_override:
            continue
        
        for metric in ["entropy", "MI", "Brier"]:
            if method is not None and metric != metric_override:
                continue

            all_predictions = np.zeros((len(dataloader_test), 704, 704))
            all_labels = np.zeros((len(dataloader_test), 704, 704))
            uncertainty_maps = np.zeros((len(dataloader_test), 704, 704))
            artefact_maps = np.zeros((len(dataloader_test), 704, 704))

            for i, sample in enumerate(tqdm(dataloader_test, desc=f"{metric}, {sampling_method}")): 
                file_name = str(sample["metadata"]["file_name"][0])
                file_name = "_".join(file_name.split("_")[0:3])
                artefact_maps[i] = artefact_dict[file_name]

                # if "NLDISALA0097_1_frame360" not in file_name:
                #     continue

                # make predictions
                # if sampling_method == "tta":
                #     outputs, image, labels = make_predictions_tta(sample, model, samples)
                # elif sampling_method == "mc":
                #     outputs, image, labels = make_predictions_mc(sample, model, samples)
                # elif sampling_method == "ens":
                #     outputs, image, labels = make_predictions_ensemble(sample, model, samples)

                # all_predictions[i] = outputs.mean(axis=0).argmax(axis=0)
                # all_labels[i] = labels.squeeze()

                # # calculate uncertainty maps
                # if metric == "entropy":
                #     uncertainty_maps[i] = entropy(outputs)
                # elif metric == "MI":
                #     uncertainty_maps[i] = MI(outputs)
                # elif metric == "Brier":
                #     uncertainty_maps[i] = brier_score(outputs)

                # # plot_polar_image(image.squeeze(), file_path=save_dir_images, file_name=file_name + "_polar")
                # # plot_image_overlay_labels(image.squeeze(), labels, file_path=save_dir_images, file_name=file_name + "_image", alpha=0)

                # if "NLDISALA0097_1_frame360" in file_name:
                #     print("make image")
                #     plot_image_overlay_labels(image.squeeze(), outputs.mean(axis=0).argmax(axis=0), file_path=save_dir_images, file_name=file_name + "_pred")
                #     plot_image_overlay_labels(image.squeeze(), labels, file_path=save_dir_images, file_name=file_name)
                #     plot_image_overlay_labels(image.squeeze(), labels, file_path=save_dir_images, file_name=file_name + "_image", alpha=0)
                #     plot_uncertainty(entropy(outputs), file_path=save_dir_images, file_name=file_name + "_uncertainty_en", metric="Entropy")
                #     plot_uncertainty(MI(outputs), file_path=save_dir_images, file_name=file_name + "_uncertainty_MI", metric="MI")
                #     plot_uncertainty(brier_score(outputs), file_path=save_dir_images, file_name=file_name + "_uncertainty_brier", metric="Brier")
                #     plot_image_prediction_certain(image.squeeze(), outputs, entropy(outputs), file_path=save_dir_images, file_name=file_name + "_certain_en", title="Filtered")

            # np.save(f"{save_dir_images}/all_labels_full_test.npy", all_labels)
            # np.save(f"{save_dir_images}/all_predictions_full_test.npy", all_predictions)
            # np.save(f"{save_dir_images}/uncertainty_maps_full_test.npy", uncertainty_maps)
            # np.save(f"{save_dir_images}/all_predictions.npy", all_predictions)
            # np.save(f"{save_dir_images}/uncertainty_maps.npy", uncertainty_maps)

            print("loading predictions")
            all_predictions = np.load(f"{save_dir_images}/all_predictions_full_test.npy")
            uncertainty_maps = np.load(f"{save_dir_images}/uncertainty_maps_full_test.npy")
            all_labels = np.load(f"{save_dir_images}/all_labels_full_test.npy")
            print("shape predictions: ", all_predictions.shape, uncertainty_maps.shape)

            group_analysis_confusion_matrices(all_predictions, uncertainty_maps, all_labels, save_dir_images)

            # dices, percentiles, images_used, standard_deviations = group_analysis_percentiles(all_predictions, uncertainty_maps, all_labels)
            
            # metric_method = metric + "_" + sampling_method
            # dices_dict[metric_method] = dices
            # dices_std_dict[metric_method] = standard_deviations
            # images_used_dict[metric_method] = images_used

            
            # dist_analysis(all_predictions, uncertainty_maps, save_dir_images)
            # vicinity_analysis(all_predictions, uncertainty_maps, save_dir_images)
            # plot_artefact_uncertainty_correlation(all_predictions, artefact_maps, uncertainty_maps, save_dir_images)
            # plot_artefact_uncertainty_correlation_violin(all_predictions, artefact_maps, uncertainty_maps, save_dir_images)
            # plot_artefact_uncertainty_correlation_classes_combined(all_predictions, artefact_maps, uncertainty_maps, save_dir_images, [4,5,9], "plaque") # plaque
            # plot_artefact_uncertainty_correlation_classes_combined(all_predictions, artefact_maps, uncertainty_maps, save_dir_images, [3,6], "wall") # wall
            # for i in range(1,10):
            #     plot_artefact_uncertainty_correlation_classes_combined(all_predictions, artefact_maps, uncertainty_maps, save_dir_images, [i], NEW_CLASS_DICT[i]) # individual structures

            # amount_of_uncertaint_per_class(all_predictions, uncertainty_maps)

    if load: 
        with open(save_dir_images + '/dices_dict.pkl', 'rb') as f:
            dices_dict = pickle.load(f)
        with open(save_dir_images + '/images_used_dict.pkl', 'rb') as f:
            images_used_dict = pickle.load(f)

    else: 
        if dices_dict:
            with open(save_dir_images + '/dices_dict.pkl', 'wb') as f:
                pickle.dump(dices_dict, f)
            with open(save_dir_images + '/images_used_dict.pkl', 'wb') as f:
                pickle.dump(images_used_dict, f)

    # plot dice vs percentile and nr of images used per class
    if dices_dict: # only plot if dices are calculated
        # order: guidewire, catheter, lumen, side branches, intima, media, lipid, calcium, and healed plaque.
        for index, c in [2,7,1,8,3,6,4,5,9]:
        # for index, c in [1,2,3,4,5,6,7,8,9]:# skipping background, catheter, and average
            fig, ax = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, layout='constrained')
            
            for key in dices_dict.keys():
                ax[index/4, index%4].plot(percentiles, dices_dict[key][:, c], label=key)
                ax.set_title(NEW_CLASS_DICT[c])
                max_dice = np.max(dices_dict[key][:, c])
                max_percentile = np.argmax(dices_dict[key][:, c])
                # print(f"for method/metric {key} and class {NEW_CLASS_DICT[c]}, the maximal dice is {max_dice}, at percentile {max_percentile}")
                print(f"dice = {dices_dict[key][5, c]} +- {dices_std_dict[key][5, c]}, for method/metric {key} and class {NEW_CLASS_DICT[c]}, the maximal dice is {max_dice}, at percentile {max_percentile}")
    
        plt.xlabel("percentile")
        plt.ylabel("Dice score")
        plt.xlim(max(percentiles)+1, min(percentiles))
        plt.ylim(0,1.01)
        plt.legend()
        plt.savefig(save_dir_images + f"/plot_filtering_{NEW_CLASS_DICT[c]}.png", bbox_inches='tight')
        plt.close()

    # if dices_dict: # only plot if dices are calculated
    #     for c in range(NEW_CLASSES+1):
    #         fig = plt.figure(figsize=(5,5))
            
    #         for key in dices_dict.keys():
    #             plt.plot(percentiles, dices_dict[key][:, c], label=key)
    #             max_dice = np.max(dices_dict[key][:, c])
    #             max_percentile = np.argmax(dices_dict[key][:, c])
    #             # print(f"for method/metric {key} and class {NEW_CLASS_DICT[c]}, the maximal dice is {max_dice}, at percentile {max_percentile}")
    #             print(f"dice = {dices_dict[[key][5, c]]} +- {dices_std_dict[[key][5, c]]}, for method/metric {key} and class {NEW_CLASS_DICT[c]}, the maximal dice is {max_dice}, at percentile {max_percentile}")

    #         plt.xlabel("percentile")
    #         plt.ylabel("Dice score")
    #         plt.xlim(max(percentiles)+1, min(percentiles))
    #         plt.ylim(0,1.01)
    #         plt.legend()
    #         plt.savefig(save_dir_images + f"/plot_filtering_{NEW_CLASS_DICT[c]}.png", bbox_inches='tight')
    #         plt.close()

    #         fig = plt.figure(figsize=(5,5))
    #         for key in images_used_dict.keys():
    #             plt.plot(percentiles, images_used_dict[key][:, c], label=key)
    #             print(f"with {images_used_dict[key][:, c][0] - images_used_dict[key][:, c][5]} images filtered out out of {images_used_dict[key][:, c][0]}.")

    #         plt.xlabel("percentile")
    #         plt.ylabel("nr of images used")
    #         plt.xlim(max(percentiles)+1, min(percentiles))
    #         plt.legend()
    #         plt.savefig(save_dir_images + f"/plot_images_used_{NEW_CLASS_DICT[c]}.png", bbox_inches='tight')
    #         plt.close()

    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/data/diag/leahheil/data", help='path to the directory that contains the data')
    parser.add_argument("--save_dir", type=str, default="/data/diag/leahheil/saved", help='path to the directory that you want to save in')
    parser.add_argument("--samples", type=int, default=10, help='number of mc samples')
    parser.add_argument("--dropout", type=float, default=0.2, help='fraction of dropout to use')
    parser.add_argument("--model_id", type=str, default="1", help='id of the model used for inference')
    parser.add_argument("--debug", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--load", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--polish", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--test", default=False, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)

    val = not args.test

    inference(data_dir, save_dir, method=args.method, dropout=args.dropout, samples=args.samples, debug=args.debug, load=args.load, polish=args.polish, validation=val)

if __name__ == "__main__":
    main()

