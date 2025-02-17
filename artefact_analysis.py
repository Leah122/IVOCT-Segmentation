from pathlib import Path

import numpy as np
import pandas as pd
# import scipy.ndimage as ndi
import SimpleITK as sitk
import torch
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader
# import torchvision.transforms.functional as tf
import pickle
# import regex as re
import torch

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

    model_mc = load_model_mc(save_dir, 2)

    file_dir = "/data/diag/leahheil/some_artifact_pullbacks/"
    file_names = ["NLD-AMPH-0067", "NLD-HMC-0004-RCx", "NLD-ISALA-0003", "NLD-ISALA-0024", "NLD-ISALA-0037-LAD-Mid-1"]#NLD-ISALA-0037-LAD-Mid-1_artifacts
    save_dir_images = str(save_dir) + f"/artefacts"

    pullbacks = []
    artefacts = []

    for file_name in file_names:
    # load dicom
        series = sitk.ReadImage(file_dir + f"data/{file_name}.dcm")
        pullbacks.append(sitk.GetArrayFromImage(series)[::4,:,:])

        # load artefacts
        artefact = sitk.ReadImage(file_dir + f"preds/{file_name}_artifacts.nii.gz")
        artefact = sitk.GetArrayFromImage(artefact)
        artefact_map = preds_to_segmasks(artefact)
        artefacts.append(artefact_map.numpy()[::4,:,:])
        print("shape of artefact maps: ", artefact_map.shape)
        # values, counts = np.unique(artefact_map, return_counts=True)
        # print("nr of pixels with specific score: ", values, counts, counts/np.sum(counts))

        # pullback = pullback[::3,:,:]
        # artefact_map = artefact_map[::3,:,:]

    full_pullback = np.concatenate((pullbacks[0], pullbacks[1], pullbacks[2], pullbacks[3], pullbacks[4]))
    artefact_maps = np.concatenate((artefacts[0], artefacts[1], artefacts[2], artefacts[3], artefacts[4]))

    print("shapes after reducing: ", full_pullback.shape)
    values, counts = np.unique(artefact_maps, return_counts=True)
    print("nr of pixels with specific score total: ", values, counts, counts/np.sum(counts))
    
    
    all_predictions = np.zeros((full_pullback.shape[0], 704, 704))
    uncertainty_maps = np.zeros((full_pullback.shape[0], 704, 704))

    for i, sample in enumerate(tqdm(full_pullback)): 
        
        image = torch.tensor(sample)
        image = image.swapaxes(1,2).swapaxes(0,1).type(torch.cuda.FloatTensor).unsqueeze(0)
        image = image.to(device)

        with torch.no_grad():
            outputs = np.zeros((samples, NEW_CLASSES, 704, 704))
            for mc in range(samples):
                outputs[mc] = model_mc(image).detach().cpu().numpy().squeeze()

        image = image.detach().cpu().numpy() 
        
        all_predictions[i] = outputs.mean(axis=0).argmax(axis=0)
        uncertainty_maps[i] = entropy(outputs)

    np.save(f"{save_dir_images}/predictions_all.npy", all_predictions)
    np.save(f"{save_dir_images}/uncertainty_maps_all.npy", uncertainty_maps)

    # print("loading predictions")
    # all_predictions = np.load(f"{save_dir_images}/predictions_{file_name}.npy")
    # uncertainty_maps = np.load(f"{save_dir_images}/uncertainty_maps_{file_name}.npy")
    print("shape predictions: ", all_predictions.shape, uncertainty_maps.shape)

    values, counts = np.unique(all_predictions, return_counts=True)
    print("nr of preds per class: ", values, counts, counts/np.sum(counts))

    
    plot_artefact_uncertainty_correlation_classes_combined(all_predictions, artefact_maps, uncertainty_maps, save_dir_images, [4,5,9], "plaque") # plaque
    plot_artefact_uncertainty_correlation_classes_combined(all_predictions, artefact_maps, uncertainty_maps, save_dir_images, [3,6], "wall") # wall
    for i in range(1,10):
        plot_artefact_uncertainty_correlation_classes_combined(all_predictions, artefact_maps, uncertainty_maps, save_dir_images, [i], NEW_CLASS_DICT[i]) # individual structures

    # plot_artefact_uncertainty_correlation(all_predictions, artefact_maps, uncertainty_maps, save_dir_images)
    # plot_artefact_uncertainty_correlation_violin(all_predictions, artefact_maps, uncertainty_maps, save_dir_images)

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


