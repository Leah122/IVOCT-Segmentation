from pathlib import Path

import numpy as np
# import pandas
# import scipy.ndimage as ndi
# import SimpleITK as sitk
import torch
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from data import OCTDataset
from model2 import U_Net
from constants import INPUT_SIZE, CHANNELS, STATE, CLASSES
from metrics import AUCPR, AUCROC, MI
from utils import plot_uncertainty_per_class, plot_image, plot_labels, plot_image_overlay_labels

def inference(
    data_dir: Path, 
    save_dir: Path, 
    dropout: float,
    ):

    #TODO: arg parse for val or train split

    model = U_Net(dropout=dropout)#.cuda()
    # model.eval()
    #TODO: eval turns off dropout and batchnorm, so find a way to only turn off batchnorm. 

    # checkpoint = torch.load(save_dir / "best_model.pth")
    # model.load_state_dict(checkpoint)

    test_set_path = Path(data_dir / "test_set" / "images")
    save_path = save_dir / "segmentations"

    if "val" in str(data_dir):
        dataset_test = OCTDataset(validation=True)
    else:
        dataset_test = OCTDataset(test=True)

    dataloader_test = DataLoader(
        dataset=dataset_test,
        batch_size=1,
    )

    raw_segmentations = []

    files = [file for file in list((data_dir / "labels").glob('./*')) if file != Path('data/test/labels/.DS_Store')]

    for sample in tqdm(dataloader_test): 
        image = sample["image"].swapaxes(2,3).swapaxes(1,2) # swap axes so the shape becomes (nr_batches, nr_channels, img, img)
        labels = sample["labels"].detach().cpu().numpy()
        # with torch.no_grad():
            # output = model(image)#.detach().cpu().numpy()
        
        # thresholded_output = torch.argmax(output, dim=1)

        # num_correct = ((thresholded_output == labels).sum())
        # print(num_correct)

        mc_samples = 2
        outputs = np.zeros((mc_samples, CLASSES, 704, 704))
        
        # MC dropout trial:
        for mc in range(mc_samples):
            
            with torch.no_grad():
                outputs[mc] = model(image).detach().cpu().numpy().squeeze()
        
            #TODO: not hardcode nr classes

            # fig, ax = plt.subplots(nrows=1, ncols=15)
            # for j in range(15):
            #     # print(outputs[0, j].shape)
            #     ax[j].imshow(outputs[0, j], cmap='viridis')
            #     ax[j].set_xticks([])
            #     ax[j].set_yticks([])
            #     ax[j].set_title(f"{j}")
            # plt.savefig(f"output_per_class{mc}.png")
            # np.mean(outputs, 0)

            # prediction = np.mean(outputs, axis=0)

            

            #TODO: for calculating MI and stuff i think the labels need to be in shape (15, 704, 704) as one hot encoding
            # maybe not one hot encoding, or at least not the way i do it here, cause it takes too long.
            # one_hot_labels = np.zeros((15, labels.shape[1], labels.shape[2]))
            # for i in range(labels.shape[1]):
            #     for j in range(labels.shape[2]):
            #         one_hot_labels[labels[0,i,j].item(),i,j] = 1
            # print(one_hot_labels.sum())

        # print(outputs.shape)
        # mutual_information = MI(np.argmax(outputs, axis=1).squeeze(), labels)
        mutual_information = MI(outputs)
        auc_roc = AUCROC(outputs, mutual_information, labels)
        auc_pr = AUCPR(outputs, mutual_information, labels)

        print("AUC_ROC / AUC_PR: ", auc_roc, auc_pr)

        plot_image(outputs[0])

        plot_uncertainty_per_class(outputs)
        
        plot_labels(labels)

        plot_image_overlay_labels(sample["image"].detach().cpu().numpy().squeeze(), labels)

        # norm, mycmap = create_color_map()
        # fig = plt.figure(figsize=(6,6)) # if figsize is higher i wont get pixel blending between classes
        # plt.imshow()
        # plt.imshow(labels.reshape((704,704,1)), cmap=mycmap, norm=norm)
        # # plt.colorbar()
        # plt.savefig(f"labels_colors.png", dpi=600)
            
        # labels = labels.squeeze()
        # AUCPR(outputs, labels.squeeze())


        # segmentation = output
        
        # Save segmentation before thresholding
        # raw_segmentations.append(segmentation)

        # segmentation = torch.argmax(raw_segmentations, dim=1) #TODO: figure out dimension

        # threshold_value = 0.5
        # segmentation = (segmentation > threshold_value).astype(np.uint8)
        
        # sitk.WriteImage(
        #     segmentation,
        #     str(test_set_path / f"{file_base}.nii.gz"),
        #     True,
        # )

    # np.save(save_dir / "segmentations" / "raw_segmentations.npy", raw_segmentations)
    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/val", help='path to the directory that contains the data')
    parser.add_argument("--save_dir", type=str, default="./data", help='path to the directory that you want to save in')

    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)

    inference(data_dir, save_dir, dropout=0.3)

if __name__ == "__main__":
    main()


# python inference.py --data_dir "./data/val" --save_dir "./data"
# /data/diag/rubenvdw/nnU-net/Codes/dataset-conversion/Carthesian_view/15_classes/segs_conversion_2d