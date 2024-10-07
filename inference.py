from pathlib import Path

import numpy as np
import pandas
import scipy.ndimage as ndi
import SimpleITK as sitk
import torch
from tqdm import tqdm

from model2 import U_Net
from constants import INPUT_SIZE, CHANNELS, STATE

def inference(
    data_dir: Path, 
    result_dir: Path, 
    dropout: float = 0.0
    ):

    model = U_Net(dropout=dropout).cuda()
    model.eval()

    checkpoint = torch.load(result_dir / "best_model.pth")
    model.load_state_dict(checkpoint)

    test_set_path = Path(data_dir / "test_set" / "images")
    save_path = result_dir / "segmentations"

    raw_segmentations = []

    files = [file for file in list((data_dir / "test" / "labels").glob('./*')) if file != Path('data/test/labels/.DS_Store')]

    for file in files: 
        file_base = str(file).split("/")[-1].split(".")[0]

        image = np.zeros((INPUT_SIZE[0], INPUT_SIZE[1], CHANNELS))
        for channel in range(CHANNELS):
            img = sitk.ReadImage(
                    data_dir / "test" / "images" / f"{file_base}_00{channel+12}.nii.gz" #TODO: remove the +12, this is just for testing one image
                )
            color_channel = sitk.GetArrayFromImage(img).squeeze() / 255.0
            image[:,:,channel] = color_channel
    
        segmentation = sitk.ReadImage(
                data_dir / "test" / "labels" / f"{file_base}.nii.gz"
            )
        
        with torch.no_grad():
            output = model(image).detach().cpu().numpy()

        print(output.shape)
        segmentation = output
        
        # Save segmentation before thresholding
        raw_segmentations.append(segmentation)

        # threshold_value = 0.5
        # segmentation = (segmentation > threshold_value).astype(np.uint8)
        
        sitk.WriteImage(
            segmentation,
            str(test_set_path / f"{file_base}.nii.gz"),
            True,
        )

    np.save(result_dir / "raw_segmentations.npy", raw_segmentations)


    return
