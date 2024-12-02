from data import OCTDataset
from torch.utils.data import DataLoader
from pathlib import Path
import torch
from tqdm import tqdm
import numpy as np
import argparse
import gc
import json

from model2 import U_Net
from constants import NEW_CLASS_DICT
from utils import make_train_val_split, plot_metrics, soft_dice


class Trainer:
    def __init__(
        self,
        data_dir: Path,
        save_dir: Path,
        job_id: str,
        ensemble_id: int,
        epochs: int = 100,
        batch_size: int = 4,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        class_weight_power: float = 0.5,
        dice_weight: int = 2,
        lipid_calcium_weight: float = 0.0
    ):
        
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.flip_chance = 0.5
        self.aug_rotations = 180
        self.aug_color_chance = 0.1
        self.aug_gaussian_sigma = 0.3
        self.dice_weight = dice_weight
        self.class_weight_power = class_weight_power
        self.job_id = job_id
        self.ensemble_id = ensemble_id
        self.lipid_calcium_weight = lipid_calcium_weight

        train_on_gpu = torch.cuda.is_available()

        if not train_on_gpu:
            print('CUDA is not available. Training on CPU')
        else:
            print('CUDA is available. Training on GPU')

        self.device = torch.device("cuda:0" if train_on_gpu else "cpu")

        # initialise model
        self.model = U_Net(dropout=dropout).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
        )

        self.scheduler = torch.optim.lr_scheduler.PolynomialLR(self.optimizer, total_iters=self.epochs, power=self.weight_decay)

        # check if val split has been made
        if not any((self.data_dir / "val" / "labels").iterdir()):
            print("INFO: did not find validation split, making one")
            make_train_val_split(data_dir)
        
        debugging = False

        # create datasets and dataloaders
        self.dataset_train = OCTDataset(data_dir=self.data_dir, 
                                        aug_rotations=self.aug_rotations, 
                                        aug_flip_chance=self.flip_chance,
                                        aug_color_chance=self.aug_color_chance, 
                                        aug_gaussian_sigma=self.aug_gaussian_sigma,
                                        class_weight_power=self.class_weight_power,
                                        debugging=debugging)
        self.dataset_val = OCTDataset(data_dir=self.data_dir, validation=True, debugging=debugging)

        self.dataloader_train = DataLoader(
            dataset=self.dataset_train,
            batch_size=self.batch_size,
        )

        self.dataloader_val = DataLoader(
            dataset=self.dataset_val,
            batch_size=self.batch_size,
        )

    def call_model(self, batch: dict):
        images = batch["image"].to(self.device)
        labels = batch["labels"].to(self.device)

        outputs = self.model(images)

        # add extra importance to lipid and calcium (4,5)
        weights = np.array(self.dataset_train.class_weights)
        weights[4] += self.lipid_calcium_weight
        weights[5] += self.lipid_calcium_weight

        class_weights = torch.tensor(weights).to(self.device).type(torch.cuda.FloatTensor)
        cross_entropy_func = torch.nn.CrossEntropyLoss(weight=class_weights) 

        cross_entropy = cross_entropy_func(outputs, labels.squeeze().type(torch.cuda.LongTensor))

        softmax = torch.nn.Softmax(dim=1)
        outputs = softmax(outputs)

        dice_loss = soft_dice(outputs, labels.squeeze())
        #TODO: calculate dice_per_class as non soft dice??
        dice_per_class = soft_dice(outputs, labels.squeeze(), reduction='dice_per_class')

        loss = cross_entropy + (dice_loss * self.dice_weight)

        outputs = outputs.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        metrics = {
            "dice_loss": dice_loss,
            "cross_entropy": loss,
            "dice": torch.mean(dice_per_class[1:]),
            "dice_per_class": dice_per_class,
        }

        return outputs, labels, loss, metrics


    def train_epoch(self):
        self.model.train()
        losses = []

        metrics = {
            "dice_loss": [],
            "cross_entropy": [],
            "dice": [],
            "dice_per_class": [],
        }

        for batch in tqdm(self.dataloader_train, desc="Training"):
            self.optimizer.zero_grad()
            batch_predictions, batch_labels, batch_loss, batch_metrics = self.call_model(batch)

            losses.append(batch_loss.item())
            metrics["dice_loss"].append(batch_metrics["dice_loss"].item())
            metrics["cross_entropy"].append(batch_metrics["cross_entropy"].item())
            metrics["dice"].append(batch_metrics["dice"].item())
            metrics["dice_per_class"].append(batch_metrics["dice_per_class"].detach().cpu().numpy())

            # temporary printing to see output and labels of model during training
            # for i, prediction in enumerate(batch_predictions):
            #         plot_image_overlay_labels(batch["image"][i].detach().cpu().numpy(), batch_labels[i], f"overlay_train_while_training_{i}")
            #         plot_image_overlay_labels(batch["image"][i].detach().cpu().numpy(), batch_predictions[i], f"overlay_prediction_train_while_training_{i}")

            batch_loss.backward()
            self.optimizer.step()

        self.scheduler.step()
        self.optimizer.zero_grad()

        epoch_metrics = { #TODO: added skipna
            "dice_loss": np.nanmean(metrics["dice_loss"]),
            "cross_entropy": np.nanmean(metrics["cross_entropy"]),
            "dice": np.nanmean(metrics["dice"]),
            "dice_per_class": np.nanmean(metrics["dice_per_class"], axis=0),
        }

        return epoch_metrics


    def validation(self):
        self.model.eval()
        losses = []
        metrics = {
            "dice_loss": [],
            "cross_entropy": [],
            "dice": [],
            "dice_per_class": [],
        }

        with torch.no_grad():
            for batch in tqdm(self.dataloader_val, desc="Validation"):
                batch_predictions, batch_labels, batch_loss, batch_metrics = self.call_model(batch)

                losses.append(batch_loss.item())
                metrics["dice_loss"].append(batch_metrics["dice_loss"].item())
                metrics["cross_entropy"].append(batch_metrics["cross_entropy"].item())
                metrics["dice"].append(batch_metrics["dice"].item())
                metrics["dice_per_class"].append(batch_metrics["dice_per_class"].detach().cpu().numpy())

                # temporary printing to see output and labels of model during training
                # for i, prediction in enumerate(batch_predictions):
                #     plot_image_overlay_labels(batch["image"][i].detach().cpu().numpy(), batch_labels[i], f"overlay_val_while_training_{i}")
                #     plot_image_overlay_labels(batch["image"][i].detach().cpu().numpy(), batch_predictions[i], f"overlay_prediction_val_while_training_{i}")

        self.optimizer.zero_grad()

        epoch_metrics = {
            "dice_loss": np.nanmean(metrics["dice_loss"]),
            "cross_entropy": np.nanmean(metrics["cross_entropy"]),
            "dice": np.nanmean(metrics["dice"]),
            "dice_per_class": np.nanmean(metrics["dice_per_class"], axis=0),
        }

        return epoch_metrics
    
    def print_run_info(self):
        print(f"\n========== Run Info (model {self.ensemble_id}) ==========\n")
        print(f"job id: {self.job_id}")
        print(f"learning rate: {self.learning_rate}")
        print(f"weight decay: {self.weight_decay}")
        print(f"dropout: {self.dropout}")
        print(f"batch size: {self.batch_size}")
        print(f"flip chance: {self.flip_chance}")
        print(f"rotation degrees: {self.aug_rotations}")
        print(f"color chance: {self.aug_color_chance}")
        print(f"gaussian sigma: {self.aug_gaussian_sigma}")
        print(f"dice weight: {self.dice_weight}")
        print(f"class weight power: {self.class_weight_power}")


    def print_metrics(self, metrics):
        print(f"dice_loss: {metrics['dice_loss']}")
        print(f"cross entropy loss: {metrics['cross_entropy']}")
        print(f"dice: {metrics['dice']}")
        print(f"dice per class: {'  '.join([str(x) for x in metrics['dice_per_class']])}")
    
    
    def train(self):
        metrics = {"train": [], "valid": []}
        best_metric = 0
        best_epoch = 0

        self.print_run_info()
        # print("\n========== Class weights ==========\n")
        # for i in range(len(NEW_CLASS_DICT)):
        #     print(f"{NEW_CLASS_DICT[i]}: {self.dataset_train.class_weights[i]}")

        epoch_valid_metrics = self.validation()

        print(f"metrics before training:")
        self.print_metrics(epoch_valid_metrics)

        for epoch in range(self.epochs):
            print(f"\n\n========== Epoch {epoch + 1} / {self.epochs} (model {self.ensemble_id}) ==========\n ")
            print(f"learning rate: {self.optimizer.param_groups[0]['lr']}")

            epoch_train_metrics = self.train_epoch()
            metrics["train"].append(epoch_train_metrics)

            self.print_metrics(epoch_train_metrics)

            epoch_valid_metrics = self.validation()
            metrics["valid"].append(epoch_valid_metrics)

            self.print_metrics(epoch_valid_metrics)

            if epoch_valid_metrics["dice"] > best_metric:
                print("\n========== Saving best model! ==========\n")
                best_metric = epoch_valid_metrics["dice"]
                best_epoch = epoch

                torch.save(self.model.state_dict(), self.save_dir / f"best_model_{self.job_id}.pth")
                np.save(self.save_dir / f"best_metrics_{self.job_id}.npy", epoch_valid_metrics)

            else:
                print(f"Model has not improved since epoch {best_epoch + 1}")

            np.save(self.save_dir / f"metrics_{self.job_id}.npy", metrics)
            torch.save(self.model.state_dict(), self.save_dir / f"last_model_{self.job_id}.pth")
            gc.collect()

        self.print_run_info()
        print(f"best validation dice: {best_metric}")

        #plot metrics
        plot_metrics(metrics, ["dice_loss", "cross_entropy"], file_path=str(self.save_dir), file_name="metrics_plot")

        self.model.cpu()

        return best_metric


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/data/diag/leahheil/data", help='path to the directory that contains the data')
    parser.add_argument("--save_dir", type=str, default="/data/diag/leahheil/saved/ensemble", help='path to the directory that you want to save in')
    parser.add_argument("--epochs", type=int, default=80, help='number of epochs to run')
    parser.add_argument("--learning_rate", type=float, default=5e-2, help='initial learning rate')
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--weight_decay", type=float, default=0.9)
    parser.add_argument("--class_weight_power", type=float, default=4.0, help='the square used to recalculate the class weights')
    parser.add_argument("--dice_weight", type=int, default=2, help='weight added to the dice score')
    parser.add_argument("--lipid_calcium_weight", type=float, default=0.0, help='float value to add to lipid and calcium')
    parser.add_argument("--job_id", type=str, default="1", help='id to add to the job when running multiple')
    parser.add_argument("--ensemble_ids", type=str, default="0,1,2,3,4")

    args = parser.parse_args()

    with open(f'/data/diag/leahheil/IVOCT-Segmentation/ensemble_config.json', 'r') as file:
        config = json.load(file)

    best_metrics = []

    print(f"\n========== Training models: {args.ensemble_ids} ==========\n")
    for id in args.ensemble_ids.split(","):
        trainer = Trainer(
            data_dir=Path(args.data_dir),
            save_dir=Path(args.save_dir + f"/model_{id}"),
            job_id=args.job_id,
            ensemble_id=id,
            epochs=args.epochs,
            batch_size=args.batch_size,
            dropout=config[str(id)]["dropout"],
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            class_weight_power=config[str(id)]["class_weight_power"],
            dice_weight=config[str(id)]["dice_weight"],
            lipid_calcium_weight=config[str(id)]["lipid_calcium_weight"],
        )
        best_metrics.append(trainer.train())
        del trainer
    
    for i, metrics in enumerate(best_metrics):
        print(i, metrics)

if __name__ == "__main__":
    main()
