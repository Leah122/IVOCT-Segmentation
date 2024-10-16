from data import OCTDataset
from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt
# from torchsummary import summary
from pathlib import Path
import torch
from tqdm import tqdm
import numpy as np
import argparse
from torch.autograd import Variable
import sklearn.metrics as skl_metrics

from model2 import U_Net
from constants import CLASSES
from utils import plot_labels, plot_image, plot_image_overlay_labels, make_train_val_split
# from model import Model


def dice_loss(input, target): #TODO: is this multiclass??
    """Function to compute dice loss
    source: https://github.com/pytorch/pytorch/issues/1249#issuecomment-305088398

    Args:
        input (torch.Tensor): predictions
        target (torch.Tensor): ground truth mask

    Returns:
        dice loss: 1 - dice coefficient
    """

    

    eps = 0.0001
    iflat = input.view(-1, 704**2)
    tflat = target.view(-1, 704**2)
    intersection = (iflat * tflat).sum(dim=1)

    dice = np.zeros(CLASSES-1)
    # calculate dice for each class except 0, since that is background
    for c in range(1, CLASSES):
        iflat_ = iflat==c
        tflat_ = tflat==c
        intersection = (iflat_ * tflat_).sum()
        union = iflat_.sum(dim=1) + tflat_.sum(dim=1)
        d = ((2.0 * intersection + eps) / (union + eps)).mean()
        dice[c-1] += d

    print(f"INFO DICE: input shape: {input.shape}, target shape: {target.shape}, intersection: {intersection}, dice per class: {dice}")

    return torch.Tensor([1-dice.mean()])


class Trainer:
    def __init__(
        self,
        data_dir: Path,
        save_dir: Path,
        epochs: int = 100,
        batch_size: int = 4,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
    ):
        
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.epochs = epochs
        self.batch_size = batch_size

        train_on_gpu = torch.cuda.is_available()

        if not train_on_gpu:
            print('CUDA is not available. Training on CPU')
        else:
            print('CUDA is available. Training on GPU')

        self.device = torch.device("cuda:0" if train_on_gpu else "cpu")

        # initialise model
        self.model = U_Net(dropout=dropout)#.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # check if val split has been made
        val_empty = not any((self.data_dir / "val" / "labels").iterdir())
        if val_empty:
            print("INFO: did not find validation split, making one")
            make_train_val_split(data_dir)
        
        # create datasets and dataloaders
        print("INFO: loading data")
        dataset_train = OCTDataset(data_dir=self.data_dir ,aug_rotations=180, aug_flip_chance=0.5,)
        dataset_val = OCTDataset(data_dir=self.data_dir, validation=True, aug_rotations=180, aug_flip_chance=0.5,)

        self.dataloader_train = DataLoader(
            dataset=dataset_train,
            batch_size=self.batch_size,
        )

        self.dataloader_val = DataLoader(
            dataset=dataset_val,
            batch_size=self.batch_size,
        )


    def call_model(self, batch: dict):
        images = batch["image"]#.swapaxes(2,3).swapaxes(1,2)#.to(self.device) # need to swap axes to put color channels at the front
        labels = batch["labels"]#.to(self.device)

        outputs = self.model(images).argmax(dim=1)
        # TODO: outputs should be argmaxed to get the labels i think.
        # print(outputs.shape, labels.shape)
        losses = dice_loss(outputs, labels)
        print("INFO LOSSES: ", outputs.shape, labels.shape, losses)
        #torch.nn.functional.cross_entropy(predictions, labels)

        outputs = outputs.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        # labels = [batch_labels.cpu() for batch_labels in labels.items()]

        # print(losses)
        return outputs, labels, losses
    


    def train_epoch(self):
        self.model.train()
        losses = []
        predictions = []
        labels = []

        for batch in tqdm(self.dataloader_train, desc="Training"):
            self.optimizer.zero_grad()
            batch_predictions, batch_labels, batch_loss = self.call_model(batch)

            # Store the predictions, labels and losses to later aggregate them.
            # print(batch_loss, batch_loss[0].item())
            losses.append(batch_loss[0].item())
            predictions.append(batch_predictions)
            labels.append(batch_labels.squeeze())

            # temporary printing to see output and labels of model during training
            # for i, prediction in enumerate(batch_predictions):
            #         plot_image_overlay_labels(batch["image"][i].detach().cpu().numpy(), batch_labels[i], f"overlay_train_while_training_{i}")
            #         plot_image_overlay_labels(batch["image"][i].detach().cpu().numpy(), batch_predictions[i], f"overlay_prediction_train_while_training_{i}")

            batch_loss = Variable(batch_loss.data, requires_grad=True) # need this to set requires_grad to true

            batch_loss.backward()
            self.optimizer.step()

        self.optimizer.zero_grad()
        loss = np.mean(losses)
        BCELoss = torch.nn.BCEWithLogitsLoss()

        metrics = {
            # "auc": np.mean([[skl_metrics.roc_auc_score(label.ravel(), prediction.ravel(), multi_class='ovo') for label, prediction in zip(batch_label, batch_prediction)] for batch_label, batch_prediction in zip(labels, predictions)]), # mean over all auc scores per sample
            "dice": 1 - loss,
        }

        return metrics


    def validation(self):
        self.model.eval()
        losses = []
        predictions = []
        labels = []

        with torch.no_grad():
            for batch in tqdm(self.dataloader_val, desc="Validation"):
                batch_predictions, batch_labels, batch_loss = self.call_model(batch)

                # Store the predictions, labels and losses to later aggregate them.
                losses.append(batch_loss[0].item())
                predictions.append(batch_predictions)
                labels.append(batch_labels)

                # temporary printing to see output and labels of model during training
                # for i, prediction in enumerate(batch_predictions):
                #     plot_image_overlay_labels(batch["image"][i].detach().cpu().numpy(), batch_labels[i], f"overlay_val_while_training_{i}")
                #     plot_image_overlay_labels(batch["image"][i].detach().cpu().numpy(), batch_predictions[i], f"overlay_prediction_val_while_training_{i}")

        self.optimizer.zero_grad()
        loss = np.mean(losses)

        metrics = {
            # "auc": np.mean([skl_metrics.roc_auc_score(label.ravel(), prediction.ravel(), multi_class='ovo') for label, prediction in zip(labels, predictions)]),
            "dice": 1 - loss,
        }

        return metrics
    
    
    
    def train(self):
        metrics = {"train": [], "valid": []}
        best_metric = 0
        best_epoch = 0

        for epoch in range(self.epochs):
            print(f"\n\n===== Epoch {epoch + 1} / {self.epochs} =====\n")

            epoch_train_metrics = self.train_epoch()
            metrics["train"].append(epoch_train_metrics)

            print(epoch_train_metrics)

            epoch_valid_metrics = self.validation()
            metrics["valid"].append(epoch_valid_metrics)

            print(epoch_valid_metrics)

            if epoch_valid_metrics["dice"] > best_metric:
                print("\n===== Saving best model! =====\n")
                best_metric = epoch_valid_metrics["dice"]
                best_epoch = epoch

                torch.save(self.model.state_dict(), self.save_dir / "best_model.pth")
                np.save(self.save_dir / "best_metrics.npy", epoch_valid_metrics)
                # with open(self.save_dir / "best_metrics.json", "w") as f:
                #     json.dump(epoch_valid_metrics, f, indent=4)

            else:
                print(f"Model has not improved since epoch {best_epoch + 1}")

            np.save(self.save_dir / "metrics.npy", metrics)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/data/diag/leahheil/data", help='path to the directory that contains the data')
    parser.add_argument("--save_dir", type=str, default="/data/diag/leahheil/saved", help='path to the directory that you want to save in')
    parser.add_argument("--epochs", type=int, default=20, help='number of epochs to run')
    parser.add_argument("--learning_rate", type=float, default=1e-4, help='initial learning rate')
    parser.add_argument("--dropout", type=float, default=0.2, help='dropout rate')

    args = parser.parse_args()

    trainer = Trainer(
        data_dir=Path(args.data_dir),
        save_dir=Path(args.save_dir),
        epochs=args.epochs,
        batch_size=16,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=0.0
    )
    trainer.train()

if __name__ == "__main__":
    main()



# for i, sample in enumerate(dataloader_train):
#     print(sample["image"].shape)
    # plt.imsave(f"train_{i}.jpg", sample["image"].reshape((704,704,3)).numpy())
    
# model = U_Net()
# #TODO: when training need to reshape to (3, 704, 704) for it to fit into the model
# print(summary(model, input_size=(3,704,704)))

# model = Model()
# print(summary(model, input_size=(3,704,704)))

