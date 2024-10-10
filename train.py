from data import OCTDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchsummary import summary
from pathlib import Path
import torch
from tqdm import tqdm
import numpy as np
import argparse

from model2 import U_Net
from model import Model


# def dice_loss(input, target): #TODO: is this multiclass??
#     """Function to compute dice loss
#     source: https://github.com/pytorch/pytorch/issues/1249#issuecomment-305088398

#     Args:
#         input (torch.Tensor): predictions
#         target (torch.Tensor): ground truth mask

#     Returns:
#         dice loss: 1 - dice coefficient
#     """
#     smooth = 1.0

#     iflat = input.view(-1, 64**3)
#     tflat = target.view(-1, 64**3)
#     intersection = (iflat * tflat).sum(dim=1)

#     return (
#         1
#         - (
#             (2.0 * intersection + smooth)
#             / (iflat.sum(dim=1) + tflat.sum(dim=1) + smooth)
#         ).mean()
#     )

def dice_loss(prediction, target):
    """Calculating the dice loss
    Args:
        prediction = predicted image
        target = Targeted image
    Output:
        dice_loss"""

    smooth = 1.0

    i_flat = prediction.view(-1)
    t_flat = target.view(-1)

    intersection = (i_flat * t_flat).sum()

    return 1 - ((2. * intersection + smooth) / (i_flat.sum() + t_flat.sum() + smooth))



class Trainer:
    def __init__(
        self,
        data_dir: Path,
        save_dir: Path,
        epochs: int = 100,
        batch_size: int = 8,
        dropout: float = 0.0,
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

        self.model = U_Net(dropout=dropout).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        dataset_train = OCTDataset()
        dataset_val = OCTDataset(validation=True)

        dataloader_train = DataLoader(
            dataset=dataset_train,
            batch_size=1,
        )

        dataloader_val = DataLoader(
            dataset=dataset_val,
            batch_size=1,
        )



    def call_model(self, batch: dict):
        images = batch["image"].swapaxes(2,3).swapaxes(1,2).to(self.device) # need to swap axes to put color channels at the front
        labels = batch["labels"].to(self.device)

        outputs = self.model(images)
        # TODO: outputs should be argmaxed to get the labels i think.
        losses = dice_loss(outputs, labels)

        outputs = outputs.detach().cpu().numpy()
        labels = [batch_labels.cpu() for batch_labels in labels.items()]

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
            for value in batch_loss.items():
                losses.append(value.item())
            for value in batch_predictions.items():
                predictions.extend(value)
            for value in batch_labels.items():
                labels.extend(value)

            batch_loss.backward()
            self.optimizer.step()

        self.optimizer.zero_grad()
        loss = np.mean(losses)
        return loss


    def validation(self):
        self.model.eval()
        losses = []
        predictions = []
        labels = []

        with torch.no_grad():
            for batch in tqdm(self.dataloader_valid, desc="Validation"):
                batch_predictions, batch_labels, batch_loss = self.call_model(batch)

                # Store the predictions, labels and losses to later aggregate them.
                for value in batch_loss.items():
                    losses.append(value.item())
                for value in batch_predictions.items():
                    predictions.extend(value)
                for value in batch_labels.items():
                    labels.extend(value)

        self.optimizer.zero_grad()
        loss = np.mean(losses)

        return loss
    
    
    
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

            if epoch_valid_metrics["overall"] > best_metric:
                print("\n===== Saving best model! =====\n")
                best_metric = epoch_valid_metrics["overall"]
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
    parser.add_argument("--data_dir", type=str, help='path to the directory that contains the data')
    parser.add_argument("--save_dir", type=str, help='path to the directory that you want to save in')

    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)

    trainer = Trainer(
        data_dir=data_dir,
        save_dir=save_dir,
        epochs=2,
        batch_size=16,
        dropout=None,
        learning_rate=1e-4,
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

