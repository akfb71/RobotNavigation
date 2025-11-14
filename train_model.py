import pickle

from Data_Loaders import Data_Loaders
from Networks import Action_Conditioned_FF

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import Adam


def train_model(no_epochs):

    batch_size = 10
    data_loaders = Data_Loaders(batch_size)
    model = Action_Conditioned_FF()
    optimizer = Adam(model.parameters(),lr=1e-3)
    loss_function = nn.MSELoss()

    losses = []
    # loss_function = nn.BCEWithLogitsLoss()
    min_loss = model.evaluate(model, data_loaders.test_loader, loss_function)
    losses.append(min_loss)

    print (f"Initial test loss: {min_loss:.4f}")

    for epoch_i in range(no_epochs):
        model.train()
        total_train_loss = 0
        for idx, sample in enumerate(data_loaders.train_loader): # sample['input'] and sample['label']
            x = sample['input'].float()
            y = sample['label'].float()
            optimizer.zero_grad()
            y_hat = model(x).squeeze(1)
            loss = loss_function(y_hat, y)

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(data_loaders.train_loader)
            val_loss = model.evaluate(model, data_loaders.test_loader, loss_function)

            losses.append(val_loss)

            print(
                f"Epoch {epoch_i + 1}/{no_epochs} "
                f"- Train Loss: {avg_train_loss:.4f} "
                f"- Test Loss: {val_loss:.4f}"
            )

            with open("saved_model.pkl", "wb") as f:
                pickle.dump(model, f)



if __name__ == '__main__':
    no_epochs = 20
    train_model(no_epochs)
