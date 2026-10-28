from torch.utils.data import DataLoader
import torch
from tqdm import trange
import sys
import time
import numpy as np

#Train
def train(dataset, model, batch_size, n_epochs):
    device = next(model.parameters()).device
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters())
    loss_fn = torch.nn.MSELoss()
    best = {"loss": sys.float_info.max}
    loss_history = []
    epoch_times = []

    for epoch in trange(n_epochs, desc="training"):
        t0 = time.time()
        epoch_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch["given"].to(device)), batch["answer"].to(device))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_times.append(time.time() - t0)
        loss_history.append(epoch_loss)
        if epoch_loss < best["loss"]:
            best.update({"state": model.state_dict(), "loss": epoch_loss, "epoch": epoch + 1})

    return best, loss_history, epoch_times

#Predict
def inference(dataset, model, batch_size):
    device = next(model.parameters()).device
    dataloader = DataLoader(dataset, batch_size=batch_size)

    with torch.no_grad():
        ts = [np.array(batch["ts"]) for batch in dataloader]
        dist = [torch.abs(batch["answer"].to(device) - model(batch["given"].to(device))).cpu().numpy() for batch in dataloader]
        att = [np.array(batch["attack"]) if "attack" in batch else np.zeros(batch_size) for batch in dataloader]

    return (np.concatenate(ts), np.concatenate(dist), np.concatenate(att))
