import torch
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch.optim import Optimizer


def train_epoch(
    model: torch.nn.Module, device: str, loader: DataLoader, optimizer: Optimizer, criterion
) -> None:
    model.train()

    loss_accum = 0
    for batch in tqdm(loader, desc="Iteration"):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred_list = model(batch)
            optimizer.zero_grad()

            loss = 0
            for i, _ in enumerate(pred_list):
                loss += criterion(pred_list[i].to(torch.float32), batch.y_arr[:, i])

            loss = loss / len(pred_list)

            loss.backward()
            optimizer.step()

            loss_accum += loss.item()

    print(f"Average training loss: {loss_accum / len(loader) }")
