import optuna
import torch
from torch import nn

from torch.utils.data import DataLoader

from multires_timeseries.src.dataset import TrafficDataset, get_time_split
from multires_timeseries.src.model_with_rotary import MultiResTrafficTransformer
from multires_timeseries.src.seeder import set_seed, get_study_name, DIM_HOURLY, DIM_5MIN, get_study_number


def collate_fn(batch):
    hourly, fivemin, y_5min, y_hourly = zip(*batch)
    return (
        torch.stack(hourly),
        torch.stack(fivemin),
        torch.stack(y_5min),
        torch.stack(y_hourly)
    )

save_path = "multires_transformer.pth"


def train_with_val(model, train_loader, val_loader, optimizer, criterion_5min, criterion_hourly, device,
                   alpha=0.5, patience=10, min_delta=1e-4, max_epochs=100):
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        total_train_loss = 0
        for hourly, fivemin, y_5min, y_hourly in train_loader:
            hourly = hourly.to(device)
            fivemin = fivemin.to(device)
            y_5min = y_5min.to(device)
            y_hourly = y_hourly.to(device)
            y_5min = y_5min.squeeze(-1)  # shape becomes [32, 72]

            optimizer.zero_grad()
            pred_5min, pred_hourly = model(hourly, fivemin)

            loss_5min = criterion_5min(pred_5min, y_5min)
            loss_hourly = criterion_hourly(pred_hourly, y_hourly)
            loss = alpha * loss_5min + (1 - alpha) * loss_hourly

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation loop
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for hourly, fivemin, y_5min, y_hourly in val_loader:
                hourly = hourly.to(device)
                fivemin = fivemin.to(device)
                y_5min = y_5min.to(device)
                y_hourly = y_hourly.to(device)
                y_5min = y_5min.squeeze(-1)  # shape becomes [32, 72]

                pred_5min, pred_hourly = model(hourly, fivemin)
                loss_5min = criterion_5min(pred_5min, y_5min)
                loss_hourly = criterion_hourly(pred_hourly, y_hourly)
                loss = alpha * loss_5min + (1 - alpha) * loss_hourly
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")

        # Early stopping
        if best_val_loss - avg_val_loss > min_delta:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break


if __name__ == "__main__":
    import torch

    # Set seed for reproducibility
    set_seed(42)

    # Load best trial from Optuna
    study = optuna.load_study(study_name=get_study_name(), storage="sqlite:///optuna_traffic.db")  # adjust path
    best_trial = study.best_trial
    params = best_trial.params

    # Model hyperparameters
    d_model = int(params["d_model"])
    n_heads_encoder = int(params["n_heads_encoder"])
    n_layers_encoder = int(params["n_layers_encoder"])
    n_heads_decoder = int(params["n_heads_decoder"])
    n_layers_decoder = int(params["n_layers_decoder"])
    lr = params["lr"]
    alpha = params["alpha"]

    dataset = TrafficDataset()
    train_size = int(0.8 * len(dataset))
    train_ds, val_ds = get_time_split(dataset)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    if torch.backends.mps.is_available():
        device = torch.device("mps") # for mac
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Keep d_model divisible by n_heads to avoid shape mismatch.
    # d_model Embedding size of inputs and outputs
    # More layers = deeper modeling, but risk overfitting
    # More heads = more parallel attention patterns
    # d_model Larger values = richer representations, but slower

    # start with small d_model to avoid overfitting
    # Options : use grid search or optuna
    model = MultiResTrafficTransformer(input_dim_hourly=DIM_HOURLY, input_dim_5min=DIM_5MIN,
                                       d_model=d_model,  # 128, 256, 512
                                       n_heads_encoder=n_heads_encoder,
                                       n_layers_encoder=n_layers_encoder,
                                       n_heads_decoder=n_heads_decoder,
                                       n_layers_decoder=n_layers_decoder,
                                       ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # if mse is used, predictions are off by roughly square root of mse
    # use mae if you don't care about outliers
    criterion_5min = nn.MSELoss()
    criterion_hourly = nn.MSELoss()

    # inverse scaling to interpret predictions in original units
    # store mean and std during training and reuse them during inference.
    # save them with the model as torch.save({
    #     'model_state_dict': model.state_dict(),
    #     'mean': dataset.mean,
    #     'std': dataset.std
    # }, "model_with_scaler.pt")
    # original_pred = y_pred[:, 0] * std[0] + mean[0]
    train_with_val(model, train_loader, val_loader, optimizer, criterion_5min, criterion_hourly, device, alpha)

    # train loss less than validation indicates overfitting
    # smaller gap between train and validation indicates good generalization or possibly some underfitting still.
