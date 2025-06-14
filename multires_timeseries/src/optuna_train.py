import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from multires_timeseries.src.dataset import TrafficDataset, get_time_split
from multires_timeseries.src.model_with_rotary import MultiResTrafficTransformer
from multires_timeseries.src.seeder import get_study_name, DIM_HOURLY, DIM_5MIN, get_study_number

# Check device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def train(model, dataloader, optimizer, criterion_5min, criterion_hourly, device, alpha=0.5):
    is_training = model.training
    total_loss = 0.0

    for hourly, fivemin, y_5min, y_hourly in dataloader:
        hourly = hourly.to(device)
        fivemin = fivemin.to(device)
        y_5min = y_5min.to(device)
        y_hourly = y_hourly.to(device)
        y_5min = y_5min.squeeze(-1)  # shape becomes [32, 72]

        if is_training:
            optimizer.zero_grad()

        pred_5min, pred_hourly = model(hourly, fivemin)

        loss_5min = criterion_5min(pred_5min, y_5min)
        loss_hourly = criterion_hourly(pred_hourly, y_hourly)
        loss = alpha * loss_5min + (1 - alpha) * loss_hourly

        if is_training:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


# Objective function for Optuna
def objective(trial):
    d_model = trial.suggest_categorical("d_model", [128, 256, 512])
    n_heads_encoder = trial.suggest_categorical("n_heads_encoder", [2, 4, 8])
    n_layers_encoder = trial.suggest_int("n_layers_encoder", 2, 4, 6)
    n_heads_decoder = trial.suggest_categorical("n_heads_decoder", [2, 4, 8])
    n_layers_decoder = trial.suggest_int("n_layers_decoder", 2, 4, 6)
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-3)
    alpha = trial.suggest_uniform("alpha", 0.3, 0.7)

    model = MultiResTrafficTransformer(
        input_dim_hourly=DIM_HOURLY,
        input_dim_5min=DIM_5MIN,
        d_model=d_model,
        n_heads_encoder=n_heads_encoder,
        n_layers_encoder=n_layers_encoder,
        n_heads_decoder=n_heads_decoder,
        n_layers_decoder=n_layers_decoder,
    ).to(device)

    dataset = TrafficDataset()

    train_ds, val_ds = get_time_split(dataset)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion_5min = nn.MSELoss()
    criterion_hourly = nn.MSELoss()

    best_val_loss = float("inf")
    epochs_no_improve = 0
    for epoch in range(100):
        train_loss = train(model, train_loader, optimizer, criterion_5min, criterion_hourly, device, alpha)
        val_loss = train(model, val_loader, optimizer, criterion_5min, criterion_hourly, device, alpha)

        trial.report(val_loss, epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= 5:
            break

    return best_val_loss

# Run Optuna study
if __name__ == "__main__":

    for n in range(get_study_number()):
        name = get_study_name(n)
        try:
            optuna.delete_study(study_name=name, storage="sqlite:///optuna_traffic.db")
            print(f"deleted {name}")
        except:
            pass

    study = optuna.create_study(study_name=get_study_name(), direction="minimize", storage="sqlite:///optuna_traffic.db")
    study.optimize(objective, n_trials=20)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Save best params to file
    import json
    with open("best_hyperparams.json", "w") as f:
        json.dump(trial.params, f, indent=4)


    print(f'created {get_study_name()}')
