import optuna
import torch
import numpy as np
from model import MultiResTrafficTransformer  # Adjust if in same file
from multires_timeseries.src.seeder import get_study_name, HOURLY_LEN, FIVEMIN_LEN, DIM_HOURLY, DIM_5MIN

# Configuration
input_dim_hourly = DIM_HOURLY
input_dim_5min = DIM_5MIN
hourly_len = HOURLY_LEN
fivemin_len = FIVEMIN_LEN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Load the model
model = MultiResTrafficTransformer(
    input_dim_hourly=input_dim_hourly,
    input_dim_5min=input_dim_5min,
    d_model=d_model,
    n_heads_encoder=n_heads_encoder,
    n_layers_encoder=n_layers_encoder,
    n_heads_decoder=n_heads_decoder,
    n_layers_decoder=n_layers_decoder,
)
model.load_state_dict(torch.load("multires_transformer.pth", map_location=device))
model.to(device)
model.eval()

# Generate or load sample input
sample_hourly = np.random.randn(1, hourly_len, input_dim_hourly).astype(np.float32)
sample_fivemin = np.random.randn(1, fivemin_len, input_dim_5min).astype(np.float32)

# Convert to torch tensors
hourly_tensor = torch.tensor(sample_hourly).to(device)
fivemin_tensor = torch.tensor(sample_fivemin).to(device)

# Run inference
with torch.no_grad():
    pred_5min, pred_hourly = model(hourly_tensor, fivemin_tensor)

# Post-process outputs
pred_5min = pred_5min.squeeze(0).cpu().numpy()    # Shape: (fivemin_len, 1)
pred_hourly = pred_hourly.squeeze(0).cpu().numpy()  # Shape: (1,)

# Print predictions
print("Hourly prediction:", pred_hourly)
print("5-minute prediction series:", pred_5min.flatten())
