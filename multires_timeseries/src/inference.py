import torch
import numpy as np
from model import MultiResTrafficTransformer  # Adjust if in same file

# Configuration
input_dim_hourly = 4
input_dim_5min = 4
hourly_len = 24
fivemin_len = 72
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = MultiResTrafficTransformer(
    input_dim_hourly=input_dim_hourly,
    input_dim_5min=input_dim_5min,
    d_model=128, n_heads=4, n_layers=2
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
