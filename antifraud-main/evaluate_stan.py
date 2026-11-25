
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

from methods.stan.stan import stan_model

# 1. Configuration
DEVICE = torch.device("cpu")
CHECKPOINT = "results/stan_3d_model.pth"

# 2. Load test data
print("Loading STAN test data...")

x_test = np.load("data/tef_3d.npy", allow_pickle=True)
y_test = np.load("data/tel_3d.npy", allow_pickle=True)

x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

print("Test features:", x_test.shape)
print("Test labels:", y_test.shape)

# 3. Recreate the trained model architecture
print("\nCreating STAN model...")

model = stan_model(
    time_windows_dim=8,
    spatio_windows_dim=5,
    feat_dim=5,
    num_classes=2,
    attention_hidden_dim=150
)

# 4. Initialize LazyLinear layers before loading weights
with torch.no_grad():
    model(torch.zeros(1, 8, 5, 5))

# 5. Load saved model weights
print("Loading trained STAN weights...")

state_dict = torch.load(CHECKPOINT, map_location=DEVICE)
model.load_state_dict(state_dict)

model.to(DEVICE)
model.eval()

print("STAN model loaded successfully!")

# 6. Run inference (no training)
print("\nRunning STAN inference...")

all_probabilities = []
all_predictions = []

with torch.no_grad():
    for start in range(0, len(x_test), 64):
        batch = x_test[start:start + 64].to(DEVICE)

        logits = model(batch)

        probabilities = torch.softmax(logits, dim=1)[:, 1]
        predictions = torch.argmax(logits, dim=1)

        all_probabilities.extend(probabilities.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())

# 7. Calculate evaluation metrics
actual = y_test.numpy()
probabilities = np.array(all_probabilities)
predictions = np.array(all_predictions)

auc = roc_auc_score(actual, probabilities)
f1 = f1_score(actual, predictions, average="macro")
ap = average_precision_score(actual, probabilities)

# 8. Display results
print("\n===================================")
print("       STAN INFERENCE RESULTS")
print("===================================")

print("Evaluated transactions:", len(actual))
print("Fraud transactions:", np.sum(actual == 1))
print("Legitimate transactions:", np.sum(actual == 0))

print("-----------------------------------")
print("Test AUC:", round(auc, 6))
print("Test F1 :", round(f1, 6))
print("Test AP :", round(ap, 6))
print("===================================")

# 9. Save results
with open("results/stan_inference_results.txt", "w") as f:
    f.write("STAN 3D Inference Results\n")
    f.write("=========================\n")
    f.write(f"Test AUC: {auc:.6f}\n")
    f.write(f"Test F1: {f1:.6f}\n")
    f.write(f"Test AP: {ap:.6f}\n")

print("\nResults saved to results/stan_inference_results.txt")
print("Inference completed successfully!")
