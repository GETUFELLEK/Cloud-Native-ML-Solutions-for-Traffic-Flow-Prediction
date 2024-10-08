import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Step 1: Load the Dataset
data = pd.read_csv("traffic_data.csv")

# Step 2: Preprocessing
# Separate the timestamp column and numeric columns
timestamps = data['timestamp']  # Extract the timestamp column
numeric_data = data.drop(columns=['timestamp'])  # Keep only numeric columns

# Apply MinMaxScaler to the numeric data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(numeric_data)

# Step 3: Prepare Time-Series Data
def create_sequences(data, seq_length=60, forecast_horizon=15):
    X, y = [], []
    for i in range(len(data) - seq_length - forecast_horizon):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length:i + seq_length + forecast_horizon])
    return np.array(X), np.array(y)

sequence_length = 60
forecast_horizon = 15
X, y = create_sequences(scaled_data, sequence_length, forecast_horizon)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Step 4: Define the PyTorch Model
class TrafficFlowModel(nn.Module):
    def __init__(self):
        super(TrafficFlowModel, self).__init__()
        self.lstm = nn.LSTM(input_size=X_train.shape[2], hidden_size=128, batch_first=True)
        self.fc = nn.Linear(128, y_train.shape[1])

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Get the output of the last LSTM cell
        output = self.fc(lstm_out)
        return output

# Step 5: Initialize Model, Loss, and Optimizer
model = TrafficFlowModel()
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 6: Train the Model
epochs = 10
batch_size = 64

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    output = model(X_train)

    # Reshape y_train to match the shape of output
    y_train_squeezed = torch.squeeze(y_train)  # Remove singleton dimensions (like [740, 15, 1] -> [740, 15])

    # Compute the loss
    loss = loss_fn(output, y_train_squeezed)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}")