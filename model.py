"""
LSTM Model for Sepsis Risk Prediction
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os


class SepsisLSTM(nn.Module):
    """LSTM model for sepsis prediction"""
    
    def __init__(self, input_size=8, hidden_size=64, num_layers=1, dropout=0.2):
        super(SepsisLSTM, self).__init__()
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                           num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.dropout(lstm_out[:, -1, :])
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out.squeeze(-1)


def train_model(X_train, y_train, X_test, y_test, epochs=10, batch_size=32, 
                model_path='data/sepsis_lstm.pth'):
    """Train the LSTM model"""
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_test_t, y_test_t)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SepsisLSTM(input_size=X_train.shape[2]).to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            preds = (y_pred >= 0.5).float()
            train_correct += (preds == y_batch).sum().item()
            total_train += y_batch.size(0)
            
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                preds = (y_pred >= 0.5).float()
                val_correct += (preds == y_batch).sum().item()
                total_val += y_batch.size(0)
                
        epoch_train_loss = train_loss / total_train
        epoch_train_acc = train_correct / total_train
        epoch_val_loss = val_loss / total_val
        epoch_val_acc = val_correct / total_val
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} || "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")

    torch.save(model.state_dict(), model_path)
    return model


def load_model(model_path='data/sepsis_lstm.pth', input_size=8):
    """Load trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SepsisLSTM(input_size=input_size).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def predict(model, patient_data):
    """Make prediction for patient data"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if isinstance(patient_data, np.ndarray):
        patient_data = torch.tensor(patient_data, dtype=torch.float32)
    
    if patient_data.ndim == 2:
        patient_data = patient_data.unsqueeze(0)
    
    patient_data = patient_data.to(device)
    
    with torch.no_grad():
        risk_score = model(patient_data).item()
    
    return risk_score
