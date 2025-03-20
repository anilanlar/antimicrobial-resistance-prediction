import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt

class FactorizationMachine(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # FM parameters
        self.linear = nn.Linear(input_dim, 1)
        self.v = nn.Parameter(torch.randn(input_dim, latent_dim))
        
    def forward(self, x):
        # Linear term
        linear_term = self.linear(x)
        
        # Interaction term
        square_of_sum = torch.pow(torch.matmul(x, self.v), 2)
        sum_of_square = torch.matmul(torch.pow(x, 2), torch.pow(self.v, 2))
        interaction_term = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)
        
        return linear_term + interaction_term

class DeepFM(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], latent_dim=16, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        # FM component
        self.fm = FactorizationMachine(input_dim, latent_dim)
        
        # Deep component
        self.deep_layers = nn.ModuleList()
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.deep_layers.append(nn.Linear(prev_dim, hidden_dim))
            self.deep_layers.append(nn.ReLU())
            self.deep_layers.append(nn.BatchNorm1d(hidden_dim))
            self.deep_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        self.output = nn.Linear(hidden_dims[-1] + 1, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # FM component
        fm_output = self.fm(x)
        
        # Deep component
        deep_output = x
        for layer in self.deep_layers:
            deep_output = layer(deep_output)
        
        # Combine FM and Deep outputs
        combined = torch.cat([fm_output, deep_output], dim=1)
        output = self.output(combined)
        return self.sigmoid(output)

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y.unsqueeze(1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            total_loss += loss.item()
            
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    return total_loss / len(val_loader), np.array(all_preds), np.array(all_labels)

def create_data_loaders(X_train, y_train, X_val, y_val, batch_size=32):
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    
    return train_loader, val_loader

def train_and_evaluate(X, y, params):
    """
    Train and evaluate DeepFM with given parameters
    """
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        X_train_scaled, y_train, X_val_scaled, y_val, 
        batch_size=params['batch_size']
    )
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepFM(
        input_dim=X.shape[1],
        hidden_dims=params['hidden_dims'],
        latent_dim=params['latent_dim'],
        dropout=params['dropout']
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(params['n_epochs']):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        val_loss, val_preds, val_labels = evaluate_model(model, val_loader, criterion, device)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{params['n_epochs']}], "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Load best model
    model.load_state_dict(best_model)
    
    # Final evaluation
    _, val_preds, val_labels = evaluate_model(model, val_loader, criterion, device)
    
    # Calculate metrics
    accuracy = accuracy_score(val_labels, (val_preds > 0.5).astype(int))
    conf_matrix = confusion_matrix(val_labels, (val_preds > 0.5).astype(int))
    
    print("\nFinal Results:")
    print(f"Accuracy: {accuracy:.2f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(classification_report(val_labels, (val_preds > 0.5).astype(int)))
    
    # ROC curve
    fpr, tpr, _ = roc_curve(val_labels, val_preds)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
    
    return accuracy, roc_auc

def main():
    # Load and prepare the dataset
    print("Loading dataset...")
    df = pd.read_csv("nit_dataset.csv")
    df.drop(columns=["Unnamed: 0.1", "Unnamed: 0", "example_id"], axis=1, inplace=True)
    
    X = df.iloc[:, :-1].values  # All columns except the last one
    y = df.iloc[:, -1].values   # Only the last column (NIT)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of samples: {X.shape[0]}")
    
    # Use single parameter setup
    params = {
        'hidden_dims': [128, 64, 32],    # Architecture
        'latent_dim': 18,                 # Latent dimension
        'dropout': 0.08,                  # Dropout rate
        'learning_rate': 0.0015,          # Learning rate
        'batch_size': 24,                 # Batch size
        'n_epochs': 60                    # Number of epochs
    }
    
    print("\nTraining DeepFM with parameters:")
    print(params)
    accuracy, roc_auc = train_and_evaluate(X, y, params)
    
    print(f"\nFinal Results:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"AUC-ROC: {roc_auc:.2f}")

if __name__ == "__main__":
    main() 
