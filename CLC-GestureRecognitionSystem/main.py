import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_obesity_data(file_path):
    # 1. Load the data
    df = pd.read_csv(file_path)
    
    # 2. Check for missing values
    print("Missing values:\n", df.isnull().sum())
    
    # 3. Simple data validation
    # Remove any rows where height or weight are unreasonable
    df = df[(df['Height'] > 1.4) & (df['Height'] < 2.2) &
            (df['Weight'] > 40) & (df['Weight'] < 200)]
    
    # 4. Encode categorical variables using Label Encoder
    categorical_cols = ['Gender', 'family_history', 'FAVC', 'CAEC', 
                       'SMOKE', 'SCC', 'CALC', 'MTRANS']
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    
    # 5. Scale numerical variables (now without Weight)
    numerical_cols = ['Age', 'Height', 'FCVC', 'NCP', 
                     'CH2O', 'FAF', 'TUE']
    
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # 6. Split features and target, dropping both BMI and Weight
    X = df.drop(['BMI', 'Weight'], axis=1)
    y = df['BMI']
    
    # 7. Print basic statistics
    print("\nDataset shape:", df.shape)
    print("\nFeature names:", list(X.columns))
    
    return X, y


x,y =preprocess_obesity_data("ObesityPrediction.csv")



import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import pandas as pd

class BMIDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X.values)
        self.y = torch.FloatTensor(y.values)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class EnhancedBMINet(nn.Module):
    def __init__(self, input_dim):
        super(EnhancedBMINet, self).__init__()
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2)
        )
        
        # Parallel branches for different feature scales
        self.branch1 = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(64)
        )
        
        self.branch2 = nn.Sequential(
            nn.Linear(128, 32),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(32)
        )
        
        # Concatenated features processing
        self.combined = nn.Sequential(
            nn.Linear(96, 48),  # 64 + 32 = 96
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(48),
            nn.Dropout(0.1),
            
            nn.Linear(48, 24),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(24),
            
            nn.Linear(24, 1)
        )
        
    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)
        
        # Process through parallel branches
        branch1_out = self.branch1(features)
        branch2_out = self.branch2(features)
        
        # Concatenate branch outputs
        combined = torch.cat((branch1_out, branch2_out), dim=1)
        
        # Final processing
        return self.combined(combined)

def train_and_evaluate_bmi_model(X, y, epochs=150, batch_size=64, threshold=2.0):
    # Split the data with stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create datasets and dataloaders
    train_dataset = BMIDataset(X_train, y_train)
    test_dataset = BMIDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model and optimizer
    model = EnhancedBMINet(X.shape[1])
    criterion = nn.HuberLoss(delta=1.0)  # Combines MSE and MAE, more robust to outliers
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Training loop
    best_val_loss = float('inf')
    patience = 150
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y.unsqueeze(1)).item()
        
        # Learning rate scheduling
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.0001)
        
        # Early stopping with best model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
            
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(test_loader):.4f}')
    
    # Load best model for final evaluation
    model.load_state_dict(best_model_state)
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        # Get all predictions
        train_predictions = []
        train_true = []
        for batch_X, batch_y in train_loader:
            outputs = model(batch_X)
            train_predictions.extend(outputs.numpy().flatten())
            train_true.extend(batch_y.numpy().flatten())
            
        test_predictions = []
        test_true = []
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            test_predictions.extend(outputs.numpy().flatten())
            test_true.extend(batch_y.numpy().flatten())
    
    # Convert to numpy arrays
    train_predictions = np.array(train_predictions)
    train_true = np.array(train_true)
    test_predictions = np.array(test_predictions)
    test_true = np.array(test_true)
    
    # Evaluate predictions
    train_within_threshold = np.abs(train_true - train_predictions) <= threshold
    test_within_threshold = np.abs(test_true - test_predictions) <= threshold
    
    # Print detailed results
    print("\nTraining Set Performance:")
    print(f"Accuracy (predictions within {threshold} BMI points): {np.mean(train_within_threshold):.4f}")
    print("\nTest Set Performance:")
    print(f"Accuracy (predictions within {threshold} BMI points): {np.mean(test_within_threshold):.4f}")
    
    # Error analysis
    errors = np.abs(test_true - test_predictions)
    print("\nDetailed Error Analysis (Test Set):")
    print(f"Mean Absolute Error: {np.mean(errors):.2f} BMI points")
    print(f"Median Absolute Error: {np.median(errors):.2f} BMI points")
    print(f"90th percentile of absolute error: {np.percentile(errors, 90):.2f} BMI points")
    print(f"Percentage of predictions within {threshold} BMI points: {(100 * np.mean(test_within_threshold)):.1f}%")
    
    # Error distribution
    error_bins = pd.cut(errors, bins=[0, 1, 2, 3, 4, 5, float('inf')], 
                       labels=['0-1', '1-2', '2-3', '3-4', '4-5', '5+'])
    error_distribution = pd.value_counts(error_bins, normalize=True).sort_index()
    print("\nError Distribution:")
    for bin_name, percentage in error_distribution.items():
        print(f"Error {bin_name} BMI points: {percentage*100:.1f}%")
    
    return model, errors

# Usage example:
if __name__ == "__main__":
    # Assuming X and y are already preprocessed using your preprocess_obesity_data function
    
    
    # Train and evaluate the model
    model, errors = train_and_evaluate_bmi_model(x, y, threshold=2.0)