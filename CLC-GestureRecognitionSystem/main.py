import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_prepare_data(csv_path='ObesityPrediction.csv'):
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Define target column
    target_column = 'BMI'
    
    # Separate features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Identify categorical columns
    categorical_columns = ['Gender', 'family_history', 'FAVC', 'CAEC', 
                         'SMOKE', 'SCC', 'CALC', 'MTRANS']
    
    # Convert categorical variables to numeric
    label_encoders = {}
    for column in categorical_columns:
        label_encoders[column] = LabelEncoder()
        X[column] = label_encoders[column].fit_transform(X[column])
    
    # Ensure all remaining columns are numeric
    numeric_columns = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    X[numeric_columns] = X[numeric_columns].astype(float)
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, 
        y, 
        test_size=0.2, 
        random_state=42
    )
    
    return X_train, X_test, y_train, y_test, scaler, label_encoders

if __name__ == "__main__":
    # Load and prepare the data
    X_train, X_test, y_train, y_test, scaler, label_encoders = load_and_prepare_data()
    
    # Print shapes and basic information
    print("Features:", X_train.shape[1])
    print("Training samples:", X_train.shape[0])
    print("Testing samples:", X_test.shape[0])
    print("\nTarget (BMI) statistics:")
    print("Mean BMI:", y_train.mean())
    print("Min BMI:", y_train.min())
    print("Max BMI:", y_train.max())
