import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
# Load X_train data
X_train = pd.read_csv(r"X_Train_Data_Input.csv")
X_test = pd.read_csv(r"X_Test_Data_Input.csv")
# Load Y_target data
Y_target = pd.read_csv(r"Y_Train_Data_Target.csv")
Y_final = pd.read_csv(r"Y_Test_Data_Target.csv")
# Drop the ID column, as it's not needed for modeling
X_train = X_train.drop('ID', axis=1)
X_test = X_test.drop('ID', axis=1)
Y_target = Y_target.drop('ID', axis=1)
Y_test = Y_final.drop('ID', axis=1)
# Handle missing values
X_train = X_train.fillna(X_train.mean())

# Check consistency
assert len(X_train) == len(Y_target), "X_train and Y_target do not have the same number of rows"

# Split the data into training and validation sets
X_train_split, X_val_split, Y_train_split, Y_val_split = train_test_split(X_train, Y_target['target'], test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestClassifier()

# Train the model
model.fit(X_train, Y_target['target']) # Changed to use the target column from Y_target

# Make predictions on the validation set
Y_val_pred = model.predict(X_test)

# Evaluate the model on the validation set
# Changed to use the target column from Y_final, and converted to numeric if needed
accuracy_val = accuracy_score(pd.to_numeric(Y_final['target']), Y_val_pred)  
print(f'Validation Accuracy: {accuracy_val}')
print('Validation Classification Report:')
print(classification_report(pd.to_numeric(Y_final['target']), Y_val_pred))
print('Validation Confusion Matrix:')
print(confusion_matrix(pd.to_numeric(Y_final['target']), Y_val_pred))