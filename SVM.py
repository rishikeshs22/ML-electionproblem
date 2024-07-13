import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Read data
data = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Function to convert amount strings to numeric
def convert_to_numeric(amount):
    if isinstance(amount, str):
        amount = amount.replace(' Crore+', 'e7').replace(' Lac+', 'e5').replace(' Thou+', 'e3')
        return pd.to_numeric(amount, errors='coerce')
    return amount

# Convert 'Total Assets' and 'Liabilities' to numeric
data['Total Assets'] = data['Total Assets'].apply(convert_to_numeric)
data['Liabilities'] = data['Liabilities'].apply(convert_to_numeric)

test['Total Assets'] = test['Total Assets'].apply(convert_to_numeric)
test['Liabilities'] = test['Liabilities'].apply(convert_to_numeric)

# Drop rows with NaN values
data.dropna(inplace=True)
test.dropna(inplace=True)

# Define target and features
target = data['Education']
variables = data.drop(columns=['Education', 'ID', 'Candidate'], axis=1)
test = test.drop(columns=['ID', 'Candidate'])

# Define mappings for categorical features
categorical_features = ['Constituency ∇', 'Party', 'state']

# Encoding categorical features using one-hot encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', ['Criminal Case', 'Total Assets', 'Liabilities']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)])

# Encode the 'Education' feature using label encoding
label_encoder = LabelEncoder()
target_encoded = label_encoder.fit_transform(target)

# Create SVM classifier pipeline
svm_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', SVC(random_state=42))
])

# Define parameter grid for SVM
param_grid = {
    'clf__C': [0.1, 1, 10],
    'clf__kernel': ['linear', 'poly', 'rbf', 'sigmoid']
}

# Define cross-validation strategy
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform GridSearchCV
grid_search = GridSearchCV(estimator=svm_pipeline, param_grid=param_grid, cv=cv_strategy, scoring='f1_micro')
grid_search.fit(variables, target_encoded)

# Get the best model
best_model = grid_search.best_estimator_

# Predict on test data
test_predictions_encoded = best_model.predict(test)

# Reverse label encoding for predicted 'Education'
predicted_education = label_encoder.inverse_transform(test_predictions_encoded)

# Create a DataFrame with ID and predicted education
submission_df = pd.DataFrame({'ID': test['ID'], 'Education': predicted_education})

# Save the DataFrame to a CSV file
submission_df.to_csv('220514.csv', index=False)
