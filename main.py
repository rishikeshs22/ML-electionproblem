import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc

# Taking data
df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# Defining features and target
features = ['Constituency ∇', 'Party', 'Criminal Case', 'Total Assets', 'Liabilities', 'state']
target = 'Education'

# Converting all variables to numeric format to feed our models
label_encoder = LabelEncoder()

# Fit LabelEncoder on our data
combined = pd.concat([df[features], df_test[features]])
for feature in features:
    label_encoder.fit(combined[feature])
    df[feature] = label_encoder.transform(df[feature])
    df_test[feature] = label_encoder.transform(df_test[feature])

# Splitting data into features and target and also into training and validation set (20% of the data)
X = df[features]
Y = label_encoder.fit_transform(df[target])
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Random Forest Classifier
randomforest_model = RandomForestClassifier(random_state=42)

# setting hyperparameters for Random Forest
param_grid_rf = {
    'n_estimators': [200, 500],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Performing Grid Search CV for Random Forest
rf_grid_search = GridSearchCV(estimator=randomforest_model, param_grid=param_grid_rf, cv=3, scoring='f1_weighted', n_jobs=-1)
rf_grid_search.fit(X_train, Y_train)

# Getting the best Random Forest model
best_randomforest_model = rf_grid_search.best_estimator_

# Evaluating the best Random Forest model on validation set
val_predictions_rf = best_randomforest_model.predict(X_val)
f1_rf = f1_score(Y_val, val_predictions_rf, average='weighted')

# Gradient Boosting Classifier
gradientboosting_model = GradientBoostingClassifier(random_state=42)

# setting hyperparameters for Gradient Boosting
param_grid_gb = {
    'n_estimators': [200, 500],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5]
}

# Performing Grid Search CV for Gradient Boosting
gb_grid_search = GridSearchCV(estimator=gradientboosting_model, param_grid=param_grid_gb, cv=3, scoring='f1_weighted', n_jobs=-1)
gb_grid_search.fit(X_train, Y_train)

# getting the best Gradient Boosting model
best_gradientboosting_model = gb_grid_search.best_estimator_

# Evaluating the best Gradient Boosting model on validation set
val_predictions_gb = best_gradientboosting_model.predict(X_val)
f1_gb = f1_score(Y_val, val_predictions_gb, average='weighted')

# the model with better F1 score on validation set wins
best_model = best_randomforest_model if f1_rf > f1_gb else best_gradientboosting_model

# Make predictions on the test set
test_predictions = best_model.predict(df_test[features])

# numeric to normal back using inverse transform feature
test_predictions = label_encoder.inverse_transform(test_predictions)

# getting it ready to submit
submission_df = pd.DataFrame({'ID': df_test['ID'], 'Education': test_predictions})
submission_df.to_csv('final3.csv', index=False)



# Feature Importance Plot for Random Forest
plt.figure(figsize=(10, 6))
sns.barplot(x=best_randomforest_model.feature_importances_, y=features)
plt.title('Feature Importance - Random Forest')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.savefig('feature_importance_rf.png')
plt.show()

# Feature Importance Plot for Gradient Boosting
plt.figure(figsize=(10, 6))
sns.barplot(x=best_gradientboosting_model.feature_importances_, y=features)
plt.title('Feature Importance - Gradient Boosting')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.savefig('feature_importance_gb.png')
plt.show()

# Confusion Matrix for Random Forest
plt.figure(figsize=(8, 6))
cm_rf = confusion_matrix(Y_val, val_predictions_rf)
sns.heatmap(cm_rf, annot=True, fmt="d")
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix_rf.png')
plt.show()

# Confusion Matrix for Gradient Boosting
plt.figure(figsize=(8, 6))
cm_gb = confusion_matrix(Y_val, val_predictions_gb)
sns.heatmap(cm_gb, annot=True, fmt="d")
plt.title('Confusion Matrix - Gradient Boosting')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix_gb.png')
plt.show()

# wealth check

    def convert_to_rupees(wealth):
    wealth = wealth.replace(' Crore+', 'e7').replace(' Lac+', 'e5').replace(' Crore', 'e7').replace(' Lac', 'e5').replace('Thou+', 'e3').replace(' Thousand', 'e3')
    wealth_parts = wealth.split()
    if len(wealth_parts) == 2:
        wealth = wealth_parts[0] + wealth_parts[1]
    return float(wealth)
df['Total Assets'] = df['Total Assets'].apply(convert_to_rupees)
wealthy_candidates = df[df['Total Assets'] > 1e7]  # 1 crore = 1e7 rupees
party_counts = df['Party'].value_counts()
wealthy_counts = wealthy_candidates['Party'].value_counts()
percentage_wealthy_candidates = (wealthy_counts / party_counts) * 100
percentage_wealthy_candidates.plot(kind='bar', figsize=(10, 6))
plt.title('Percentage of Candidates with More Than One Crore Rupees in Total Assets by Party')
plt.ylabel('Percentage')
plt.xlabel('Party')
plt.ylim(0, 100)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
