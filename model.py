# %% [markdown]
# Import the neccessary libraries

# %%
import pandas as pd
import yfinance as yf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,roc_curve,auc,f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# %% [markdown]
# Data Acquisition

# %%
# Define the sticker symbol for Visa (V)
TICKER = "V"
# Dowload historical stock data for the last 5 years
print(f"Downloading data for {TICKER}..........")
data = yf.download(TICKER,period="5y")

# %% [markdown]
# Data Preprocessing

# %%
# Check for missing values
data_missing = data.isnull().sum()
print("===== Missing Values =====")
print(data_missing)

# %%
# Check for duplicated rows
data_duplicates = data.duplicated().sum()
print("===== Duplicated Rows =====")
print(data_duplicates)

# %% [markdown]
# Feature Engineering

# %%
# Create the Target Variable: Price Movement (Classification)
# Shift the 'Close' price one day back to get 'Next_Close'
# This is what we want to predict: the closing price for the next day.
data['Next_Close'] = data['Close'].shift(-1)
# Create the binary target: 1 if Next_Close > Close (price went UP), 0 otherwise (price went DOWN/flat)
# Use positional (numpy) comparison to avoid pandas index alignment issues when comparing Series
data['Target'] = (data['Next_Close'].to_numpy() > data['Close'].to_numpy()).astype(int)

# Drop the last row, as it will have NaN for 'Next_Close' and 'Target'
# because we cannot know the future closing price.
data.dropna(inplace=True)

# %%
# Function to calculate Relative Strength Index (RSI) - a momentum indicator
def calculate_rsi(df,window=14):
    # Calculte daily price changes
    delta = df["Close"].diff()
    #  Separate gains (upward changes) and losses (downward changes)
    gain = delta.where(delta > 0,0)
    loss = -delta.where(delta < 0,0)

    # Calculate the Exponential Moving Avearge (EMA) of gains and losses
    avg_gain = gain.ewm(com=window-1,min_periods=window).mean()
    avg_loss = loss.ewm(com=window-1,min_periods=window).mean()

    # Calculate Relative Strength (RS)
    rs = avg_gain / avg_loss
    
    # Calculate RS1
    rsi = 100 - (100 / (1 + rs))

    return rsi

# %%
# Create a Simple Moving Average (SMAs) - trend indicators
data["SMA_5"] = data["Close"].rolling(window=5).mean() # 5-day Moving Average
data["SMA_10"] = data["Close"].rolling(window=10).mean() # 10-day Moving Average

# %%
# Create a Relative Strength Index (RSI)
data["RSI"] = calculate_rsi(data)

# %%
# Create Moving Average Convergence Divergence (MACD) - another momentum indicator
# Typically uses 12-day EMA, 26-day EMA, and a 9-day EMA signal line
data["EMA_12"] =  data["Close"].ewm(span=12,adjust=False).mean()
data["EMA_26"] = data["Close"].ewm(span=26,adjust=False).mean()
data["MACD"] = data["EMA_12"] - data["EMA_26"]
data["MACD_Signal"] = data["MACD"].ewm(span=9,adjust=False).mean()

# %%
# Add Volume as a feature, normalized by the mean value
data["Volume_Norm"] = data["Volume"] / data["Volume"].mean()

# %%
# Drop rows with NaN values created by the rolling windows/EMAs (the first 26 days)
data.dropna(inplace=True)

# %% [markdown]
# Visualization Before Training

# %%
'''print("Generating pre-training visualizations...........")

# Set up the plotting style
sns.set_style("whitegrid")
plt.Figure(figsize=(14,10))

# Subplot1: Closing Price Trend
plt.subplot(2,1,1)
data["Target"].plot(title=f"{TICKER} Stock Price Trend (5 years)",color="black",linewidth=1.5)
plt.xlabel("Date")
plt.ylabel("Closing Price (USD)")
plt.grid(True,linestyle="--",alpha=0.6)

# Subplot 2:  Target Variable Distribution (Balance Check)
plt.subplot(2,1,2)
data["Target"].value_counts().plot(kind="bar",color=["red","blue"])
plt.title("Distribution of Target Variable (Price Movement)")
plt.xticks([0,1],["Down/Flat (0)","Up (1)"])
plt.ylabel("Count of Days")
plt.grid(axis="y",linestyle="--",alpha=0.6)
plt.tight_layout(rect=[0,0.03,1,0.95])
plt.suptitle("Pre-Training Data Analysis",fontsize=16,fontweight="bold")
plt.show()'''

# %% [markdown]
# Feature Engineering

# %%
# Define the features (X) to be used for training. Exclude the original 'Open', 'High', 'Low', 'Close', 'Adj Close'
FEATUERES = ["SMA_5","SMA_10","RSI","MACD","MACD_Signal","Volume_Norm"]
X = data[FEATUERES]
y = data["Target"]

# %%
print(data.shape)

# %% [markdown]
# Data Splitting

# %%
# Split data into training (70%) and testing (30%) sets
# shuffle=False is cruical for time series data to maintain chronological order
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,shuffle=False)

# %% [markdown]
# Data Scaling

# %%
# Initialize the StandardScaler (Standardization)
# Scaling is essential for models sensitive to feature magnitudes, like LogisticRegression and SVM
scaler = StandardScaler()

# Fit the scaler only on the training data to prevent data leakage
X_train_scaled = scaler.fit_transform(X_train)

# Apply the fitted scaler to both training and test data
X_test_scaled = scaler.transform(X_test)

# %%
# Convert the scaled arrays back to DataFrames for easier handling (optional but good practice)
X_train_scaled = pd.DataFrame(X_train_scaled,columns=FEATUERES,index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled,columns=FEATUERES,index=X_test.index)

# %% [markdown]
# Model Comparison

# %%
# Define a dictionary of classifiers to compare
classifiers = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gaussian Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(probability=True, random_state=42)
}

results = {}

print("\n--- Model Training and Comparison ---")
# Loop through each classifier, train it, evaluate performance, and store results
for name, model in classifiers.items():
    print(f"Training {name}...")
    # Train the model using the scaled training data
    if name in ["Logistic Regression", "Support Vector Machine"]:
        model.fit(X_train_scaled, y_train)
        # Make predictions on the scaled test data
        y_pred = model.predict(X_test_scaled)
    else:
        # Tree-based models and Naive Bayes are less sensitive to scaling, but we use scaled data for consistency
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    # Calculate performance metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Store the F1-Score (often better than Accuracy for financial data which may be imbalanced)
    results[name] = {'Accuracy': acc, 'F1-Score': f1}
    print(f"{name} - F1 Score: {f1:.4f}, Accuracy: {acc:.4f}")

# Convert results dictionary to a DataFrame for easy visualization
results_df = pd.DataFrame(results).T.sort_values(by='F1-Score', ascending=False)
print("\nModel Comparison Results:")
print(results_df)

# %% [markdown]
# Visualization After Training

# %%
# Plot the comparison of the models' performance
plt.figure(figsize=(12, 6))
sns.barplot(x=results_df.index, y='F1-Score', data=results_df, palette='viridis')
plt.title('Classification Model F1-Score Comparison', fontsize=14)
plt.ylabel('F1-Score')
plt.xlabel('Model')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# %% [markdown]
# Hyperparameter Tuning on the Best Model

# %%
# Select Random Forest as the target for tuning, as it is robust and often performs well
best_model_name = "Random Forest"
rf_model = RandomForestClassifier(random_state=42)

# Define the parameter grid to search through
param_grid = {
    'n_estimators': [50, 100, 200],         # Number of trees in the forest
    'max_depth': [None, 10, 20],            # Maximum depth of the tree
    'min_samples_split': [2, 5],            # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2]              # Minimum number of samples required to be at a leaf node
}

# Initialize GridSearchCV. Scoring is set to 'f1' since we prioritize balanced performance.
# We use the unscaled data for tree-based models
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='f1', verbose=1, n_jobs=-1)

print(f"\n--- Hyperparameter Tuning using GridSearchCV on {best_model_name} ---")
grid_search.fit(X_train, y_train)

# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print(f"Best parameters found: {best_params}")
print(f"Best F1 Score on training data (CV): {best_score:.4f}")

# Train the final model with the best parameters
tuned_model = grid_search.best_estimator_

# Evaluate the tuned model on the test set
y_pred_tuned = tuned_model.predict(X_test)
y_proba_tuned = tuned_model.predict_proba(X_test)[:, 1]

# Final Metrics
final_f1 = f1_score(y_test, y_pred_tuned)
final_acc = accuracy_score(y_test, y_pred_tuned)
print(f"\n--- Tuned {best_model_name} Final Evaluation ---")
print(f"Test F1 Score: {final_f1:.4f}")
print(f"Test Accuracy: {final_acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_tuned))


# %% [markdown]
# Final Visualization

# %%
# Calculate ROC curve components
fpr, tpr, thresholds = roc_curve(y_test, y_proba_tuned)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(14, 6))

# Subplot 1: ROC Curve
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Recall)')
plt.title(f'Receiver Operating Characteristic (ROC) - Tuned {best_model_name}', fontsize=12)
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.6)

# Subplot 2: Confusion Matrix
cm = confusion_matrix(y_test, y_pred_tuned)
plt.subplot(1, 2, 2)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted Down', 'Predicted Up'],
            yticklabels=['Actual Down', 'Actual Up'])
plt.title(f'Confusion Matrix - Tuned {best_model_name}', fontsize=12)
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.suptitle('Post-Training Model Evaluation', fontsize=16, fontweight='bold')
plt.show()

# %% [markdown]
# Prediction Input for New Data

# %%
# Function to make a prediction for a new day
def predict_new_day(model, scaler, new_data_dict, feature_list):
    """
    Takes a dictionary of new data, scales it, and makes a prediction.
    NOTE: In a real-world scenario, you would need to calculate the 
    technical indicators for the new day based on the past N days of data.
    This function simulates the input features being pre-calculated.
    """
    print("\n--- Simulating New Day Prediction ---")

    # Convert the input dictionary to a DataFrame
    new_df = pd.DataFrame([new_data_dict], columns=feature_list)
    
    # Determine if the model needs scaling (Logistic Regression, SVM)
    if isinstance(model, (LogisticRegression, SVC)):
        # Scale the new data using the fitted scaler (CRITICAL: DO NOT refit)
        new_data_scaled = scaler.transform(new_df)
        prediction_input = new_data_scaled
        print("Data scaled before prediction.")
    else:
        # Use unscaled data for tree-based models
        prediction_input = new_df
        print("Using unscaled data for tree model prediction.")
    
    # Make the prediction (0 or 1)
    prediction = model.predict(prediction_input)[0]
    
    # Try to get the probability if the model supports it
    try:
        probability = model.predict_proba(prediction_input)[0]
        prob_up = probability[1]
        
        result_text = "UP (1)" if prediction == 1 else "DOWN/FLAT (0)"
        
        print(f"Raw Input Features: {new_data_dict}")
        print(f"Predicted Outcome: {result_text}")
        print(f"Probability of Price UP (1): {prob_up*100:.2f}%")
        
    except AttributeError:
        # Some models (like standard SVC without probability=True) don't have predict_proba
        result_text = "UP (1)" if prediction == 1 else "DOWN/FLAT (0)"
        print(f"Predicted Outcome: {result_text}")
        print("Probability estimate not available for this model.")
        
    return prediction

# --- Example of New Prediction Input ---
# These values are based on the expected input features:
# ['SMA_5', 'SMA_10', 'RSI', 'MACD', 'MACD_Signal', 'Volume_Norm']

# Simulate new data for the next day (all features must be provided)
simulated_new_data = {
    'SMA_5': 200.5,           # Current 5-day moving average
    'SMA_10': 199.8,          # Current 10-day moving average
    'RSI': 55.0,              # Current Relative Strength Index (above 50 suggests momentum)
    'MACD': 0.5,              # Current MACD value (positive suggests bullish momentum)
    'MACD_Signal': 0.4,       # Current MACD Signal Line
    'Volume_Norm': 1.1        # Current Volume (10% higher than average)
}

# Run the prediction using the final, tuned Random Forest model
final_prediction = predict_new_day(tuned_model, scaler, simulated_new_data, FEATURES)
# The prediction result is now stored in the final_prediction variable
print(f"The model's final binary prediction for {TICKER} tomorrow is: {final_prediction}")


