# Visa Stock
![Visa-Logo](/visa_image.jpg)

## Procedures
- Import Libraries
    - Scikit-learn
    - pandas
    - numpy
    - seaborn
    - matplotlib
    - yfinance
- Data Acquisition
    - Data aquired from the Yahoo Finance api (yfinance)
- Data Preprocessing
    - Check for missing values
    - Check for duplicated rows
- Feature Engineering
- Pre-Training Visualization
- Feature Engineering
- Data Splitting
    - Split data into training (70%) and testing sets (30%)
    - shuffle=False is cruical for time series data to maintain the chronological order
- Data Scaling
    - Initializae the StandardScaler
    - Scaling is essential for models sensitive to feature magnitudes
- Model Comparison
    - Logistic Regression
    - K-Nearest Neighbors
    - Decison Tree
    - Random Forest
    - Gaussian Naive Bayes
    - Support Vector Machine
- Post-Training Visualization
- Hyperparameter Tuning
    - n_estimators
    - max_depth
    - min_samples_split
    - min_samples_leaf
- Final Visualization
- Prediction Input for New Data

## Tech Stack and Tools
- Programming language
    - Python 
- libraries
    - scikit-learn
    - pandas
    - numpy
    - seaborn
    - matplotlib
- Environment
    - Jupyter Notebook
    - Anaconda
    - Google Colab
- IDE
    - VSCode

You can install all dependencies via:
```
pip install -r requirements.txt
```

## Usage Instructions
To run this project locally:
1. Clone the repository:
```
git clone https://github.com/charlesakinnurun/visa-stock.git
cd visa-stock
```
2. Install required packages
```
pip install -r requirements.txt
```
3. Open the notebook:
```
jupyter notebook model.ipynb

```

## Project Structure
```
visa-stock/
│
├── model.ipynb  
|── model.py    
|── visa_stock_data.csv  
├── requirements.txt 
├── visa_logo.jpg 
├── visa_image.jpg      
├── output1.png        
├── output2.png        
├── SECURITY.md        
├── CONTRIBUTING.md    
├── CODE_OF_CONDUCT.md 
├── LICENSE
└── README.md          

```