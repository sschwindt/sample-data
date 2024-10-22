import pandas as pd

# Step 1: Load the CSV file with Pandas and check the data consistency
file = 'XS-morphology.csv'
data = pd.read_csv(file)

# Display the first few rows of the dataset to understand its structure
data.head()

# Check on missing values and duplicated rows
print("Sum of missing values: " + str(data.isnull().sum()))
print("Sum of duplicated rows: " + str(data.duplicated().sum()))

# Check on global descriptive statistics
data.describe()

# Analyze histograms
for column in df.columns:
    data[column].plot(kind='hist')

# Detect outliers with the InterQuartile Range IQR: values below Q1 - 1.5 * IQR and above Q3 + 1.5 * IQR are potential outliers.
for column in df.columns:
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    outliers = data[(data[column] < Q1 - 1.5 * IQR) | (data[column] > Q3 + 1.5 * IQR)]
    print("Outliers in column" + str(column) + ": " + str(outliers))



# Step 2A: Use numpy and scipy to test for normality of each numeric column using the Shapiro-Wilk test
import scipy.stats as stats
import numpy as np

# Extract numeric columns
numeric_columns = data.select_dtypes(include=[np.number])
# alternative:
# numeric_data = data.select_dtypes(include=[float, int])

# Apply Shapiro-Wilk test for normality
normality_results = {}
for column in numeric_columns:
    stat, p_value = stats.shapiro(numeric_columns[column])
    normality_results[column] = {"Statistic": stat, "p-value": p_value, "Normality": p_value > 0.05}

# Display normality test results
normality_results

# Note: If the data is skewed (especially right-skewed), a log transformation can help normalize it:
data['log_Q'] = np.log1p(data["Q"])

# Step 2B: Use QQ plots
import matplotlib.pyplot as plt

# Create a function to create QQ plots for each column
def create_qq_plots(df):
    for column in df.columns:
        # Drop missing values
        column_data = df[column].dropna()
        
        # Generate QQ plot
        plt.figure()
        stats.probplot(column_data, dist="norm", plot=plt)
        plt.title(f"QQ Plot for {column}")
        plt.xlabel("Theoretical Quantiles")
        plt.ylabel("Sample Quantiles")
        plt.show()

# Create QQ plots for numeric columns
create_qq_plots(numeric_columns)


# Step 3 - OPTION A: Investigate the best fitting statistical distribution for each column with the fit method 
# and calculate the goodness-of-fit using the Kolmogorov-Smirnov test (for continuous distributions).

distributions = ['norm', 'lognorm', 'expon', 'gamma', 'weibull_min']
best_fit_results = {}

for column in numeric_columns:
    column_data = numeric_columns[column].dropna()  # Removing NaNs
    best_fit = {'distribution': None, 'p_value': 0}
    
    for dist_name in distributions:
        dist = getattr(stats, dist_name)
        params = dist.fit(column_data)
        
        # Perform the Kolmogorov-Smirnov test
        ks_stat, p_value = stats.kstest(column_data, dist_name, args=params)
        
        # Keep track of the distribution with the highest p-value
        if p_value > best_fit['p_value']:
            best_fit = {'distribution': dist_name, 'p_value': p_value, 'params': params}
    
    best_fit_results[column] = best_fit


# Display best fitting distribution results
best_fit_results

# Step 3 - OPTION B: Investigate the best fitting statistical distribution for each column with the fitter 
# package.

from fitter import Fitter


# Loop over each column and find the best fitting distribution
for column in numeric_columns:
    print(f"Analyzing best fit for column: {column}")
    
    # Drop missing values
    column_data = numeric_columns[column].dropna()

    # Fit distributions using Fitter
    f = Fitter(column_data, distributions=['norm', 'lognorm', 'gamma', 'weibull_min', 'expon'])
    f.fit()

    # Print the summary of the best fit
    print(f.summary())

    # To get the best distribution directly:
    print(f"Best fitting distribution for {column}: {f.get_best(method='sumsquare_error')}")


# Step 4: Check on correlation and multicollinearity
# As found in Step 2, none of the columns are normally distributed, so we'll calculate Spearman rank correlations for all numeric columns.

# Calculate Spearman correlation
spearman_correlation = numeric_columns.corr(method='spearman')
import ace_tools as tools; tools.display_dataframe_to_user(name="Spearman Rank Correlation", dataframe=spearman_correlation)

# Variance Inflation Factor VIF checks for multicollinearity among features, which can distort model performance
from statsmodels.stats.outliers_influence import variance_inflation_factor
VIF = pd.DataFrame()
VIF['VIF Factor'] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]


# Step 5: Run Multivariate analysis
# A: Create pairplots:

import seaborn as sns

# Plot pairplot for all numeric columns
sns.pairplot(numeric_columns)
plt.show()

# B: Run a PCA
from sklearn.decomposition import PCA

# Standardize the data before PCA (important!)
# Center the data to have a mean of zero and a variance of 1
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)

# Apply PCA
pca = PCA(n_components=2)  # Reduce to 2 components for visualization
principal_components = pca.fit_transform(scaled_data)

# Create a DataFrame with the PCA results
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Plot the PCA results
plt.figure(figsize=(8, 6))
plt.scatter(pca_df['PC1'], pca_df['PC2'])
plt.title('PCA of Sample Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()

# Explained variance by each component
print(f"Explained variance ratio by each component: {pca.explained_variance_ratio_}")

# Step 6: SVM
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Prepare the data for SVM analysis
X = data[['w', 'S0', 'Q', 'U', 'h']]  # Feature columns
y = data['Morphology']  # Target column (Morphology)

# Convert the target variable to numerical values (SVM needs numeric targets)
y_encoded = pd.factorize(y)[0]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Standardize the feature data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = svm_model.predict(X_test_scaled)

# Get classification report to check performance
classification_rep = classification_report(y_test, y_pred)

# Get the coefficients to identify which features are most important
svm_coefficients = svm_model.coef_

classification_rep, svm_coefficients


