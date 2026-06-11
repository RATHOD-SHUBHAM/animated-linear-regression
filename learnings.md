Standardize the data¶
Standardization is a preprocessing technique used in machine learning to rescale and transform the features (variables) of a dataset to have a mean of 0 and a standard deviation of 1. It is also known as "z-score normalization" or "z-score scaling." Standardization is an essential step in the data preprocessing pipeline for various reasons:

Why Use Standardization in Machine Learning?
Mean Centering: Standardization centers the data by subtracting the mean from each feature. This ensures that the transformed data has a mean of 0. Mean centering is crucial because it helps in capturing the relative variations in the data.

Scale Invariance: Standardization scales the data by dividing each feature by its standard deviation. This makes the data scale-invariant, meaning that the scale of the features no longer affects the performance of many machine learning algorithms. Without standardization, features with larger scales may dominate the learning process.

Improved Convergence: Many machine learning algorithms, such as gradient-based optimization algorithms (e.g., gradient descent), converge faster when the features are standardized. It reduces the potential for numerical instability and overflow/underflow issues during training.

Comparability: Standardizing the features makes it easier to compare and interpret the importance of each feature. This is especially important in models like linear regression, where the coefficients represent the feature's impact on the target variable.

Regularization: In regularization techniques like Ridge and Lasso regression, the regularization strength is applied uniformly to all features. Standardization ensures that the regularization term applies fairly to all features.

How to Standardize Data
The standardization process involves the following steps:

Calculate the mean (μ) and standard deviation (σ) for each feature in the dataset.
For each data point (sample), subtract the mean (μ) of the feature and then divide by the standard deviation (σ) of the feature.
Mathematically, the standardized value for a feature x in a dataset is calculated as:

Standardized value = (x - μ) / σ
Here, x is the original value of the feature, 
μ is the mean of the feature, and 
σ is the standard deviation of the feature.