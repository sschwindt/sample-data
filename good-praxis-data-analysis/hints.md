## Interpret QQ Plots

To interpret QQ (Quantile-Quantile) plots, compare the distribution of your data to a theoretical normal distribution:

1. **Perfect Fit (Normal Distribution)**:  
   - If the data is normally distributed, the points in the QQ plot will lie approximately along a straight diagonal line.
   - A closer alignment of the points to the line indicates a stronger adherence to normality.

2. **Deviations from Normality**:  
   - **Upward or Downward Curving at the Ends (Tails)**:  
     If the points curve away from the line at the ends, it suggests that the distribution has heavier or lighter tails than a normal distribution. This is common in skewed distributions.
     - **Upward curve**: The data has heavy tails (possibly a leptokurtic or fat-tailed distribution).
     - **Downward curve**: The data has light tails (possibly a platykurtic distribution).
   
   - **S-Shape (Middle Deviation)**:  
     If the points form an S-shape, with a bulge near the center, it suggests that the distribution has more central data (a peak) than a normal distribution, which may indicate a bimodal or non-normal distribution.

3. **Outliers**:  
   - Points that fall far from the line at either end (outliers) may indicate anomalies or extreme values in your data.

### Example Interpretations:
- **Linearity**: If the points in a QQ plot for a column mostly follow the diagonal line, the data for that column is approximately normal.
- **Curved Tail Ends**: If the points curve away from the line at either end, the data likely exhibits skewness or non-normality.
- **Systematic Patterns**: If the points show a distinct pattern (such as S-shape or curvature), it signals that the data does not follow a normal distribution and might fit better to another distribution, such as lognormal or gamma.

In the case of the morphology dataset, none of the columns follow a normal distribution, as the points in the QQ plots deviate from the straight line, confirming the results of the normality tests. These visualizations can be used in conjunction with statistical tests like the Shapiro-Wilk test to conclude whether the normality assumption holds for your dataset.

## SVM

### Notes on Factorization

Factorization here refers to converting a categorical "Morphology" variable into a numeric form that can be used by machine learning algorithms, which typically require numeric inputs. Factorization also helps to treat **No Ordinality**.  By assigning numeric values to categories without implying any order or ranking, factorization is different from ordinal encoding, where the categories have a specific order (e.g., low, medium, high).

In Python, the `pd.factorize()` function from the `pandas` library is used to accomplish this. It assigns a unique integer to each distinct category (class) in the categorical column. Therefore, consider the column `Morphology`, which may look like this:

| Index | Morphology   |
|-------|--------------|
| 0     | Riffle-pool  |
| 1     | Step-pool    |
| 2     | Cascade      |
| 3     | Riffle-pool  |
| 4     | Step-pool    |

Applying `pd.factorize()` to this column assigns a unique integer to each distinct value (category), like so:

```python
import pandas as pd

# Example data
morphology = pd.Series(['Riffle-pool', 'Step-pool', 'Cascade', 'Riffle-pool', 'Step-pool'])

# Factorize the data
morphology_encoded, unique_labels = pd.factorize(morphology)

# Output the encoded values and the unique labels
print(morphology_encoded)  # Output: array([0, 1, 2, 0, 1])
print(unique_labels)        # Output: Index(['Riffle-pool', 'Step-pool', 'Cascade'], dtype='object')
```

The Factorization steps involve:
1. Create **Encoded Values**: 
   - Factorization converts each unique category in the `Morphology` column into a corresponding integer.
   - In this case, `'Riffle-pool'` is encoded as `0`, `'Step-pool'` as `1`, and `'Cascade'` as `2`.
2. **Unique Labels**:
   - The second output of `pd.factorize()` is the array of unique labels (the original category names). This array lets us map the encoded integers back to their original categories if needed.


### Analyze Results

1. **Classification Performance:**
   The SVM model's overall accuracy is **56%**, and the detailed classification performance (precision, recall, and f1-score) for each class is shown in the classification report. This suggests that the SVM model has moderate performance in predicting the "Morphology" classes based on the input features (`w`, `S0`, `Q`, `U`, `h`).

2. **Feature Importance (SVM Coefficients):**
   The coefficients of the linear SVM model indicate the contribution of each feature in distinguishing between the different classes of "Morphology". Here's a brief interpretation:

   - **Negative coefficients** imply that higher values of the feature decrease the likelihood of certain morphology classes.
   - **Positive coefficients** suggest that higher values of the feature increase the likelihood of certain morphology classes.

### Feature Importance Summary
- The feature **`S0` (slope)** seems to have significant contributions across multiple morphology classes, as seen from the strong positive and negative coefficients.
- **`Q` (flow rate)** also plays an important role, with some substantial coefficients.
- **`w` (width)**, **`U` (velocity)**, and **`h` (depth)** also influence the predictions, but their contributions are generally smaller compared to `S0` and `Q`.

These results indicate that the **slope** and **flow rate** are likely the most influential features in determining the morphology, based on the SVM analysis.

