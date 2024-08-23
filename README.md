# Target 
This is a project using [LightGBM](https://lightgbm.readthedocs.io/en/latest/index.html) to do the traditinoal machine learning using the [Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality) dataset.

# Model Evaluation Methods

In this project, we use the following metrics to evaluate model performance:

## 1. Mean Absolute Error (MAE)

**Mean Absolute Error (MAE)** measures the average absolute difference between the predicted and true values. It is a common metric for regression tasks.

- **Calculation**:
  $$ \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} | y_{i,\text{true}} - y_{i,\text{pred}} | $$

## 2. Range-MAE

**Range-MAE** calculates the mean absolute error for predictions and true values within a specific range (i.e., [0, 30]). This metric helps evaluate model performance within the desired range.

- **Calculation**:
  $$ \text{Range-MAE} = \frac{1}{n_{\text{range}}} \sum_{i=1}^{n_{\text{range}}} | y_{i,\text{true}} - y_{i,\text{pred}} | $$
  where \( n_{\text{range}} \) is the number of samples within the range [0, 30].

## 3. F1-score

**F1-score** is a metric that combines precision and recall into a single score for binary classification. For this evaluation, we binarize predictions and true values into two classes: within range (<= 30) and out of range (> 30).

- **Calculation**:
  $$ \text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} $$

  Where:
  - **Precision** = \( \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}} \)
  - **Recall** = \( \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}} \)

## Final Score

The final score is a weighted combination of the above metrics. The calculation formula is:

- **Calculation**:
  $$ \text{Score} = 0.5 \cdot \left(1 - \frac{\text{MAE}}{100}\right) + 0.5 \cdot \text{F1} \cdot \left(1 - \frac{\text{Range-MAE}}{100}\right) $$

Where:
- MAE is the Mean Absolute Error, ranging from [0, 100].
- Range-MAE is the Range-MAE, ranging from [0, 100].
- F1 is the F1-score, ranging from [0, 1].

The final score ranges from [0, 1], taking into account the reduction in MAE and the increase in F1-score.

