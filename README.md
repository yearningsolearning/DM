# SECOM Predictive Maintenance System

## Overview

This project implements a machine learning pipeline for semiconductor manufacturing failure prediction using the SECOM (SEmiCONductor Manufacturing) dataset. The system combines ADASYN (Adaptive Synthetic Sampling) technique with Balanced Random Forest classifier to handle class imbalance and predict manufacturing failures.

## Features

- **Advanced Preprocessing Pipeline**: Comprehensive data cleaning and feature engineering
- **Imbalanced Data Handling**: ADASYN sampling technique for minority class augmentation
- **Robust Model Training**: Balanced Random Forest classifier optimized for imbalanced datasets
- **Threshold Optimization**: Automated threshold tuning for optimal F1-score performance
- **Comprehensive Evaluation**: Multiple metrics including precision, recall, F1-score, and ROC-AUC
- **Visualization**: Confusion matrix and performance metrics visualization

## Dataset

The SECOM dataset contains:
- **Features**: 590 sensor measurements from semiconductor manufacturing process
- **Target**: Binary classification (Pass/Fail) for manufacturing quality
- **Characteristics**: Highly imbalanced dataset with significant class imbalance
- **Source**: UCI Machine Learning Repository

## Prerequisites

### Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

### Specific Dependencies

- `pandas >= 1.3.0`
- `numpy >= 1.21.0`
- `matplotlib >= 3.4.0`
- `seaborn >= 0.11.0`
- `scikit-learn >= 1.0.0`
- `imbalanced-learn >= 0.8.0`

## Installation

1. Clone or download the repository
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure SECOM dataset files are available:
   - `secom.data` (sensor measurements)
   - `secom_labels.data` (failure labels)

## Dataset Setup

Update the file paths in the `load_data()` method to match your dataset location:

```python
data = pd.read_csv(r'path/to/your/secom.data', delim_whitespace=True, header=None)
labels = pd.read_csv(r'path/to/your/secom_labels.data', delim_whitespace=True, header=None)
```

## Usage

### Basic Usage

```python
from secom_analyzer import SECOM_ADASYN_BalancedRF

# Initialize the analyzer
analyzer = SECOM_ADASYN_BalancedRF()

# Run complete analysis
success = analyzer.run_complete_analysis()

if success:
    print("Analysis completed successfully!")
    print(f"F1-Score: {analyzer.results['f1_score']:.4f}")
    print(f"ROC-AUC: {analyzer.results['roc_auc']:.4f}")
```

### Step-by-Step Execution

```python
# Initialize
analyzer = SECOM_ADASYN_BalancedRF()

# Load data
analyzer.load_data()

# Preprocess data
analyzer.advanced_preprocessing()

# Split data
analyzer.split_data()

# Train model with ADASYN
analyzer.apply_adasyn_and_train()

# Evaluate model
results = analyzer.evaluate_model()

# Generate reports and visualizations
analyzer.show_classification_report()
analyzer.plot_confusion_matrix()
```

## Architecture

### Class Structure

```
SECOM_ADASYN_BalancedRF
├── Data Loading & Preprocessing
│   ├── load_data()
│   ├── advanced_preprocessing()
│   └── split_data()
├── Model Training
│   ├── apply_adasyn_and_train()
│   └── optimize_threshold()
├── Evaluation & Visualization
│   ├── evaluate_model()
│   ├── plot_confusion_matrix()
│   └── show_classification_report()
└── Main Pipeline
    └── run_complete_analysis()
```

### Preprocessing Pipeline

1. **Data Loading**: Import sensor data and failure labels
2. **KNN Imputation**: Handle missing values using 5-nearest neighbors
3. **Constant Feature Removal**: Remove features with ≤1 unique values or std < 1e-6
4. **Correlation Analysis**: Remove highly correlated features (r > 0.9)
5. **Robust Scaling**: Scale features using median and IQR
6. **Feature Selection**: RFE with Random Forest to select top 200 features

### Model Training Pipeline

1. **Data Splitting**: Stratified 80/20 train-test split
2. **ADASYN Sampling**: Generate synthetic samples for minority class
3. **Balanced Random Forest**: Train with 200 estimators, max_depth=15
4. **Threshold Optimization**: Find optimal classification threshold for F1-score

## Output

The system provides comprehensive analysis output:

### Performance Metrics
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC score
- Optimal classification threshold
- Class-wise performance metrics

### Visualizations
- Confusion Matrix with performance annotations
- Classification Report with detailed metrics

### Example Output
```
FINAL SUMMARY:
Sampling Method: ADASYN
Model Name: Balanced Random Forest
Optimal Threshold: 0.423
Accuracy: 0.9456
Precision: 0.8234
Recall: 0.7891
F1-Score: 0.8058
ROC-AUC: 0.9123
```

## Configuration

### Model Parameters

```python
# ADASYN Parameters
adasyn = ADASYN(random_state=42)

# Balanced Random Forest Parameters
model = BalancedRandomForestClassifier(
    n_estimators=200,      # Number of trees
    max_depth=15,          # Maximum tree depth
    random_state=42        # Reproducibility seed
)

# RFE Parameters
n_features_to_select=min(200, total_features)  # Maximum features to select
```

### Threshold Optimization

```python
# Threshold range for optimization
thresholds = np.arange(0.1, 0.9, 0.01)
# Optimization metric: F1-score
metric = 'f1'
```

## Troubleshooting

### Common Issues

1. **File Not Found Error**
   - Verify dataset file paths in `load_data()` method
   - Ensure files exist and are accessible

2. **Memory Issues**
   - Reduce `n_features_to_select` in RFE
   - Consider using smaller `n_estimators` for Random Forest

3. **ADASYN Sampling Failure**
   - Check class distribution in training data
   - Ensure sufficient minority class samples exist

4. **Import Errors**
   - Verify all required libraries are installed
   - Check library versions compatibility

### Performance Optimization

- **Faster Training**: Reduce `n_estimators` or `max_depth`
- **Memory Optimization**: Limit feature selection to fewer features
- **Reproducibility**: Maintain consistent `random_state` values

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open-source and available under the MIT License.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{secom_predictive_maintenance,
  title={SECOM Predictive Maintenance System},
  author={Your Name},
  year={2024},
  description={Machine Learning Pipeline for Semiconductor Manufacturing Failure Prediction}
}
```

## Contact

For questions, issues, or contributions, please contact [sanskriti.khatiwada002@gmail.com],[Agrimaremi2004@gmail.com]

---

**Note**: This implementation is designed for research and educational purposes. For production use, additional validation and testing are recommended.
