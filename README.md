# SECOM Multi-Model Predictive Maintenance System

## Overview

This project implements a comprehensive machine learning pipeline for semiconductor manufacturing failure prediction using the SECOM (SEmiCONductor Manufacturing) dataset. The system combines ADASYN (Adaptive Synthetic Sampling) technique with multiple advanced classifiers including Balanced Random Forest and XGBoost to handle class imbalance and predict manufacturing failures with optimal performance.

## Features

- **Advanced Preprocessing Pipeline**: Comprehensive data cleaning and feature engineering
- **Multi-Model Architecture**: Supports Balanced Random Forest and XGBoost classifiers
- **Imbalanced Data Handling**: ADASYN sampling technique for minority class augmentation
- **Robust Model Training**: Multiple optimized classifiers for imbalanced datasets
- **Threshold Optimization**: Automated threshold tuning for optimal F1-score performance
- **Comprehensive Evaluation**: Multiple metrics including precision, recall, F1-score, and ROC-AUC
- **Advanced Visualization**: Model comparison charts, confusion matrices, and performance metrics
- **Feature Importance Analysis**: Detailed feature importance analysis for tree-based models

## Dataset

The SECOM dataset contains:
- **Features**: 590 sensor measurements from semiconductor manufacturing process
- **Target**: Binary classification (Pass/Fail) for manufacturing quality
- **Characteristics**: Highly imbalanced dataset with significant class imbalance
- **Source**: UCI Machine Learning Repository

## Prerequisites

### Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost
```

### Specific Dependencies

- `pandas >= 1.3.0`
- `numpy >= 1.21.0`
- `matplotlib >= 3.4.0`
- `seaborn >= 0.11.0`
- `scikit-learn >= 1.0.0`
- `imbalanced-learn >= 0.8.0`
- `xgboost >= 1.5.0`

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
from secom_multi_model_analyzer import SECOM_MultiModel_Analysis

# Initialize the analyzer
analyzer = SECOM_MultiModel_Analysis()

# Run complete analysis with all models
success = analyzer.run_complete_analysis()

if success:
    print("Multi-model analysis completed successfully!")
    # Access results for all models
    for model_key, result in analyzer.results.items():
        print(f"{result['model_name']} - F1-Score: {result['f1_score']:.4f}")
```

### Step-by-Step Execution

```python
# Initialize
analyzer = SECOM_MultiModel_Analysis()

# Load and preprocess data
analyzer.load_data()
analyzer.advanced_preprocessing()
analyzer.split_data()

# Apply ADASYN sampling
analyzer.apply_adasyn_sampling()

# Train individual models
analyzer.train_balanced_random_forest()
analyzer.train_xgboost_model()

# Evaluate all models
for model_key in analyzer.models.keys():
    analyzer.evaluate_model(model_key)

# Generate comprehensive reports and visualizations
analyzer.plot_model_comparison()
analyzer.plot_confusion_matrices()
analyzer.show_feature_importance()
analyzer.show_classification_report()
```

### Individual Model Training

```python
# Train only Balanced Random Forest
analyzer.train_balanced_random_forest()
analyzer.evaluate_model('Balanced_RF')

# Train only XGBoost
analyzer.train_xgboost_model()
analyzer.evaluate_model('XGBoost')
```

## Architecture

### Class Structure

```
SECOM_MultiModel_Analysis
├── Data Loading & Preprocessing
│   ├── load_data()
│   ├── advanced_preprocessing()
│   ├── split_data()
│   └── apply_adasyn_sampling()
├── Model Training
│   ├── train_balanced_random_forest()
│   ├── train_xgboost_model()
│   └── optimize_threshold()
├── Evaluation & Visualization
│   ├── evaluate_model()
│   ├── plot_model_comparison()
│   ├── plot_confusion_matrices()
│   ├── show_feature_importance()
│   └── show_classification_report()
└── Main Pipeline
    └── run_complete_analysis()
```

### Preprocessing Pipeline

1. **Data Loading**: Import sensor data and failure labels with automatic label mapping
2. **KNN Imputation**: Handle missing values using 5-nearest neighbors
3. **Constant Feature Removal**: Remove features with ≤1 unique values or std < 1e-6
4. **Correlation Analysis**: Remove highly correlated features (r > 0.9)
5. **Robust Scaling**: Scale features using median and IQR for outlier resistance
6. **Feature Selection**: RFE with Random Forest to select top 200 features

### Model Training Pipeline

1. **Data Splitting**: Stratified 80/20 train-test split
2. **ADASYN Sampling**: Generate synthetic samples for minority class
3. **Multi-Model Training**:
   - **Balanced Random Forest**: 200 estimators, max_depth=15
   - **XGBoost**: 200 estimators with scale_pos_weight for imbalance handling
4. **Threshold Optimization**: Find optimal classification threshold for each model

## Supported Models

### Balanced Random Forest
- **Purpose**: Handles class imbalance through balanced bootstrap sampling
- **Parameters**: 200 estimators, max_depth=15, built-in class balancing
- **Advantages**: Robust to overfitting, handles imbalanced data natively

### XGBoost Classifier
- **Purpose**: Gradient boosting with advanced regularization
- **Parameters**: 200 estimators, max_depth=6, scale_pos_weight for imbalance
- **Advantages**: High performance, built-in regularization, handles missing values

## Results Summary

Based on comprehensive testing with the SECOM dataset, the analysis reveals:

### Performance Comparison
| Model | F1-Score | Precision | Recall | ROC-AUC | Accuracy |
|-------|----------|-----------|--------|---------|----------|
| **Balanced Random Forest** | **0.400** | **0.324** | **0.524** | **0.794** | **0.895** |
| XGBoost | 0.281 | 0.222 | 0.381 | 0.721 | 0.869 |

### Key Findings
- **Balanced Random Forest is the clear winner** across all performance metrics
- **42% better F1-Score** demonstrates superior balance between precision and recall
- **37% higher recall** is crucial for manufacturing failure detection
- **ROC-AUC of 0.794** indicates strong discriminative ability for imbalanced data
- Both models benefit significantly from ADASYN sampling for handling class imbalance

## Output

The system provides comprehensive multi-model analysis output:

### Performance Metrics
- Accuracy, Precision, Recall, F1-Score for each model
- ROC-AUC score comparison
- Optimal classification threshold for each model
- Class-wise performance metrics

### Visualizations
- **Model Comparison Chart**: Side-by-side metric comparison
- **Confusion Matrices**: Individual confusion matrices for each model
- **Feature Importance**: Top 10 important features for tree-based models

### Sample Output
```
FINAL SUMMARY - ALL MODELS
================================================================================
Best F1-Score: Balanced Random Forest (0.4000)
Best ROC-AUC: Balanced Random Forest (0.7940)

Balanced Random Forest (ADASYN):
  F1: 0.4000 | Precision: 0.3240 | Recall: 0.5240
  Accuracy: 0.8950 | ROC-AUC: 0.7940 | Threshold: 0.320

XGBoost (ADASYN):
  F1: 0.2810 | Precision: 0.2220 | Recall: 0.3810
  Accuracy: 0.8690 | ROC-AUC: 0.7210 | Threshold: 0.280
```

## Configuration

### Model Parameters

#### Balanced Random Forest
```python
model = BalancedRandomForestClassifier(
    n_estimators=200,      # Number of trees
    max_depth=15,          # Maximum tree depth
    random_state=42,       # Reproducibility seed
    n_jobs=-1             # Use all available cores
)
```

#### XGBoost
```python
model = xgb.XGBClassifier(
    n_estimators=200,      # Number of boosting rounds
    max_depth=6,           # Maximum tree depth
    learning_rate=0.1,     # Step size shrinkage
    subsample=0.8,         # Subsample ratio of training instances
    colsample_bytree=0.8,  # Subsample ratio of columns
    scale_pos_weight=ratio, # Balance class weights
    random_state=42,       # Reproducibility seed
    eval_metric='logloss', # Evaluation metric
    n_jobs=-1             # Use all available cores
)
```

### ADASYN Parameters
```python
adasyn = ADASYN(random_state=42)  # Adaptive synthetic sampling
```

### Threshold Optimization
```python
# Threshold range for optimization
thresholds = np.arange(0.1, 0.9, 0.01)
# Optimization metric: F1-score
metric = 'f1'
```

## Advanced Features

### Model Comparison
The system automatically compares all trained models across multiple metrics:
- F1-Score, Precision, Recall
- ROC-AUC, Accuracy
- Visual bar charts for easy comparison

### Feature Importance Analysis
For tree-based models, the system provides:
- Top 10 most important features
- Feature importance scores
- Model-specific importance rankings

### Threshold Optimization
Each model undergoes individual threshold optimization:
- Grid search across threshold values (0.1 to 0.9)
- F1-score maximization
- Model-specific optimal thresholds

## Troubleshooting

### Common Issues

1. **File Not Found Error**
   - Verify dataset file paths in `load_data()` method
   - Ensure files exist and are accessible

2. **Memory Issues**
   - Reduce `n_features_to_select` in RFE (currently 200)
   - Consider using smaller `n_estimators` for models
   - Use `n_jobs=1` instead of `n_jobs=-1` if memory is limited

3. **ADASYN Sampling Failure**
   - Check class distribution in training data
   - Ensure sufficient minority class samples exist
   - The system automatically falls back to original data if ADASYN fails

4. **XGBoost Installation Issues**
   - Ensure XGBoost is properly installed: `pip install xgboost`
   - For conda users: `conda install -c conda-forge xgboost`

5. **Import Errors**
   - Verify all required libraries are installed
   - Check library versions compatibility
   - Install missing dependencies: `pip install -r requirements.txt`

### Performance Optimization

- **Faster Training**: Reduce `n_estimators` for both models
- **Memory Optimization**: Limit feature selection to fewer features
- **Parallel Processing**: Adjust `n_jobs` parameter based on available cores
- **Reproducibility**: Maintain consistent `random_state` values across all components

## Model Selection Guidelines

### Balanced Random Forest (Recommended)
**Based on actual results, this is the superior model for SECOM data:**
- **Higher F1-Score**: 0.400 vs 0.281 (42% better)
- **Better Precision**: 0.324 vs 0.222 (46% better)  
- **Superior Recall**: 0.524 vs 0.381 (37% better)
- **Better ROC-AUC**: 0.794 vs 0.721 (10% better)
- **Higher Accuracy**: 0.895 vs 0.869 (3% better)

**Use when:**
- Failure detection is critical (high recall importance)
- You need robust performance with minimal tuning
- Interpretability and feature importance are valued
- Working with imbalanced manufacturing data

### XGBoost
**Shows lower performance on SECOM dataset but may be useful for:**
- Different datasets with different characteristics
- When extensive hyperparameter tuning is possible
- Baseline comparison purposes
- Research and experimentation

**Key Insight**: For semiconductor manufacturing failure prediction, Balanced Random Forest consistently outperforms XGBoost across all metrics, making it the recommended choice for production systems.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-model`)
3. Add your new model to the training pipeline
4. Update the evaluation and visualization methods
5. Add tests if applicable
6. Submit a pull request

### Adding New Models

To add a new model to the pipeline:

1. Create a new training method (e.g., `train_new_model()`)
2. Add model to `self.models` dictionary with appropriate metadata
3. Ensure model supports `predict()` and `predict_proba()` methods
4. Update visualization methods if needed

## License

This project is open-source and available under the MIT License.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{,
  title={SECOM Multi-Model Predictive Maintenance System},
  author={Sanskriti khatiwada,Agrima Regmi},
  year={2025},
  description={Multi-Model Machine Learning Pipeline for Semiconductor Manufacturing Failure Prediction},
  url={https://github.com/yearningsolearning/DM}
}
```

## Contact

For questions, issues, or contributions, please contact:
- [sanskriti.khatiwada002@gmail.com]
- [Agrimaremi2004@gmail.com]

## Changelog

### Version 2.0
- Added XGBoost classifier support
- Implemented multi-model comparison framework
- Enhanced visualization with model comparison charts
- Added feature importance analysis for all tree-based models
- Improved threshold optimization for individual models
- Enhanced error handling and fallback mechanisms

### Version 1.0
- Initial release with Balanced Random Forest
- Basic ADASYN sampling implementation
- Standard evaluation metrics and visualization

---

**Note**: This implementation is designed for research and educational purposes. For production use, additional validation, hyperparameter tuning, and testing are recommended. Consider implementing cross-validation and more sophisticated model selection techniques for critical applications.
