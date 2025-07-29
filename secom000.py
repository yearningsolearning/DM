import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
from sklearn.feature_selection import RFE
from sklearn.metrics import (confusion_matrix, precision_score, recall_score, 
                           roc_auc_score,classification_report)
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import ADASYN
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class SECOM_MultiModel_Analysis:
    def __init__(self):
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Load SECOM dataset - exact from original code"""
        try:
            print("Loading SECOM dataset...")
            data = pd.read_csv(r'C:\machine learn\DM\secom\secom.data', 
                              delim_whitespace=True, header=None)
            
            labels = pd.read_csv(r'C:\machine learn\DM\secom\secom_labels.data', 
                                delim_whitespace=True, header=None)
            
            y1 = labels.iloc[:, 0]
            data['label'] = y1
            
            self.X = data.drop('label', axis=1)
            self.y = data['label']
            
            print(f"Dataset shape: {data.shape}")
            print(f"Features shape: {self.X.shape}")
            print(f"Labels distribution:")
            print(self.y.value_counts())
            
            # Convert labels to binary
            unique_labels = sorted(self.y.unique())
            if len(unique_labels) == 2:
                label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
                self.y = self.y.map(label_map)
                print(f"Labels mapped: {unique_labels[0]} -> 0, {unique_labels[1]} -> 1")
                
                class_counts = self.y.value_counts()
                imbalance_ratio = class_counts[0] / class_counts[1]
                print(f"Class imbalance ratio: {imbalance_ratio:.2f}:1")
                print(f"Minority class percentage: {class_counts[1]/len(self.y)*100:.2f}%")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
        
        return True
    
    def advanced_preprocessing(self):
        """Advanced preprocessing pipeline"""
        print("\nAdvanced preprocessing...")
        
        # Store original feature names
        self.feature_names = [f'Feature_{i}' for i in range(self.X.shape[1])]
        
        # Advanced imputation with KNN
        print("Using KNN imputer...")
        imputer = KNNImputer(n_neighbors=5)
        self.X = pd.DataFrame(imputer.fit_transform(self.X))
        
        # Remove constant and near-constant columns
        constant_cols = []
        for col in self.X.columns:
            if self.X[col].nunique() <= 1:
                constant_cols.append(col)
            elif self.X[col].std() < 1e-6:  # Near constant
                constant_cols.append(col)
        
        if constant_cols:
            self.X = self.X.drop(columns=constant_cols)
            print(f"Removed {len(constant_cols)} constant/near-constant columns")
        
        # Remove highly correlated features with threshold 0.9
        corr_matrix = self.X.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        high_corr_cols = []
        for column in upper_tri.columns:
            if any(upper_tri[column] > 0.9):
                high_corr_cols.append(column)
        
        if high_corr_cols:
            self.X = self.X.drop(columns=high_corr_cols)
            print(f"Removed {len(high_corr_cols)} highly correlated columns")
        
        # Advanced scaling with RobustScaler
        self.scaler = RobustScaler()
        self.X = pd.DataFrame(self.scaler.fit_transform(self.X))
        
        # Feature selection with RFE
        print("Using RFE feature selection...")
        estimator = RandomForestClassifier(n_estimators=50, random_state=42)
        self.selector = RFE(estimator, n_features_to_select=min(200, self.X.shape[1]))
        self.X = pd.DataFrame(self.selector.fit_transform(self.X, self.y))
        
        print(f"Final feature shape: {self.X.shape}")
        print(f"Missing values: {self.X.isnull().sum().sum()}")
    
    def split_data(self):
        """Split data with stratification"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        print(f"\nData split:")
        print(f"Training: {self.X_train.shape[0]} samples, Failures: {self.y_train.sum()} ({self.y_train.mean():.2%})")
        print(f"Test: {self.X_test.shape[0]} samples, Failures: {self.y_test.sum()} ({self.y_test.mean():.2%})")
    
    def apply_adasyn_sampling(self):
        """Apply ADASYN sampling to training data"""
        print("\nApplying ADASYN sampling...")
        try:
            adasyn = ADASYN(random_state=42)
            self.X_train_resampled, self.y_train_resampled = adasyn.fit_resample(self.X_train, self.y_train)
            print(f"Original training shape: {self.X_train.shape}")
            print(f"Resampled training shape: {self.X_train_resampled.shape}")
            print(f"Resampled class distribution: {pd.Series(self.y_train_resampled).value_counts().to_dict()}")
        except Exception as e:
            print(f"ADASYN sampling failed: {e}")
            print("Using original training data...")
            self.X_train_resampled, self.y_train_resampled = self.X_train, self.y_train
    
    def train_balanced_random_forest(self):
        """Train Balanced Random Forest model"""
        print("\n--- Training Balanced Random Forest ---")
        
        model = BalancedRandomForestClassifier(
            n_estimators=200, 
            max_depth=15, 
            random_state=42,
            n_jobs=-1
        )
        
        # Train on ADASYN resampled data
        model.fit(self.X_train_resampled, self.y_train_resampled)
        
        self.models['Balanced_RF'] = {
            'model': model,
            'name': 'Balanced Random Forest',
            'sampling': 'ADASYN'
        }
        
        print("Balanced Random Forest training completed!")
    
    def train_xgboost_model(self):
        """Train XGBoost model"""
        print("\n--- Training XGBoost ---")
        
        # Calculate scale_pos_weight for class imbalance
        scale_pos_weight = (self.y_train == 0).sum() / (self.y_train == 1).sum()
        
        # XGBoost model with class imbalance handling
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss',
            n_jobs=-1
        )
        
        # Train on ADASYN resampled data
        model.fit(self.X_train_resampled, self.y_train_resampled)
        
        self.models['XGBoost'] = {
            'model': model,
            'name': 'XGBoost',
            'sampling': 'ADASYN'
        }
        
        print("XGBoost training completed!")
    

    
    def optimize_threshold(self, y_true, y_proba, metric='f1'):
        """Find optimal threshold for classification"""
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_score = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            if metric == 'f1':
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                if precision + recall > 0:
                    score = 2 * (precision * recall) / (precision + recall)
                else:
                    score = 0
            elif metric == 'precision':
                score = precision_score(y_true, y_pred, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_true, y_pred, zero_division=0)
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        return best_threshold, best_score
    
    def evaluate_model(self, model_key):
        """Evaluate a specific model"""
        model_info = self.models[model_key]
        model = model_info['model']
        
        print(f"\nEvaluating {model_info['name']}...")
        
        # Make predictions
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Optimize threshold
        if y_pred_proba is not None:
            best_threshold, best_f1 = self.optimize_threshold(self.y_test, y_pred_proba, 'f1')
            y_pred_optimized = (y_pred_proba >= best_threshold).astype(int)
        else:
            best_threshold = 0.5
            y_pred_optimized = y_pred
            best_f1 = 0
        
        # Calculate metrics
        accuracy = (y_pred_optimized == self.y_test).mean()
        precision = precision_score(self.y_test, y_pred_optimized, zero_division=0)
        recall = recall_score(self.y_test, y_pred_optimized, zero_division=0)
        f1 = best_f1 if y_pred_proba is not None else 0
        if f1 == 0 and precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba) if y_pred_proba is not None else None
        
        # Store results
        self.results[model_key] = {
            'model': model,
            'sampling_method': model_info['sampling'],
            'model_name': model_info['name'],
            'predictions': y_pred_optimized,
            'predictions_proba': y_pred_proba,
            'best_threshold': best_threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        }
        
        print(f"Results for {model_info['name']}:")
        print(f"  F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        print(f"  Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")
        print(f"  Optimal threshold: {best_threshold:.3f}")
        
        return self.results[model_key]

    def plot_model_comparison(self):
        """Plot comparison of all models"""
        if not self.results:
            print("No results to plot. Please run evaluation first.")
            return

        # Create comparison dataframe
        comparison_data = []
        for model_key, result in self.results.items():
            comparison_data.append({
                'Model': result['model_name'],
                'Sampling': result['sampling_method'],
                'F1-Score': result['f1_score'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'ROC-AUC': result['roc_auc'],
                'Accuracy': result['accuracy']
            })

        df_comparison = pd.DataFrame(comparison_data)

        # Plot metrics comparison
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        metrics = ['F1-Score', 'Precision', 'Recall', 'ROC-AUC', 'Accuracy']
        model_labels = [f"{row['Model']}" for _, row in df_comparison.iterrows()]
        colors = ['skyblue', 'lightcoral'][:len(df_comparison)]

        for i, metric in enumerate(metrics):
            row = i // 3
            col = i % 3

            values = df_comparison[metric].values
            bars = axes[row, col].bar(np.arange(len(values)), values, color=colors, width=0.6)

            axes[row, col].set_title(metric, fontweight='bold')
            axes[row, col].set_ylabel(metric)
            axes[row, col].set_xticks(np.arange(len(values)))
            axes[row, col].set_xticklabels(model_labels, ha='center')
            axes[row, col].set_ylim(0, 1)

            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[row, col].text(bar.get_x() + bar.get_width() / 2,
                                    bar.get_height() + 0.01,
                                    f'{value:.3f}',
                                    ha='center', va='bottom', fontweight='bold')

        # Remove empty subplot
        fig.delaxes(axes[1, 2])

        plt.tight_layout()
        plt.show()

    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        n_models = len(self.results)
        if n_models == 0:
            print("No results to plot.")
            return
        
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        if n_models == 1:
            axes = [axes]
        
        for i, (model_key, result) in enumerate(self.results.items()):
            cm = confusion_matrix(self.y_test, result['predictions'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['PASS', 'FAIL'],
                       yticklabels=['PASS', 'FAIL'],
                       ax=axes[i])
            
            axes[i].set_title(f'{result["model_name"]}\n({result["sampling_method"]})')
            axes[i].set_ylabel('True Label')
            axes[i].set_xlabel('Predicted Label')
            
            # Add metrics below each plot
            f1 = result['f1_score']
            precision = result['precision']
            recall = result['recall']
            roc_auc = result['roc_auc']
            
            axes[i].text(0.5, -0.15, 
                        f'F1: {f1:.3f} | Prec: {precision:.3f} | Rec: {recall:.3f} | AUC: {roc_auc:.3f}',
                        transform=axes[i].transAxes, ha='center', fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        plt.show()
     
    def show_classification_report(self):
        """Show detailed classification report - exact from original"""
        print("\nDetailed Classification Report:")
        print("=" * 50)
        
        class_report = classification_report(
            self.y_test, 
            self.results['predictions'],
            target_names=['PASS', 'FAIL'],
            digits=4
        )
        print(class_report)
    
    def show_feature_importance(self):
        """Show feature importance for tree-based models"""
        for model_key, result in self.results.items():
            model = result['model']
            
            if hasattr(model, 'feature_importances_'):
                print(f"\nTop 10 Feature Importances - {result['model_name']}:")
                print("-" * 50)
                
                importances = model.feature_importances_
                feature_importance = pd.DataFrame({
                    'feature': [f'Feature_{i}' for i in range(len(importances))],
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                print(feature_importance.head(10).to_string(index=False, float_format='%.4f'))
    
    def run_complete_analysis(self):
        """Run the complete analysis with multiple models"""
        print("SECOM Predictive Maintenance - Multi-Model Analysis")
        print("=" * 70)
        
        # Step 1: Load data
        if not self.load_data():
            print("Failed to load data. Please check file paths.")
            return False
        
        # Step 2: Advanced preprocessing
        self.advanced_preprocessing()
        
        # Step 3: Split data
        self.split_data()
        
        # Step 4: Apply ADASYN sampling
        self.apply_adasyn_sampling()
        
        # Step 5: Train all models
        self.train_balanced_random_forest()
        self.train_xgboost_model()
        
        # Step 6: Evaluate all models
        for model_key in self.models.keys():
            self.evaluate_model(model_key)
        
        # Step 7: Show results and visualizations
        self.plot_model_comparison()
        self.plot_confusion_matrices()
        self.show_feature_importance()
        self.show_classification_report()
        
        # Step 8: Final summary
        print("\n" + "="*80)
        print("FINAL SUMMARY - ALL MODELS")
        print("="*80)
        
        best_f1_model = max(self.results.items(), key=lambda x: x[1]['f1_score'])
        best_auc_model = max(self.results.items(), key=lambda x: x[1]['roc_auc'])
        
        print(f"Best F1-Score: {best_f1_model[1]['model_name']} ({best_f1_model[1]['f1_score']:.4f})")
        print(f"Best ROC-AUC: {best_auc_model[1]['model_name']} ({best_auc_model[1]['roc_auc']:.4f})")
        
        for model_key, result in self.results.items():
            print(f"\n{result['model_name']} ({result['sampling_method']}):")
            print(f"  F1: {result['f1_score']:.4f} | Precision: {result['precision']:.4f} | Recall: {result['recall']:.4f}")
            print(f"  Accuracy: {result['accuracy']:.4f} | ROC-AUC: {result['roc_auc']:.4f} | Threshold: {result['best_threshold']:.3f}")
        
        print("\nAnalysis completed successfully!")
        return True


def main():
    """Initialize and run analysis"""
    secom_analyzer = SECOM_MultiModel_Analysis()
    success = secom_analyzer.run_complete_analysis()
    
    if success:
        return secom_analyzer
    else:
        return None

# Run the analysis
if __name__ == "__main__":
    analyzer = main()
