import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
from sklearn.feature_selection import RFE
from sklearn.metrics import (confusion_matrix, precision_score, recall_score, 
                           roc_auc_score, classification_report)
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import ADASYN
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

class SECOM_ADASYN_BalancedRF:
    def __init__(self):
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.results = {}
        
    def load_data(self):
        "Load SECOM dataset - exact from original code"
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
        "preprocessing "
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
        scaler = RobustScaler()
        self.X = pd.DataFrame(scaler.fit_transform(self.X))
        
        # Feature selection with RFE
        print("Using RFE feature selection...")
        estimator = RandomForestClassifier(n_estimators=50, random_state=42)
        selector = RFE(estimator, n_features_to_select=min(200, self.X.shape[1]))
        self.X = pd.DataFrame(selector.fit_transform(self.X, self.y))
        
        print(f"Final feature shape: {self.X.shape}")
        print(f"Missing values: {self.X.isnull().sum().sum()}")
    
    def split_data(self):
        """Split data with stratification - exact from original"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        print(f"\nData split:")
        print(f"Training: {self.X_train.shape[0]} samples, Failures: {self.y_train.sum()} ({self.y_train.mean():.2%})")
        print(f"Test: {self.X_test.shape[0]} samples, Failures: {self.y_test.sum()} ({self.y_test.mean():.2%})")
    
    def apply_adasyn_and_train(self):
        """Apply ADASYN sampling and train Balanced Random Forest - exact from original"""
        print("\n--- ADASYN + Balanced Random Forest ---")
        
        # Apply ADASYN sampling
        try:
            adasyn = ADASYN(random_state=42)
            X_resampled, y_resampled = adasyn.fit_resample(self.X_train, self.y_train)
            print(f"Resampled shape: {X_resampled.shape}")
            print(f"Class distribution: {pd.Series(y_resampled).value_counts().to_dict()}")
        except Exception as e:
            print(f"Sampling failed: {e}")
            X_resampled, y_resampled = self.X_train, self.y_train
        
        # Train Balanced Random Forest 
        print("Training Balanced Random Forest...")
        self.model = BalancedRandomForestClassifier(
            n_estimators=200, 
            max_depth=15, 
            random_state=42
        )
        
        # Fit model on ADASYN resampled data
        self.model.fit(X_resampled, y_resampled)
        
        print("Training completed!")
    
    def optimize_threshold(self, y_true, y_proba, metric='f1'):
        "Find optimal threshold for classification - exact from original"
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_score = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            if metric == 'f1':
                # Calculate F1 score manually to handle zero division
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
    
    def evaluate_model(self):
        "Evaluate model"
        print("\nEvaluating model...")
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1] if hasattr(self.model, 'predict_proba') else None
        
        # Optimize threshold 
        if y_pred_proba is not None:
            best_threshold, best_f1 = self.optimize_threshold(self.y_test, y_pred_proba, 'f1')
            y_pred_optimized = (y_pred_proba >= best_threshold).astype(int)
        else:
            best_threshold = 0.5
            y_pred_optimized = y_pred
        
        # Calculate metrics (exact from original)
        accuracy = (y_pred_optimized == self.y_test).mean()
        precision = precision_score(self.y_test, y_pred_optimized, zero_division=0)
        recall = recall_score(self.y_test, y_pred_optimized, zero_division=0)
        f1 = best_f1 if y_pred_proba is not None else 0
        if f1 == 0 and precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba) if y_pred_proba is not None else None
        
        # Store results (exact structure from original)
        self.results = {
            'model': self.model,
            'sampling_method': 'ADASYN',
            'model_name': 'Balanced Random Forest',
            'predictions': y_pred_optimized,
            'predictions_proba': y_pred_proba,
            'best_threshold': best_threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        }
        
        print(f"F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        print(f"Optimal threshold: {best_threshold:.3f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        
        return self.results
    
    def plot_confusion_matrix(self):
        """Plot confusion matrix - enhanced version"""
        cm = confusion_matrix(self.y_test, self.results['predictions'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['PASS', 'FAIL'],
                   yticklabels=['PASS', 'FAIL'])
        plt.title('Confusion Matrix - ADASYN + Balanced Random Forest')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Add performance metrics
        precision = self.results['precision']
        recall = self.results['recall']
        f1_score = self.results['f1_score']
        roc_auc = self.results['roc_auc']
        threshold = self.results['best_threshold']
        
        plt.figtext(0.02, 0.02, 
                   f'Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1_score:.4f} | ROC-AUC: {roc_auc:.4f} | Threshold: {threshold:.3f}',
                   fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        plt.show()
        
        # Print confusion matrix details
        tn, fp, fn, tp = cm.ravel()
        print(f"\nConfusion Matrix Details:")
        print(f"True Negatives (Correctly predicted PASS): {tn}")
        print(f"False Positives (Incorrectly predicted FAIL): {fp}")
        print(f"False Negatives (Incorrectly predicted PASS): {fn}")
        print(f"True Positives (Correctly predicted FAIL): {tp}")
    

    
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

    def run_complete_analysis(self):
        """Run the complete analysis - exact pipeline from original"""
        print("SECOM Predictive Maintenance - ADASYN + Balanced Random Forest")
        print("=" * 70)
        
        # Step 1: Load data
        if not self.load_data():
            print("Failed to load data. Please check file paths.")
            return False
        
        # Step 2: Advanced preprocessing (exact from original)
        self.advanced_preprocessing()
        
        # Step 3: Split data
        self.split_data()
        
        # Step 4: Apply ADASYN + Train Balanced Random Forest
        self.apply_adasyn_and_train()
        
        # Step 5: Evaluate model (exact evaluation from original)
        results = self.evaluate_model()
        
        # Step 6: Show detailed results
        self.show_classification_report()
        

        # Step 7: Plot visualizations
        self.plot_confusion_matrix()

        
        print("\n Analysis completed successfully!")
        
        # Final summary (exact format from original)
        print(f"\FINAL SUMMARY:")
        print(f"Sampling Method: {results['sampling_method']}")
        print(f"Model Name: {results['model_name']}")
        print(f"Optimal Threshold: {results['best_threshold']:.3f}")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"F1-Score: {results['f1_score']:.4f}")
        print(f"ROC-AUC: {results['roc_auc']:.4f}")
        
        return True


def main():
    # Initialize and run analysis
    secom_analyzer = SECOM_ADASYN_BalancedRF()
    success = secom_analyzer.run_complete_analysis()
    
    if success:
        return secom_analyzer
    else:
        return None

# Run the analysis
if __name__ == "__main__":
    analyzer = main()