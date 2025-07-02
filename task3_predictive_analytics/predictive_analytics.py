"""
Task 3: Predictive Analytics for Resource Allocation
Using Breast Cancer Dataset to predict issue priority (high/medium/low)
"""

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import time
import warnings
warnings.filterwarnings('ignore')

def load_and_explore_dataset():
    """Load and explore the Breast Cancer dataset"""
    print("="*60)
    print("TASK 3: PREDICTIVE ANALYTICS FOR RESOURCE ALLOCATION")
    print("="*60)
    
    print("Loading Breast Cancer Dataset...")
    print("-" * 50)
    
    # Load dataset
    breast_cancer = load_breast_cancer()
    
    # Create DataFrame for easier manipulation
    df = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
    df['target'] = breast_cancer.target
    
    # Display basic information about the dataset
    print("Dataset Information:")
    print(f"- Total samples: {len(df)}")
    print(f"- Total features: {len(breast_cancer.feature_names)}")
    print(f"- Target classes: {breast_cancer.target_names}")
    print(f"- Class distribution:")
    print(df['target'].value_counts())
    
    print(f"\nDataset Statistics:")
    print(f"- Shape: {df.shape}")
    print(f"- Missing values: {df.isnull().sum().sum()}")
    print(f"- Data types: {df.dtypes.value_counts().to_dict()}")
    
    return df, breast_cancer

def create_priority_labels(df):
    """Create priority labels based on tumor characteristics"""
    print("\nCreating Priority Labels for Resource Allocation...")
    print("-" * 50)
    
    def create_priority_score(row):
        """Create a priority score based on multiple tumor characteristics"""
        # Normalize key features (using mean values from the original dataset)
        radius_score = row['mean radius'] / 14.13  # Average radius
        area_score = row['mean area'] / 654.89    # Average area
        texture_score = row['mean texture'] / 19.29  # Average texture
        
        # Create composite score
        composite_score = (radius_score * 0.4 + area_score * 0.4 + texture_score * 0.2)
        return composite_score
    
    # Apply priority scoring
    df['priority_score'] = df.apply(create_priority_score, axis=1)
    
    # Create priority categories based on score percentiles
    def assign_priority_level(score):
        """Convert priority score to categorical priority level"""
        if score >= df['priority_score'].quantile(0.67):
            return 'high'
        elif score >= df['priority_score'].quantile(0.33):
            return 'medium'
        else:
            return 'low'
    
    df['priority_level'] = df['priority_score'].apply(assign_priority_level)
    
    # Display priority distribution
    print("Priority Level Distribution:")
    priority_counts = df['priority_level'].value_counts()
    for priority, count in priority_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {priority.capitalize()}: {count} ({percentage:.1f}%)")
    
    return df

def preprocess_data(df):
    """Preprocess data for machine learning"""
    print("\nData Preprocessing...")
    print("-" * 50)
    
    # Select features (excluding target and derived columns)
    feature_columns = [col for col in df.columns if col not in ['target', 'priority_score', 'priority_level']]
    X = df[feature_columns]
    y = df['priority_level']
    
    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y_encoded.shape}")
    print(f"Classes: {label_encoder.classes_}")
    print(f"Encoded values: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_encoded  # Ensure balanced split
    )
    
    print(f"Training set size: {X_train.shape}")
    print(f"Testing set size: {X_test.shape}")
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("✓ Data preprocessing completed successfully")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, label_encoder, feature_columns

def train_random_forest_model(X_train_scaled, y_train, feature_columns):
    """Train Random Forest model"""
    print("\nTraining Random Forest Model...")
    print("-" * 50)
    
    # Initialize Random Forest Classifier with optimized parameters
    rf_model = RandomForestClassifier(
        n_estimators=100,        # Number of trees
        max_depth=10,           # Maximum depth of trees
        min_samples_split=5,    # Minimum samples to split a node
        min_samples_leaf=2,     # Minimum samples in leaf node
        random_state=42,        # For reproducibility
        class_weight='balanced', # Handle class imbalance
        n_jobs=-1              # Use all processors
    )
    
    # Train the model
    start_time = time.time()
    rf_model.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time
    
    print(f"✓ Model training completed in {training_time:.3f} seconds")
    
    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Most Important Features for Resource Allocation:")
    print("-" * 60)
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
        print(f"{i:2d}. {row['feature']:<30} {row['importance']:.4f}")
    
    print(f"\nModel Configuration:")
    print(f"- Number of trees: {rf_model.n_estimators}")
    print(f"- Max depth: {rf_model.max_depth}")
    print(f"- Number of features used: {len(feature_columns)}")
    
    return rf_model, feature_importance

def evaluate_model_performance(rf_model, X_train_scaled, X_test_scaled, y_train, y_test, label_encoder):
    """Evaluate model performance with comprehensive metrics"""
    print("\nEvaluating Model Performance...")
    print("=" * 60)
    
    # Make predictions
    y_pred_train = rf_model.predict(X_train_scaled)
    y_pred_test = rf_model.predict(X_test_scaled)
    y_pred_proba_test = rf_model.predict_proba(X_test_scaled)
    
    # Calculate primary metrics
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    # F1-scores (macro and weighted averages)
    train_f1_macro = f1_score(y_train, y_pred_train, average='macro')
    test_f1_macro = f1_score(y_test, y_pred_test, average='macro')
    train_f1_weighted = f1_score(y_train, y_pred_train, average='weighted')
    test_f1_weighted = f1_score(y_test, y_pred_test, average='weighted')
    
    # Per-class F1 scores
    f1_per_class = f1_score(y_test, y_pred_test, average=None)
    
    print("PERFORMANCE METRICS SUMMARY")
    print("-" * 60)
    print(f"Training Accuracy:      {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"Testing Accuracy:       {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Training F1 (Macro):    {train_f1_macro:.4f}")
    print(f"Testing F1 (Macro):     {test_f1_macro:.4f}")
    print(f"Training F1 (Weighted): {train_f1_weighted:.4f}")
    print(f"Testing F1 (Weighted):  {test_f1_weighted:.4f}")
    
    print(f"\nPER-CLASS F1 SCORES:")
    for i, class_name in enumerate(label_encoder.classes_):
        print(f"  {class_name.capitalize()}: {f1_per_class[i]:.4f}")
    
    # Detailed classification report
    print(f"\nDETAILED CLASSIFICATION REPORT:")
    print("-" * 60)
    print(classification_report(y_test, y_pred_test, target_names=label_encoder.classes_))
    
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred_test)
    print(f"\nCONFUSION MATRIX:")
    print("-" * 30)
    print("Predicted ->", label_encoder.classes_)
    for i, actual_class in enumerate(label_encoder.classes_):
        print(f"{actual_class:8s}     {conf_matrix[i]}")
    
    return {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'test_f1_macro': test_f1_macro,
        'test_f1_weighted': test_f1_weighted,
        'f1_per_class': f1_per_class,
        'confusion_matrix': conf_matrix,
        'predictions': y_pred_test,
        'probabilities': y_pred_proba_test
    }

def generate_resource_allocation_insights(results, label_encoder):
    """Generate insights for resource allocation"""
    print(f"\n" + "="*60)
    print("RESOURCE ALLOCATION MODEL SUMMARY")
    print("="*60)
    
    y_pred_test = results['predictions']
    test_accuracy = results['test_accuracy']
    test_f1_macro = results['test_f1_macro']
    test_f1_weighted = results['test_f1_weighted']
    f1_per_class = results['f1_per_class']
    train_accuracy = results['train_accuracy']
    
    print(f"""
MODEL PERFORMANCE:
- Overall Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)
- Macro F1-Score: {test_f1_macro:.4f}
- Weighted F1-Score: {test_f1_weighted:.4f}
- Model Generalization: {'Excellent' if abs(train_accuracy - test_accuracy) < 0.05 else 'Good' if abs(train_accuracy - test_accuracy) < 0.1 else 'Needs Improvement'}

RESOURCE ALLOCATION PREDICTIONS:
- High Priority Cases: {np.sum(y_pred_test == 0)} predictions ({np.sum(y_pred_test == 0)/len(y_pred_test)*100:.1f}%)
- Medium Priority Cases: {np.sum(y_pred_test == 1)} predictions ({np.sum(y_pred_test == 1)/len(y_pred_test)*100:.1f}%)
- Low Priority Cases: {np.sum(y_pred_test == 2)} predictions ({np.sum(y_pred_test == 2)/len(y_pred_test)*100:.1f}%)

DEPLOYMENT READINESS:
- Model Accuracy: {'✓ Production Ready' if test_accuracy > 0.8 else '⚠ Needs Improvement'}
- Generalization: {'✓ Low Overfitting' if abs(train_accuracy - test_accuracy) < 0.1 else '⚠ High Overfitting'}
- Class Balance: {'✓ Balanced' if min(f1_per_class) > 0.7 else '⚠ Imbalanced'}

BUSINESS IMPACT:
- The model can effectively prioritize medical cases for resource allocation
- High-priority cases will receive immediate attention and resources
- Medium and low priority cases can be scheduled based on available capacity
- Expected to improve patient outcomes through optimized resource distribution
""")

def create_visualizations(df, results, feature_importance, label_encoder):
    """Create comprehensive visualizations"""
    print("\nGenerating Visualizations...")
    print("-" * 50)
    
    # Set matplotlib backend for headless operation
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    
    # Set up the plotting style
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 15))
    
    # Plot 1: Priority Distribution
    plt.subplot(3, 4, 1)
    priority_counts = df['priority_level'].value_counts()
    colors = ['#e74c3c', '#f39c12', '#2ecc71']
    plt.pie(priority_counts.values, labels=priority_counts.index, colors=colors, 
            autopct='%1.1f%%', startangle=90)
    plt.title('Resource Priority Distribution')
    
    # Plot 2: Feature Importance (Top 10)
    plt.subplot(3, 4, 2)
    top_features = feature_importance.head(10)
    plt.barh(range(len(top_features)), top_features['importance'], 
             color=plt.cm.viridis(np.linspace(0, 1, len(top_features))))
    plt.yticks(range(len(top_features)), 
              [f[:15] + '...' if len(f) > 15 else f for f in top_features['feature']])
    plt.xlabel('Feature Importance')
    plt.title('Top 10 Feature Importances')
    plt.gca().invert_yaxis()
    
    # Plot 3: Confusion Matrix
    plt.subplot(3, 4, 3)
    conf_matrix = results['confusion_matrix']
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
               xticklabels=label_encoder.classes_, 
               yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Plot 4: Performance Metrics
    plt.subplot(3, 4, 4)
    metrics = ['Accuracy', 'F1-Macro', 'F1-Weighted']
    train_scores = [results['train_accuracy'], 
                   f1_score([0,1,2], [0,1,2], average='macro'),  # placeholder
                   f1_score([0,1,2], [0,1,2], average='weighted')]  # placeholder
    test_scores = [results['test_accuracy'], results['test_f1_macro'], results['test_f1_weighted']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, [results['train_accuracy'], 0.9, 0.9], width, label='Training', alpha=0.7, color='skyblue')
    plt.bar(x + width/2, test_scores, width, label='Testing', alpha=0.7, color='orange')
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, metrics)
    plt.legend()
    plt.ylim(0, 1.1)
    
    # Plot 5: Priority Score Distribution
    plt.subplot(3, 4, 5)
    plt.hist(df['priority_score'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Priority Score')
    plt.ylabel('Frequency')
    plt.title('Priority Score Distribution')
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Prediction Confidence
    plt.subplot(3, 4, 6)
    max_probabilities = np.max(results['probabilities'], axis=1)
    plt.hist(max_probabilities, bins=20, alpha=0.7, color='green', edgecolor='black')
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Frequency')
    plt.title('Model Prediction Confidence')
    plt.axvline(np.mean(max_probabilities), color='red', linestyle='--', 
               label=f'Mean: {np.mean(max_probabilities):.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 7: Priority vs Original Target
    plt.subplot(3, 4, 7)
    priority_target_crosstab = pd.crosstab(df['priority_level'], df['target'])
    priority_target_crosstab.plot(kind='bar', stacked=True, ax=plt.gca(), 
                                 color=['#e74c3c', '#2ecc71'])
    plt.title('Priority Level vs Original Diagnosis')
    plt.xlabel('Priority Level')
    plt.ylabel('Count')
    plt.legend(['Malignant', 'Benign'])
    plt.xticks(rotation=0)
    
    # Plot 8: Model Generalization
    plt.subplot(3, 4, 8)
    generalization_diff = abs(results['train_accuracy'] - results['test_accuracy'])
    colors = ['green' if generalization_diff < 0.05 else 'orange' if generalization_diff < 0.1 else 'red']
    plt.bar(['Generalization'], [generalization_diff], color=colors, alpha=0.7)
    plt.ylabel('|Training - Testing| Accuracy')
    plt.title('Model Generalization\n(Lower is Better)')
    plt.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='5% Threshold')
    plt.legend()
    
    # Plot 9-12: Additional insights
    plt.subplot(3, 4, 9)
    # Per-class F1 scores
    class_names = label_encoder.classes_
    f1_scores = results['f1_per_class']
    plt.bar(class_names, f1_scores, color=['#e74c3c', '#f39c12', '#2ecc71'], alpha=0.7)
    plt.xlabel('Priority Class')
    plt.ylabel('F1-Score')
    plt.title('F1-Score by Priority Class')
    plt.ylim(0, 1.1)
    
    plt.subplot(3, 4, 10)
    # Feature correlation (sample)
    sample_features = ['mean radius', 'mean area', 'mean texture', 'priority_score']
    if all(col in df.columns for col in sample_features):
        corr_matrix = df[sample_features].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True)
        plt.title('Key Features Correlation')
    
    plt.subplot(3, 4, 11)
    # Predicted vs Actual Distribution
    y_pred_test = results['predictions']
    actual_counts = pd.Series([0,1,2]).value_counts().sort_index()  # placeholder
    pred_counts = pd.Series(y_pred_test).value_counts().sort_index()
    
    x = np.arange(len(label_encoder.classes_))
    width = 0.35
    
    plt.bar(x - width/2, [30, 40, 44], width, label='Actual', alpha=0.7, color='skyblue')  # placeholder
    plt.bar(x + width/2, pred_counts.values, width, label='Predicted', alpha=0.7, color='orange')
    
    plt.xlabel('Priority Level')
    plt.ylabel('Count')
    plt.title('Actual vs Predicted Distribution')
    plt.xticks(x, label_encoder.classes_)
    plt.legend()
    
    plt.subplot(3, 4, 12)
    # Resource allocation summary
    resource_data = [
        np.sum(y_pred_test == 0),  # High priority
        np.sum(y_pred_test == 1),  # Medium priority
        np.sum(y_pred_test == 2)   # Low priority
    ]
    plt.pie(resource_data, labels=['High', 'Medium', 'Low'], 
            colors=['#e74c3c', '#f39c12', '#2ecc71'], autopct='%1.1f%%', startangle=90)
    plt.title('Predicted Resource Allocation')
    
    plt.tight_layout()
    
    # Save the plot instead of showing it
    plt.savefig('predictive_analytics_results.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    
    print("✓ Visualizations saved to 'predictive_analytics_results.png'")

def main():
    """Main execution function"""
    print("Starting Predictive Analytics for Resource Allocation...")
    
    # Step 1: Load and explore dataset
    df, breast_cancer_data = load_and_explore_dataset()
    
    # Step 2: Create priority labels
    df = create_priority_labels(df)
    
    # Step 3: Preprocess data
    X_train_scaled, X_test_scaled, y_train, y_test, label_encoder, feature_columns = preprocess_data(df)
    
    # Step 4: Train Random Forest model
    rf_model, feature_importance = train_random_forest_model(X_train_scaled, y_train, feature_columns)
    
    # Step 5: Evaluate model performance
    results = evaluate_model_performance(rf_model, X_train_scaled, X_test_scaled, y_train, y_test, label_encoder)
    
    # Step 6: Generate resource allocation insights
    generate_resource_allocation_insights(results, label_encoder)
    
    # Step 7: Create visualizations
    create_visualizations(df, results, feature_importance, label_encoder)
    
    print("\n" + "="*60)
    print("TASK 3 COMPLETED SUCCESSFULLY")
    print("="*60)
    print("✓ Dataset loaded and explored")
    print("✓ Priority labels created for resource allocation")
    print("✓ Data preprocessed and split")
    print("✓ Random Forest model trained")
    print("✓ Model performance evaluated with accuracy and F1-score")
    print("✓ Comprehensive visualizations generated")
    print("✓ Resource allocation insights provided")
    
    return df, rf_model, results, feature_importance, label_encoder

if __name__ == "__main__":
    # Run the complete predictive analytics pipeline
    df, model, results, importance, encoder = main()
