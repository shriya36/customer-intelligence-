"""
Machine Learning Models for Customer Intelligence
Implements Classification, Regression, and Clustering models
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                           precision_score, recall_score, f1_score, r2_score, 
                           mean_squared_error, mean_absolute_error, silhouette_score)
import warnings
warnings.filterwarnings('ignore')

class CustomerMLModels:
    """
    Comprehensive ML model suite for customer intelligence
    """
    
    def __init__(self, df):
        """
        Initialize with customer dataset
        
        Parameters:
        df (pd.DataFrame): Customer dataset
        """
        self.df = df.copy()
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
        # Prepare features
        self._prepare_features()
        
        # Train models
        self._train_churn_models()
        self._train_clv_models()
        self._train_cluster_model()
    
    def _prepare_features(self):
        """Prepare features for ML models"""
        
        # Define all possible numerical features
        all_numerical_features = [
            'age', 'tenure_days', 'purchase_frequency', 'total_spent', 
            'recency', 'satisfaction_score', 'avg_order_value', 'support_tickets',
            'annual_income', 'email_open_rate', 'email_click_rate', 
            'mobile_sessions', 'social_engagement_score', 'has_mobile_app',
            'social_media_follower'
        ]
        
        # Only use features that actually exist in the dataframe
        numerical_features = [f for f in all_numerical_features if f in self.df.columns]
        
        # Ensure we have at least some features to work with
        if len(numerical_features) == 0:
            raise ValueError("No valid numerical features found in the dataset")
        
        print(f"Using {len(numerical_features)} numerical features: {numerical_features}")
        
        # Select categorical features
        categorical_features = []
        potential_categorical = ['gender', 'acquisition_channel', 'city']
        
        for feature in potential_categorical:
            if feature in self.df.columns:
                categorical_features.append(feature)
        
        # Prepare feature matrix
        X_numerical = self.df[numerical_features]
        
        # Encode categorical variables
        X_categorical = pd.DataFrame()
        for feature in categorical_features:
            if feature in self.df.columns:
                le = LabelEncoder()
                X_categorical[feature] = le.fit_transform(self.df[feature])
                self.label_encoders[feature] = le
        
        # Combine features
        self.X = pd.concat([X_numerical, X_categorical], axis=1)
        
        # Scale features
        self.X_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.X),
            columns=self.X.columns,
            index=self.X.index
        )
        
        # Target variables
        self.y_churn = self.df['churn']
        self.y_clv = self.df['total_spent']
        
        print(f"Prepared features: {self.X.shape[1]} features, {self.X.shape[0]} samples")
    
    def _train_churn_models(self):
        """Train churn prediction models"""
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_scaled, self.y_churn, test_size=0.2, random_state=42, stratify=self.y_churn
        )
        
        # Store test sets for evaluation
        self.X_test_churn = X_test
        self.y_test_churn = y_test
        
        # Train models
        self.churn_models = {}
        
        # Random Forest
        rf_churn = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_churn.fit(X_train, y_train)
        self.churn_models['Random Forest'] = rf_churn
        
        # Logistic Regression
        lr_churn = LogisticRegression(random_state=42, max_iter=1000)
        lr_churn.fit(X_train, y_train)
        self.churn_models['Logistic Regression'] = lr_churn
        
        print("Churn prediction models trained successfully")
    
    def _train_clv_models(self):
        """Train CLV prediction models"""
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_scaled, self.y_clv, test_size=0.2, random_state=42
        )
        
        # Store test sets for evaluation
        self.X_test_clv = X_test
        self.y_test_clv = y_test
        
        # Train models
        self.clv_models = {}
        
        # Random Forest Regressor
        rf_clv = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_clv.fit(X_train, y_train)
        self.clv_models['Random Forest'] = rf_clv
        
        # Linear Regression
        lr_clv = LinearRegression()
        lr_clv.fit(X_train, y_train)
        self.clv_models['Linear Regression'] = lr_clv
        
        print("CLV prediction models trained successfully")
    
    def _train_cluster_model(self):
        """Train customer segmentation model"""
        
        # Use key features for clustering
        cluster_features = ['total_spent', 'purchase_frequency', 'recency', 'satisfaction_score']
        X_cluster = self.df[cluster_features]
        
        # Scale features for clustering
        X_cluster_scaled = StandardScaler().fit_transform(X_cluster)
        
        # Determine optimal number of clusters using elbow method
        inertias = []
        silhouette_scores = []
        K_range = range(2, 11)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_cluster_scaled)
            inertias.append(kmeans.inertia_)
            if len(np.unique(kmeans.labels_)) > 1:
                silhouette_scores.append(silhouette_score(X_cluster_scaled, kmeans.labels_))
            else:
                silhouette_scores.append(0)
        
        # Choose optimal k (highest silhouette score)
        optimal_k = K_range[np.argmax(silhouette_scores)]
        
        # Train final clustering model
        self.cluster_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        self.cluster_labels = self.cluster_model.fit_predict(X_cluster_scaled)
        
        # Store cluster features and scaler
        self.cluster_features = cluster_features
        self.cluster_scaler = StandardScaler().fit(X_cluster)
        
        print(f"Clustering model trained with {optimal_k} clusters")
    
    def get_churn_metrics(self):
        """Get churn prediction model metrics"""
        
        metrics = {}
        
        for name, model in self.churn_models.items():
            y_pred = model.predict(self.X_test_churn)
            
            metrics[name] = {
                'accuracy': accuracy_score(self.y_test_churn, y_pred),
                'precision': precision_score(self.y_test_churn, y_pred, zero_division=0),
                'recall': recall_score(self.y_test_churn, y_pred, zero_division=0),
                'f1_score': f1_score(self.y_test_churn, y_pred, zero_division=0)
            }
        
        return metrics
    
    def get_clv_metrics(self):
        """Get CLV prediction model metrics"""
        
        metrics = {}
        
        for name, model in self.clv_models.items():
            y_pred = model.predict(self.X_test_clv)
            
            metrics[name] = {
                'r2': r2_score(self.y_test_clv, y_pred),
                'rmse': np.sqrt(mean_squared_error(self.y_test_clv, y_pred)),
                'mae': mean_absolute_error(self.y_test_clv, y_pred)
            }
        
        return metrics
    
    def get_churn_feature_importance(self):
        """Get feature importance for churn prediction"""
        
        # Use Random Forest feature importance
        rf_model = self.churn_models['Random Forest']
        feature_importance = dict(zip(self.X.columns, rf_model.feature_importances_))
        
        # Sort by importance
        feature_importance = dict(sorted(feature_importance.items(), 
                                       key=lambda x: x[1], reverse=True))
        
        return feature_importance
    
    def get_clv_feature_importance(self):
        """Get feature importance for CLV prediction"""
        
        # Use Random Forest feature importance
        rf_model = self.clv_models['Random Forest']
        feature_importance = dict(zip(self.X.columns, rf_model.feature_importances_))
        
        # Sort by importance
        feature_importance = dict(sorted(feature_importance.items(), 
                                       key=lambda x: x[1], reverse=True))
        
        return feature_importance
    
    def predict_churn(self, data=None):
        """Predict churn for customers"""
        
        if data is None:
            data = self.X_scaled
        
        # Use best performing model (Random Forest)
        best_model = self.churn_models['Random Forest']
        return best_model.predict(data)
    
    def predict_churn_proba(self, data=None):
        """Predict churn probability for customers"""
        
        if data is None:
            data = self.X_scaled
        
        # Use best performing model (Random Forest)
        best_model = self.churn_models['Random Forest']
        return best_model.predict_proba(data)[:, 1]  # Probability of churn
    
    def predict_clv(self, data=None):
        """Predict CLV for customers"""
        
        if data is None:
            data = self.X_scaled
        
        # Use best performing model (Random Forest)
        best_model = self.clv_models['Random Forest']
        return best_model.predict(data)
    
    def perform_clustering(self, data=None):
        """Perform customer segmentation"""
        
        if data is None:
            return self.cluster_labels
        
        # Scale new data
        X_cluster = data[self.cluster_features]
        X_cluster_scaled = self.cluster_scaler.transform(X_cluster)
        
        return self.cluster_model.predict(X_cluster_scaled)
    
    def get_segment_summary(self):
        """Get summary of customer segments"""
        
        segments = pd.Series(self.cluster_labels)
        segment_counts = segments.value_counts().sort_index()
        
        # Create meaningful segment names
        segment_names = {}
        df_with_segments = self.df.copy()
        df_with_segments['segment'] = self.cluster_labels
        
        segment_stats = df_with_segments.groupby('segment').agg({
            'total_spent': 'mean',
            'purchase_frequency': 'mean',
            'recency': 'mean',
            'satisfaction_score': 'mean'
        })
        
        # Name segments based on characteristics
        for i in range(len(segment_stats)):
            stats = segment_stats.iloc[i]
            
            if stats['total_spent'] > self.df['total_spent'].median() and stats['satisfaction_score'] > 7:
                segment_names[i] = 'High-Value Loyal'
            elif stats['total_spent'] > self.df['total_spent'].median():
                segment_names[i] = 'High-Value At-Risk'
            elif stats['purchase_frequency'] > self.df['purchase_frequency'].median():
                segment_names[i] = 'Frequent Buyers'
            elif stats['recency'] > self.df['recency'].median():
                segment_names[i] = 'Inactive Customers'
            else:
                segment_names[i] = f'Segment {i}'
        
        # Return segment counts with names
        named_segments = {}
        for i, count in segment_counts.items():
            named_segments[segment_names.get(i, f'Segment {i}')] = count
        
        return named_segments
    
    def get_customer_risk_score(self):
        """Calculate risk score for each customer"""
        
        churn_proba = self.predict_churn_proba()
        
        # Combine churn probability with other risk factors
        risk_factors = {
            'churn_probability': churn_proba,
            'recency_risk': (self.df['recency'] / self.df['recency'].max()),
            'satisfaction_risk': (10 - self.df['satisfaction_score']) / 9,
            'frequency_risk': 1 - (self.df['purchase_frequency'] / self.df['purchase_frequency'].max())
        }
        
        # Weighted risk score
        risk_score = (
            risk_factors['churn_probability'] * 0.4 +
            risk_factors['recency_risk'] * 0.3 +
            risk_factors['satisfaction_risk'] * 0.2 +
            risk_factors['frequency_risk'] * 0.1
        )
        
        return risk_score
    
    def get_model_summary(self):
        """Get comprehensive model summary"""
        
        summary = {
            'dataset_size': len(self.df),
            'features_used': len(self.X.columns),
            'churn_models': list(self.churn_models.keys()),
            'clv_models': list(self.clv_models.keys()),
            'clusters_found': len(np.unique(self.cluster_labels)),
            'churn_rate': f"{self.df['churn'].mean():.3f}",
            'avg_clv': f"${self.df['total_spent'].mean():.2f}"
        }
        
        return summary
