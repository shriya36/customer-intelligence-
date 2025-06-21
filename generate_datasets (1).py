"""
Generate enhanced datasets and model outputs for download
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')

def generate_customer_data(n_customers=5000):
    """Generate comprehensive customer dataset"""
    np.random.seed(42)
    random.seed(42)
    
    # Demographics
    customer_ids = range(1, n_customers + 1)
    ages = np.random.normal(40, 15, n_customers).astype(int)
    ages = np.clip(ages, 18, 80)
    
    genders = np.random.choice(['Male', 'Female', 'Other'], n_customers, p=[0.48, 0.50, 0.02])
    
    cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 
              'San Antonio', 'San Diego', 'Dallas', 'San Jose', 'Austin', 'Jacksonville']
    customer_cities = np.random.choice(cities, n_customers)
    
    # Financial data
    income_levels = np.random.lognormal(10.5, 0.5, n_customers)
    income_levels = np.clip(income_levels, 25000, 200000)
    
    tenure_days = np.random.exponential(365, n_customers).astype(int)
    tenure_days = np.clip(tenure_days, 1, 1095)
    
    # Purchase behavior
    base_frequency = 2 + (income_levels - income_levels.min()) / (income_levels.max() - income_levels.min()) * 20
    age_factor = 1 - (ages - 18) / (80 - 18) * 0.3
    purchase_frequency = (base_frequency * age_factor).astype(int)
    purchase_frequency = np.clip(purchase_frequency, 1, 50)
    
    # Total spending
    base_spending = income_levels * 0.1
    frequency_factor = 1 + (purchase_frequency - 1) / 49 * 0.5
    tenure_factor = 1 + tenure_days / 1095 * 0.3
    
    total_spent = base_spending * frequency_factor * tenure_factor
    total_spent = total_spent + np.random.normal(0, total_spent * 0.1)
    total_spent = np.clip(total_spent, 50, 50000)
    
    avg_order_value = total_spent / purchase_frequency
    
    # Recency
    recency = np.random.exponential(30, n_customers).astype(int)
    recency = np.clip(recency, 1, 365)
    
    # Satisfaction
    base_satisfaction = 6 + np.random.normal(0, 1.5, n_customers)
    spending_percentile = (total_spent - total_spent.min()) / (total_spent.max() - total_spent.min())
    satisfaction_boost = spending_percentile * 1.5
    satisfaction_score = base_satisfaction + satisfaction_boost
    satisfaction_score = np.clip(satisfaction_score, 1, 10)
    
    # Support interactions
    satisfaction_factor = (10 - satisfaction_score) / 9
    support_tickets = np.random.poisson(satisfaction_factor * 3, n_customers)
    
    # Marketing channels
    channels = ['Organic Search', 'Social Media', 'Email Marketing', 'Paid Ads', 
                'Referral', 'Direct', 'Content Marketing', 'Affiliate']
    channel_weights = [0.25, 0.20, 0.15, 0.15, 0.10, 0.08, 0.04, 0.03]
    acquisition_channel = np.random.choice(channels, n_customers, p=channel_weights)
    
    # Digital engagement
    email_open_rate = np.random.beta(2, 3, n_customers)
    email_click_rate = email_open_rate * np.random.beta(2, 5, n_customers)
    
    has_mobile_app = np.random.choice([0, 1], n_customers, p=[0.3, 0.7])
    mobile_sessions = np.where(has_mobile_app, 
                              np.random.poisson(purchase_frequency * 2, n_customers), 0)
    
    social_media_follower = np.random.choice([0, 1], n_customers, p=[0.6, 0.4])
    social_engagement_score = np.where(social_media_follower,
                                      np.random.uniform(0, 100, n_customers), 0)
    
    # Churn calculation
    churn_probability = (
        0.3 * (10 - satisfaction_score) / 9 +
        0.25 * recency / 365 +
        0.2 * (1 - (purchase_frequency - 1) / 49) +
        0.15 * (support_tickets == 0).astype(int) +
        0.1 * (tenure_days < 90).astype(int) / 90
    )
    
    churn_probability += np.random.normal(0, 0.1, n_customers)
    churn_probability = np.clip(churn_probability, 0, 1)
    churn = (np.random.random(n_customers) < churn_probability).astype(int)
    
    # Product categories
    categories = ['Electronics', 'Clothing', 'Books', 'Home & Garden', 'Sports', 
                  'Beauty', 'Automotive', 'Health', 'Toys', 'Food']
    preferred_categories = []
    for _ in range(n_customers):
        n_categories = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
        customer_categories = np.random.choice(categories, n_categories, replace=False)
        preferred_categories.append(', '.join(customer_categories))
    
    # Seasonal preferences
    seasonal_preference = np.random.choice(['Winter', 'Summer', 'Balanced'], n_customers, p=[0.3, 0.3, 0.4])
    
    # Create DataFrame
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'age': ages,
        'gender': genders,
        'city': customer_cities,
        'annual_income': income_levels.round(2),
        'tenure_days': tenure_days,
        'purchase_frequency': purchase_frequency,
        'total_spent': total_spent.round(2),
        'avg_order_value': avg_order_value.round(2),
        'recency': recency,
        'satisfaction_score': satisfaction_score.round(1),
        'support_tickets': support_tickets,
        'acquisition_channel': acquisition_channel,
        'preferred_categories': preferred_categories,
        'seasonal_preference': seasonal_preference,
        'email_open_rate': email_open_rate.round(3),
        'email_click_rate': email_click_rate.round(3),
        'has_mobile_app': has_mobile_app,
        'mobile_sessions': mobile_sessions,
        'social_media_follower': social_media_follower,
        'social_engagement_score': social_engagement_score.round(1),
        'churn': churn
    })
    
    return df

def train_ml_models(df):
    """Train ML models and generate predictions"""
    
    # Prepare features
    numerical_features = [
        'age', 'annual_income', 'tenure_days', 'purchase_frequency',
        'avg_order_value', 'recency', 'satisfaction_score', 'support_tickets',
        'email_open_rate', 'email_click_rate', 'mobile_sessions', 'social_engagement_score'
    ]
    
    categorical_features = ['gender', 'acquisition_channel']
    
    # Numerical features
    X_numerical = df[numerical_features]
    
    # Encode categorical features
    X_categorical = pd.DataFrame()
    label_encoders = {}
    
    for feature in categorical_features:
        if feature in df.columns:
            le = LabelEncoder()
            X_categorical[feature] = le.fit_transform(df[feature])
            label_encoders[feature] = le
    
    # Combine features
    X = pd.concat([X_numerical, X_categorical], axis=1)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    
    # Target variables
    y_churn = df['churn']
    y_clv = df['total_spent']
    
    # Train churn prediction model
    X_train_churn, X_test_churn, y_train_churn, y_test_churn = train_test_split(
        X_scaled, y_churn, test_size=0.2, random_state=42, stratify=y_churn
    )
    
    churn_model = RandomForestClassifier(n_estimators=100, random_state=42)
    churn_model.fit(X_train_churn, y_train_churn)
    
    # Train CLV prediction model
    X_train_clv, X_test_clv, y_train_clv, y_test_clv = train_test_split(
        X_scaled, y_clv, test_size=0.2, random_state=42
    )
    
    clv_model = RandomForestRegressor(n_estimators=100, random_state=42)
    clv_model.fit(X_train_clv, y_train_clv)
    
    # Generate predictions
    churn_proba = churn_model.predict_proba(X_scaled)[:, 1]
    clv_pred = clv_model.predict(X_scaled)
    
    # Model performance
    churn_accuracy = accuracy_score(y_test_churn, churn_model.predict(X_test_churn))
    clv_r2 = r2_score(y_test_clv, clv_model.predict(X_test_clv))
    
    # Feature importance
    churn_importance = dict(zip(X.columns, churn_model.feature_importances_))
    clv_importance = dict(zip(X.columns, clv_model.feature_importances_))
    
    return {
        'churn_proba': churn_proba,
        'clv_pred': clv_pred,
        'churn_accuracy': churn_accuracy,
        'clv_r2': clv_r2,
        'churn_importance': churn_importance,
        'clv_importance': clv_importance
    }

def calculate_rfm_scores(df):
    """Calculate RFM analysis"""
    
    # RFM scores (1-5 scale)
    r_score = pd.qcut(df['recency'].rank(method='first'), 
                     q=5, labels=[5, 4, 3, 2, 1])  # Lower recency = higher score
    f_score = pd.qcut(df['purchase_frequency'].rank(method='first'), 
                     q=5, labels=[1, 2, 3, 4, 5])  # Higher frequency = higher score
    m_score = pd.qcut(df['total_spent'].rank(method='first'), 
                     q=5, labels=[1, 2, 3, 4, 5])  # Higher monetary = higher score
    
    # Convert to numeric
    r_score = pd.to_numeric(r_score)
    f_score = pd.to_numeric(f_score)
    m_score = pd.to_numeric(m_score)
    
    # Calculate weighted RFM score
    rfm_score = (r_score * 0.3 + f_score * 0.4 + m_score * 0.3)
    
    # Assign RFM segments
    segments = []
    for i in range(len(df)):
        r, f, m = r_score.iloc[i], f_score.iloc[i], m_score.iloc[i]
        
        if r >= 4 and f >= 4 and m >= 4:
            segment = 'Champions'
        elif r >= 3 and f >= 3 and m >= 3:
            segment = 'Loyal Customers'
        elif r >= 4 and f <= 2:
            segment = 'New Customers'
        elif r >= 3 and f >= 3 and m <= 2:
            segment = 'Potential Loyalists'
        elif r <= 2 and f >= 3 and m >= 3:
            segment = 'At Risk'
        elif r <= 2 and f >= 4 and m >= 4:
            segment = 'Cannot Lose Them'
        elif r <= 2 and f <= 2 and m >= 3:
            segment = 'Hibernating'
        elif r <= 2 and f <= 2 and m <= 2:
            segment = 'Lost'
        else:
            segment = 'Others'
        
        segments.append(segment)
    
    return {
        'r_score': r_score,
        'f_score': f_score,
        'm_score': m_score,
        'rfm_score': rfm_score,
        'rfm_segment': segments
    }

def perform_clustering(df):
    """Perform customer clustering"""
    from sklearn.cluster import KMeans
    
    # Select features for clustering
    cluster_features = ['total_spent', 'purchase_frequency', 'recency', 'satisfaction_score']
    X_cluster = df[cluster_features]
    
    # Scale features
    X_cluster_scaled = StandardScaler().fit_transform(X_cluster)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_cluster_scaled)
    
    return cluster_labels

def generate_business_insights(df):
    """Generate business insights"""
    
    insights = {
        'total_customers': len(df),
        'total_revenue': df['total_spent'].sum(),
        'avg_clv': df['total_spent'].mean(),
        'churn_rate': (df['churn'].sum() / len(df)) * 100,
        'avg_satisfaction': df['satisfaction_score'].mean(),
        'high_value_customers': len(df[df['total_spent'] > df['total_spent'].quantile(0.8)]),
        'avg_purchase_frequency': df['purchase_frequency'].mean(),
        'avg_recency': df['recency'].mean(),
        'mobile_app_adoption': (df['has_mobile_app'].sum() / len(df)) * 100,
        'social_media_followers': (df['social_media_follower'].sum() / len(df)) * 100
    }
    
    return insights

def main():
    """Generate all datasets and model outputs"""
    
    print("Generating customer dataset...")
    df = generate_customer_data(5000)
    
    print("Training ML models...")
    ml_results = train_ml_models(df)
    
    print("Calculating RFM analysis...")
    rfm_results = calculate_rfm_scores(df)
    
    print("Performing customer clustering...")
    cluster_labels = perform_clustering(df)
    
    print("Generating business insights...")
    insights = generate_business_insights(df)
    
    # Create enhanced dataset
    df_enhanced = df.copy()
    df_enhanced['churn_probability'] = ml_results['churn_proba']
    df_enhanced['predicted_clv'] = ml_results['clv_pred']
    df_enhanced['r_score'] = rfm_results['r_score']
    df_enhanced['f_score'] = rfm_results['f_score']
    df_enhanced['m_score'] = rfm_results['m_score']
    df_enhanced['rfm_score'] = rfm_results['rfm_score']
    df_enhanced['rfm_segment'] = rfm_results['rfm_segment']
    df_enhanced['ml_cluster'] = cluster_labels
    
    # Save datasets
    print("Saving datasets...")
    
    # 1. Main customer dataset
    df.to_csv('customer_data.csv', index=False)
    
    # 2. Enhanced dataset with ML predictions
    df_enhanced.to_csv('customer_data_enhanced.csv', index=False)
    
    # 3. Model performance metrics
    model_metrics = {
        'Churn Model Accuracy': ml_results['churn_accuracy'],
        'CLV Model R²': ml_results['clv_r2']
    }
    
    model_df = pd.DataFrame([model_metrics])
    model_df.to_csv('model_performance.csv', index=False)
    
    # 4. Feature importance
    churn_importance_df = pd.DataFrame(
        list(ml_results['churn_importance'].items()),
        columns=['Feature', 'Churn_Importance']
    ).sort_values('Churn_Importance', ascending=False)
    
    clv_importance_df = pd.DataFrame(
        list(ml_results['clv_importance'].items()),
        columns=['Feature', 'CLV_Importance']
    ).sort_values('CLV_Importance', ascending=False)
    
    feature_importance = churn_importance_df.merge(
        clv_importance_df, on='Feature', how='outer'
    )
    feature_importance.to_csv('feature_importance.csv', index=False)
    
    # 5. RFM segment analysis
    rfm_summary = df_enhanced.groupby('rfm_segment').agg({
        'customer_id': 'count',
        'total_spent': ['mean', 'sum'],
        'purchase_frequency': 'mean',
        'recency': 'mean',
        'satisfaction_score': 'mean',
        'churn_probability': 'mean',
        'rfm_score': 'mean'
    }).round(2)
    
    rfm_summary.columns = ['Customer_Count', 'Avg_CLV', 'Total_Revenue', 
                          'Avg_Frequency', 'Avg_Recency', 'Avg_Satisfaction',
                          'Avg_Churn_Risk', 'Avg_RFM_Score']
    rfm_summary.to_csv('rfm_segment_analysis.csv')
    
    # 6. Business insights
    insights_df = pd.DataFrame([insights]).T
    insights_df.columns = ['Value']
    insights_df.to_csv('business_insights.csv')
    
    # 7. Customer clustering analysis
    cluster_summary = df_enhanced.groupby('ml_cluster').agg({
        'customer_id': 'count',
        'total_spent': ['mean', 'sum'],
        'purchase_frequency': 'mean',
        'satisfaction_score': 'mean',
        'churn_probability': 'mean'
    }).round(2)
    
    cluster_summary.columns = ['Customer_Count', 'Avg_CLV', 'Total_Revenue',
                              'Avg_Frequency', 'Avg_Satisfaction', 'Avg_Churn_Risk']
    cluster_summary.to_csv('customer_clusters.csv')
    
    # 8. High-risk customers
    high_risk = df_enhanced[df_enhanced['churn_probability'] > 0.7].copy()
    high_risk = high_risk.sort_values('churn_probability', ascending=False)
    high_risk.to_csv('high_risk_customers.csv', index=False)
    
    # 9. Top customers by CLV
    top_customers = df_enhanced.nlargest(100, 'total_spent')
    top_customers.to_csv('top_customers.csv', index=False)
    
    print("All datasets generated successfully!")
    print(f"\\nFiles created:")
    print(f"• customer_data.csv - Original dataset ({len(df)} customers)")
    print(f"• customer_data_enhanced.csv - Dataset with ML predictions")
    print(f"• model_performance.csv - Model accuracy metrics")
    print(f"• feature_importance.csv - Feature importance rankings")
    print(f"• rfm_segment_analysis.csv - RFM segment performance")
    print(f"• business_insights.csv - Key business metrics")
    print(f"• customer_clusters.csv - ML clustering analysis")
    print(f"• high_risk_customers.csv - Customers at risk of churning")
    print(f"• top_customers.csv - Top 100 customers by CLV")
    
    print(f"\\nKey Statistics:")
    print(f"• Total Revenue: ${insights['total_revenue']:,.2f}")
    print(f"• Average CLV: ${insights['avg_clv']:.2f}")
    print(f"• Churn Rate: {insights['churn_rate']:.1f}%")
    print(f"• Model Accuracy: Churn={ml_results['churn_accuracy']:.1%}, CLV R²={ml_results['clv_r2']:.3f}")

if __name__ == "__main__":
    main()