"""
RFM (Recency, Frequency, Monetary) Analysis for Customer Segmentation
"""

import pandas as pd
import numpy as np
from datetime import datetime

def calculate_rfm_scores(df):
    """
    Calculate RFM scores for customer segmentation
    
    Parameters:
    df (pd.DataFrame): Customer dataset with recency, purchase_frequency, and total_spent
    
    Returns:
    dict: Dictionary containing R, F, M scores and overall RFM score
    """
    
    # Ensure we have the required columns
    required_columns = ['recency', 'purchase_frequency', 'total_spent']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in dataset")
    
    # Calculate RFM scores (1-5 scale)
    # Recency: Lower recency (more recent) = higher score
    r_score = pd.qcut(df['recency'].rank(method='first'), 
                     q=5, labels=[5, 4, 3, 2, 1], duplicates='drop')
    
    # Frequency: Higher frequency = higher score
    f_score = pd.qcut(df['purchase_frequency'].rank(method='first'), 
                     q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
    
    # Monetary: Higher monetary value = higher score
    m_score = pd.qcut(df['total_spent'].rank(method='first'), 
                     q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
    
    # Convert to numeric
    r_score = pd.to_numeric(r_score, errors='coerce').fillna(3)
    f_score = pd.to_numeric(f_score, errors='coerce').fillna(3)
    m_score = pd.to_numeric(m_score, errors='coerce').fillna(3)
    
    # Calculate weighted RFM score
    rfm_score = (r_score * 0.3 + f_score * 0.4 + m_score * 0.3)
    
    return {
        'r_score': r_score,
        'f_score': f_score,
        'm_score': m_score,
        'rfm_score': rfm_score
    }

def create_rfm_segments(rfm_scores):
    """
    Create meaningful customer segments based on RFM scores
    
    Parameters:
    rfm_scores (dict): Dictionary containing RFM scores
    
    Returns:
    list: List of segment names for each customer
    """
    
    r_score = rfm_scores['r_score']
    f_score = rfm_scores['f_score']
    m_score = rfm_scores['m_score']
    
    segments = []
    
    for i in range(len(r_score)):
        r, f, m = r_score.iloc[i], f_score.iloc[i], m_score.iloc[i]
        
        # Define segments based on RFM combinations
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
    
    return segments

def get_rfm_segment_descriptions():
    """
    Get descriptions and recommendations for each RFM segment
    
    Returns:
    dict: Dictionary with segment descriptions and recommendations
    """
    
    descriptions = {
        'Champions': {
            'description': 'Your best customers who bought recently, buy often and spend the most',
            'characteristics': ['High recency', 'High frequency', 'High monetary'],
            'recommendations': [
                'Reward them for their loyalty',
                'Ask for referrals and reviews',
                'Offer exclusive products or early access',
                'Make them brand ambassadors'
            ],
            'marketing_focus': 'Retention and advocacy'
        },
        
        'Loyal Customers': {
            'description': 'Good customers who buy regularly and spend decent amounts',
            'characteristics': ['Good recency', 'Good frequency', 'Good monetary'],
            'recommendations': [
                'Upsell higher value products',
                'Ask for product reviews',
                'Engage them in loyalty programs',
                'Offer personalized recommendations'
            ],
            'marketing_focus': 'Upselling and cross-selling'
        },
        
        'Potential Loyalists': {
            'description': 'Recent customers with potential for higher frequency and spend',
            'characteristics': ['Good recency', 'Good frequency', 'Low monetary'],
            'recommendations': [
                'Offer membership or loyalty programs',
                'Recommend higher-value products',
                'Increase engagement through content',
                'Provide personalized offers'
            ],
            'marketing_focus': 'Development and growth'
        },
        
        'New Customers': {
            'description': 'Customers who have made recent purchases but low frequency',
            'characteristics': ['High recency', 'Low frequency', 'Varies monetary'],
            'recommendations': [
                'Start building relationships',
                'Provide onboarding support',
                'Offer welcome series and education',
                'Focus on product satisfaction'
            ],
            'marketing_focus': 'Onboarding and education'
        },
        
        'At Risk': {
            'description': 'Customers who used to be valuable but haven\'t purchased recently',
            'characteristics': ['Low recency', 'High frequency', 'High monetary'],
            'recommendations': [
                'Send personalized reactivation campaigns',
                'Offer limited-time promotions',
                'Ask for feedback on their experience',
                'Provide incentives to return'
            ],
            'marketing_focus': 'Reactivation and win-back'
        },
        
        'Cannot Lose Them': {
            'description': 'High-value customers who haven\'t purchased in a while',
            'characteristics': ['Low recency', 'Very high frequency', 'Very high monetary'],
            'recommendations': [
                'Win them back via renewals or newer products',
                'Offer exclusive deals or VIP treatment',
                'Conduct personal outreach',
                'Address any service issues immediately'
            ],
            'marketing_focus': 'High-priority retention'
        },
        
        'Hibernating': {
            'description': 'Customers with low recent activity but some past engagement',
            'characteristics': ['Low recency', 'Low frequency', 'Good monetary'],
            'recommendations': [
                'Re-engage with special offers',
                'Share valuable content and updates',
                'Conduct surveys to understand needs',
                'Offer product recommendations'
            ],
            'marketing_focus': 'Re-engagement'
        },
        
        'Lost': {
            'description': 'Customers with the lowest recency, frequency, and monetary scores',
            'characteristics': ['Very low recency', 'Very low frequency', 'Very low monetary'],
            'recommendations': [
                'Revive interest with relevant content',
                'Offer significant discounts or incentives',
                'Consider removing from active campaigns',
                'Focus resources on higher-value segments'
            ],
            'marketing_focus': 'Revival or removal'
        },
        
        'Others': {
            'description': 'Customers who don\'t fit clearly into other segments',
            'characteristics': ['Mixed scores'],
            'recommendations': [
                'Analyze individual customer patterns',
                'Provide targeted communications',
                'Monitor for segment migration',
                'Apply general best practices'
            ],
            'marketing_focus': 'Personalized approach'
        }
    }
    
    return descriptions

def analyze_rfm_trends(df, rfm_scores, time_period_days=90):
    """
    Analyze RFM trends and segment movements
    
    Parameters:
    df (pd.DataFrame): Customer dataset
    rfm_scores (dict): RFM scores
    time_period_days (int): Time period for trend analysis
    
    Returns:
    dict: Analysis results
    """
    
    segments = create_rfm_segments(rfm_scores)
    segment_counts = pd.Series(segments).value_counts()
    
    # Calculate segment percentages
    segment_percentages = (segment_counts / len(segments) * 100).round(1)
    
    # Identify high-value segments
    high_value_segments = ['Champions', 'Loyal Customers', 'Cannot Lose Them']
    high_value_customers = sum([segment_percentages.get(seg, 0) for seg in high_value_segments])
    
    # Identify at-risk segments
    at_risk_segments = ['At Risk', 'Hibernating', 'Lost']
    at_risk_customers = sum([segment_percentages.get(seg, 0) for seg in at_risk_segments])
    
    # Calculate business impact metrics
    df_with_segments = df.copy()
    df_with_segments['rfm_segment'] = segments
    
    segment_metrics = df_with_segments.groupby('rfm_segment').agg({
        'total_spent': ['mean', 'sum', 'count'],
        'purchase_frequency': 'mean',
        'satisfaction_score': 'mean',
        'churn': 'mean'
    }).round(2)
    
    # Flatten column names
    segment_metrics.columns = ['_'.join(col).strip() for col in segment_metrics.columns]
    
    analysis_results = {
        'segment_distribution': segment_counts.to_dict(),
        'segment_percentages': segment_percentages.to_dict(),
        'high_value_percentage': high_value_customers,
        'at_risk_percentage': at_risk_customers,
        'segment_metrics': segment_metrics.to_dict(),
        'total_customers': len(segments),
        'total_revenue': df['total_spent'].sum(),
        'analysis_date': datetime.now().strftime('%Y-%m-%d')
    }
    
    return analysis_results

def create_rfm_action_plan(segment_analysis):
    """
    Create actionable recommendations based on RFM analysis
    
    Parameters:
    segment_analysis (dict): Results from analyze_rfm_trends
    
    Returns:
    dict: Action plan with priorities and recommendations
    """
    
    segment_dist = segment_analysis['segment_distribution']
    descriptions = get_rfm_segment_descriptions()
    
    action_plan = {
        'priority_actions': [],
        'segment_actions': {},
        'overall_strategy': {}
    }
    
    # Determine priority actions based on segment sizes
    total_customers = segment_analysis['total_customers']
    
    for segment, count in segment_dist.items():
        percentage = (count / total_customers) * 100
        
        if segment in descriptions:
            action_plan['segment_actions'][segment] = {
                'count': count,
                'percentage': round(percentage, 1),
                'description': descriptions[segment]['description'],
                'recommendations': descriptions[segment]['recommendations'],
                'marketing_focus': descriptions[segment]['marketing_focus']
            }
            
            # Add to priority actions if significant segment
            if percentage >= 10:  # Segments with 10% or more of customers
                action_plan['priority_actions'].append({
                    'segment': segment,
                    'percentage': round(percentage, 1),
                    'priority': 'High' if percentage >= 20 else 'Medium',
                    'focus': descriptions[segment]['marketing_focus']
                })
    
    # Overall strategy recommendations
    high_value_pct = segment_analysis['high_value_percentage']
    at_risk_pct = segment_analysis['at_risk_percentage']
    
    strategy_recommendations = []
    
    if high_value_pct < 30:
        strategy_recommendations.append("Focus on customer development and loyalty programs to increase high-value segments")
    
    if at_risk_pct > 25:
        strategy_recommendations.append("Implement immediate retention campaigns for at-risk customers")
    
    if segment_dist.get('New Customers', 0) / total_customers > 0.15:
        strategy_recommendations.append("Strengthen onboarding processes to convert new customers to loyal ones")
    
    action_plan['overall_strategy'] = {
        'high_value_percentage': high_value_pct,
        'at_risk_percentage': at_risk_pct,
        'recommendations': strategy_recommendations
    }
    
    return action_plan
