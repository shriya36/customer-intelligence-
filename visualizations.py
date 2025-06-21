"""
Visualization utilities for Customer Intelligence Dashboard
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_churn_analysis_plots(data, churn_proba=None):
    """
    Create comprehensive churn analysis visualizations
    
    Parameters:
    data (pd.DataFrame): Customer dataset
    churn_proba (np.array): Churn probabilities from ML model
    
    Returns:
    dict: Dictionary of plotly figures
    """
    
    plots = {}
    
    # Churn rate by satisfaction score
    churn_by_satisfaction = data.groupby('satisfaction_score')['churn'].mean().reset_index()
    plots['churn_by_satisfaction'] = px.line(
        churn_by_satisfaction, 
        x='satisfaction_score', 
        y='churn',
        title='Churn Rate by Satisfaction Score',
        labels={'churn': 'Churn Rate', 'satisfaction_score': 'Satisfaction Score'}
    )
    
    # Churn rate by recency
    data_temp = data.copy()
    data_temp['recency_bucket'] = pd.cut(data_temp['recency'], bins=5, labels=['0-73', '74-146', '147-219', '220-292', '293-365'])
    churn_by_recency = data_temp.groupby('recency_bucket')['churn'].mean().reset_index()
    plots['churn_by_recency'] = px.bar(
        churn_by_recency,
        x='recency_bucket',
        y='churn',
        title='Churn Rate by Recency (Days Since Last Purchase)',
        labels={'churn': 'Churn Rate', 'recency_bucket': 'Days Since Last Purchase'}
    )
    
    # Churn distribution by acquisition channel
    churn_by_channel = data.groupby('acquisition_channel')['churn'].agg(['mean', 'count']).reset_index()
    plots['churn_by_channel'] = px.scatter(
        churn_by_channel,
        x='count',
        y='mean',
        size='count',
        hover_name='acquisition_channel',
        title='Churn Rate by Acquisition Channel',
        labels={'mean': 'Churn Rate', 'count': 'Customer Count'}
    )
    
    if churn_proba is not None:
        # Churn probability distribution
        plots['churn_probability_dist'] = px.histogram(
            x=churn_proba,
            nbins=30,
            title='Distribution of Churn Probabilities',
            labels={'x': 'Churn Probability', 'y': 'Number of Customers'}
        )
    
    return plots

def create_clv_analysis_plots(data, clv_pred=None):
    """
    Create CLV analysis visualizations
    
    Parameters:
    data (pd.DataFrame): Customer dataset
    clv_pred (np.array): CLV predictions from ML model
    
    Returns:
    dict: Dictionary of plotly figures
    """
    
    plots = {}
    
    # CLV by age groups
    data_temp = data.copy()
    data_temp['age_group'] = pd.cut(data_temp['age'], bins=[18, 30, 40, 50, 60, 80], labels=['18-30', '31-40', '41-50', '51-60', '60+'])
    clv_by_age = data_temp.groupby('age_group')['total_spent'].agg(['mean', 'count']).reset_index()
    plots['clv_by_age'] = px.bar(
        clv_by_age,
        x='age_group',
        y='mean',
        title='Average CLV by Age Group',
        labels={'mean': 'Average CLV ($)', 'age_group': 'Age Group'}
    )
    
    # CLV vs Purchase Frequency
    plots['clv_vs_frequency'] = px.scatter(
        data,
        x='purchase_frequency',
        y='total_spent',
        color='satisfaction_score',
        title='CLV vs Purchase Frequency',
        labels={'total_spent': 'CLV ($)', 'purchase_frequency': 'Purchase Frequency'}
    )
    
    # CLV distribution by acquisition channel
    plots['clv_by_channel'] = px.box(
        data,
        x='acquisition_channel',
        y='total_spent',
        title='CLV Distribution by Acquisition Channel',
        labels={'total_spent': 'CLV ($)'}
    )
    
    if clv_pred is not None:
        # Actual vs Predicted CLV
        plots['actual_vs_predicted'] = px.scatter(
            x=data['total_spent'],
            y=clv_pred,
            title='Actual vs Predicted CLV',
            labels={'x': 'Actual CLV ($)', 'y': 'Predicted CLV ($)'}
        )
        # Add perfect prediction line
        min_val = min(data['total_spent'].min(), clv_pred.min())
        max_val = max(data['total_spent'].max(), clv_pred.max())
        plots['actual_vs_predicted'].add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='Perfect Prediction', line=dict(dash='dash'))
        )
    
    return plots

def create_segmentation_plots(data, segments):
    """
    Create customer segmentation visualizations
    
    Parameters:
    data (pd.DataFrame): Customer dataset
    segments (np.array): Segment labels
    
    Returns:
    dict: Dictionary of plotly figures
    """
    
    plots = {}
    
    # Add segments to data
    data_with_segments = data.copy()
    data_with_segments['segment'] = segments
    
    # Segment distribution
    segment_counts = pd.Series(segments).value_counts()
    plots['segment_distribution'] = px.pie(
        values=segment_counts.values,
        names=[f'Segment {i}' for i in segment_counts.index],
        title='Customer Segment Distribution'
    )
    
    # 3D scatter plot of key metrics by segment
    plots['segment_3d'] = px.scatter_3d(
        data_with_segments,
        x='total_spent',
        y='purchase_frequency',
        z='satisfaction_score',
        color='segment',
        title='Customer Segments in 3D Space',
        labels={
            'total_spent': 'Total Spent ($)',
            'purchase_frequency': 'Purchase Frequency',
            'satisfaction_score': 'Satisfaction Score'
        }
    )
    
    # Segment characteristics heatmap
    segment_stats = data_with_segments.groupby('segment').agg({
        'total_spent': 'mean',
        'purchase_frequency': 'mean',
        'recency': 'mean',
        'satisfaction_score': 'mean',
        'age': 'mean'
    }).round(2)
    
    # Normalize for heatmap
    segment_stats_norm = (segment_stats - segment_stats.min()) / (segment_stats.max() - segment_stats.min())
    
    plots['segment_heatmap'] = px.imshow(
        segment_stats_norm.T,
        labels=dict(x="Segment", y="Metric", color="Normalized Value"),
        x=[f'Segment {i}' for i in segment_stats.index],
        y=segment_stats.columns,
        title='Segment Characteristics Heatmap'
    )
    
    return plots

def create_rfm_plots(data, rfm_scores, rfm_segments):
    """
    Create RFM analysis visualizations
    
    Parameters:
    data (pd.DataFrame): Customer dataset
    rfm_scores (dict): RFM scores
    rfm_segments (list): RFM segment labels
    
    Returns:
    dict: Dictionary of plotly figures
    """
    
    plots = {}
    
    # RFM segment distribution
    segment_counts = pd.Series(rfm_segments).value_counts()
    plots['rfm_segments'] = px.pie(
        values=segment_counts.values,
        names=segment_counts.index,
        title='RFM Segment Distribution'
    )
    
    # RFM scores correlation
    rfm_df = pd.DataFrame(rfm_scores)
    correlation_matrix = rfm_df[['r_score', 'f_score', 'm_score']].corr()
    plots['rfm_correlation'] = px.imshow(
        correlation_matrix,
        title='RFM Scores Correlation',
        labels=dict(color="Correlation"),
        color_continuous_scale='RdBu'
    )
    
    # 3D RFM plot
    plots['rfm_3d'] = px.scatter_3d(
        x=rfm_scores['r_score'],
        y=rfm_scores['f_score'],
        z=rfm_scores['m_score'],
        color=rfm_segments,
        title='RFM Analysis in 3D Space',
        labels={
            'x': 'Recency Score',
            'y': 'Frequency Score',
            'z': 'Monetary Score'
        }
    )
    
    # RFM segment value analysis
    data_with_rfm = data.copy()
    data_with_rfm['rfm_segment'] = rfm_segments
    
    segment_value = data_with_rfm.groupby('rfm_segment').agg({
        'total_spent': ['sum', 'mean', 'count']
    }).round(2)
    segment_value.columns = ['Total_Revenue', 'Avg_CLV', 'Customer_Count']
    segment_value = segment_value.reset_index()
    
    plots['segment_value'] = px.scatter(
        segment_value,
        x='Customer_Count',
        y='Total_Revenue',
        size='Avg_CLV',
        hover_name='rfm_segment',
        title='RFM Segment Business Value',
        labels={
            'Customer_Count': 'Number of Customers',
            'Total_Revenue': 'Total Revenue ($)'
        }
    )
    
    return plots

def create_performance_metrics_plot(metrics_dict, model_type):
    """
    Create model performance comparison visualization
    
    Parameters:
    metrics_dict (dict): Dictionary of model metrics
    model_type (str): Type of model (e.g., 'Churn Prediction', 'CLV Prediction')
    
    Returns:
    plotly.graph_objects.Figure: Performance comparison plot
    """
    
    if model_type == 'Churn Prediction':
        metric_names = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    else:  # CLV Prediction
        metric_names = ['r2', 'rmse', 'mae']
        metric_labels = ['R² Score', 'RMSE', 'MAE']
    
    fig = go.Figure()
    
    for model_name, metrics in metrics_dict.items():
        if model_type == 'Churn Prediction':
            values = [metrics[metric] for metric in metric_names]
        else:
            # For regression metrics, normalize RMSE and MAE to 0-1 scale for comparison
            values = [
                metrics['r2'],
                1 - min(metrics['rmse'] / 5000, 1),  # Normalize RMSE
                1 - min(metrics['mae'] / 2000, 1)    # Normalize MAE
            ]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metric_labels if model_type == 'Churn Prediction' else ['R² Score', 'RMSE (Inverted)', 'MAE (Inverted)'],
            fill='toself',
            name=model_name
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title=f'{model_type} Model Comparison'
    )
    
    return fig

def create_business_overview_plots(data):
    """
    Create business overview dashboard plots
    
    Parameters:
    data (pd.DataFrame): Customer dataset
    
    Returns:
    dict: Dictionary of plotly figures
    """
    
    plots = {}
    
    # Customer acquisition trend (simulated monthly data)
    months = pd.date_range(start='2023-01-01', end='2024-12-01', freq='MS')
    acquisition_data = pd.DataFrame({
        'month': months,
        'new_customers': np.random.poisson(len(data) / 24, len(months))
    })
    
    plots['acquisition_trend'] = px.line(
        acquisition_data,
        x='month',
        y='new_customers',
        title='Customer Acquisition Trend',
        labels={'new_customers': 'New Customers', 'month': 'Month'}
    )
    
    # Revenue distribution
    plots['revenue_distribution'] = px.histogram(
        data,
        x='total_spent',
        nbins=30,
        title='Customer Lifetime Value Distribution',
        labels={'total_spent': 'CLV ($)', 'count': 'Number of Customers'}
    )
    
    # Satisfaction vs Churn
    plots['satisfaction_churn'] = px.box(
        data,
        x='churn',
        y='satisfaction_score',
        title='Satisfaction Score by Churn Status',
        labels={'churn': 'Churn Status (0=Active, 1=Churned)', 'satisfaction_score': 'Satisfaction Score'}
    )
    
    # Geographic distribution (if city data available)
    if 'city' in data.columns:
        city_stats = data.groupby('city').agg({
            'customer_id': 'count',
            'total_spent': 'sum'
        }).reset_index()
        city_stats.columns = ['city', 'customer_count', 'total_revenue']
        
        plots['geographic_distribution'] = px.bar(
            city_stats.sort_values('customer_count', ascending=True),
            x='customer_count',
            y='city',
            orientation='h',
            title='Customer Distribution by City',
            labels={'customer_count': 'Number of Customers', 'city': 'City'}
        )
    
    return plots
