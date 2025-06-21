"""
Customer Intelligence Dashboard
A comprehensive web application for customer analytics and machine learning insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import io
from datetime import datetime

# Import custom modules
from ml_models import CustomerMLModels
from data_generator import generate_customer_data
from rfm_analysis import calculate_rfm_scores, create_rfm_segments
from visualizations import (
    create_churn_analysis_plots, create_clv_analysis_plots,
    create_segmentation_plots, create_rfm_plots, create_performance_metrics_plot
)
from report_generator import generate_pdf_report, create_excel_export

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Customer Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'ml_models' not in st.session_state:
    st.session_state.ml_models = None
if 'customer_data' not in st.session_state:
    st.session_state.customer_data = None

def load_sample_data():
    """Load sample customer data"""
    with st.spinner("Generating sample customer data..."):
        data = generate_customer_data(2000)
        st.session_state.customer_data = data
        st.session_state.data_loaded = True
        st.success("Sample data loaded successfully!")
        st.rerun()

def train_models():
    """Train ML models on the loaded data"""
    if st.session_state.customer_data is not None:
        with st.spinner("Training machine learning models..."):
            try:
                ml_models = CustomerMLModels(st.session_state.customer_data)
                st.session_state.ml_models = ml_models
                st.success("Models trained successfully!")
                return True
            except Exception as e:
                st.error(f"Error training models: {str(e)}")
                return False
    return False

def main():
    # Header
    st.title("🎯 Customer Intelligence Dashboard")
    st.markdown("**Comprehensive ML-powered customer analytics for business growth**")
    st.divider()

    # Sidebar for data management
    with st.sidebar:
        st.header("📁 Data Management")
        
        # Data upload section
        uploaded_file = st.file_uploader(
            "Upload Customer Data (CSV)",
            type=['csv'],
            help="Upload a CSV file with customer data"
        )
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.session_state.customer_data = data
                st.session_state.data_loaded = True
                st.success(f"Data loaded: {len(data)} customers")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
        
        # Sample data button
        if st.button("🎲 Generate Sample Data", use_container_width=True):
            load_sample_data()
        
        # Model training section
        if st.session_state.data_loaded:
            st.divider()
            st.header("🤖 ML Models")
            
            if st.session_state.ml_models is None:
                if st.button("🚀 Train Models", use_container_width=True):
                    train_models()
            else:
                st.success("✅ Models Ready")
                if st.button("🔄 Retrain Models", use_container_width=True):
                    train_models()
        
        # Data summary
        if st.session_state.data_loaded:
            st.divider()
            st.header("📊 Data Summary")
            data = st.session_state.customer_data
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Customers", f"{len(data):,}")
                st.metric("Churn Rate", f"{data['churn'].mean():.1%}")
            with col2:
                st.metric("Avg CLV", f"${data['total_spent'].mean():.0f}")
                st.metric("Avg Satisfaction", f"{data['satisfaction_score'].mean():.1f}/10")

    # Main content area
    if not st.session_state.data_loaded:
        # Welcome screen
        st.markdown("""
        ## Welcome to Customer Intelligence Dashboard
        
        This comprehensive analytics platform provides:
        
        - 🎯 **Churn Prediction**: Identify customers at risk of leaving
        - 💰 **Customer Lifetime Value**: Predict future customer value
        - 👥 **Customer Segmentation**: Group customers for targeted marketing
        - 📈 **RFM Analysis**: Recency, Frequency, Monetary analysis
        - 📊 **Interactive Dashboards**: Real-time analytics and insights
        - 📄 **Exportable Reports**: PDF and Excel downloads
        
        **Get started by uploading your data or generating sample data from the sidebar.**
        """)
        
        # Features showcase
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🤖 Machine Learning
            - Random Forest Models
            - Logistic Regression
            - K-Means Clustering
            - Cross-validation
            """)
        
        with col2:
            st.markdown("""
            ### 📊 Analytics
            - Churn Analysis
            - CLV Forecasting
            - Customer Segments
            - Feature Importance
            """)
        
        with col3:
            st.markdown("""
            ### 📈 Visualizations
            - Interactive Charts
            - Business Dashboards
            - Performance Metrics
            - Export Options
            """)
    
    else:
        # Main dashboard tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Overview", "🎯 Churn Analysis", "💰 CLV Analysis", 
            "👥 Segmentation", "📈 RFM Analysis", "📄 Reports"
        ])
        
        data = st.session_state.customer_data
        ml_models = st.session_state.ml_models
        
        with tab1:
            st.header("Business Overview Dashboard")
            
            # Key metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            total_customers = len(data)
            total_revenue = data['total_spent'].sum()
            avg_clv = data['total_spent'].mean()
            churn_rate = data['churn'].mean()
            avg_satisfaction = data['satisfaction_score'].mean()
            
            with col1:
                st.metric("Total Customers", f"{total_customers:,}")
            with col2:
                st.metric("Total Revenue", f"${total_revenue:,.0f}")
            with col3:
                st.metric("Average CLV", f"${avg_clv:.0f}")
            with col4:
                st.metric("Churn Rate", f"{churn_rate:.1%}")
            with col5:
                st.metric("Avg Satisfaction", f"{avg_satisfaction:.1f}/10")
            
            st.divider()
            
            # Overview charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Customer distribution by age
                if 'age' in data.columns:
                    fig_age = px.histogram(
                        data, x='age', nbins=20,
                        title="Customer Age Distribution",
                        color_discrete_sequence=['#1f77b4']
                    )
                    fig_age.update_layout(showlegend=False)
                    st.plotly_chart(fig_age, use_container_width=True)
                else:
                    st.info("Age data not available")
                
                # Revenue by acquisition channel  
                if 'acquisition_channel' in data.columns and 'total_spent' in data.columns:
                    channel_data = data.groupby('acquisition_channel')['total_spent'].sum().reset_index()
                    channel_data = channel_data.sort_values('total_spent', ascending=True)
                    fig_channel = px.bar(
                        channel_data,
                        x='total_spent',
                        y='acquisition_channel',
                        orientation='h',
                        title="Revenue by Acquisition Channel",
                        color_discrete_sequence=['#2ca02c']
                    )
                    st.plotly_chart(fig_channel, use_container_width=True)
                else:
                    if 'gender' in data.columns:
                        segment_counts = pd.Series(data['gender']).value_counts()
                        fig_gender = px.pie(
                            values=segment_counts.values,
                            names=segment_counts.index,
                            title="Customer Distribution by Gender"
                        )
                        st.plotly_chart(fig_gender, use_container_width=True)
                    else:
                        st.info("Additional data not available")
            
            with col2:
                # CLV distribution
                if 'total_spent' in data.columns:
                    fig_clv = px.histogram(
                        data, x='total_spent', nbins=30,
                        title="Customer Lifetime Value Distribution",
                        color_discrete_sequence=['#ff7f0e']
                    )
                    fig_clv.update_layout(showlegend=False)
                    st.plotly_chart(fig_clv, use_container_width=True)
                else:
                    st.info("CLV data not available")
                
                # Satisfaction vs Churn
                if 'churn' in data.columns and 'satisfaction_score' in data.columns:
                    fig_sat = px.box(
                        data, x='churn', y='satisfaction_score',
                        title="Satisfaction Score by Churn Status",
                        color='churn',
                        color_discrete_map={0: '#2ca02c', 1: '#d62728'}
                    )
                    fig_sat.update_xaxes(tickvals=[0, 1], ticktext=['Active', 'Churned'])
                    st.plotly_chart(fig_sat, use_container_width=True)
                else:
                    st.info("Satisfaction/churn data not available")
        
        with tab2:
            st.header("🎯 Churn Prediction Analysis")
            
            if ml_models is not None:
                # Model performance metrics
                churn_metrics = ml_models.get_churn_metrics()
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Performance comparison
                    fig_metrics = create_performance_metrics_plot(churn_metrics, "Churn Prediction")
                    st.plotly_chart(fig_metrics, use_container_width=True)
                
                with col2:
                    st.subheader("Model Performance")
                    for model_name, metrics in churn_metrics.items():
                        st.markdown(f"**{model_name}**")
                        st.write(f"Accuracy: {metrics['accuracy']:.3f}")
                        st.write(f"Precision: {metrics['precision']:.3f}")
                        st.write(f"Recall: {metrics['recall']:.3f}")
                        st.write(f"F1-Score: {metrics['f1_score']:.3f}")
                        st.divider()
                
                # Feature importance
                feature_importance = ml_models.get_churn_feature_importance()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Feature importance chart
                    features = list(feature_importance.keys())[:10]
                    importance = list(feature_importance.values())[:10]
                    
                    fig_importance = px.bar(
                        x=importance, y=features,
                        orientation='h',
                        title="Top 10 Features for Churn Prediction",
                        color_discrete_sequence=['#d62728']
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)
                
                with col2:
                    # Churn risk distribution
                    churn_proba = ml_models.predict_churn_proba()
                    
                    fig_risk = px.histogram(
                        x=churn_proba, nbins=20,
                        title="Churn Risk Distribution",
                        labels={'x': 'Churn Probability', 'y': 'Number of Customers'},
                        color_discrete_sequence=['#ff7f0e']
                    )
                    st.plotly_chart(fig_risk, use_container_width=True)
                
                # High-risk customers table
                st.subheader("High-Risk Customers")
                
                # Add predictions to data
                data_with_predictions = data.copy()
                data_with_predictions['churn_probability'] = churn_proba
                
                high_risk = data_with_predictions[
                    data_with_predictions['churn_probability'] > 0.7
                ].sort_values('churn_probability', ascending=False)
                
                if len(high_risk) > 0:
                    st.dataframe(
                        high_risk[['customer_id', 'satisfaction_score', 'total_spent', 
                                 'recency', 'churn_probability']].head(10),
                        use_container_width=True
                    )
                else:
                    st.info("No high-risk customers identified (probability > 0.7)")
                    
            else:
                st.warning("Please train the ML models first using the sidebar.")
        
        with tab3:
            st.header("💰 Customer Lifetime Value Analysis")
            
            if ml_models is not None:
                # CLV model performance
                clv_metrics = ml_models.get_clv_metrics()
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Performance comparison
                    fig_metrics = create_performance_metrics_plot(clv_metrics, "CLV Prediction")
                    st.plotly_chart(fig_metrics, use_container_width=True)
                
                with col2:
                    st.subheader("Model Performance")
                    for model_name, metrics in clv_metrics.items():
                        st.markdown(f"**{model_name}**")
                        st.write(f"R² Score: {metrics['r2']:.3f}")
                        st.write(f"RMSE: ${metrics['rmse']:.2f}")
                        st.write(f"MAE: ${metrics['mae']:.2f}")
                        st.divider()
                
                # Feature importance for CLV
                clv_importance = ml_models.get_clv_feature_importance()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Feature importance chart
                    features = list(clv_importance.keys())[:10]
                    importance = list(clv_importance.values())[:10]
                    
                    fig_importance = px.bar(
                        x=importance, y=features,
                        orientation='h',
                        title="Top 10 Features for CLV Prediction",
                        color_discrete_sequence=['#2ca02c']
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)
                
                with col2:
                    # Actual vs Predicted CLV
                    clv_pred = ml_models.predict_clv()
                    
                    fig_scatter = px.scatter(
                        x=data['total_spent'], y=clv_pred,
                        title="Actual vs Predicted CLV",
                        labels={'x': 'Actual CLV', 'y': 'Predicted CLV'},
                        color_discrete_sequence=['#ff7f0e']
                    )
                    # Add diagonal line
                    min_val = min(data['total_spent'].min(), clv_pred.min())
                    max_val = max(data['total_spent'].max(), clv_pred.max())
                    fig_scatter.add_trace(go.Scatter(
                        x=[min_val, max_val], y=[min_val, max_val],
                        mode='lines', name='Perfect Prediction',
                        line=dict(dash='dash', color='red')
                    ))
                    st.plotly_chart(fig_scatter, use_container_width=True)
                
                # CLV segments
                st.subheader("CLV Customer Segments")
                
                # Create CLV segments
                data_with_clv = data.copy()
                data_with_clv['predicted_clv'] = clv_pred
                data_with_clv['clv_segment'] = pd.qcut(
                    data_with_clv['predicted_clv'], 
                    q=4, 
                    labels=['Low Value', 'Medium Value', 'High Value', 'Premium']
                )
                
                segment_summary = data_with_clv.groupby('clv_segment').agg({
                    'customer_id': 'count',
                    'predicted_clv': 'mean',
                    'satisfaction_score': 'mean',
                    'churn': 'mean'
                }).round(2)
                segment_summary.columns = ['Customer Count', 'Avg Predicted CLV', 'Avg Satisfaction', 'Churn Rate']
                
                st.dataframe(segment_summary, use_container_width=True)
                
            else:
                st.warning("Please train the ML models first using the sidebar.")
        
        with tab4:
            st.header("👥 Customer Segmentation")
            
            if ml_models is not None:
                # Get clustering results
                cluster_labels = ml_models.perform_clustering()
                segment_summary = ml_models.get_segment_summary()
                
                # Add cluster labels to data
                data_with_clusters = data.copy()
                data_with_clusters['segment'] = cluster_labels
                
                # Segment overview
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.subheader("Segment Distribution")
                    segment_counts = pd.Series(cluster_labels).value_counts().sort_index()
                    
                    fig_pie = px.pie(
                        values=segment_counts.values,
                        names=[f"Segment {i}" for i in segment_counts.index],
                        title="Customer Segments"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    st.subheader("Segment Characteristics")
                    
                    # Calculate segment characteristics
                    segment_stats = data_with_clusters.groupby('segment').agg({
                        'total_spent': 'mean',
                        'purchase_frequency': 'mean',
                        'recency': 'mean',
                        'satisfaction_score': 'mean',
                        'churn': 'mean'
                    }).round(2)
                    
                    st.dataframe(segment_stats, use_container_width=True)
                
                # Segment visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    # Spend vs Frequency by segment
                    fig_scatter = px.scatter(
                        data_with_clusters, 
                        x='purchase_frequency', 
                        y='total_spent',
                        color='segment',
                        title="Customer Segments: Spend vs Frequency",
                        hover_data=['satisfaction_score', 'recency']
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                
                with col2:
                    # Recency vs Satisfaction by segment
                    fig_scatter2 = px.scatter(
                        data_with_clusters,
                        x='recency',
                        y='satisfaction_score',
                        color='segment',
                        title="Customer Segments: Recency vs Satisfaction",
                        hover_data=['total_spent', 'purchase_frequency']
                    )
                    st.plotly_chart(fig_scatter2, use_container_width=True)
                
            else:
                st.warning("Please train the ML models first using the sidebar.")
        
        with tab5:
            st.header("📈 RFM Analysis")
            
            # Calculate RFM scores
            rfm_results = calculate_rfm_scores(data)
            rfm_segments = create_rfm_segments(rfm_results)
            
            # Add RFM data to customer data
            data_with_rfm = data.copy()
            for key, values in rfm_results.items():
                data_with_rfm[key] = values
            data_with_rfm['rfm_segment'] = rfm_segments
            
            # RFM Overview
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Avg Recency Score", f"{rfm_results['r_score'].mean():.1f}")
            with col2:
                st.metric("Avg Frequency Score", f"{rfm_results['f_score'].mean():.1f}")
            with col3:
                st.metric("Avg Monetary Score", f"{rfm_results['m_score'].mean():.1f}")
            
            # RFM Segment Distribution
            col1, col2 = st.columns(2)
            
            with col1:
                segment_counts = pd.Series(rfm_segments).value_counts()
                
                fig_rfm_pie = px.pie(
                    values=segment_counts.values,
                    names=segment_counts.index,
                    title="RFM Segment Distribution"
                )
                st.plotly_chart(fig_rfm_pie, use_container_width=True)
            
            with col2:
                # RFM segment characteristics
                rfm_segment_stats = data_with_rfm.groupby('rfm_segment').agg({
                    'customer_id': 'count',
                    'total_spent': 'mean',
                    'purchase_frequency': 'mean',
                    'recency': 'mean',
                    'churn': 'mean'
                }).round(2)
                rfm_segment_stats.columns = ['Count', 'Avg Spend', 'Avg Frequency', 'Avg Recency', 'Churn Rate']
                
                st.subheader("Segment Characteristics")
                st.dataframe(rfm_segment_stats, use_container_width=True)
            
            # RFM Score distributions
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fig_r = px.histogram(
                    x=rfm_results['r_score'],
                    title="Recency Score Distribution",
                    color_discrete_sequence=['#1f77b4']
                )
                st.plotly_chart(fig_r, use_container_width=True)
            
            with col2:
                fig_f = px.histogram(
                    x=rfm_results['f_score'],
                    title="Frequency Score Distribution",
                    color_discrete_sequence=['#ff7f0e']
                )
                st.plotly_chart(fig_f, use_container_width=True)
            
            with col3:
                fig_m = px.histogram(
                    x=rfm_results['m_score'],
                    title="Monetary Score Distribution",
                    color_discrete_sequence=['#2ca02c']
                )
                st.plotly_chart(fig_m, use_container_width=True)
            
            # Detailed RFM analysis
            st.subheader("Segment Recommendations")
            
            recommendations = {
                'Champions': '🏆 Your best customers! Reward them and ask for referrals.',
                'Loyal Customers': '❤️ Upsell higher value products and ask for reviews.',
                'Potential Loyalists': '⭐ Offer membership or loyalty programs.',
                'New Customers': '🆕 Start building relationships and provide onboarding support.',
                'At Risk': '⚠️ Send personalized reactivation campaigns.',
                'Cannot Lose Them': '🚨 Win them back via renewals or newer products.',
                'Hibernating': '😴 Re-engage with special offers or surveys.',
                'Lost': '💔 Revive interest with relevant content.'
            }
            
            for segment, recommendation in recommendations.items():
                if segment in segment_counts.index:
                    st.write(f"**{segment}** ({segment_counts[segment]} customers): {recommendation}")
        
        with tab6:
            st.header("📄 Reports & Export")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Data Export")
                
                # Prepare enhanced dataset for export
                if ml_models is not None:
                    enhanced_data = data.copy()
                    enhanced_data['churn_probability'] = ml_models.predict_churn_proba()
                    enhanced_data['predicted_clv'] = ml_models.predict_clv()
                    enhanced_data['ml_segment'] = ml_models.perform_clustering()
                    
                    # Add RFM data
                    rfm_results = calculate_rfm_scores(data)
                    for key, values in rfm_results.items():
                        enhanced_data[key] = values
                    enhanced_data['rfm_segment'] = create_rfm_segments(rfm_results)
                    
                    # CSV Export
                    csv = enhanced_data.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Enhanced Dataset (CSV)",
                        data=csv,
                        file_name=f"customer_intelligence_data_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    # Excel Export
                    excel_buffer = create_excel_export(enhanced_data, ml_models)
                    st.download_button(
                        label="📊 Download Complete Analysis (Excel)",
                        data=excel_buffer,
                        file_name=f"customer_intelligence_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                
                else:
                    # Basic CSV export
                    csv = data.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Dataset (CSV)",
                        data=csv,
                        file_name=f"customer_data_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            with col2:
                st.subheader("📄 PDF Report")
                
                if ml_models is not None:
                    if st.button("🔄 Generate PDF Report", use_container_width=True):
                        with st.spinner("Generating comprehensive PDF report..."):
                            try:
                                pdf_buffer = generate_pdf_report(data, ml_models)
                                st.download_button(
                                    label="📄 Download PDF Report",
                                    data=pdf_buffer,
                                    file_name=f"customer_intelligence_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                                    mime="application/pdf",
                                    use_container_width=True
                                )
                                st.success("PDF report generated successfully!")
                            except Exception as e:
                                st.error(f"Error generating PDF: {str(e)}")
                else:
                    st.info("Train ML models to generate comprehensive reports")
            
            # Report preview
            if ml_models is not None:
                st.divider()
                st.subheader("📋 Report Summary")
                
                # Business insights
                col1, col2, col3 = st.columns(3)
                
                churn_proba = ml_models.predict_churn_proba()
                high_risk_count = np.sum(churn_proba > 0.7)
                
                with col1:
                    st.metric("High Risk Customers", high_risk_count)
                    st.caption("Churn probability > 70%")
                
                with col2:
                    avg_predicted_clv = ml_models.predict_clv().mean()
                    st.metric("Avg Predicted CLV", f"${avg_predicted_clv:.0f}")
                    st.caption("ML model prediction")
                
                with col3:
                    segments = ml_models.get_segment_summary()
                    st.metric("Customer Segments", len(segments))
                    st.caption("ML clustering")

if __name__ == "__main__":
    main()
