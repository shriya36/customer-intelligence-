"""
Report generation utilities for Customer Intelligence Dashboard
"""

import pandas as pd
import numpy as np
from fpdf import FPDF
import io
from datetime import datetime
import xlsxwriter
from xlsxwriter.utility import xl_range

class CustomerIntelligencePDF(FPDF):
    """
    Custom PDF class for Customer Intelligence Reports
    """
    
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Customer Intelligence Report', 0, 1, 'C')
        self.set_font('Arial', '', 10)
        self.cell(0, 5, f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1, 'C')
        self.ln(10)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    
    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)
    
    def chapter_body(self, body):
        self.set_font('Arial', '', 11)
        # Split text into lines
        lines = body.split('\n')
        for line in lines:
            self.cell(0, 6, line.encode('latin-1', 'replace').decode('latin-1'), 0, 1)
        self.ln()

def generate_pdf_report(data, ml_models):
    """
    Generate comprehensive PDF report
    
    Parameters:
    data (pd.DataFrame): Customer dataset
    ml_models (CustomerMLModels): Trained ML models
    
    Returns:
    bytes: PDF report as bytes
    """
    
    pdf = CustomerIntelligencePDF()
    pdf.add_page()
    
    # Executive Summary
    pdf.chapter_title('Executive Summary')
    
    total_customers = len(data)
    total_revenue = data['total_spent'].sum()
    avg_clv = data['total_spent'].mean()
    churn_rate = data['churn'].mean()
    avg_satisfaction = data['satisfaction_score'].mean()
    
    # Get ML predictions
    churn_proba = ml_models.predict_churn_proba()
    high_risk_customers = np.sum(churn_proba > 0.7)
    
    summary_text = f"""
Customer Base Overview:
- Total Customers: {total_customers:,}
- Total Revenue: ${total_revenue:,.2f}
- Average Customer Lifetime Value: ${avg_clv:.2f}
- Current Churn Rate: {churn_rate:.1%}
- Average Satisfaction Score: {avg_satisfaction:.1f}/10

Key Risk Indicators:
- High-Risk Customers (>70% churn probability): {high_risk_customers}
- At-Risk Revenue: ${data[churn_proba > 0.7]['total_spent'].sum():.2f}

Model Performance Summary:
- Churn Prediction Accuracy: {ml_models.get_churn_metrics()['Random Forest']['accuracy']:.3f}
- CLV Prediction R² Score: {ml_models.get_clv_metrics()['Random Forest']['r2']:.3f}
- Customer Segments Identified: {len(np.unique(ml_models.perform_clustering()))}
"""
    
    pdf.chapter_body(summary_text)
    
    # Customer Segmentation Analysis
    pdf.chapter_title('Customer Segmentation Analysis')
    
    segments = ml_models.perform_clustering()
    segment_summary = ml_models.get_segment_summary()
    
    segmentation_text = "Customer Segments Distribution:\n"
    for segment, count in segment_summary.items():
        percentage = (count / total_customers) * 100
        segmentation_text += f"- {segment}: {count} customers ({percentage:.1f}%)\n"
    
    pdf.chapter_body(segmentation_text)
    
    # Churn Analysis
    pdf.chapter_title('Churn Risk Analysis')
    
    # Risk distribution
    low_risk = np.sum(churn_proba < 0.3)
    medium_risk = np.sum((churn_proba >= 0.3) & (churn_proba < 0.7))
    high_risk = np.sum(churn_proba >= 0.7)
    
    churn_text = f"""
Churn Risk Distribution:
- Low Risk (<30%): {low_risk} customers ({low_risk/total_customers:.1%})
- Medium Risk (30-70%): {medium_risk} customers ({medium_risk/total_customers:.1%})
- High Risk (>70%): {high_risk} customers ({high_risk/total_customers:.1%})

Top Risk Factors:
"""
    
    # Feature importance
    feature_importance = ml_models.get_churn_feature_importance()
    top_features = list(feature_importance.items())[:5]
    
    for feature, importance in top_features:
        churn_text += f"- {feature}: {importance:.3f}\n"
    
    pdf.chapter_body(churn_text)
    
    # CLV Analysis
    pdf.chapter_title('Customer Lifetime Value Analysis')
    
    clv_pred = ml_models.predict_clv()
    
    # CLV segments
    clv_quartiles = np.percentile(clv_pred, [25, 50, 75])
    
    clv_text = f"""
CLV Distribution:
- 25th Percentile: ${clv_quartiles[0]:.2f}
- Median (50th Percentile): ${clv_quartiles[1]:.2f}
- 75th Percentile: ${clv_quartiles[2]:.2f}
- Average Predicted CLV: ${clv_pred.mean():.2f}

High-Value Customers (Top 20%):
- Count: {int(total_customers * 0.2)}
- Minimum CLV: ${np.percentile(clv_pred, 80):.2f}
- Total Predicted Value: ${np.sum(clv_pred[clv_pred >= np.percentile(clv_pred, 80)]):.2f}

Key CLV Drivers:
"""
    
    clv_importance = ml_models.get_clv_feature_importance()
    top_clv_features = list(clv_importance.items())[:5]
    
    for feature, importance in top_clv_features:
        clv_text += f"- {feature}: {importance:.3f}\n"
    
    pdf.chapter_body(clv_text)
    
    # Recommendations
    pdf.chapter_title('Strategic Recommendations')
    
    recommendations_text = """
1. Churn Prevention:
   - Implement immediate outreach for high-risk customers
   - Focus on improving satisfaction scores for at-risk segments
   - Develop retention campaigns based on key risk factors

2. Customer Development:
   - Identify upselling opportunities in medium-value segments
   - Create loyalty programs for high-value customers
   - Improve onboarding for new customer acquisition

3. Revenue Optimization:
   - Focus marketing spend on high-CLV potential customers
   - Develop targeted pricing strategies by segment
   - Optimize acquisition channels based on CLV performance

4. Operational Improvements:
   - Monitor key satisfaction drivers to prevent churn
   - Implement predictive alerts for customer risk changes
   - Regular model retraining to maintain accuracy
"""
    
    pdf.chapter_body(recommendations_text)
    
    # Convert to bytes
    pdf_output = io.BytesIO()
    pdf_string = pdf.output(dest='S').encode('latin1')
    pdf_output.write(pdf_string)
    pdf_output.seek(0)
    
    return pdf_output.getvalue()

def create_excel_export(data, ml_models):
    """
    Create comprehensive Excel export with multiple sheets
    
    Parameters:
    data (pd.DataFrame): Customer dataset
    ml_models (CustomerMLModels): Trained ML models
    
    Returns:
    bytes: Excel file as bytes
    """
    
    # Create in-memory Excel file
    output = io.BytesIO()
    workbook = xlsxwriter.Workbook(output, {'in_memory': True})
    
    # Define formats
    header_format = workbook.add_format({
        'bold': True,
        'text_wrap': True,
        'valign': 'top',
        'fg_color': '#D7E4BC',
        'border': 1
    })
    
    number_format = workbook.add_format({'num_format': '#,##0.00'})
    percent_format = workbook.add_format({'num_format': '0.0%'})
    
    # Sheet 1: Enhanced Customer Data
    enhanced_data = data.copy()
    enhanced_data['churn_probability'] = ml_models.predict_churn_proba()
    enhanced_data['predicted_clv'] = ml_models.predict_clv()
    enhanced_data['ml_segment'] = ml_models.perform_clustering()
    enhanced_data['risk_score'] = ml_models.get_customer_risk_score()
    
    worksheet1 = workbook.add_worksheet('Customer Data')
    
    # Write headers
    for col, header in enumerate(enhanced_data.columns):
        worksheet1.write(0, col, header, header_format)
    
    # Write data
    for row, (_, customer) in enumerate(enhanced_data.iterrows(), 1):
        for col, value in enumerate(customer):
            if isinstance(value, (int, float)):
                worksheet1.write(row, col, value, number_format)
            else:
                worksheet1.write(row, col, value)
    
    # Sheet 2: Model Performance
    worksheet2 = workbook.add_worksheet('Model Performance')
    
    # Churn model metrics
    churn_metrics = ml_models.get_churn_metrics()
    clv_metrics = ml_models.get_clv_metrics()
    
    row = 0
    worksheet2.write(row, 0, 'Churn Prediction Models', header_format)
    row += 1
    
    worksheet2.write(row, 0, 'Model', header_format)
    worksheet2.write(row, 1, 'Accuracy', header_format)
    worksheet2.write(row, 2, 'Precision', header_format)
    worksheet2.write(row, 3, 'Recall', header_format)
    worksheet2.write(row, 4, 'F1-Score', header_format)
    row += 1
    
    for model_name, metrics in churn_metrics.items():
        worksheet2.write(row, 0, model_name)
        worksheet2.write(row, 1, metrics['accuracy'], percent_format)
        worksheet2.write(row, 2, metrics['precision'], percent_format)
        worksheet2.write(row, 3, metrics['recall'], percent_format)
        worksheet2.write(row, 4, metrics['f1_score'], percent_format)
        row += 1
    
    row += 2
    worksheet2.write(row, 0, 'CLV Prediction Models', header_format)
    row += 1
    
    worksheet2.write(row, 0, 'Model', header_format)
    worksheet2.write(row, 1, 'R² Score', header_format)
    worksheet2.write(row, 2, 'RMSE', header_format)
    worksheet2.write(row, 3, 'MAE', header_format)
    row += 1
    
    for model_name, metrics in clv_metrics.items():
        worksheet2.write(row, 0, model_name)
        worksheet2.write(row, 1, metrics['r2'], percent_format)
        worksheet2.write(row, 2, metrics['rmse'], number_format)
        worksheet2.write(row, 3, metrics['mae'], number_format)
        row += 1
    
    # Sheet 3: Feature Importance
    worksheet3 = workbook.add_worksheet('Feature Importance')
    
    churn_importance = ml_models.get_churn_feature_importance()
    clv_importance = ml_models.get_clv_feature_importance()
    
    # Churn feature importance
    worksheet3.write(0, 0, 'Churn Prediction Features', header_format)
    worksheet3.write(1, 0, 'Feature', header_format)
    worksheet3.write(1, 1, 'Importance', header_format)
    
    row = 2
    for feature, importance in churn_importance.items():
        worksheet3.write(row, 0, feature)
        worksheet3.write(row, 1, importance, number_format)
        row += 1
    
    # CLV feature importance
    start_col = 3
    worksheet3.write(0, start_col, 'CLV Prediction Features', header_format)
    worksheet3.write(1, start_col, 'Feature', header_format)
    worksheet3.write(1, start_col + 1, 'Importance', header_format)
    
    row = 2
    for feature, importance in clv_importance.items():
        worksheet3.write(row, start_col, feature)
        worksheet3.write(row, start_col + 1, importance, number_format)
        row += 1
    
    # Sheet 4: Segment Analysis
    worksheet4 = workbook.add_worksheet('Segment Analysis')
    
    segments = ml_models.perform_clustering()
    segment_summary = ml_models.get_segment_summary()
    
    # Add segment analysis data
    data_with_segments = data.copy()
    data_with_segments['segment'] = segments
    
    segment_stats = data_with_segments.groupby('segment').agg({
        'customer_id': 'count',
        'total_spent': ['mean', 'sum'],
        'purchase_frequency': 'mean',
        'recency': 'mean',
        'satisfaction_score': 'mean',
        'churn': 'mean'
    }).round(2)
    
    # Flatten column names
    segment_stats.columns = ['Customer_Count', 'Avg_CLV', 'Total_Revenue', 
                           'Avg_Frequency', 'Avg_Recency', 'Avg_Satisfaction', 'Churn_Rate']
    segment_stats = segment_stats.reset_index()
    
    # Write segment data
    for col, header in enumerate(segment_stats.columns):
        worksheet4.write(0, col, header, header_format)
    
    for row, (_, segment_data) in enumerate(segment_stats.iterrows(), 1):
        for col, value in enumerate(segment_data):
            if col in [1, 2, 3, 4, 5] and isinstance(value, (int, float)):  # Numeric columns
                worksheet4.write(row, col, value, number_format)
            elif col == 6 and isinstance(value, (int, float)):  # Churn rate
                worksheet4.write(row, col, value, percent_format)
            else:
                worksheet4.write(row, col, value)
    
    # Close workbook and return bytes
    workbook.close()
    output.seek(0)
    
    return output.getvalue()

def generate_summary_report(data, ml_models):
    """
    Generate a summary dictionary for quick insights
    
    Parameters:
    data (pd.DataFrame): Customer dataset
    ml_models (CustomerMLModels): Trained ML models
    
    Returns:
    dict: Summary report dictionary
    """
    
    churn_proba = ml_models.predict_churn_proba()
    clv_pred = ml_models.predict_clv()
    segments = ml_models.perform_clustering()
    
    summary = {
        'customer_metrics': {
            'total_customers': len(data),
            'total_revenue': data['total_spent'].sum(),
            'average_clv': data['total_spent'].mean(),
            'churn_rate': data['churn'].mean(),
            'average_satisfaction': data['satisfaction_score'].mean()
        },
        'risk_analysis': {
            'high_risk_customers': np.sum(churn_proba > 0.7),
            'medium_risk_customers': np.sum((churn_proba >= 0.3) & (churn_proba < 0.7)),
            'low_risk_customers': np.sum(churn_proba < 0.3),
            'at_risk_revenue': data[churn_proba > 0.7]['total_spent'].sum()
        },
        'clv_insights': {
            'predicted_total_clv': clv_pred.sum(),
            'high_value_customers': np.sum(clv_pred > np.percentile(clv_pred, 80)),
            'clv_75th_percentile': np.percentile(clv_pred, 75),
            'clv_median': np.percentile(clv_pred, 50)
        },
        'segmentation': {
            'total_segments': len(np.unique(segments)),
            'segment_distribution': dict(zip(*np.unique(segments, return_counts=True)))
        },
        'model_performance': {
            'churn_accuracy': ml_models.get_churn_metrics()['Random Forest']['accuracy'],
            'clv_r2_score': ml_models.get_clv_metrics()['Random Forest']['r2']
        },
        'generated_at': datetime.now().isoformat()
    }
    
    return summary