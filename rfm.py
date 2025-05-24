"""
RFM segmentation module for the recommendation system using K-means clustering.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import os

logger = logging.getLogger("FedPerRecommender")

class RFMSegmentation:
    """Handles customer segmentation using RFM analysis and K-means clustering."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.kmeans = None
        self.model_path = "rfm_kmeans_model.pkl"
        self.scaler_path = "rfm_scaler.pkl"
        
    def compute_rfm(self, df, reference_date=None):
        """Calculate RFM metrics and segment customers using K-means."""
        logger.info("Computing RFM metrics with K-means segmentation...")
        
        # Use provided reference date or get the latest date plus one day
        if reference_date is None:
            reference_date = df['InvoiceDate'].max() + timedelta(days=1)
        
        # Group by customer and calculate metrics
        rfm = df.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (reference_date - x.max()).days,  # Recency
            'InvoiceNo': 'nunique',                                    # Frequency
            'UnitPrice': lambda x: (x * df.loc[x.index, 'Quantity']).sum()  # Monetary
        })
        
        rfm.columns = ['Recency', 'Frequency', 'Monetary']
        
        # Perform K-means clustering
        segmented_rfm = self._apply_kmeans_clustering(rfm)
        
        # Show distribution
        segment_dist = segmented_rfm['Segment'].value_counts()
        logger.info("Customer segments distribution:")
        for segment, count in segment_dist.items():
            logger.info(f"  {segment}: {count} customers")
        
        return segmented_rfm.reset_index()  # Reset to make CustomerID a column
    
    def _apply_kmeans_clustering(self, rfm):
        """Apply K-means clustering to RFM data."""
        # Create a copy to avoid modifying the original
        rfm_data = rfm.copy()
        
        # Prepare data for clustering
        cluster_data = rfm_data[['Recency', 'Frequency', 'Monetary']].copy()
        
        # Handle zeros in Monetary and Frequency with small offset to allow log transformation
        cluster_data['Monetary'] = cluster_data['Monetary'] + 1  # Add 1 to handle zero values
        cluster_data['Frequency'] = cluster_data['Frequency'] + 1  # Add 1 to handle zero values
        
        # Apply log transformation to reduce skewness
        cluster_data['Recency_Log'] = np.log1p(cluster_data['Recency'])
        cluster_data['Frequency_Log'] = np.log1p(cluster_data['Frequency'])
        cluster_data['Monetary_Log'] = np.log1p(cluster_data['Monetary'])
        
        # Use transformed values for clustering
        features = cluster_data[['Recency_Log', 'Frequency_Log', 'Monetary_Log']]
        
        # Scale the data
        if os.path.exists(self.scaler_path):
            try:
                self.scaler = joblib.load(self.scaler_path)
                scaled_data = self.scaler.transform(features)
                logger.info("Loaded existing scaler for consistent scaling")
            except Exception as e:
                logger.warning(f"Error loading scaler, fitting new one: {e}")
                scaled_data = self.scaler.fit_transform(features)
                joblib.dump(self.scaler, self.scaler_path)
        else:
            scaled_data = self.scaler.fit_transform(features)
            try:
                joblib.dump(self.scaler, self.scaler_path)
                logger.info("Saved scaler for future use")
            except Exception as e:
                logger.warning(f"Could not save scaler: {e}")
        
        # Load or train K-means model
        if os.path.exists(self.model_path):
            try:
                self.kmeans = joblib.load(self.model_path)
                logger.info(f"Loaded existing K-means model with {self.kmeans.n_clusters} clusters")
            except Exception as e:
                logger.warning(f"Error loading model, training new one: {e}")
                self._train_kmeans_model(scaled_data)
        else:
            self._train_kmeans_model(scaled_data)
        
        # Apply clustering
        cluster_labels = self.kmeans.predict(scaled_data)
        rfm_data['Cluster'] = cluster_labels
        
        # Analyze clusters and assign meaningful names
        rfm_data['Segment'] = self._get_segment_names(rfm_data)
        
        return rfm_data
    
    def _train_kmeans_model(self, scaled_data):
        """Train K-means model with optimal number of clusters."""
        # Find optimal number of clusters using elbow method
        inertias = []
        k_range = range(2, 12)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(scaled_data)
            inertias.append(kmeans.inertia_)
        
        # Find elbow point using second derivative method
        if len(inertias) > 3:
            # First derivative (negative of the slope)
            deltas = np.diff(inertias)
            # Second derivative 
            delta_deltas = np.diff(deltas)
            # Elbow is where second derivative is maximum
            elbow_idx = np.argmax(delta_deltas) + 1
            optimal_k = k_range[elbow_idx]
            
            # Bound to reasonable range
            optimal_k = max(min(optimal_k, 8), 3)
        else:
            # Default fallback
            optimal_k = 5
            
        logger.info(f"Selected optimal number of clusters: {optimal_k}")
        
        # Train final model with optimal clusters
        self.kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        self.kmeans.fit(scaled_data)
        
        # Save model
        try:
            joblib.dump(self.kmeans, self.model_path)
            logger.info("K-means clustering model saved")
        except Exception as e:
            logger.warning(f"Could not save K-means model: {e}")
    
    def _get_segment_names(self, clustered_rfm):
        """Assign meaningful names to clusters based on their characteristics."""
        # Calculate cluster centers in the original RFM space
        cluster_summary = clustered_rfm.groupby('Cluster').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean'
        })
        
        # Normalize values relative to all clusters (0-1 scale for comparison)
        normalized = cluster_summary.copy()
        for col in ['Recency', 'Frequency', 'Monetary']:
            min_val = cluster_summary[col].min()
            max_val = cluster_summary[col].max()
            range_val = max_val - min_val
            if range_val > 0:
                normalized[col] = (cluster_summary[col] - min_val) / range_val
            else:
                normalized[col] = 0.5  # Default if all values are the same
        
        # Invert Recency so higher is better (lower actual recency)
        normalized['Recency'] = 1 - normalized['Recency']
        
        # Calculate RFM scores (higher is better for all)
        normalized['R_Score'] = normalized['Recency'] 
        normalized['F_Score'] = normalized['Frequency']
        normalized['M_Score'] = normalized['Monetary']
        normalized['Total_Score'] = normalized['R_Score'] + normalized['F_Score'] + normalized['M_Score']
        
        # Create mapping for segment names
        segment_mapping = {}
        
        for cluster_id, row in normalized.iterrows():
            r_score = row['R_Score']
            f_score = row['F_Score']
            m_score = row['M_Score']
            total = row['Total_Score']
            
            # Classification thresholds
            high = 0.75
            medium = 0.45
            low = 0.25
            
            # Determine segment based on RFM characteristics
            if r_score >= high and f_score >= high and m_score >= high:
                segment = "Champions"
            elif r_score >= high and f_score >= medium and m_score >= medium:
                segment = "Loyal Customers"
            elif r_score <= low and f_score >= medium and m_score >= medium:
                segment = "At Risk"
            elif r_score <= low and f_score <= low and m_score <= low:
                segment = "Lost Customers"
            elif r_score <= low and f_score >= high and m_score >= medium:
                segment = "Can't Lose Them"
            elif r_score >= high and f_score <= low and m_score <= medium:
                segment = "New Customers"
            elif r_score >= medium and f_score <= low and m_score <= low:
                segment = "Promising"
            elif r_score <= medium and r_score > low and f_score <= medium:
                segment = "Need Attention"
            elif r_score <= low and f_score <= low and m_score >= medium:
                segment = "About To Sleep"
            elif r_score >= medium and f_score >= medium and m_score <= low:
                segment = "Potential Loyalists"
            else:
                # Calculate relative strength
                strengths = [r_score, f_score, m_score]
                max_score = max(strengths)
                max_idx = strengths.index(max_score)
                
                if max_idx == 0:
                    segment = "Recent Shoppers"
                elif max_idx == 1:
                    segment = "Frequent Shoppers"
                else:
                    segment = "Big Spenders"
            
            segment_mapping[cluster_id] = segment
            
        # Add additional info to logs
        logger.info("Cluster characteristics and assigned segments:")
        for cluster_id, segment_name in segment_mapping.items():
            r = cluster_summary.loc[cluster_id, 'Recency']
            f = cluster_summary.loc[cluster_id, 'Frequency']
            m = cluster_summary.loc[cluster_id, 'Monetary']
            logger.info(f"  Cluster {cluster_id}: R={r:.1f}, F={f:.1f}, M={m:.1f} -> '{segment_name}'")
        
        # Map cluster IDs to segment names
        return clustered_rfm['Cluster'].map(segment_mapping)
    
    def update_segments(self, df, rfm_table, new_transactions):
        """Update RFM segments with new transaction data."""
        # Ensure new_transactions has required columns
        required_cols = ['InvoiceDate', 'CustomerID', 'InvoiceNo', 'UnitPrice', 'Quantity']
        missing_cols = [col for col in required_cols if col not in new_transactions.columns]
        
        if missing_cols:
            logger.error(f"Missing columns in new transactions: {missing_cols}")
            return rfm_table
        
        # Preprocess new transactions
        new_transactions['InvoiceDate'] = pd.to_datetime(new_transactions['InvoiceDate'])
        new_transactions['CustomerID'] = pd.to_numeric(new_transactions['CustomerID'], errors='coerce')
        
        # Get reference date (current date + 1 day to make "today" = 0 days ago)
        reference_date = datetime.now() + timedelta(days=1)
        
        # Combine with existing data for customers with new transactions
        updated_customers = new_transactions['CustomerID'].unique()
        
        # For existing customers in RFM table
        existing_customer_mask = rfm_table['CustomerID'].isin(updated_customers)
        existing_customers = rfm_table.loc[existing_customer_mask, 'CustomerID'].unique()
        
        if len(existing_customers) > 0:
            # Get transactions for existing customers to update
            customer_df = df[df['CustomerID'].isin(existing_customers)]
            combined_df = pd.concat([customer_df, new_transactions[new_transactions['CustomerID'].isin(existing_customers)]])
            
            # Recompute RFM for these customers
            updated_rfm = self.compute_rfm(combined_df, reference_date)
            
            # Update existing customer records
            for idx, row in updated_rfm.iterrows():
                customer_id = row['CustomerID']
                # Get all columns except CustomerID for update
                update_cols = [col for col in updated_rfm.columns if col != 'CustomerID']
                
                # Ensure all columns exist in rfm_table
                missing_cols = [col for col in update_cols if col not in rfm_table.columns]
                if missing_cols:
                    for col in missing_cols:
                        rfm_table[col] = None
                
                # Update the customer's record
                rfm_table.loc[rfm_table['CustomerID'] == customer_id, update_cols] = row[update_cols].values
            
            logger.info(f"Updated RFM for {len(existing_customers)} existing customers")
        
        # For new customers not in RFM table
        new_customer_mask = ~new_transactions['CustomerID'].isin(rfm_table['CustomerID'])
        new_customers = new_transactions.loc[new_customer_mask, 'CustomerID'].unique()
        
        if len(new_customers) > 0:
            # Compute RFM for new customers
            new_customer_df = new_transactions[new_transactions['CustomerID'].isin(new_customers)]
            new_rfm = self.compute_rfm(new_customer_df, reference_date)
            
            # Ensure new_rfm has same columns as rfm_table
            for col in rfm_table.columns:
                if col not in new_rfm.columns:
                    new_rfm[col] = None
                    
            for col in new_rfm.columns:
                if col not in rfm_table.columns:
                    rfm_table[col] = None
            
            # Append to RFM table
            common_cols = list(set(rfm_table.columns) & set(new_rfm.columns))
            rfm_table = pd.concat([rfm_table[common_cols], new_rfm[common_cols]])
            
            logger.info(f"Added {len(new_customers)} new customers to RFM table")
        
        return rfm_table
    
    def get_segment_migration(self, old_rfm, new_rfm):
        """
        Analyze how customers migrated between segments.
        Returns a migration matrix and a list of notable migrations.
        """
        if 'Segment' not in old_rfm.columns or 'Segment' not in new_rfm.columns:
            logger.error("RFM tables must contain Segment column")
            return None, []
        
        # Merge the old and new RFM tables
        migration_df = old_rfm[['CustomerID', 'Segment']].merge(
            new_rfm[['CustomerID', 'Segment']],
            on='CustomerID',
            how='inner',
            suffixes=('_old', '_new')
        )
        
        # Create migration matrix
        segments = list(set(old_rfm['Segment'].unique()) | set(new_rfm['Segment'].unique()))
        segments.sort()
        
        matrix = pd.DataFrame(0, index=segments, columns=segments)
        
        for _, row in migration_df.iterrows():
            matrix.loc[row['Segment_old'], row['Segment_new']] += 1
        
        # Convert to percentages (row-wise)
        matrix_pct = matrix.div(matrix.sum(axis=1), axis=0).fillna(0) * 100
        
        # Find notable migrations (movements of more than 10% of a segment)
        notable = []
        for old_segment in segments:
            for new_segment in segments:
                if old_segment != new_segment and matrix_pct.loc[old_segment, new_segment] > 10:
                    count = matrix.loc[old_segment, new_segment]
                    pct = matrix_pct.loc[old_segment, new_segment]
                    notable.append({
                        'from': old_segment,
                        'to': new_segment,
                        'count': int(count),
                        'percentage': float(pct)
                    })
        
        # Sort by percentage
        notable.sort(key=lambda x: x['percentage'], reverse=True)
        
        return matrix, notable
    
    def predict_churn_risk(self, rfm_table):
        """
        Predict churn risk for each customer based on RFM metrics and segment.
        Adds a 'ChurnRisk' column to the RFM table with values:
        'High', 'Medium', 'Low'
        """
        # Copy RFM table to avoid modifying the original
        rfm = rfm_table.copy()
        
        # If required columns don't exist, create them
        required_cols = ['Recency', 'Frequency', 'Monetary']
        for col in required_cols:
            if col not in rfm.columns:
                logger.warning(f"Required column {col} missing for churn prediction")
                return rfm
        
        # Get overall statistics for normalization
        r_max = rfm['Recency'].max()
        f_med = rfm['Frequency'].median()
        m_med = rfm['Monetary'].median()
        
        # Calculate churn risk based on RFM values
        def get_churn_risk(row):
            # Recency is the strongest predictor - normalized to 0-1 scale
            r_norm = row['Recency'] / r_max if r_max > 0 else 0
            
            # High recency (hasn't bought recently) = higher churn risk
            if r_norm > 0.7:  # Hasn't purchased in a long time
                if row['Frequency'] < f_med and row['Monetary'] < m_med:
                    return 'High'
                elif row['Frequency'] < f_med or row['Monetary'] < m_med:
                    return 'High'
                else:
                    return 'Medium'
            elif r_norm > 0.4:  # Moderate recency
                if row['Frequency'] < f_med and row['Monetary'] < m_med:
                    return 'Medium'
                elif row['Segment'] in ['About To Sleep', 'Need Attention', 'At Risk']:
                    return 'Medium'
                else:
                    return 'Low'
            else:  # Recent purchaser
                if row['Frequency'] < f_med/2 and row['Monetary'] < m_med/2:
                    return 'Medium'
                elif row['Segment'] in ['New Customers', 'Promising']:
                    return 'Medium'
                else:
                    return 'Low'
        
        rfm['ChurnRisk'] = rfm.apply(get_churn_risk, axis=1)
        
        # Log the distribution
        risk_dist = rfm['ChurnRisk'].value_counts()
        logger.info(f"Churn risk distribution: {dict(risk_dist)}")
        
        return rfm