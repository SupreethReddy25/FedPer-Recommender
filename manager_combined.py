"""
Recommendation manager module.
"""

import pandas as pd
import numpy as np
import os
import pickle
import logging
from datetime import datetime, timedelta
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder

from .data_processor import DataProcessor
from .rfm import RFMSegmentation
from .embeddings import ProductEmbeddings
from .recommender_model import FedPerRecommender

logger = logging.getLogger("FedPerRecommender")

class RecommendationManager:
    """Manages the training and recommendation processes."""
    
    def __init__(self, n_factors=50):
        """Initialize the recommendation manager."""
        self.data_processor = DataProcessor()
        self.rfm_segmentation = RFMSegmentation()
        self.embeddings_creator = ProductEmbeddings()
        self.n_factors = n_factors
        
        # Model storage
        self.segment_models = {}
        self.segment_encoders = {}
        self.global_model = None
        self.product_info = None
        self.product_embeddings = None
        self.word2vec_model = None
        self.df = None
        self.rfm_table = None
        
        # New customer tracking
        self.pending_customers = set()  # Customers waiting for segment assignment
        self.last_model_update = datetime.now()
        self.new_transactions = []  # Buffer for new transactions
        
        # Schedule settings
        self.update_frequency_days = 3  # Update models weekly
    
    def load_and_prepare_data(self, file_path, rfm_table_path=None):
        """Load and prepare the retail dataset."""
        # Load raw data
        df = self.data_processor.load_data(file_path)
        
        # Preprocess
        df = self.data_processor.preprocess(df)
        
        # Load or compute RFM segments
        if rfm_table_path and os.path.exists(rfm_table_path):
            rfm_table = pd.read_csv(rfm_table_path)
            rfm_table['CustomerID'] = pd.to_numeric(rfm_table['CustomerID'], errors='coerce')
            logger.info(f"RFM table loaded from {rfm_table_path}")
        else:
            logger.info("Computing RFM table...")
            rfm_table = self.rfm_segmentation.compute_rfm(df)
        
        # Merge RFM segments into dataframe
        df = df.merge(rfm_table[['CustomerID', 'Segment']], on='CustomerID', how='left')
        
        # Create indices
        df = self.data_processor.encode_indices(df)
        
        # Generate product embeddings
        product_embeddings = self.embeddings_creator.create_product_embeddings(df)
        self.word2vec_model = self.embeddings_creator.model
        
        # Create product info dictionary
        self.product_info = self.data_processor.get_product_info(df)
        
        # Store for later use
        self.df = df
        self.rfm_table = rfm_table
        self.product_embeddings = product_embeddings
        
        return df, rfm_table, product_embeddings
    
    def train_models(self, df, product_embeddings, epochs=10):
        """Train segmented models and a global model."""
        logger.info("Training FedPer recommendation models...")
        
        # Split data by segment
        segments = df['Segment'].dropna().unique()
        
        for segment in segments:
            logger.info(f"Training model for segment: {segment}")
            segment_data = df[df['Segment'] == segment].copy()
            
            # Skip if not enough data
            if len(segment_data) < 10:
                logger.info(f"  Skipping segment {segment}: insufficient data")
                continue
                
            # Create segment-specific customer encoding
            segment_encoder = LabelEncoder()
            segment_data['LocalCustomerIdx'] = segment_encoder.fit_transform(segment_data['CustomerID'])
            
            # Create interaction data (1 for purchase, 0 for no purchase)
            interactions = segment_data.groupby(['LocalCustomerIdx', 'ItemIdx'])['Quantity'].sum().reset_index()
            interactions['Rating'] = 1  # Implicit feedback - all purchases are positive signals
            
            # Get dimensions
            n_users = segment_data['LocalCustomerIdx'].nunique()
            n_items = len(self.data_processor.item_encoder.classes_)
            
            logger.info(f"  Users: {n_users}, Items: {n_items}, Interactions: {len(interactions)}")
            
            # Create and train model
            model = FedPerRecommender(n_factors=self.n_factors)
            try:
                model.fit_local(
                    interactions[['LocalCustomerIdx', 'ItemIdx', 'Rating']].values,
                    n_users=n_users,
                    n_items=n_items,
                    product_emb=product_embeddings,
                    item_encoder=self.data_processor.item_encoder,
                    epochs=epochs
                )
                self.segment_models[segment] = model
                self.segment_encoders[segment] = segment_encoder
                logger.info(f"  Model for {segment} trained successfully")
            except Exception as e:
                logger.error(f"  Error training model for {segment}: {str(e)}")
        
        # Create global model by aggregating segment models
        self.global_model = FedPerRecommender(n_factors=self.n_factors)
        
        if self.segment_models:
            self.global_model.aggregate(list(self.segment_models.values()))
            logger.info("Global model created by aggregating segment models")
        else:
            logger.error("No segment models available for aggregation!")
            
            # Train a global model directly if no segments available
            logger.info("Training global model directly...")
            global_interactions = df.groupby(['CustomerIdx', 'ItemIdx'])['Quantity'].sum().reset_index()
            global_interactions['Rating'] = 1
            
            n_users = df['CustomerIdx'].nunique()
            n_items = df['ItemIdx'].nunique()
            
            try:
                self.global_model.fit_local(
                    global_interactions[['CustomerIdx', 'ItemIdx', 'Rating']].values,
                    n_users=n_users,
                    n_items=n_items,
                    product_emb=product_embeddings,
                    item_encoder=self.data_processor.item_encoder,
                    epochs=epochs
                )
                logger.info("Global model trained successfully")
            except Exception as e:
                logger.error(f"Error training global model: {str(e)}")
                
        return self.segment_models, self.global_model
    
    def add_new_customer(self, customer_id):
        """Register a new customer in the system."""
        # Check if customer already exists
        if customer_id in self.data_processor.customer_encoder.classes_:
            logger.info(f"Customer {customer_id} already exists in the system")
            return False
        
        # Add to customer encoder
        self.data_processor.customer_encoder.classes_ = np.append(
            self.data_processor.customer_encoder.classes_, [customer_id]
        )
        
        # Add to pending customers list
        self.pending_customers.add(customer_id)
        logger.info(f"New customer {customer_id} added to the system")
        return True

    def record_transaction(self, transaction_data):
        """
        Record a new transaction in the system.
        
        Args:
            transaction_data: dict with keys:
                - customer_id: Customer identifier
                - invoice_no: Invoice number
                - invoice_date: Date of transaction
                - stock_code: Product code
                - description: Product description
                - quantity: Number of items
                - unit_price: Price per unit
        """
        # Validate input
        required_fields = ['customer_id', 'invoice_no', 'invoice_date', 
                           'stock_code', 'description', 'quantity', 'unit_price']
        
        for field in required_fields:
            if field not in transaction_data:
                logger.error(f"Missing required field: {field}")
                return False
        
        # Convert to proper format
        transaction = {
            'CustomerID': transaction_data['customer_id'],
            'InvoiceNo': transaction_data['invoice_no'],
            'InvoiceDate': pd.to_datetime(transaction_data['invoice_date']),
            'StockCode': str(transaction_data['stock_code']),
            'Description': transaction_data['description'],
            'Quantity': int(transaction_data['quantity']),
            'UnitPrice': float(transaction_data['unit_price'])
        }
        
        # Add to new transactions buffer
        self.new_transactions.append(transaction)
        
        # Track in appropriate models for recommendation
        try:
            # Get indices
            customer_idx = -1
            if transaction['CustomerID'] in self.data_processor.customer_encoder.classes_:
                customer_idx = self.data_processor.customer_encoder.transform([transaction['CustomerID']])[0]
            
            # Add stock code if it's new
            if transaction['StockCode'] not in self.data_processor.item_encoder.classes_:
                self.data_processor.item_encoder.classes_ = np.append(
                    self.data_processor.item_encoder.classes_, [transaction['StockCode']]
                )
                
                # Update product info
                self.product_info[transaction['StockCode']] = {
                    'description': transaction['Description'],
                    'median_price': transaction['UnitPrice'],
                    'total_sold': transaction['Quantity']
                }
                
                # Update embeddings if possible
                if self.word2vec_model:
                    words = str(transaction['Description']).lower().split()
                    valid_words = [w for w in words if w in self.word2vec_model.wv]
                    
                    if valid_words:
                        vector_size = list(self.product_embeddings.values())[0].shape[0]
                        self.product_embeddings[transaction['StockCode']] = np.mean(
                            [self.word2vec_model.wv[w] for w in valid_words], axis=0
                        )[:vector_size]
                    else:
                        vector_size = list(self.product_embeddings.values())[0].shape[0]
                        self.product_embeddings[transaction['StockCode']] = np.zeros(vector_size)
            
            item_idx = -1
            if transaction['StockCode'] in self.data_processor.item_encoder.classes_:
                item_idx = self.data_processor.item_encoder.transform([transaction['StockCode']])[0]
            
            # Update global model's purchase tracking
            if customer_idx >= 0 and item_idx >= 0 and self.global_model:
                self.global_model.track_purchase(
                    customer_idx, 
                    item_idx,
                    transaction['InvoiceDate'],
                    transaction['Quantity']
                )
            
            # Check if customer has a segment and update segment model
            if transaction['CustomerID'] in self.rfm_table['CustomerID'].values:
                segment = self.rfm_table[
                    self.rfm_table['CustomerID'] == transaction['CustomerID']
                ]['Segment'].iloc[0]
                
                if segment in self.segment_models and segment in self.segment_encoders:
                    # Get local customer index
                    segment_encoder = self.segment_encoders[segment]
                    if transaction['CustomerID'] in segment_encoder.classes_:
                        local_idx = segment_encoder.transform([transaction['CustomerID']])[0]
                        
                        # Update segment model's purchase tracking
                        if item_idx >= 0:
                            self.segment_models[segment].track_purchase(
                                local_idx,
                                item_idx,
                                transaction['InvoiceDate'],
                                transaction['Quantity']
                            )
        except Exception as e:
            logger.error(f"Error processing transaction: {str(e)}")
        
        # Check if we need to update models
        if len(self.new_transactions) >= 100 or (datetime.now() - self.last_model_update).days >= self.update_frequency_days:
            self.update_models()
        
        return True

    def update_models(self):
        """Update models with new transaction data."""
        if not self.new_transactions:
            logger.info("No new transactions to process")
            return False
        
        logger.info(f"Updating models with {len(self.new_transactions)} new transactions")
        
        # Convert transactions to DataFrame
        new_df = pd.DataFrame(self.new_transactions)
        
        # Update RFM table
        self.rfm_table = self.rfm_segmentation.update_segments(
            self.df, self.rfm_table, new_df
        )
        
        # Process new customers that now have segments
        newly_segmented = []
        for customer_id in list(self.pending_customers):
            if customer_id in self.rfm_table['CustomerID'].values:
                segment = self.rfm_table[
                    self.rfm_table['CustomerID'] == customer_id
                ]['Segment'].iloc[0]
                
                # Add to segment encoder and remove from pending
                if segment in self.segment_encoders:
                    if customer_id not in self.segment_encoders[segment].classes_:
                        self.segment_encoders[segment].classes_ = np.append(
                            self.segment_encoders[segment].classes_, [customer_id]
                        )
                    self.pending_customers.remove(customer_id)
                    newly_segmented.append((customer_id, segment))
        
        if newly_segmented:
            logger.info(f"Assigned segments to {len(newly_segmented)} new customers")
        
        # Update product embeddings for new products
        new_products = new_df[['StockCode', 'Description']].drop_duplicates()
        self.product_embeddings = self.embeddings_creator.update_embeddings(
            self.product_embeddings, self.word2vec_model, new_products
        )

    def update_models_continued(self):
        """Second part of the update_models method."""
        new_df = pd.DataFrame(self.new_transactions)
        
        # Update global model
        interactions = []
        for _, row in new_df.iterrows():
            try:
                # Skip if missing required data
                if pd.isna(row['CustomerID']) or pd.isna(row['StockCode']):
                    continue
                    
                # Get indices
                customer_idx = self.data_processor.customer_encoder.transform([row['CustomerID']])[0]
                item_idx = self.data_processor.item_encoder.transform([row['StockCode']])[0]
                
                # Add interaction (using quantity as implicit rating strength)
                rating = min(5, max(1, int(row['Quantity'] / 2) + 1))  # Normalize to 1-5
                interactions.append((customer_idx, item_idx, rating))
            except Exception as e:
                logger.error(f"Error processing interaction: {str(e)}")
        
        # Update global model
        if interactions and self.global_model:
            self.global_model.update_model(
                interactions, 
                self.data_processor.customer_encoder,
                self.data_processor.item_encoder,
                self.product_embeddings
            )
        
        # Update segment models
        segment_updates = defaultdict(list)
        for _, row in new_df.iterrows():
            try:
                # Skip if missing required data
                if pd.isna(row['CustomerID']) or pd.isna(row['StockCode']):
                    continue
                
                # Skip if customer has no segment
                if row['CustomerID'] not in self.rfm_table['CustomerID'].values:
                    continue
                    
                segment = self.rfm_table[
                    self.rfm_table['CustomerID'] == row['CustomerID']
                ]['Segment'].iloc[0]
                
                # Skip if segment has no model
                if segment not in self.segment_models or segment not in self.segment_encoders:
                    continue
                
                # Get local indices
                segment_encoder = self.segment_encoders[segment]
                
                # Add customer to segment encoder if new
                if row['CustomerID'] not in segment_encoder.classes_:
                    segment_encoder.classes_ = np.append(
                        segment_encoder.classes_, [row['CustomerID']]
                    )
                
                local_idx = segment_encoder.transform([row['CustomerID']])[0]
                item_idx = self.data_processor.item_encoder.transform([row['StockCode']])[0]
                
                # Add interaction
                rating = min(5, max(1, int(row['Quantity'] / 2) + 1))
                segment_updates[segment].append((local_idx, item_idx, rating))
            except Exception as e:
                logger.error(f"Error processing segment interaction: {str(e)}")
        
        # Update each segment model
        for segment, interactions in segment_updates.items():
            if interactions:
                try:
                    self.segment_models[segment].update_model(
                        interactions,
                        self.segment_encoders[segment],
                        self.data_processor.item_encoder,
                        self.product_embeddings
                    )
                    logger.info(f"Updated model for segment {segment} with {len(interactions)} interactions")
                except Exception as e:
                    logger.error(f"Error updating segment {segment} model: {str(e)}")
        
        # Synchronize models (local models learn from global and vice versa)
        self._federated_synchronization()
        
        # Clear transaction buffer and update timestamp
        self.new_transactions = []
        self.last_model_update = datetime.now()
        
        # Append new transactions to main dataframe
        self.df = pd.concat([self.df, new_df])
        
        logger.info("Models updated successfully")
        return True

    def _federated_synchronization(self):
        """Perform federated learning synchronization between global and local models."""
        # First: Local models learn from global model
        for segment, model in self.segment_models.items():
            model.synchronize_with_global(self.global_model, learning_rate=0.3)
        
        # Then: Update global model by aggregating updated local models
        self.global_model.aggregate(list(self.segment_models.values()))
        
        logger.info("Completed federated synchronization of models")

    def generate_recommendations(self, customer_id, n_items=5, use_global=False,
                              personalization_level='auto', price_range=None):
        """
        Generate recommendations for a specific customer.
        
        Args:
            customer_id: Customer identifier
            n_items: Number of recommendations to generate
            use_global: Force using the global model instead of segment model
            personalization_level: How to personalize ('auto', 'segment', 'global')
            price_range: Optional tuple of (min_price, max_price) to filter results
        """
        # Validate customer_id type (handle float conversion)
        customer_id = int(customer_id) if pd.notna(customer_id) else None
        
        # Check if customer exists
        if customer_id is None or customer_id not in self.data_processor.customer_encoder.classes_:
            logger.warning(f"Customer {customer_id} not found in system")
            return []
        
        # Detailed logging for model selection
        logger.info(f"Generating recommendations for customer {customer_id}")
        logger.info(f"Current personalization level: {personalization_level}")
        logger.info(f"Pending customers: {self.pending_customers}")
        logger.info(f"Global model exists: {self.global_model is not None}")
        
        # Decide which model to use based on customer status and settings
        if personalization_level == 'auto':
            # New customer with no segment yet = use global
            if customer_id in self.pending_customers or customer_id not in self.rfm_table['CustomerID'].values:
                use_global = True
                logger.info(f"Using global model for new customer {customer_id}")
            else:
                # Has segment, check if segment has a model
                segment = self.rfm_table[self.rfm_table['CustomerID'] == customer_id]['Segment'].iloc[0]
                if segment not in self.segment_models:
                    use_global = True
                    logger.info(f"Using global model for customer {customer_id} (no model for segment {segment})")
        elif personalization_level == 'global':
            use_global = True
        # else use segment as specified by caller
            
        # Fetch recommendations using appropriate model
        if use_global and self.global_model:
            # Use global model
            try:
                # Detailed logging for global model recommendation
                logger.info("Attempting to generate global recommendations")
                
                # Validate customer index
                try:
                    customer_idx = self.data_processor.customer_encoder.transform([customer_id])[0]
                except Exception as idx_error:
                    logger.error(f"Error encoding customer index: {idx_error}")
                    logger.error(f"Available customer classes: {self.data_processor.customer_encoder.classes_}")
                    return []
                
                # Get items the customer has already purchased
                purchased_items = self.df[self.df['CustomerID'] == customer_id]['ItemIdx'].unique().tolist()
                
                # Log additional model details
                logger.info(f"Global model dimensions: Users={self.global_model.U.shape[0]}, Items={self.global_model.V.shape[0]}")
                logger.info(f"Number of purchased items to exclude: {len(purchased_items)}")
                
                # Generate recommendations
                recommendations = self.global_model.recommend(
                    customer_idx,
                    item_encoder=self.data_processor.item_encoder,
                    product_info=self.product_info,
                    n_items=n_items,
                    exclude_items=purchased_items,
                    product_emb=self.product_embeddings,
                    price_range=price_range
                )
                
                logger.info(f"Generated {len(recommendations)} global recommendations")
                return recommendations
            
            except Exception as e:
                logger.error(f"Comprehensive error generating global recommendations: {str(e)}")
                logger.error(f"Error details: {traceback.format_exc()}")
                return []
        
        # Use segment-specific model
        try:
            segment = self.rfm_table[self.rfm_table['CustomerID'] == customer_id]['Segment'].iloc[0]
            
            if segment not in self.segment_models:
                logger.error(f"No model available for segment {segment}")
                return []
            
            model = self.segment_models[segment]
            segment_encoder = self.segment_encoders[segment]
            
            if customer_id not in segment_encoder.classes_:
                logger.warning(f"Customer {customer_id} not found in segment {segment} encoder")
                segment_encoder.classes_ = np.append(segment_encoder.classes_, [customer_id])
            
            # Convert customer ID to segment-specific index
            local_idx = segment_encoder.transform([customer_id])[0]
            
            # Get items the customer has already purchased
            purchased_items = self.df[self.df['CustomerID'] == customer_id]['ItemIdx'].unique().tolist()
            
            # Generate recommendations
            recommendations = model.recommend(
                local_idx,
                item_encoder=self.data_processor.item_encoder,
                product_info=self.product_info,
                n_items=n_items,
                exclude_items=purchased_items,
                product_emb=self.product_embeddings,
                price_range=price_range
            )
            
            return recommendations
        except Exception as e:
            logger.error(f"Error generating segment recommendations: {str(e)}")
            return []

    def save_state(self, base_path='recommender_state'):
        """Save the entire recommender state to disk."""
        if not os.path.exists(base_path):
            os.makedirs(base_path)
            
        try:
            # Save models
            with open(f"{base_path}/models.pkl", 'wb') as f:
                pickle.dump({
                    'segment_models': self.segment_models,
                    'segment_encoders': self.segment_encoders,
                    'global_model': self.global_model,
                }, f)
            
            # Save data processor state
            with open(f"{base_path}/data_processor.pkl", 'wb') as f:
                pickle.dump({
                    'item_encoder': self.data_processor.item_encoder,
                    'customer_encoder': self.data_processor.customer_encoder
                }, f)
            
            # Save embeddings
            with open(f"{base_path}/embeddings.pkl", 'wb') as f:
                pickle.dump({
                    'product_embeddings': self.product_embeddings,
                    'word2vec_model': self.word2vec_model
                }, f)
            
            # Save RFM table
            self.rfm_table.to_csv(f"{base_path}/rfm_table.csv", index=False)
            
            # Save product info
            with open(f"{base_path}/product_info.pkl", 'wb') as f:
                pickle.dump(self.product_info, f)
            
            # Save pending customers
            with open(f"{base_path}/pending_customers.pkl", 'wb') as f:
                pickle.dump(self.pending_customers, f)
            
            # Save dataframe
            self.df.to_csv(f"{base_path}/transactions.csv", index=False)
            
            logger.info(f"Recommender state saved to {base_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving recommender state: {str(e)}")
            return False

    def load_state(self, base_path='recommender_state'):
        """Load recommender state from disk."""
        if not os.path.exists(base_path):
            logger.error(f"State directory {base_path} not found")
            return False
            
        try:
            # Load models
            with open(f"{base_path}/models.pkl", 'rb') as f:
                models_data = pickle.load(f)
                self.segment_models = models_data['segment_models']
                self.segment_encoders = models_data['segment_encoders']
                self.global_model = models_data['global_model']
            
            # Load data processor state
            with open(f"{base_path}/data_processor.pkl", 'rb') as f:
                processor_data = pickle.load(f)
                self.data_processor.item_encoder = processor_data['item_encoder']
                self.data_processor.customer_encoder = processor_data['customer_encoder']
            
            # Load embeddings
            with open(f"{base_path}/embeddings.pkl", 'rb') as f:
                embeddings_data = pickle.load(f)
                self.product_embeddings = embeddings_data['product_embeddings']
                self.word2vec_model = embeddings_data['word2vec_model']
            
            # Load RFM table
            self.rfm_table = pd.read_csv(f"{base_path}/rfm_table.csv")
            self.rfm_table['CustomerID'] = pd.to_numeric(self.rfm_table['CustomerID'], errors='coerce')
            
            # Load product info
            with open(f"{base_path}/product_info.pkl", 'rb') as f:
                self.product_info = pickle.load(f)
            
            # Load pending customers
            with open(f"{base_path}/pending_customers.pkl", 'rb') as f:
                self.pending_customers = pickle.load(f)
            
            # Load dataframe
            self.df = pd.read_csv(f"{base_path}/transactions.csv")
            self.df['InvoiceDate'] = pd.to_datetime(self.df['InvoiceDate'])
            
            logger.info(f"Recommender state loaded from {base_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading recommender state: {str(e)}")
            return False

    def evaluate_recommendations(self, test_data, k=5):
        """
        Evaluate recommendation quality using test data.
        Returns precision@k, recall@k, and ndcg@k metrics.
        """
        metrics = {
            'precision@k': [],
            'recall@k': [],
            'ndcg@k': []
        }
        
        for customer_id in test_data['CustomerID'].unique():
            try:
                # Get actual items purchased by customer in test set
                actual_items = set(test_data[test_data['CustomerID'] == customer_id]['StockCode'])
                
                if not actual_items:
                    continue
                
                # Get recommendations
                recs = self.generate_recommendations(customer_id, n_items=k)
                recommended_items = {rec['stock_code'] for rec in recs}
                
                # Calculate metrics
                hits = len(actual_items & recommended_items)
                
                # Precision@k and Recall@k
                precision = hits / k if k > 0 else 0
                recall = hits / len(actual_items) if actual_items else 0
                
                metrics['precision@k'].append(precision)
                metrics['recall@k'].append(recall)
                
                # NDCG@k
                dcg = 0
                idcg = sum(1 / np.log2(i + 2) for i in range(min(k, len(actual_items))))
                
                for i, rec in enumerate(recs):
                    if rec['stock_code'] in actual_items:
                        dcg += 1 / np.log2(i + 2)
                
                ndcg = dcg / idcg if idcg > 0 else 0
                metrics['ndcg@k'].append(ndcg)
                
            except Exception as e:
                logger.error(f"Error evaluating recommendations for customer {customer_id}: {str(e)}")
                continue
        
        # Calculate average metrics
        return {
            'precision@k': np.mean(metrics['precision@k']),
            'recall@k': np.mean(metrics['recall@k']),
            'ndcg@k': np.mean(metrics['ndcg@k'])
        }

    def get_segment_insights(self, segment):
        """Get insights about a customer segment."""
        if segment not in self.segment_models:
            return None
            
        insights = {
            'customer_count': 0,
            'top_products': [],
            'avg_order_value': 0,
            'purchase_frequency': 0
        }
        
        try:
            # Get customers in segment
            segment_customers = self.rfm_table[
                self.rfm_table['Segment'] == segment
            ]['CustomerID'].values
            
            insights['customer_count'] = len(segment_customers)
            
            # Get transactions for segment
            segment_transactions = self.df[
                self.df['CustomerID'].isin(segment_customers)
            ]
            
            # Calculate average order value
            if 'InvoiceNo' in segment_transactions.columns:
                insights['avg_order_value'] = segment_transactions.groupby('InvoiceNo')[
                    'UnitPrice'
                ].sum().mean()
                
                # Calculate purchase frequency (orders per customer)
                insights['purchase_frequency'] = segment_transactions['InvoiceNo'].nunique() / len(segment_customers)
            
            # Get top products
            top_products = segment_transactions.groupby('StockCode').agg({
                'Quantity': 'sum',
                'UnitPrice': 'mean'
            }).sort_values('Quantity', ascending=False).head(5)
            
            for stock_code, row in top_products.iterrows():
                if stock_code in self.product_info:
                    insights['top_products'].append({
                        'stock_code': stock_code,
                        'name': self.product_info[stock_code]['description'],
                        'total_quantity': int(row['Quantity']),
                        'avg_price': float(row['UnitPrice'])
                    })
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting segment insights: {str(e)}")
            return None
