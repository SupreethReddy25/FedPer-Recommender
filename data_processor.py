"""
Data processing module for the recommendation system.
"""

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

logger = logging.getLogger("FedPerRecommender")

class DataProcessor:
    """Handles loading and preprocessing of retail data."""
    
    def __init__(self):
        self.customer_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        
    def load_data(self, file_path):
        """Load data from file and perform basic preprocessing."""
        logger.info(f"Loading data from: {file_path}")
        
        # Load based on file extension
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            raise ValueError("Unsupported file format. Please use .xlsx, .xls, or .csv")
            
        logger.info(f"Data loaded: {df.shape[0]} rows and {df.shape[1]} columns")
        return df
    
    def preprocess(self, df):
        """Clean and prepare the dataset."""
        logger.info("Preprocessing data...")
        
        # Convert types
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df['CustomerID'] = pd.to_numeric(df['CustomerID'], errors='coerce')
        df['StockCode'] = df['StockCode'].astype(str)
        
        # Remove rows with missing CustomerID
        missing_customers = df['CustomerID'].isna().sum()
        if missing_customers > 0:
            logger.info(f"Dropping {missing_customers} rows with missing CustomerID")
            df = df.dropna(subset=['CustomerID'])
        
        # Remove canceled transactions
        if df['InvoiceNo'].dtype == object:
            canceled_count = df['InvoiceNo'].str.contains('C', na=False).sum()
            if canceled_count > 0:
                logger.info(f"Removing {canceled_count} canceled transactions")
                df = df[~df['InvoiceNo'].str.contains('C', na=False)]
        
        # Remove negative quantities
        neg_quantity = (df['Quantity'] <= 0).sum()
        if neg_quantity > 0:
            logger.info(f"Removing {neg_quantity} rows with non-positive quantity")
            df = df[df['Quantity'] > 0]
        
        # Handle extreme values in Quantity
        if 'Quantity' in df.columns:
            q_upper = df['Quantity'].quantile(0.95)               #'''Changable'''
            extreme_quantity = (df['Quantity'] > q_upper).sum()
            if extreme_quantity > 0:
                logger.info(f"Capping {extreme_quantity} extreme quantity values")
                df.loc[df['Quantity'] > q_upper, 'Quantity'] = q_upper
        
        # Handle extreme values in UnitPrice
        if 'UnitPrice' in df.columns:
            p_upper = df['UnitPrice'].quantile(0.95)
            extreme_price = (df['UnitPrice'] > p_upper).sum()
            if extreme_price > 0:
                logger.info(f"Capping {extreme_price} extreme price values")
                df.loc[df['UnitPrice'] > p_upper, 'UnitPrice'] = p_upper
            
        return df
    
    def encode_indices(self, df, update=False):
        """Create numerical indices for customers and items."""
        if update:
            # For updates, fit_transform only on new values and transform all
            all_customers = df['CustomerID'].unique()
            all_items = df['StockCode'].unique()
            
            # Find new customers and items
            new_customers = [c for c in all_customers if c not in self.customer_encoder.classes_]
            new_items = [i for i in all_items if i not in self.item_encoder.classes_]
            
            # Update encoders if there are new values
            if new_customers:
                logger.info(f"Adding {len(new_customers)} new customers to encoder")
                self.customer_encoder.classes_ = np.append(self.customer_encoder.classes_, new_customers)
                
            if new_items:
                logger.info(f"Adding {len(new_items)} new items to encoder")
                self.item_encoder.classes_ = np.append(self.item_encoder.classes_, new_items)
        else:
            # Initial fit
            logger.info("Creating customer and item encoders")
            self.customer_encoder.fit(df['CustomerID'].unique())
            self.item_encoder.fit(df['StockCode'].unique())
        
        # Transform all values
        df['CustomerIdx'] = self.customer_encoder.transform(df['CustomerID'])
        df['ItemIdx'] = self.item_encoder.transform(df['StockCode'])
        return df
    
    def get_product_info(self, df):
        """Create a dictionary of product descriptions and metadata."""
        product_info = {}
        
        # Group by StockCode to get primary information
        grouped = df.groupby('StockCode').agg({
            'Description': 'first',
            'UnitPrice': 'median',  # Use median price to handle outliers
            'Quantity': 'sum'       # Total quantity sold
        })
        
        for stock_code, row in grouped.iterrows():
            product_info[stock_code] = {
                'description': row['Description'],
                'median_price': row['UnitPrice'],
                'total_sold': row['Quantity']
            }
            
        # Calculate price history if possible
        if 'InvoiceDate' in df.columns:
            # Get monthly price averages
            price_history = df.groupby(['StockCode', pd.Grouper(key='InvoiceDate', freq='M')]) ['UnitPrice'].mean().reset_index()
            
            # Add to product info
            for stock_code in product_info:
                history = price_history[price_history['StockCode'] == stock_code]
                if not history.empty:
                    product_info[stock_code]['price_history']=dict(zip(history['InvoiceDate'].dt.strftime('%Y-%m'), history['UnitPrice']))
        logger.info(f"Product info created for {len(product_info)} items")
        return product_info
    
    def create_train_test_split(self, df, test_ratio=0.2, by_time=True):
        """Split the data into training and test sets."""
        if by_time and 'InvoiceDate' in df.columns:
            # Time-based split (last test_ratio of the data timespan)
            min_date = df['InvoiceDate'].min()
            max_date = df['InvoiceDate'].max()
            date_range = (max_date - min_date).days
            
            split_date = min_date + pd.Timedelta(days=int(date_range * (1 - test_ratio)))
            
            train = df[df['InvoiceDate'] < split_date]
            test = df[df['InvoiceDate'] >= split_date]
            
            logger.info(f"Time-based split: {len(train)} train, {len(test)} test")
            logger.info(f"Split date: {split_date}")
        else:
            # Random split
            from sklearn.model_selection import train_test_split
            train, test = train_test_split(df, test_size=test_ratio, random_state=42)
            
            logger.info(f"Random split: {len(train)} train, {len(test)} test")
        
        return train, test
