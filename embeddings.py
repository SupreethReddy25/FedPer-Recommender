"""
Product embeddings module for the recommendation system.
"""

import pandas as pd
import numpy as np
import logging
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import re
import time

logger = logging.getLogger("FedPerRecommender")

class ProductEmbeddings:
    """
    Handles product embeddings generation using Word2Vec.
    
    This class supports two types of embeddings:
    1. Transaction-based: Uses purchase sequences to learn product relationships
       based on co-occurrence patterns (similar to word context in sentences)
    2. Description-based: Uses product text descriptions to learn semantic
       relationships between products
    """
    
    def __init__(self, vector_size=100, window=6, min_count=2, workers=5, seed=42):
        """Initialize the product embeddings generator with Word2Vec parameters."""
        self.vector_size = vector_size  # Embedding dimension
        self.window = window            # Context window size
        self.min_count = min_count      # Minimum occurrence for a product to be included
        self.workers = workers          # Parallel processing threads
        self.seed = seed                # Random seed for reproducibility
        self.model = None               # Word2Vec model (initialized later)
        self.product_to_id = {}         # Mapping of product codes to numeric IDs
        self.id_to_product = {}         # Reverse mapping
        self.description_model = None   # Model for description-based embeddings
        self.description_word_cache = defaultdict(list)  # Cache for processed descriptions
    
    def create_product_embeddings(self, df):
        """Alias for generate_embeddings to maintain compatibility with manager."""
        return self.generate_embeddings(df)
        
    def generate_embeddings(self, df):
        """
        Generate product embeddings from transaction data.
        
        Args:
            df: DataFrame with columns: CustomerID, InvoiceNo, StockCode, InvoiceDate
            
        Returns:
            Dictionary mapping product codes to their embeddings
        """
        if df.empty:
            logger.warning("Empty DataFrame provided. Cannot generate embeddings.")
            return {}
            
        # Validate required columns exist
        required_columns = ['CustomerID', 'InvoiceNo', 'StockCode', 'InvoiceDate']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return {}
            
        logger.info("Generating transaction-based product embeddings...")
        
        # Create product sequences by invoice
        invoice_products = df.groupby('InvoiceNo')['StockCode'].apply(list).values
        
        # Create sequences by customer (time-ordered baskets)
        customer_sequences = []
        for customer_id, customer_df in df.sort_values('InvoiceDate').groupby('CustomerID'):
            # Get all invoices for this customer in chronological order
            invoices = customer_df['InvoiceNo'].unique()
            customer_seq = []
            for invoice in invoices:
                # Add all products from this invoice
                products = df[df['InvoiceNo'] == invoice]['StockCode'].astype(str).tolist()
                customer_seq.extend(products)
            
            if len(customer_seq) > 1:  # Only add if there's more than one product
                customer_sequences.append(customer_seq)
        
        # Combine both types of sequences
        all_sequences = list(invoice_products) + customer_sequences
        
        # Handle case with no sequences
        if not all_sequences:
            logger.warning("No valid product sequences found in the data.")
            return {}
            
        logger.info(f"Created {len(all_sequences)} product sequences")
        if all_sequences:
            logger.info(f"Average sequence length: {sum(len(seq) for seq in all_sequences) / len(all_sequences):.2f} products")
        
        # Build vocabulary (product -> numeric ID mapping)
        unique_products = set()
        for sequence in all_sequences:
            unique_products.update(sequence)
        
        self.product_to_id = {str(product): i for i, product in enumerate(unique_products)}
        self.id_to_product = {i: product for product, i in self.product_to_id.items()}
        
        # Convert product codes to IDs for Word2Vec
        id_sequences = [[self.product_to_id[str(product)] for product in sequence] for sequence in all_sequences]
        
        # Train Word2Vec model
        logger.info(f"Training Word2Vec model with {len(unique_products)} unique products...")
        try:
            self.model = Word2Vec(
                sentences=id_sequences,
                vector_size=self.vector_size,
                window=self.window,
                min_count=self.min_count,
                workers=self.workers,
                seed=self.seed
            )
            
            # Create product-to-embedding dictionary
            embeddings = {}
            for product, product_id in self.product_to_id.items():
                if product_id in self.model.wv.key_to_index:
                    embeddings[product] = self.model.wv[product_id]
            
            logger.info(f"Generated transaction-based embeddings for {len(embeddings)} products")
            return embeddings
        except Exception as e:
            logger.error(f"Error training Word2Vec model: {str(e)}")
            return {}
    
    def _preprocess_description(self, description):
        """
        Preprocess a product description for better word embeddings.
        
        Args:
            description: The raw product description string
            
        Returns:
            List of preprocessed words
        """
        if pd.isna(description):
            return []
            
        # Convert to string and lowercase
        text = str(description).lower()
        
        # Remove special characters but keep spaces between words
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        
        # Split by whitespace and filter empty strings
        words = [word for word in text.split() if word]
        
        # Remove very short words (likely not meaningful)
        words = [word for word in words if len(word) > 1]
        
        return words
    
    def create_description_embeddings(self, df, vector_size=None):
        """
        Generate embeddings from product descriptions.
        
        Args:
            df: DataFrame with columns: StockCode, Description
            vector_size: Dimension of embeddings (defaults to self.vector_size)
            
        Returns:
            Dictionary mapping product codes to their embeddings and the trained model
        """
        if vector_size is None:
            vector_size = self.vector_size
            
        if df.empty:
            logger.warning("Empty DataFrame provided. Cannot create description embeddings.")
            return {}
            
        # Validate required columns exist
        if 'StockCode' not in df.columns or 'Description' not in df.columns:
            logger.error("Missing required columns: StockCode and/or Description")
            return {}
            
        logger.info("Creating description-based product embeddings...")
        
        # Get unique descriptions
        product_descriptions = df.groupby('StockCode')['Description'].first()
        
        # Clear the description word cache
        self.description_word_cache.clear()
        
        # Prepare text for Word2Vec
        product_words = []
        for stock_code, desc in product_descriptions.items():
            processed_words = self._preprocess_description(desc)
            if processed_words:
                product_words.append(processed_words)
                # Cache the processed words for this product
                self.description_word_cache[str(stock_code)] = processed_words
        
        if not product_words:
            logger.warning("No valid descriptions found for embedding generation.")
            return {}
            
        # Train Word2Vec model
        try:
            self.description_model = Word2Vec(
                product_words, 
                vector_size=vector_size,
                window=self.window, 
                min_count=1, 
                workers=self.workers,
                seed=self.seed
            )
            
            # Generate embeddings for each product
            product_embeddings = {}
            for stock_code in product_descriptions.index:
                stock_code_str = str(stock_code)
                if stock_code_str in self.description_word_cache:
                    words = self.description_word_cache[stock_code_str]
                    valid_words = [w for w in words if w in self.description_model.wv]
                    
                    if valid_words:
                        # Average word vectors to get product vector
                        product_embeddings[stock_code_str] = np.mean(
                            [self.description_model.wv[w] for w in valid_words], axis=0
                        )
                    else:
                        product_embeddings[stock_code_str] = np.zeros(vector_size)
                else:
                    product_embeddings[stock_code_str] = np.zeros(vector_size)
            
            logger.info(f"Created description-based embeddings for {len(product_embeddings)} products")
            return product_embeddings
        except Exception as e:
            logger.error(f"Error creating description embeddings: {str(e)}")
            return {}
    
    def update_embeddings(self, df):
        """
        Update transaction-based embeddings with new transaction data.
        This performs incremental training on the existing model.
        """
        if self.model is None:
            logger.info("No existing model. Generating new embeddings.")
            return self.generate_embeddings(df)
        
        if df.empty:
            logger.warning("Empty DataFrame provided. Cannot update embeddings.")
            return None
            
        # Validate required columns exist
        required_columns = ['InvoiceNo', 'StockCode']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns for update: {missing_columns}")
            return None
            
        logger.info("Updating transaction-based product embeddings with new data...")
        
        # Create sequences from new data
        invoice_products = df.groupby('InvoiceNo')['StockCode'].apply(
            lambda x: [str(item) for item in x]
        ).values
        
        # Update vocabulary with new products
        new_products = set()
        for sequence in invoice_products:
            new_products.update(sequence)
        
        # Add new products to mappings
        max_id = max(self.id_to_product.keys()) if self.id_to_product else -1
        new_product_count = 0
        
        for product in new_products:
            product_str = str(product)
            if product_str not in self.product_to_id:
                max_id += 1
                self.product_to_id[product_str] = max_id
                self.id_to_product[max_id] = product_str
                new_product_count += 1
        
        logger.info(f"Added {new_product_count} new products to vocabulary")
        
        # Convert sequences to IDs
        id_sequences = [[self.product_to_id[str(product)] for product in sequence 
                         if str(product) in self.product_to_id] 
                        for sequence in invoice_products]
        
        # Skip empty sequences
        id_sequences = [seq for seq in id_sequences if seq]
        
        if id_sequences:
            logger.info(f"Updating model with {len(id_sequences)} new sequences...")
            
            try:
                # Update existing model with new data (incremental training)
                self.model.build_vocab(id_sequences, update=True)
                self.model.train(
                    id_sequences,
                    total_examples=len(id_sequences),
                    epochs=self.model.epochs
                )
                
                # Create updated embeddings dictionary
                embeddings = {}
                for product, product_id in self.product_to_id.items():
                    if product_id in self.model.wv.key_to_index:
                        embeddings[product] = self.model.wv[product_id]
                
                logger.info(f"Updated embeddings for {len(embeddings)} products")
                return embeddings
            except Exception as e:
                logger.error(f"Error updating embeddings: {str(e)}")
                return None
        else:
            logger.warning("No valid sequences for updating embeddings.")
            
        return None
    
    def update_description_embeddings(self, product_emb, new_products):
        """
        Update description-based embeddings with new product descriptions.
        
        Args:
            product_emb: Existing embeddings dictionary
            new_products: DataFrame with new product data (StockCode, Description)
            
        Returns:
            Updated embeddings dictionary
        """
        if self.description_model is None:
            logger.error("Description model not trained. Call create_description_embeddings first.")
            return product_emb
            
        if new_products.empty:
            logger.warning("No new products provided for description update.")
            return product_emb
            
        # Validate required columns exist
        if 'StockCode' not in new_products.columns or 'Description' not in new_products.columns:
            logger.error("Missing required columns: StockCode and/or Description")
            return product_emb
            
        logger.info("Updating description-based embeddings with new product data...")
            
        # Get unique descriptions for new products
        new_descriptions = new_products.groupby('StockCode')['Description'].first()
        
        # Collect all new descriptions for model update
        new_product_words = []
        updated_stock_codes = []
        
        for stock_code, desc in new_descriptions.items():
            stock_code_str = str(stock_code)
            processed_words = self._preprocess_description(desc)
            
            if processed_words:
                new_product_words.append(processed_words)
                updated_stock_codes.append(stock_code_str)
                # Update the description word cache
                self.description_word_cache[stock_code_str] = processed_words
        
        if new_product_words:
            logger.info(f"Updating description model with {len(new_product_words)} new products...")
            
            try:
                # Update the description model vocabulary with new words
                self.description_model.build_vocab(new_product_words, update=True)
                
                # Train the model on new descriptions
                self.description_model.train(
                    new_product_words,
                    total_examples=len(new_product_words),
                    epochs=self.description_model.epochs
                )
            except Exception as e:
                logger.error(f"Error updating description model: {str(e)}")
                # Continue with existing model to update embeddings
        
        # Update existing embeddings
        vector_size = list(product_emb.values())[0].shape[0] if product_emb else self.vector_size
        
        # Create copies of the updated embeddings to avoid modifying during iteration
        updated_emb = product_emb.copy()
        
        for stock_code, desc in new_descriptions.items():
            stock_code_str = str(stock_code)
            
            if stock_code_str in self.description_word_cache:
                words = self.description_word_cache[stock_code_str]
                valid_words = [w for w in words if w in self.description_model.wv]
                
                if valid_words:
                    # Average word vectors to get product vector
                    updated_emb[stock_code_str] = np.mean(
                        [self.description_model.wv[w] for w in valid_words], axis=0
                    )
                else:
                    # If no valid words, create a zero vector
                    updated_emb[stock_code_str] = np.zeros(vector_size)
            else:
                # If no processed description available, create a zero vector
                updated_emb[stock_code_str] = np.zeros(vector_size)
        
        logger.info(f"Updated description-based embeddings for {len(new_descriptions)} products")
        return updated_emb
    
    def find_similar_products(self, product_code, n=10):
        """
        Find products similar to the given product code using transaction-based embeddings.
        
        Args:
            product_code: The product code to find similar items for
            n: Number of similar products to return
            
        Returns:
            List of dictionaries with similar product codes and similarity scores
        """
        if self.model is None:
            logger.error("Transaction model not trained yet. Call generate_embeddings first.")
            return []
        
        # Convert product code to string for consistency
        product_code = str(product_code)
        
        # Check if product exists in model
        if product_code not in self.product_to_id:
            logger.warning(f"Product {product_code} not found in embeddings")
            return []
            
        product_id = self.product_to_id[product_code]
        if product_id not in self.model.wv.key_to_index:
            logger.warning(f"Product {product_code} was filtered out during training")
            return []
        
        try:
            # Get similar products
            similar_ids = self.model.wv.most_similar(product_id, topn=n)
            similar_products = [
                {
                    'product_code': self.id_to_product[id],
                    'similarity': float(similarity)  # Convert numpy float to Python float
                }
                for id, similarity in similar_ids
            ]
            
            return similar_products
        except Exception as e:
            logger.error(f"Error finding similar products: {str(e)}")
            return []
    
    def get_similar_products_by_description(self, product_id, product_embeddings, n=10):
        """
        Find products with similar descriptions.
        
        Args:
            product_id: The product code to find similar items for
            product_embeddings: Dictionary of product embeddings from description
            n: Number of similar products to return
            
        Returns:
            List of dictionaries with similar product codes and similarity scores
        """
        # Convert product ID to string for consistency
        product_id = str(product_id)
        
        if not product_embeddings:
            logger.error("No product embeddings provided")
            return []
            
        if product_id not in product_embeddings:
            logger.error(f"Product {product_id} not found in description embeddings")
            return []
        
        # Get embedding for the target product
        target_embedding = product_embeddings[product_id]
        
        # Check if target embedding is a valid vector (not all zeros)
        if np.all(target_embedding == 0):
            logger.warning(f"Product {product_id} has a zero embedding vector")
            return []
            
        try:
            # Calculate similarity with all other products
            similarities = {}
            for stock_code, embedding in product_embeddings.items():
                if stock_code == product_id:
                    continue
                
                # Skip zero embeddings
                if np.all(embedding == 0):
                    continue
                    
                # Calculate cosine similarity
                similarity = np.dot(target_embedding, embedding) / (
                    np.linalg.norm(target_embedding) * np.linalg.norm(embedding) + 1e-10  # Avoid division by zero
                )
                
                similarities[stock_code] = similarity
            
            # Get top N similar products
            similar_products = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:n]
            
            return [{'product_code': p[0], 'similarity': float(p[1])} for p in similar_products]
        except Exception as e:
            logger.error(f"Error finding similar products by description: {str(e)}")
            return []
    
    def create_category_embeddings(self, df, product_embeddings):
        """
        Create category embeddings by aggregating product embeddings.
        
        Args:
            df: DataFrame with product data
            product_embeddings: Dictionary of product embeddings
            
        Returns:
            Dictionary mapping categories to embedding vectors
        """
        if not product_embeddings:
            logger.error("No product embeddings provided for category embedding creation")
            return {}
            
        if df.empty:
            logger.warning("Empty DataFrame provided. Cannot create category embeddings.")
            return {}
            
        if 'Category' not in df.columns:
            logger.warning("No 'Category' column found. Trying to infer categories from descriptions.")
            # Try to infer categories from product descriptions
            try:
                if 'Description' not in df.columns:
                    logger.error("No Description column available for inferring categories")
                    return {}
                    
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.cluster import KMeans
                
                # Get product descriptions
                descriptions = df.groupby('StockCode')['Description'].first().dropna()
                
                if descriptions.empty:
                    logger.warning("No valid descriptions for category inference")
                    return {}
                    
                # Vectorize descriptions
                vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
                X = vectorizer.fit_transform(descriptions)
                
                # Cluster products
                n_clusters = min(10, X.shape[0] - 1) if X.shape[0] > 1 else 1
                kmeans = KMeans(n_clusters=n_clusters, random_state=self.seed)
                clusters = kmeans.fit_predict(X)
                
                # Assign cluster as category
                category_map = {str(code): f"Category_{cluster}" for code, cluster in 
                                zip(descriptions.index, clusters)}
                
                df = df.copy()  # Create a copy to avoid modifying the original
                df['Category'] = df['StockCode'].astype(str).map(category_map).fillna("Uncategorized")
                logger.info(f"Inferred {len(set(clusters))} categories from descriptions")
                
            except Exception as e:
                logger.error(f"Error inferring categories: {str(e)}")
                return {}
        
        # Create category embeddings by averaging product embeddings
        category_embeddings = {}
        
        for category in df['Category'].unique():
            # Get products in this category
            category_products = df[df['Category'] == category]['StockCode'].astype(str).unique()
            
            # Get embeddings for these products
            embeddings = [product_embeddings[p] for p in category_products 
                         if p in product_embeddings]
            
            if embeddings:
                # Average embeddings
                category_embeddings[category] = np.mean(embeddings, axis=0)
                
        logger.info(f"Created embeddings for {len(category_embeddings)} categories")
        return category_embeddings
    
    def visualize_embeddings(self, embeddings, top_n=100, save_path=None):
        """
        Create a 2D visualization of product embeddings using t-SNE.
        
        Args:
            embeddings: Dictionary mapping product codes to embedding vectors
            top_n: Number of most frequent products to visualize
            save_path: If provided, saves the plot to this path
            
        Returns:
            matplotlib figure
        """
        if not embeddings:
            logger.error("No embeddings provided for visualization")
            return None
            
        logger.info(f"Visualizing top {top_n} product embeddings...")
        
        # Get products and vectors
        products = list(embeddings.keys())[:min(top_n, len(embeddings))]
        vectors = np.array([embeddings[p] for p in products])
        
        if len(products) < 2:
            logger.error("Need at least 2 products to create visualization")
            return None
            
        # Apply t-SNE for dimensionality reduction
        try:
            perplexity = min(30, len(products) - 1)
            tsne = TSNE(n_components=2, random_state=self.seed, perplexity=perplexity)
            reduced_vectors = tsne.fit_transform(vectors)
            
            # Create plot
            plt.figure(figsize=(12, 10))
            plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], alpha=0.6)
            
            # Add labels for some points (top 20)
            for i, product in enumerate(products[:min(20, len(products))]):
                plt.annotate(product, (reduced_vectors[i, 0], reduced_vectors[i, 1]), 
                             fontsize=9, alpha=0.8)
            
            plt.title(f't-SNE visualization of {len(products)} product embeddings')
            plt.tight_layout()
            
            # Save if path provided
            if save_path:
                plt.savefig(save_path, dpi=300)
                logger.info(f"Visualization saved to {save_path}")
            
            return plt.gcf()  # Return the figure
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            return None