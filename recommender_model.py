"""
Core recommendation model using FedPer approach.
Integrates with ProductEmbeddings class from embedder.py
"""

import numpy as np
import logging
from datetime import datetime, timedelta

logger = logging.getLogger("FedPerRecommender")

class FedPerRecommender:
    """
    Hybrid recommendation system using FedPer (Federated Personalization) approach.
    Combines collaborative filtering with content-based recommendations.
    
    Integration with ProductEmbeddings:
    - Uses product embeddings for content-based component
    - Leverages both transaction and description-based embeddings
    - Maintains separation between global shared parameters and local personalized parameters
    """
    
    def __init__(self, n_factors=20, content_weight=0.3, learning_rate=0.01, reg=0.1):
        """Initialize the recommender with FedPer approach."""
        self.n_factors = n_factors
        self.content_weight = content_weight
        self.learning_rate = learning_rate
        self.reg = reg
        
        # Core matrices
        self.U = None  # User matrix (personalized local parameters)
        self.V = None  # Item matrix (shared global parameters)
        
        # Bias terms
        self.bias_global = 0
        self.bias_user = None  # Personalized local parameters
        self.bias_item = None  # Shared global parameters
        
        # Tracking
        self.customer_purchases = {}  # Track customer purchase history
        self.embedding_vectors = {}   # Cache for embedding vectors

    def fit_local(self, interactions, n_users, n_items, product_emb, item_encoder, epochs=10):
        """
        Train a local model on segment data - core FedPer approach.
        
        In FedPer:
        - Item matrices (V) are shared global parameters
        - User matrices (U) are personalized local parameters
        
        Args:
            interactions: List of (user_idx, item_idx, rating) tuples
            n_users: Number of users in the segment
            n_items: Number of items in the catalog
            product_emb: Dictionary of product embeddings from ProductEmbeddings
            item_encoder: Encoder to map item IDs to indices
            epochs: Number of training epochs
        """
        # Initialize user and item matrices with small random values
        self.U = np.random.normal(0, 0.1, (n_users, self.n_factors))  # Local
        self.V = np.random.normal(0, 0.1, (n_items, self.n_factors))  # Global
        
        # Initialize bias terms
        self.bias_global = 0  # Global
        self.bias_user = np.zeros(n_users)  # Local
        self.bias_item = np.zeros(n_items)  # Global
        
        # Cache embedding vectors for performance
        self.embedding_vectors = {}
        for item_idx in range(n_items):
            try:
                stock_code = item_encoder.inverse_transform([item_idx])[0]
                if stock_code in product_emb:
                    raw_emb = product_emb[stock_code]
                    # Ensure embedding has correct dimensions
                    content_emb = np.zeros(self.n_factors)
                    if len(raw_emb) >= self.n_factors:
                        content_emb = raw_emb[:self.n_factors]
                    else:
                        # Pad if needed
                        content_emb[:len(raw_emb)] = raw_emb
                    self.embedding_vectors[item_idx] = content_emb
            except Exception as e:
                pass  # Silently continue
        
        # Training loop
        for epoch in range(epochs):
            np.random.shuffle(interactions)  # Shuffle to improve convergence
            
            # Track metrics
            rmse = 0
            n_samples = 0
            
            for user_idx, item_idx, rating in interactions:
                # Skip invalid indices
                if user_idx >= n_users or item_idx >= n_items:
                    continue
                    
                # Get content embedding for this item
                content_emb = self.embedding_vectors.get(item_idx, np.zeros(self.n_factors))
                
                # Combine content and collaborative filtering
                combined_V = (1 - self.content_weight) * self.V[item_idx] + self.content_weight * content_emb
                
                # Calculate prediction with bias terms
                prediction = self.bias_global + self.bias_user[user_idx] + self.bias_item[item_idx] + np.dot(self.U[user_idx], combined_V)
                
                # Calculate error
                error = rating - prediction
                rmse += error ** 2
                n_samples += 1
                
                # Update bias terms
                self.bias_global += self.learning_rate * (error - self.reg * self.bias_global)
                self.bias_user[user_idx] += self.learning_rate * (error - self.reg * self.bias_user[user_idx])
                self.bias_item[item_idx] += self.learning_rate * (error - self.reg * self.bias_item[item_idx])
                
                # Update parameters with gradient descent and regularization
                self.U[user_idx] += self.learning_rate * (error * combined_V - self.reg * self.U[user_idx])
                self.V[item_idx] += self.learning_rate * (error * self.U[user_idx] * (1 - self.content_weight) - self.reg * self.V[item_idx])
            
            # Report progress
            if n_samples > 0:
                epoch_rmse = np.sqrt(rmse / n_samples)
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    logger.info(f"    Epoch {epoch + 1}/{epochs}, RMSE: {epoch_rmse:.4f}")

    def update_model(self, new_interactions, customer_encoder, item_encoder, product_emb, epochs=5):
        """
        Update local model with new interaction data.
        
        Args:
            new_interactions: List of (user_idx, item_idx, rating) tuples
            customer_encoder: Encoder to map customer IDs to indices
            item_encoder: Encoder to map item IDs to indices
            product_emb: Dictionary of product embeddings from ProductEmbeddings
            epochs: Number of fine-tuning epochs
        """
        if self.U is None or self.V is None:
            logger.error("Cannot update model - not initialized")
            return False
            
        # Ensure matrices are large enough
        n_users = max(np.max([ui[0] for ui in new_interactions]) + 1, self.U.shape[0]) if new_interactions else self.U.shape[0]
        n_items = max(np.max([ui[1] for ui in new_interactions]) + 1, self.V.shape[0]) if new_interactions else self.V.shape[0]
        
        # Resize if needed
        if n_users > self.U.shape[0]:
            old_size = self.U.shape[0]
            new_rows = np.random.normal(0, 0.1, (n_users - old_size, self.n_factors))
            self.U = np.vstack([self.U, new_rows])
            
            # Extend bias term
            if self.bias_user is not None:
                self.bias_user = np.append(self.bias_user, np.zeros(n_users - old_size))
                
            logger.info(f"Expanded user matrix from {old_size} to {n_users} users")
            
        if n_items > self.V.shape[0]:
            old_size = self.V.shape[0]
            new_rows = np.random.normal(0, 0.1, (n_items - old_size, self.n_factors))
            self.V = np.vstack([self.V, new_rows])
            
            # Extend bias term
            if self.bias_item is not None:
                self.bias_item = np.append(self.bias_item, np.zeros(n_items - old_size))
                
            logger.info(f"Expanded item matrix from {old_size} to {n_items} items")
        
        # Update embedding cache for new items
        for item_idx in range(old_size, n_items):
            try:
                stock_code = item_encoder.inverse_transform([item_idx])[0]
                if stock_code in product_emb:
                    raw_emb = product_emb[stock_code]
                    content_emb = np.zeros(self.n_factors)
                    if len(raw_emb) >= self.n_factors:
                        content_emb = raw_emb[:self.n_factors]
                    else:
                        content_emb[:len(raw_emb)] = raw_emb
                    self.embedding_vectors[item_idx] = content_emb
            except Exception as e:
                pass  # Continue silently
        
        # Fine-tune on new interactions
        for epoch in range(epochs):
            np.random.shuffle(new_interactions)
            
            # Track metrics
            rmse = 0
            n_samples = 0
            
            for user_idx, item_idx, rating in new_interactions:
                # Skip invalid indices
                if user_idx >= n_users or item_idx >= n_items:
                    continue
                
                # Get content embedding
                content_emb = self.embedding_vectors.get(item_idx, np.zeros(self.n_factors))
                
                # Combine content and collaborative filtering
                combined_V = (1 - self.content_weight) * self.V[item_idx] + self.content_weight * content_emb
                
                # Calculate prediction with bias terms
                prediction = self.bias_global + self.bias_user[user_idx] + self.bias_item[item_idx] + np.dot(self.U[user_idx], combined_V)
                
                # Calculate error
                error = rating - prediction
                rmse += error ** 2
                n_samples += 1
                
                # Use a smaller learning rate for updates
                fine_tune_lr = self.learning_rate * 0.5
                
                # Update bias terms
                self.bias_global += fine_tune_lr * (error - self.reg * self.bias_global)
                self.bias_user[user_idx] += fine_tune_lr * (error - self.reg * self.bias_user[user_idx])
                self.bias_item[item_idx] += fine_tune_lr * (error - self.reg * self.bias_item[item_idx])
                
                # Update parameters
                self.U[user_idx] += fine_tune_lr * (error * combined_V - self.reg * self.U[user_idx])
                self.V[item_idx] += fine_tune_lr * (error * self.U[user_idx] * (1 - self.content_weight) - self.reg * self.V[item_idx])
        
            # Report progress for the last epoch
            if n_samples > 0 and epoch == epochs - 1:
                epoch_rmse = np.sqrt(rmse / n_samples)
                logger.info(f"Update complete. Final RMSE: {epoch_rmse:.4f}")
                
        logger.info(f"Model updated with {len(new_interactions)} new interactions")
        return True

    def aggregate(self, local_models):
        """
        Aggregate multiple local models into a global model following FedPer approach.
        
        In FedPer:
        - Only global parameters (V, bias_global, bias_item) are aggregated
        - Local parameters (U, bias_user) remain separate for personalization
        
        Args:
            local_models: List of local FedPerRecommender instances
        """
        if not local_models:
            raise ValueError("No local models to aggregate")
        
        # Find the maximum dimensions across all models
        max_users = max(model.U.shape[0] for model in local_models)
        max_items = max(model.V.shape[0] for model in local_models)
        
        # Initialize global matrices
        self.U = np.zeros((max_users, self.n_factors))
        self.V = np.zeros((max_items, self.n_factors))
        self.bias_global = 0
        self.bias_user = np.zeros(max_users)
        self.bias_item = np.zeros(max_items)
        
        # Track how many models contribute to each row
        item_counts = np.zeros(max_items)
        global_count = 0
        
        # Sum all models - but only the shared parameters (V, bias_global, bias_item)
        for model in local_models:
            n_items, _ = model.V.shape
            
            # Add item embeddings (shared across segments)
            self.V[:n_items] += model.V
            item_counts[:n_items] += 1
            
            # Add global bias terms
            if model.bias_global is not None:
                self.bias_global += model.bias_global
                global_count += 1
                
            if model.bias_item is not None and len(model.bias_item) == n_items:
                self.bias_item[:n_items] += model.bias_item
        
        # Average the values where we have contributions
        # Only average the shared parameters!
        for i in range(max_items):
            if item_counts[i] > 0:
                self.V[i] /= item_counts[i]
                self.bias_item[i] /= item_counts[i]
        
        if global_count > 0:
            self.bias_global /= global_count
        
        logger.info(f"Global model created with shape: Items {self.V.shape}")
        logger.info(f"Note: User matrices are not aggregated in FedPer - they remain personalized")

    def synchronize_with_global(self, global_model, learning_rate=0.2):
        """
        Update local model to incorporate knowledge from global model.
        
        In FedPer:
        - Only global parameters (V, bias_global, bias_item) are synchronized
        - Local parameters (U, bias_user) remain untouched
        
        Args:
            global_model: The global FedPerRecommender model
            learning_rate: Controls how much the local model adopts global parameters
        """
        if self.V is None or global_model.V is None:
            logger.error("Cannot synchronize - models not initialized")
            return False
            
        # Get common dimensions - only for global parameters
        n_items = min(self.V.shape[0], global_model.V.shape[0])
        
        # Update local model as weighted average with global model - only shared parameters
        self.V[:n_items] = (1 - learning_rate) * self.V[:n_items] + learning_rate * global_model.V[:n_items]
        
        # Update bias terms - only global ones
        if self.bias_global is not None and global_model.bias_global is not None:
            self.bias_global = (1 - learning_rate) * self.bias_global + learning_rate * global_model.bias_global
            
        if self.bias_item is not None and global_model.bias_item is not None:
            n_item_bias = min(len(self.bias_item), len(global_model.bias_item))
            self.bias_item[:n_item_bias] = (1 - learning_rate) * self.bias_item[:n_item_bias] + learning_rate * global_model.bias_item[:n_item_bias]
        
        logger.info(f"Local model synchronized with global model (rate: {learning_rate})")
        logger.info(f"Note: Only global parameters were updated according to FedPer approach")
        return True

    def recommend(self, user_idx, item_encoder, product_info, n_items=5, exclude_items=None, 
                 product_emb=None, price_range=None, diversity_weight=0.3, 
                 rec_type='hybrid'):
        """
        Generate recommendations for a specific user.
        
        Args:
            user_idx: Index of the user
            item_encoder: Encoder to map item IDs to indices
            product_info: Dictionary of product information
            n_items: Number of items to recommend
            exclude_items: List of item indices to exclude
            product_emb: Dictionary of product embeddings from ProductEmbeddings
            price_range: Tuple of (min_price, max_price) to filter by price
            diversity_weight: Weight given to diversity in ranking (0-1)
            rec_type: Type of recommendation ('collaborative', 'content', or 'hybrid')
        
        Returns:
            List of recommendation dictionaries
        """
        if self.U is None or self.V is None:
            return []
        
        if user_idx >= self.U.shape[0]:
            return []  # User index out of bounds
        
        # Update embedding cache if new embeddings are provided
        if product_emb:
            for item_idx in range(self.V.shape[0]):
                try:
                    stock_code = item_encoder.inverse_transform([item_idx])[0]
                    if stock_code in product_emb:
                        raw_emb = product_emb[stock_code]
                        content_emb = np.zeros(self.n_factors)
                        if len(raw_emb) >= self.n_factors:
                            content_emb = raw_emb[:self.n_factors]
                        else:
                            content_emb[:len(raw_emb)] = raw_emb
                        self.embedding_vectors[item_idx] = content_emb
                except Exception as e:
                    pass  # Continue silently
        
        # Adjust content weight based on recommendation type
        content_weight = 0.0  # Collaborative filtering
        if rec_type == 'content':
            content_weight = 1.0  # Pure content-based
        elif rec_type == 'hybrid':
            content_weight = self.content_weight  # Hybrid approach
        
        # Calculate recommendation scores
        scores = np.zeros(self.V.shape[0])
        for i in range(self.V.shape[0]):
            # Get content embedding
            content_emb = self.embedding_vectors.get(i, np.zeros(self.n_factors))
            
            # Combine content and collaborative filtering based on rec_type
            combined_V = (1 - content_weight) * self.V[i] + content_weight * content_emb
            
            # Calculate prediction with bias terms
            if i < len(self.bias_item):
                scores[i] = self.bias_global + self.bias_user[user_idx] + self.bias_item[i] + np.dot(self.U[user_idx], combined_V)
            else:
                scores[i] = self.bias_global + self.bias_user[user_idx] + np.dot(self.U[user_idx], combined_V)
        
        # Remove excluded items
        if exclude_items:
            for idx in exclude_items:
                if 0 <= idx < len(scores):
                    scores[idx] = -np.inf
        
        # Apply price range filter if specified
        if price_range and product_info:
            min_price, max_price = price_range
            for i in range(len(scores)):
                try:
                    stock_code = item_encoder.inverse_transform([i])[0]
                    if stock_code in product_info:
                        price = product_info[stock_code].get('median_price', 0)
                        if price < min_price or price > max_price:
                            scores[i] = -np.inf
                except Exception as e:
                    logger.debug(f"Error applying price filter: {str(e)}")
        
        # Get top items
        top_indices = np.argsort(scores)[::-1][:n_items*2]  # Get more than needed for filtering
        
        # Convert to product info with scores
        recommendations = []
        for idx in top_indices:
            if scores[idx] == -np.inf:
                continue
                
            try:
                stock_code = item_encoder.inverse_transform([idx])[0]
                if stock_code in product_info:
                    info = product_info[stock_code]
                    recommendations.append({
                        'stock_code': stock_code,
                        'name': info.get('description', 'Unknown Product'),
                        'price': info.get('median_price', 0),
                        'score': float(scores[idx]),
                        'popularity': info.get('total_sold', 0),
                        'recommendation_type': rec_type
                    })
            except Exception as e:
                logger.error(f"Error getting info for item index {idx}: {str(e)}")
        
        # Final diversity-aware ranking
        recommendations = self._diversity_ranking(recommendations, n_items, diversity_weight)
        return recommendations[:n_items]
    
    def _diversity_ranking(self, recommendations, n_items, diversity_weight=0.3):
        """
        Re-rank recommendations to improve diversity.
        
        Args:
            recommendations: List of recommendation dictionaries
            n_items: Number of items to return
            diversity_weight: Weight given to diversity (0-1)
        """
        if len(recommendations) <= n_items:
            return recommendations
            
        # Sort by score initially
        sorted_recs = sorted(recommendations, key=lambda x: x['score'], reverse=True)
        
        # Always include the top item
        selected = [sorted_recs[0]]
        candidates = sorted_recs[1:]
        
        # Select remaining items with diversity in mind
        while len(selected) < n_items and candidates:
            # Calculate a diversity score for each candidate
            diversity_scores = []
            
            for i, item in enumerate(candidates):
                # Base score (original recommendation score)
                base_score = item['score'] 
                
                # Price diversity (prefer items with different price points)
                price_diversity = min(
                    abs(item['price'] - s['price']) / (max(item['price'], s['price']) + 0.1) 
                    for s in selected
                )
                
                # Combine scores
                diversity_scores.append((1 - diversity_weight) * base_score + diversity_weight * price_diversity)
            
            # Select the best candidate
            best_idx = np.argmax(diversity_scores)
            selected.append(candidates[best_idx])
            candidates.pop(best_idx)
        
        return selected

    def track_purchase(self, customer_id, item_id, timestamp=None, quantity=1):
        """
        Track a new purchase for use in recommendations.
        
        Args:
            customer_id: ID of the customer
            item_id: ID of the purchased item
            timestamp: When the purchase occurred (defaults to now)
            quantity: Number of items purchased
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        if customer_id not in self.customer_purchases:
            self.customer_purchases[customer_id] = []
            
        self.customer_purchases[customer_id].append({
            'item_id': item_id,
            'timestamp': timestamp,
            'quantity': quantity
        })
        
        # Keep only recent purchases (last 90 days)
        cutoff = datetime.now() - timedelta(days=90)
        self.customer_purchases[customer_id] = [
            p for p in self.customer_purchases[customer_id] 
            if p['timestamp'] >= cutoff
        ]
        
        return True
        
    def calculate_metrics(self, test_interactions, item_encoder, product_emb=None):
        """
        Calculate evaluation metrics on test data.
        
        Args:
            test_interactions: List of (user_idx, item_idx, rating) tuples
            item_encoder: Encoder to map item IDs to indices
            product_emb: Dictionary of product embeddings from ProductEmbeddings
        
        Returns:
            Dictionary of evaluation metrics
        """
        if self.U is None or self.V is None:
            logger.error("Model not trained")
            return {}
        
        n_test = len(test_interactions)
        if n_test == 0:
            return {}
            
        # Calculate RMSE and MAE
        rmse_sum = 0
        mae_sum = 0
        
        for user_idx, item_idx, true_rating in test_interactions:
            # Skip invalid indices
            if user_idx >= self.U.shape[0] or item_idx >= self.V.shape[0]:
                continue
                
            # Get content embedding from cache or product_emb
            content_emb = self.embedding_vectors.get(item_idx, None)
            
            # If not in cache, try to get from product_emb
            if content_emb is None and product_emb:
                try:
                    stock_code = item_encoder.inverse_transform([item_idx])[0]
                    if stock_code in product_emb:
                        raw_emb = product_emb[stock_code]
                        content_emb = np.zeros(self.n_factors)
                        if len(raw_emb) >= self.n_factors:
                            content_emb = raw_emb[:self.n_factors]
                        else:
                            content_emb[:len(raw_emb)] = raw_emb
                        # Cache for future use
                        self.embedding_vectors[item_idx] = content_emb
                except Exception as e:
                    content_emb = np.zeros(self.n_factors)
            
            # Default to zeros if still None
            if content_emb is None:
                content_emb = np.zeros(self.n_factors)
            
            # Combine content and collaborative filtering
            combined_V = (1 - self.content_weight) * self.V[item_idx] + self.content_weight * content_emb
            
            # Calculate prediction
            pred_rating = self.bias_global + self.bias_user[user_idx] + self.bias_item[item_idx] + np.dot(self.U[user_idx], combined_V)
            
            # Update error sums
            rmse_sum += (true_rating - pred_rating) ** 2
            mae_sum += abs(true_rating - pred_rating)
        
        # Calculate average metrics
        metrics = {
            'rmse': np.sqrt(rmse_sum / n_test),
            'mae': mae_sum / n_test
        }
        
        return metrics
    
    def get_collaborative_recommendations(self, user_idx, item_encoder, product_info, n_items=5, exclude_items=None):
        """
        Generate recommendations using only collaborative filtering.
        
        Args:
            user_idx: Index of the user
            item_encoder: Encoder to map item IDs to indices
            product_info: Dictionary of product information
            n_items: Number of items to recommend
            exclude_items: List of item indices to exclude
        """
        return self.recommend(
            user_idx=user_idx,
            item_encoder=item_encoder,
            product_info=product_info,
            n_items=n_items,
            exclude_items=exclude_items,
            rec_type='collaborative'
        )
    
    def get_content_recommendations(self, user_idx, item_encoder, product_info, product_emb, n_items=5, exclude_items=None):
        """
        Generate recommendations using only content-based filtering.
        
        Args:
            user_idx: Index of the user
            item_encoder: Encoder to map item IDs to indices
            product_info: Dictionary of product information
            product_emb: Dictionary of product embeddings from ProductEmbeddings
            n_items: Number of items to recommend
            exclude_items: List of item indices to exclude
        """
        return self.recommend(
            user_idx=user_idx,
            item_encoder=item_encoder,
            product_info=product_info,
            n_items=n_items,
            exclude_items=exclude_items,
            product_emb=product_emb,
            rec_type='content'
        )
    
    def get_hybrid_recommendations(self, user_idx, item_encoder, product_info, product_emb, n_items=5, exclude_items=None):
        """
        Generate recommendations using hybrid approach (collaborative + content).
        
        Args:
            user_idx: Index of the user
            item_encoder: Encoder to map item IDs to indices
            product_info: Dictionary of product information
            product_emb: Dictionary of product embeddings from ProductEmbeddings
            n_items: Number of items to recommend
            exclude_items: List of item indices to exclude
        """
        return self.recommend(
            user_idx=user_idx,
            item_encoder=item_encoder,
            product_info=product_info,
            n_items=n_items,
            exclude_items=exclude_items,
            product_emb=product_emb,
            rec_type='hybrid'
        )