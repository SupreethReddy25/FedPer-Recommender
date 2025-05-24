"""
REST API for the recommendation system.
"""

import logging
from datetime import datetime

logger = logging.getLogger("FedPerRecommender")

class RecommendationAPI:
    """REST API for the recommendation system."""
    
    def __init__(self, recommendation_manager):
        """Initialize with a recommendation manager."""
        self.manager = recommendation_manager
        self.last_background_update = datetime.now()
    
    def add_customer(self, customer_data):
        """
        Add a new customer to the system.
        
        Args:
            customer_data: Dict with customer information
                - customer_id: Unique identifier
                - name: Customer name (optional)
                - email: Customer email (optional)
        
        Returns:
            Dict with status and message
        """
        if 'customer_id' not in customer_data:
            return {
                'success': False,
                'message': 'Missing required field: customer_id'
            }
        
        try:
            customer_id = float(customer_data['customer_id'])
            success = self.manager.add_new_customer(customer_id)
            
            return {
                'success': success,
                'message': 'Customer added successfully' if success else 'Customer already exists'
            }
        except Exception as e:
            logger.error(f"Error adding customer: {str(e)}")
            return {
                'success': False,
                'message': f'Error: {str(e)}'
            }
    
    def record_purchase(self, transaction_data):
        """
        Record a purchase transaction.
        
        Args:
            transaction_data: Dict with transaction details
        
        Returns:
            Dict with status and message
        """
        try:
            success = self.manager.record_transaction(transaction_data)
            
            return {
                'success': success,
                'message': 'Transaction recorded successfully' if success else 'Failed to record transaction'
            }
        except Exception as e:
            logger.error(f"Error recording purchase: {str(e)}")
            return {
                'success': False,
                'message': f'Error: {str(e)}'
            }
    
    def get_recommendations(self, params):
        """
        Get product recommendations for a customer.
        
        Args:
            params: Dict with parameters
                - customer_id: Customer identifier
                - count: Number of recommendations (default: 5)
                - model: 'auto', 'segment', or 'global' (default: 'auto')
                - min_price: Minimum price filter (optional)
                - max_price: Maximum price filter (optional)
        
        Returns:
            Dict with recommendations list
        """
        if 'customer_id' not in params:
            return {
                'success': False,
                'message': 'Missing required parameter: customer_id',
                'recommendations': []
            }
        
        try:
            customer_id = float(params['customer_id'])
            count = int(params.get('count', 5))
            model = params.get('model', 'auto')
            
            # Parse price range if provided
            price_range = None
            if 'min_price' in params and 'max_price' in params:
                min_price = float(params['min_price'])
                max_price = float(params['max_price'])
                price_range = (min_price, max_price)
            
            # Map model parameter to function arguments
            use_global = model == 'global'
            personalization_level = model
            
            # Get recommendations
            recommendations = self.manager.generate_recommendations(
                customer_id,
                n_items=count,
                use_global=use_global,
                personalization_level=personalization_level,
                price_range=price_range
            )
            
            # Check for scheduled background updates
            self._check_background_update()
            
            return {
                'success': True,
                'recommendations': recommendations
            }
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return {
                'success': False,
                'message': f'Error: {str(e)}',
                'recommendations': []
            }
    
    def get_segment_info(self, segment_name):
        """
        Get information about a customer segment.
        
        Args:
            segment_name: Name of the segment to get info for
            
        Returns:
            Dict with segment insights
        """
        try:
            insights = self.manager.get_segment_insights(segment_name)
            
            if insights:
                return {
                    'success': True,
                    'segment': segment_name,
                    'insights': insights
                }
            else:
                return {
                    'success': False,
                    'message': f'Segment {segment_name} not found or has no data',
                    'insights': {}
                }
        except Exception as e:
            logger.error(f"Error getting segment info: {str(e)}")
            return {
                'success': False,
                'message': f'Error: {str(e)}',
                'insights': {}
            }
    
    def update_models(self):
        """
        Force an update of the recommendation models.
        
        Returns:
            Dict with status and message
        """
        try:
            success = self.manager.update_models()
            
            return {
                'success': success,
                'message': 'Models updated successfully' if success else 'No new transactions to process'
            }
        except Exception as e:
            logger.error(f"Error updating models: {str(e)}")
            return {
                'success': False,
                'message': f'Error: {str(e)}'
            }
    
    def _check_background_update(self):
        """Check if we need to run a background update of the models."""
        days_since_update = (datetime.now() - self.last_background_update).days
        
        if days_since_update >= self.manager.update_frequency_days:
            # Force model update if there are pending transactions
            if self.manager.new_transactions:
                logger.info("Running scheduled background model update")
                self.manager.update_models()
                self.last_background_update = datetime.now()
