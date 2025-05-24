"""
Website integration module for the recommendation system.
"""

import logging
import numpy as np
from datetime import datetime

logger = logging.getLogger("FedPerRecommender")

class WebsiteIntegration:
    """Handles integration with website backend."""
    
    def __init__(self, recommendation_manager):
        """Initialize with a recommendation manager."""
        self.manager = recommendation_manager
        
    def user_login(self, user_id):
        """Handle user login event."""
        # Check if user exists in our system
        try:
            customer_id = float(user_id)
            if customer_id not in self.manager.data_processor.customer_encoder.classes_:
                # New customer - add to system
                self.manager.add_new_customer(customer_id)
                return {
                    'success': True,
                    'is_new_customer': True,
                    'message': 'New customer added to recommendation system'
                }
            else:
                return {
                    'success': True,
                    'is_new_customer': False,
                    'has_segment': customer_id in self.manager.rfm_table['CustomerID'].values
                }
        except Exception as e:
            logger.error(f"Error in user_login: {str(e)}")
            return {
                'success': False,
                'message': f'Error: {str(e)}'
            }
    
    def user_views_product(self, user_id, product_id):
        """Record product view event."""
        # Could be extended to store product view history
        # For now, just check if both user and product exist
        try:
            customer_id = float(user_id)
            stock_code = str(product_id)
            
            user_exists = customer_id in self.manager.data_processor.customer_encoder.classes_
            product_exists = stock_code in self.manager.data_processor.item_encoder.classes_
            
            return {
                'success': True,
                'user_exists': user_exists,
                'product_exists': product_exists
            }
        except Exception as e:
            logger.error(f"Error in user_views_product: {str(e)}")
            return {
                'success': False,
                'message': f'Error: {str(e)}'
            }
    
    def get_homepage_recommendations(self, user_id, max_items=12):
        """Get recommendations for website homepage."""
        try:
            customer_id = float(user_id)
            
            # Divide recommendations into categories
            # 1/3 personalized recommendations
            # 1/3 popular items recommendations
            # 1/3 new items recommendations
            
            personalized_count = max(1, max_items // 3)
            popular_count = max(1, max_items // 3)
            new_count = max_items - personalized_count - popular_count
            
            recommendations = {
                'personalized': [],
                'popular': [],
                'new_arrivals': []
            }
            
            # Get personalized recommendations
            if customer_id in self.manager.data_processor.customer_encoder.classes_:
                recs = self.manager.generate_recommendations(
                    customer_id,
                    n_items=personalized_count,
                    personalization_level='auto'
                )
                
                if recs:
                    recommendations['personalized'] = recs
            
            # Get popular items (highest total_sold)
            if self.manager.product_info:
                popular_items = sorted(
                    self.manager.product_info.items(), 
                    key=lambda x: x[1].get('total_sold', 0),
                    reverse=True
                )[:popular_count*2]  # Get extra for filtering
                
                # Convert to recommendation format
                for stock_code, info in popular_items:
                    if len(recommendations['popular']) >= popular_count:
                        break
                        
                    # Skip if already in personalized recommendations
                    if any(rec['stock_code'] == stock_code for rec in recommendations['personalized']):
                        continue
                        
                    recommendations['popular'].append({
                        'stock_code': stock_code,
                        'name': info.get('description', 'Unknown Product'),
                        'price': info.get('median_price', 0),
                        'popularity': info.get('total_sold', 0),
                        'score': 1.0  # Default score
                    })
            
            # Get new arrivals (recent additions to catalog)
            # In a real system, you would track product addition dates
            # Here we'll simulate with random selection from less popular items
            if self.manager.product_info and len(self.manager.product_info) > 100:
                # Get random selection of items that aren't already in recommendations
                all_codes = set(self.manager.product_info.keys())
                existing_codes = set(rec['stock_code'] for rec in recommendations['personalized'])
                existing_codes.update(rec['stock_code'] for rec in recommendations['popular'])
                
                available_codes = list(all_codes - existing_codes)
                
                # Select new_count random items
                if len(available_codes) >= new_count:
                    selected_codes = np.random.choice(available_codes, new_count, replace=False)
                    
                    for stock_code in selected_codes:
                        info = self.manager.product_info[stock_code]
                        recommendations['new_arrivals'].append({
                            'stock_code': stock_code,
                            'name': info.get('description', 'Unknown Product'),
                            'price': info.get('median_price', 0),
                            'popularity': info.get('total_sold', 0),
                            'score': 0.5  # Default score
                        })
            
            return {
                'success': True,
                'recommendations': recommendations
            }
        except Exception as e:
            logger.error(f"Error in get_homepage_recommendations: {str(e)}")
            return {
                'success': False,
                'message': f'Error: {str(e)}',
                'recommendations': {'personalized': [], 'popular': [], 'new_arrivals': []}
            }
    
    def get_product_detail_recommendations(self, user_id, product_id, count=6):
        """Get 'customers who bought this also bought' recommendations."""
        try:
            customer_id = float(user_id)
            stock_code = str(product_id)
            
            # This could use collaborative filtering directly to find similar products
            # For now, we'll use our existing recommendation system
            
            # First, get personalized recommendations
            recommendations = []
            
            if customer_id in self.manager.data_processor.customer_encoder.classes_:
                recs = self.manager.generate_recommendations(
                    customer_id,
                    n_items=count * 2,  # Get more than needed to filter
                    personalization_level='auto'
                )
                
                if recs:
                    # Filter out the current product
                    recommendations = [
                        rec for rec in recs
                        if rec['stock_code'] != stock_code
                    ][:count]
            
            # If we couldn't get personalized recommendations or not enough,
            # fall back to popular items
            if len(recommendations) < count and self.manager.product_info:
                # Get popular items not already in recommendations
                existing_codes = set(rec['stock_code'] for rec in recommendations)
                existing_codes.add(stock_code)  # Exclude current product
                
                popular_items = sorted(
                    [(k, v) for k, v in self.manager.product_info.items() if k not in existing_codes],
                    key=lambda x: x[1].get('total_sold', 0),
                    reverse=True
                )[:count - len(recommendations)]
                
                # Add to recommendations
                for stock_code, info in popular_items:
                    recommendations.append({
                        'stock_code': stock_code,
                        'name': info.get('description', 'Unknown Product'),
                        'price': info.get('median_price', 0),
                        'popularity': info.get('total_sold', 0),
                        'score': 0.5  # Default score
                    })
            
            return {
                'success': True,
                'recommendations': recommendations
            }
        except Exception as e:
            logger.error(f"Error in get_product_detail_recommendations: {str(e)}")
            return {
                'success': False,
                'message': f'Error: {str(e)}',
                'recommendations': []
            }
    
    def record_purchase_from_website(self, order_data):
        """
        Record purchases from website checkout.
        
        Args:
            order_data: dict with keys:
                - user_id: Customer ID
                - order_id: Order number
                - order_date: Date of order
                - items: List of dicts with product info:
                    - product_id: Product ID
                    - product_name: Product name
                    - quantity: Number of items 
                    - price: Price per unit
        """
        try:
            customer_id = float(order_data['user_id'])
            order_id = order_data['order_id']
            order_date = order_data['order_date']
            
            # Process each item in the order
            for item in order_data['items']:
                # Create transaction record
                transaction_data = {
                    'customer_id': customer_id,
                    'invoice_no': order_id,
                    'invoice_date': order_date,
                    'stock_code': str(item['product_id']),
                    'description': item['product_name'],
                    'quantity': int(item['quantity']),
                    'unit_price': float(item['price'])
                }
                
                # Record transaction
                self.manager.record_transaction(transaction_data)
            
            return {
                'success': True,
                'message': f'Order {order_id} recorded successfully'
            }
        except Exception as e:
            logger.error(f"Error in record_purchase_from_website: {str(e)}")
            return {
                'success': False, 
                'message': f'Error: {str(e)}'
            }
