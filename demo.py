"""
Interactive demonstration of the recommendation system.
"""

import logging
from datetime import datetime

from .api import RecommendationAPI

logger = logging.getLogger("FedPerRecommender")

class RecommendationDemo:
    """Interactive demonstration of the recommendation system."""
    
    def __init__(self, recommendation_manager):
        """Initialize with a recommendation manager."""
        self.manager = recommendation_manager
        self.api = RecommendationAPI(recommendation_manager)
    
    def run(self):
        """Run the recommendation system demo."""
        print("\n===== FedPer Recommendation System Demo =====")
        print("A hybrid recommendation system using federated learning for customer segments")
        
        # Check if models are already loaded
        if not self.manager.global_model:
            # Try to load existing state
            if not self.load_existing_state():
                # If no state available, train new models
                self.train_new_models()
        
        # Show menu
        while True:
            print("\n===== FedPer Recommendation System Demo =====")
            print("1. Get recommendations for a customer")
            print("2. Add a new customer")
            print("3. Record a purchase")
            print("4. Compare recommendations (segment vs global)")
            print("5. Update models with new data")
            print("6. Save system state")
            print("7. Show customer segments")
            print("8. Exit")
            
            choice = input("\nEnter your choice (1-8): ")
            
            if choice == '1':
                self._get_recommendations()
            elif choice == '2':
                self._add_customer()
            elif choice == '3':
                self._record_purchase()
            elif choice == '4':
                self._compare_recommendations()
            elif choice == '5':
                self._update_models()
            elif choice == '6':
                self._save_state()
            elif choice == '7':
                self._show_segments()
            elif choice == '8':
                print("Thank you for using FedPer Recommendation System!")
                break
            else:
                print("Invalid choice. Please enter a number between 1 and 8.")
    
    def load_existing_state(self):
        """Load system state if available."""
        state_path = input("Enter path to saved state (or press Enter to train new models): ")
        if not state_path:
            return False
            
        success=self.manager.load_state(state_path)
        return success
    
    def train_new_models(self):
        """Train new models from scratch."""
        data_file = input("Enter path to retail data file (Excel or CSV): ")
        df, rfm_table, product_embeddings = self.manager.load_and_prepare_data(data_file)
        
        # Store in manager
        self.manager.df = df
        self.manager.rfm_table = rfm_table
        self.manager.product_embeddings = product_embeddings
        
        # Ask for training parameters
        epochs=int(input("Enter number of training epochs (default: 10): ") or 10)
        
        # Train models
        self.manager.train_models(df, product_embeddings, epochs=epochs)
        
        # Ask to save state
        save = input("Save system state? (y/n): ").lower() == 'y'
        if save:
            self._save_state()
    
    def _get_recommendations(self):
        """Get recommendations for a specific customer."""
        try:
            customer_id = float(input("Enter customer ID: "))
            count = int(input("Number of recommendations (default: 5): ") or 5)
            model_type = input("Model type (auto, segment, global) [default: auto]: ").lower() or 'auto'
            
            # Get price range if desired
            use_price_filter = input("Apply price filter? (y/n): ").lower() == 'y'
            price_range = None
            
            if use_price_filter:
                min_price = float(input("Minimum price: ") or 0)
                max_price = float(input("Maximum price: ") or 1000)
                price_range = (min_price, max_price)
            
            # Check if customer exists
            if customer_id not in self.manager.data_processor.customer_encoder.classes_:
                print(f"Customer {customer_id} not found in system.")
                add_new = input("Would you like to add this customer? (y/n): ").lower() == 'y'
                if add_new:
                    self.manager.add_new_customer(customer_id)
                    print(f"Customer {customer_id} added. They will initially receive global recommendations.")
                else:
                    return
            
            # Get recommendations using API
            params = {
                'customer_id': customer_id,
                'count': count,
                'model': model_type
            }
            
            if price_range:
                params['min_price'] = price_range[0]
                params['max_price'] = price_range[1]
            
            result = self.api.get_recommendations(params)
            
            if result['success']:
                recommendations = result['recommendations']
                if recommendations:
                    print(f"\nTop {len(recommendations)} recommendations for customer {customer_id}:")
                    print("=" * 60)
                    print(f"{'Stock Code':<12} {'Product Name':<30} {'Price':>8} {'Score':>8}")
                    print("-" * 60)
                    for i, rec in enumerate(recommendations):
                        print(f"{rec['stock_code']:<12} {rec['name'][:30]:<30} {rec['price']:>8.2f} {rec['score']:>8.2f}")
                    print("=" * 60)
                    
                    # Show customer segment if available
                    if customer_id in self.manager.rfm_table['CustomerID'].values:
                        segment = self.manager.rfm_table[
                            self.manager.rfm_table['CustomerID'] == customer_id
                        ]['Segment'].iloc[0]
                        print(f"Customer Segment: {segment}")
                    else:
                        print("Customer Segment: Not yet classified (new customer)")
                        
                    # Show which model was used
                    if customer_id in self.manager.pending_customers:
                        print("Model Used: Global (customer not yet segmented)")
                    elif model_type == 'global':
                        print("Model Used: Global (forced)")
                    elif model_type == 'segment':
                        print("Model Used: Segment-specific")
                    else:
                        # For 'auto' - show what was actually used
                        if customer_id in self.manager.pending_customers or customer_id not in self.manager.rfm_table['CustomerID'].values:
                            print("Model Used: Global (auto-selected for new customer)")
                        else:
                            segment = self.manager.rfm_table[self.manager.rfm_table['CustomerID'] == customer_id]['Segment'].iloc[0]
                            if segment in self.manager.segment_models:
                                print(f"Model Used: Segment-specific ({segment})")
                            else:
                                print("Model Used: Global (segment has no specific model)")
                else:
                    print("No recommendations available for this customer.")
            else:
                print(f"Error: {result.get('message', 'Unknown error')}")
        except Exception as e:
            print(f"Error: {str(e)}")

    def _add_customer(self):
        """Add a new customer to the system."""
        try:
            customer_id = float(input("Enter new customer ID: "))
            name = input("Enter customer name (optional): ")
            email = input("Enter customer email (optional): ")
            
            # Add customer via the API
            customer_data = {
                'customer_id': customer_id,
                'name': name,
                'email': email
            }
            
            result = self.api.add_customer(customer_data)
            
            if result['success']:
                print(f"Customer {customer_id} added successfully.")
                print("This customer will initially receive recommendations from the global model.")
                print("After they make purchases, they will be classified into a segment and receive personalized recommendations.")
            else:
                print(f"Error: {result.get('message', 'Failed to add customer')}")
        except Exception as e:
            print(f"Error: {str(e)}")

    def _record_purchase(self):
        """Record a purchase transaction."""
        try:
            # Get transaction details
            customer_id = float(input("Enter customer ID: "))
            
            # Check if customer exists
            if customer_id not in self.manager.data_processor.customer_encoder.classes_:
                print(f"Customer {customer_id} not found in system.")
                add_new = input("Would you like to add this customer? (y/n): ").lower() == 'y'
                if add_new:
                    self.manager.add_new_customer(customer_id)
                    print(f"Customer {customer_id} added.")
                else:
                    return
            
            invoice_no = input("Enter invoice number: ")
            invoice_date = input("Enter invoice date (YYYY-MM-DD) [default: today]: ")
            if not invoice_date:
                invoice_date = datetime.now().strftime("%Y-%m-%d")
                
            # Get product details
            stock_code = input("Enter product code: ")
            
            # Check if product exists and show info if it does
            if stock_code in self.manager.product_info:
                product_info = self.manager.product_info[stock_code]
                print(f"Product: {product_info.get('description', 'Unknown')}")
                print(f"Current price: {product_info.get('median_price', 0):.2f}")
                use_existing = input("Use this description and price? (y/n): ").lower() == 'y'
                
                if use_existing:
                    description = product_info.get('description', '')
                    unit_price = product_info.get('median_price', 0)
                else:
                    description = input("Enter product description: ")
                    unit_price = float(input("Enter unit price: "))
            else:
                description = input("Enter product description: ")
                unit_price = float(input("Enter unit price: "))
            
            quantity = int(input("Enter quantity: "))
            
            # Record transaction via API
            transaction_data = {
                'customer_id': customer_id,
                'invoice_no': invoice_no,
                'invoice_date': invoice_date,
                'stock_code': stock_code,
                'description': description,
                'quantity': quantity,
                'unit_price': unit_price
            }
            
            result = self.api.record_purchase(transaction_data)
            
            if result['success']:
                print("Transaction recorded successfully.")
                
                # Show updated customer status
                in_pending = customer_id in self.manager.pending_customers
                has_segment = customer_id in self.manager.rfm_table['CustomerID'].values
                
                if in_pending and not has_segment:
                    print("Customer is still being classified. Using global model for recommendations.")
                elif has_segment:
                    segment = self.manager.rfm_table[
                        self.manager.rfm_table['CustomerID'] == customer_id
                    ]['Segment'].iloc[0]
                    print(f"Customer belongs to segment: {segment}")
                
                # Ask if user wants to see updated recommendations
                if input("Would you like to see updated recommendations for this customer? (y/n): ").lower() == 'y':
                    # Use auto model selection
                    params = {
                        'customer_id': customer_id,
                        'count': 5,
                        'model': 'auto'
                    }
                    
                    rec_result = self.api.get_recommendations(params)
                    
                    if rec_result['success'] and rec_result['recommendations']:
                        recommendations = rec_result['recommendations']
                        print(f"\nTop {len(recommendations)} recommendations after purchase:")
                        print("=" * 60)
                        print(f"{'Stock Code':<12} {'Product Name':<30} {'Price':>8} {'Score':>8}")
                        print("-" * 60)
                        for i, rec in enumerate(recommendations):
                            print(f"{rec['stock_code']:<12} {rec['name'][:30]:<30} {rec['price']:>8.2f} {rec['score']:>8.2f}")
                        print("=" * 60)
                    else:
                        print("No recommendations available.")
            else:
                print(f"Error: {result.get('message', 'Failed to record transaction')}")
        except Exception as e:
            print(f"Error: {str(e)}")

    def _compare_recommendations(self):
        """Compare recommendations from segment-specific and global models."""
        try:
            customer_id = float(input("Enter customer ID: "))
            count = int(input("Number of recommendations (default: 5): ") or 5)
            
            # Check if customer exists
            if customer_id not in self.manager.data_processor.customer_encoder.classes_:
                print(f"Customer {customer_id} not found in system.")
                return
            
            # Check if customer has a segment
            has_segment = customer_id in self.manager.rfm_table['CustomerID'].values
            if has_segment:
                segment = self.manager.rfm_table[
                    self.manager.rfm_table['CustomerID'] == customer_id
                ]['Segment'].iloc[0]
                print(f"Customer belongs to segment: {segment}")
            else:
                print("Customer is not yet classified into a segment.")
                
            # Get price range if desired
            use_price_filter = input("Apply price filter? (y/n): ").lower() == 'y'
            price_range = None
            
            if use_price_filter:
                min_price = float(input("Minimum price: ") or 0)
                max_price = float(input("Maximum price: ") or 1000)
                price_range = (min_price, max_price)
            
            # Get recommendations from both models
            params_global = {
                'customer_id': customer_id,
                'count': count,
                'model': 'global'
            }
            
            params_segment = {
                'customer_id': customer_id,
                'count': count,
                'model': 'segment'
            }
            
            if price_range:
                params_global['min_price'] = price_range[0]
                params_global['max_price'] = price_range[1]
                params_segment['min_price'] = price_range[0]
                params_segment['max_price'] = price_range[1]
            
            global_result = self.api.get_recommendations(params_global)
            segment_result = self.api.get_recommendations(params_segment)
            
            # Display global recommendations
            print("\n=== GLOBAL MODEL RECOMMENDATIONS ===")
            if global_result['success'] and global_result['recommendations']:
                recommendations = global_result['recommendations']
                print(f"Top {len(recommendations)} recommendations:")
                print("=" * 60)
                print(f"{'Stock Code':<12} {'Product Name':<30} {'Price':>8} {'Score':>8}")
                print("-" * 60)
                for i, rec in enumerate(recommendations):
                    print(f"{rec['stock_code']:<12} {rec['name'][:30]:<30} {rec['price']:>8.2f} {rec['score']:>8.2f}")
                print("=" * 60)
            else:
                print("No global recommendations available.")
            
            # Display segment recommendations
            print("\n=== SEGMENT MODEL RECOMMENDATIONS ===")
            if not has_segment:
                print("No segment-specific recommendations available (customer not classified).")
            elif segment_result['success'] and segment_result['recommendations']:
                recommendations = segment_result['recommendations']
                print(f"Top {len(recommendations)} recommendations for segment {segment}:")
                print("=" * 60)
                print(f"{'Stock Code':<12} {'Product Name':<30} {'Price':>8} {'Score':>8}")
                print("-" * 60)
                for i, rec in enumerate(recommendations):
                    print(f"{rec['stock_code']:<12} {rec['name'][:30]:<30} {rec['price']:>8.2f} {rec['score']:>8.2f}")
                print("=" * 60)
            else:
                print(f"No segment-specific recommendations available for segment {segment}.")
            
            # Compare overlap between recommendations
            if (global_result['success'] and global_result['recommendations'] and 
                segment_result['success'] and segment_result['recommendations']):
                
                global_codes = [rec['stock_code'] for rec in global_result['recommendations']]
                segment_codes = [rec['stock_code'] for rec in segment_result['recommendations']]
                
                common_codes = set(global_codes).intersection(set(segment_codes))
                
                print(f"\nOverlap between recommendations: {len(common_codes)} products")
                if common_codes:
                    print("Common products:", ", ".join(common_codes))
                
                print(f"Agreement rate: {len(common_codes) / count:.1%}")
                
                if len(common_codes) < count * 0.5:
                    print("The models have significant differences in their recommendations.")
                    print("The segment model is providing more personalized recommendations based on the customer's segment.")
                else:
                    print("The models have substantial agreement in their recommendations.")
        except Exception as e:
            print(f"Error: {str(e)}")

    def _update_models(self):
        """Force update of the recommendation models."""
        try:
            result = self.api.update_models()
            
            if result['success']:
                print("Models updated successfully.")
                
                # Show updated stats
                num_segments = len(self.manager.segment_models)
                print(f"System now has {num_segments} segment models.")
                
                if self.manager.rfm_table is not None:
                    segment_counts = self.manager.rfm_table['Segment'].value_counts()
                    print("\nCustomer segment distribution:")
                    for segment, count in segment_counts.items():
                        print(f"  {segment}: {count} customers")
                        
                pending_count = len(self.manager.pending_customers)
                print(f"\nCustomers waiting for segment assignment: {pending_count}")
                
                if input("Would you like to save the updated state? (y/n): ").lower() == 'y':
                    self._save_state()
            else:
                print(f"Update failed: {result.get('message')}")
        except Exception as e:
            print(f"Error: {str(e)}")

    def _save_state(self):
        """Save the current system state."""
        try:
            path = input("Enter directory path to save state [default: recommender_state]: ") or "recommender_state"
            
            print(f"Saving system state to {path}...")
            success = self.manager.save_state(path)
            
            if success:
                print(f"System state saved successfully to {path}")
            else:
                print("Failed to save system state.")
        except Exception as e:
            print(f"Error: {str(e)}")

    def _show_segments(self):
        """Show information about customer segments."""
        try:
            if self.manager.rfm_table is None or len(self.manager.rfm_table) == 0:
                print("No segment information available.")
                return
                
            segment_counts = self.manager.rfm_table['Segment'].value_counts()
            
            print("\n===== CUSTOMER SEGMENTS =====")
            print(f"Total customers: {len(self.manager.rfm_table)}")
            print(f"Customers pending segment assignment: {len(self.manager.pending_customers)}")
            print("\nSegment distribution:")
            
            for segment, count in segment_counts.items():
                print(f"  {segment}: {count} customers ({count/len(self.manager.rfm_table):.1%})")
                
                # Show if this segment has a trained model
                if segment in self.manager.segment_models:
                    model = self.manager.segment_models[segment]
                    if model.U is not None and model.V is not None:
                        print(f"    - Model: Trained ({model.U.shape[0]} users, {model.V.shape[0]} items)")
                else:
                    print("    - Model: Not available")
            
            # Show detailed information for a specific segment
            show_details = input("\nShow details for a specific segment? (y/n): ").lower() == 'y'
            if show_details:
                segment_name = input("Enter segment name: ")
                
                if segment_name not in segment_counts.index:
                    print(f"Segment '{segment_name}' not found.")
                    return
                    
                segment_data = self.manager.rfm_table[self.manager.rfm_table['Segment'] == segment_name]
                
                print(f"\n===== SEGMENT: {segment_name} =====")
                print(f"Customers: {len(segment_data)}")
                
                # Show RFM stats
                print("\nRFM Statistics:")
                print(f"  Recency (days since last purchase):")
                print(f"    Min: {segment_data['Recency'].min():.1f}")
                print(f"    Avg: {segment_data['Recency'].mean():.1f}")
                print(f"    Max: {segment_data['Recency'].max():.1f}")
                
                print(f"  Frequency (number of orders):")
                print(f"    Min: {segment_data['Frequency'].min():.1f}")
                print(f"    Avg: {segment_data['Frequency'].mean():.1f}")
                print(f"    Max: {segment_data['Frequency'].max():.1f}")
                
                print(f"  Monetary (total spending):")
                print(f"    Min: ${segment_data['Monetary'].min():.2f}")
                print(f"    Avg: ${segment_data['Monetary'].mean():.2f}")
                print(f"    Max: ${segment_data['Monetary'].max():.2f}")
                
                # Show sample customers
                show_samples = input("\nShow sample customers from this segment? (y/n): ").lower() == 'y'
                if show_samples:
                    num_samples = min(5, len(segment_data))
                    samples = segment_data.sample(num_samples)
                    
                    print(f"\nSample of {num_samples} customers:")
                    for _, row in samples.iterrows():
                        print(f"  Customer ID: {row['CustomerID']}")
                        print(f"    Recency: {row['Recency']:.1f} days")
                        print(f"    Frequency: {row['Frequency']:.1f} orders")
                        print(f"    Monetary: ${row['Monetary']:.2f}")
                        print()
        except Exception as e:
            print(f"Error: {str(e)}")
