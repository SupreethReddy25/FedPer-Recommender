# File: recommender_system/__init__.py

from .data_processor      import DataProcessor
from .rfm                 import RFMSegmentation
from .embeddings          import ProductEmbeddings
from .recommender_model   import FedPerRecommender
from .manager_combined    import RecommendationManager   # <— note the “_combined”
from .api                 import RecommendationAPI
from .demo                import RecommendationDemo
from .web_integration     import WebsiteIntegration

__version__ = '1.0.0'
