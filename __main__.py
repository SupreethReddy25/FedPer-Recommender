# File: recommender_system/__main__.py

import sys

def main():
    # build the core manager once
    from .manager_combined import RecommendationManager
    recommendation_manager = RecommendationManager()

    if "--demo" in sys.argv:
        from .demo import RecommendationDemo
        demo = RecommendationDemo(recommendation_manager)
        demo.run()

    elif "--api" in sys.argv:
        from .api import RecommendationAPI
        api = RecommendationAPI(recommendation_manager)
        api.start()

    elif "--manager" in sys.argv:
        # if you want a bare‐bones manager CLI
        recommendation_manager.launch()

    else:
        print("Usage: python -m recommender_system [--demo | --api | --manager]")
        sys.exit(1)

if __name__ == "__main__":
    main()
