#!/usr/bin/env python3
"""
Robotics Trends Prediction Pipeline Runner

Usage:
    python scripts/run_pipeline.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.graph.workflow import create_workflow


def main():
    """Run the complete pipeline"""
    print("="*60)
    print("🤖 Robotics Trends Prediction Pipeline")
    print("="*60)
    
    # Get user input
    print("\n📝 Please enter your research topic:")
    print("   (e.g., 'humanoid robots in manufacturing')")
    print("   (Press Ctrl+C to exit)\n")
    
    try:
        user_input = input("Topic: ").strip()
        
        if not user_input:
            print("❌ Empty topic. Exiting...")
            return
        
        print(f"\n✅ Topic received: {user_input}")
        print(f"\n{'='*60}")
        print(f"🚀 Starting pipeline...")
        print(f"{'='*60}\n")
        
        # Create and run workflow
        workflow = create_workflow()
        
        # Initial state
        initial_state = {
            "user_input": user_input
        }
        
        # Run
        result = workflow.invoke(initial_state)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"✅ Pipeline Complete!")
        print(f"{'='*60}")
        
        if "planning_output" in result:
            print(f"\n📋 Planning:")
            print(f"   Topic: {result['planning_output'].normalized_topic}")
            print(f"   Keywords: {len(result['keywords'])} keywords")
        
        if "data_collection_status" in result:
            status = result["data_collection_status"]
            print(f"\n📦 Data Collection:")
            print(f"   ArXiv: {status.arxiv_count} papers")
            print(f"   RAG: {status.rag_count} documents")
            print(f"   News: {status.news_count} articles")
            print(f"   Quality: {status.quality_score:.2f}")
            print(f"   Status: {status.status}")
        
        if "folder_name" in result:
            print(f"\n💾 Data saved to: data/raw/{result['folder_name']}/")
        
        print(f"\n{'='*60}\n")
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Pipeline interrupted by user. Exiting...")
        sys.exit(0)
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

