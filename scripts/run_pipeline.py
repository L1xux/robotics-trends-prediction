#!/usr/bin/env python3
"""
Robotics Trends Prediction Pipeline Runner (with Design Patterns)

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --topic "your topic here"

Features:
- Factory Pattern for Agent/Tool creation
- IoC Container for dependency injection
- Singleton for settings management
- Interactive prompt for user input
- Full async workflow execution
"""

import sys
import asyncio
import platform
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Use new workflow creation method
from src.graph.workflow import create_workflow_manager


async def run_pipeline_async(user_input: str):
    """
    Run the complete pipeline asynchronously

    Args:
        user_input: User's research topic

    Returns:
        Final pipeline state
    """
    print(f"\n{'='*60}")
    print(f"🚀 Starting pipeline...")
    print(f"{'='*60}\n")

    # Create workflow manager (initializes with Design Patterns)
    workflow_manager = create_workflow_manager()

    # Run the workflow
    final_state = await workflow_manager.run_workflow(user_input)

    return final_state


def main():
    """Run the complete pipeline"""
    # Windows asyncio policy
    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Robotics Trends Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=None,
        help="Research topic (if not provided, will prompt interactively)"
    )
    args = parser.parse_args()

    print("="*60)
    print("🤖 Robotics Trends Prediction Pipeline")
    print("="*60)

    try:
        # Get user input
        if args.topic:
            user_input = args.topic
            print(f"\n📝 Topic: {user_input}")
        else:
            print("\n📝 Please enter your research topic:")
            print("   (e.g., 'humanoid robots in manufacturing')")
            print("   (Press Ctrl+C to exit)\n")

            user_input = input("Topic: ").strip()

            if not user_input:
                print("❌ Empty topic. Exiting...")
                return

        print(f"\n✅ Topic received: {user_input}")

        # Run pipeline asynchronously
        result = asyncio.run(run_pipeline_async(user_input))

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

        if "final_report" in result:
            print(f"\n📄 Report Generated!")
            print(f"   Status: {result.get('status', 'unknown')}")

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

