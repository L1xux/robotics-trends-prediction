"""
News Crawler Tool 디버깅 스크립트
"""
from src.tools.news_crawler_tool import NewsCrawlerTool
from src.tools.base.tool_config import ToolConfig

def main():
    print("=" * 80)
    print("📰 News Crawler Tool - Debug Test")
    print("=" * 80)
    print()
    
    # Tool 초기화
    config = ToolConfig(
        name="NewsCrawlerTool",
        description="News crawler tool",
        retry_count=1,
        timeout=60
    )
    tool = NewsCrawlerTool(config)
    
    # Test 1: 간단한 키워드
    print("Test 1: Simple Keywords")
    print("-" * 80)
    keywords = ["Tesla", "robotics"]
    result = tool._run(keywords=keywords, date_range="3 years", sources=3)
    
    print(f"\n📊 Result:")
    print(f"   Keywords: {result.get('keywords', [])}")
    print(f"   Total articles: {result.get('total_articles', 0)}")
    print(f"   Unique sources: {result.get('unique_sources', 0)}")
    
    if result.get('articles'):
        print(f"\n   First 3 articles:")
        for i, article in enumerate(result['articles'][:3], 1):
            print(f"   {i}. {article.get('title', 'N/A')[:80]}...")
            print(f"      Source: {article.get('source', 'N/A')}")
            print(f"      Date: {article.get('published', 'N/A')}")
    
    if 'error' in result:
        print(f"   ❌ Error: {result['error']}")
    
    print("\n" + "=" * 80)
    print()
    
    # Test 2: 기술 용어
    print("Test 2: Technical Terms")
    print("-" * 80)
    keywords = ["factory automation", "humanoid robots"]
    result = tool._run(keywords=keywords, date_range="1 year", sources=2)
    
    print(f"\n📊 Result:")
    print(f"   Keywords: {result.get('keywords', [])}")
    print(f"   Total articles: {result.get('total_articles', 0)}")
    print(f"   Unique sources: {result.get('unique_sources', 0)}")

if __name__ == "__main__":
    main()

