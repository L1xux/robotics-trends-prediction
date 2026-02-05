"""
Evaluation Agent (Full Content / Thread Isolated)
워크플로우 외부에서 독립적으로 실행되며, 
샘플링 없이 전체 리포트와 전체 컨텍스트를 사용하여 정밀 평가를 수행합니다.
"""
import asyncio
import traceback
from typing import List, Any

# [필수] 비동기 충돌 방지
import nest_asyncio
nest_asyncio.apply()

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from src.agents.base.base_agent import BaseAgent
from src.agents.base.agent_config import AgentConfig
from src.graph.state import PipelineState
from src.core.settings import Settings

from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy
from langchain_community.embeddings import HuggingFaceEmbeddings
from datasets import Dataset
from ragas.run_config import RunConfig

class EvaluationAgent(BaseAgent):
    def __init__(self, llm: BaseChatModel, config: AgentConfig, tools: List[Any] = None, settings: Settings = None):
        super().__init__(llm, tools or [], config)
        self.settings = settings or Settings()
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.settings.EMBEDDING_MODEL,
            model_kwargs={'trust_remote_code': True, 'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # JSON Mode 설정
        if isinstance(llm, ChatOpenAI):
            self.ragas_llm = ChatOpenAI(
                model=llm.model_name,
                temperature=0, 
                api_key=llm.openai_api_key,
                model_kwargs={"response_format": {"type": "json_object"}},
                request_timeout=600  # 데이터가 많아지므로 타임아웃을 5분 -> 10분으로 증가
            )
        else:
            self.ragas_llm = llm

    async def execute(self, state: PipelineState) -> PipelineState:
        print(f"\n{'='*60}\n🔍 [EvaluationAgent] Starting Full-Context Evaluation\n{'='*60}")

        try:
            question = state.get("user_input", "")
            answer = state.get("final_report", "")
            
            # [변경] 리포트 샘플링 로직 제거 (전체 내용 사용)
            if not answer:
                print("   ❌ No report content to evaluate.")
                state["evaluation_results"] = {"faithfulness": 0.0, "answer_relevancy": 0.0}
                return state
            
            print(f"   📄 Evaluating Report Length: {len(answer)} chars")

            # 컨텍스트 추출
            rag_results = state.get("rag_results", {})
            contexts = self._extract_contexts(rag_results)
            
            # [변경] 컨텍스트 샘플링 로직 제거 (전체 문서 사용)
            # RAG 데이터가 없으면 뉴스 데이터 사용
            if not contexts:
                news_data = state.get("news_data", {})
                contexts = self._extract_news_contexts(news_data)

            if not contexts:
                print("   ❌ No context data available for evaluation.")
                state["evaluation_results"] = {"faithfulness": 0.0, "answer_relevancy": 0.0}
                return state

            print(f"   📚 Using All Contexts: {len(contexts)} documents")

            # 데이터셋 준비
            data_dict = {"question": [question], "answer": [answer], "contexts": [contexts]}
            dataset = Dataset.from_dict(data_dict)
            
            print(f"   🚀 Offloading Ragas to separate thread (Full Data)...")


            def run_ragas_sync():
                try:
                    return evaluate(
                        dataset=dataset,
                        metrics=[Faithfulness(llm=self.ragas_llm), AnswerRelevancy(embeddings=self.embeddings, llm=self.ragas_llm)],
                        llm=self.ragas_llm,
                        embeddings=self.embeddings,
                        raise_exceptions=True,
                        run_config=RunConfig(timeout=600, max_retries=2) # 타임아웃/재시도 증가
                    )
                except Exception as inner_e:
                    print(f"   ⚠️ Ragas Internal Error: {inner_e}")
                    # traceback.print_exc() # 필요시 주석 해제하여 상세 로그 확인
                    return None

            results = await asyncio.to_thread(run_ragas_sync)
            # ---------------------------------------------------------

            scores = {}
            if results and hasattr(results, 'scores') and len(results.scores) > 0:
                scores = results.scores[0]
                print(f"   ✅ Success: F:{scores.get('faithfulness', 0):.2f}, R:{scores.get('answer_relevancy', 0):.2f}")
            else:
                print(f"   ⚠️ Empty results returned.")

            # 점수 저장 (NaN 처리)
            f_score = float(scores.get("faithfulness", 0.0) or 0.0)
            r_score = float(scores.get("answer_relevancy", 0.0) or 0.0)

            state["evaluation_results"] = {
                "faithfulness": f_score,
                "answer_relevancy": r_score,
                "details": str(scores)
            }

        except Exception as e:
            print(f"   ❌ Critical Evaluation Error: {e}")
            traceback.print_exc()
            state["evaluation_results"] = {"faithfulness": 0.0, "answer_relevancy": 0.0, "error": str(e)}

        return state

    def _extract_contexts(self, rag_results: Any) -> List[str]:
        """RAG 결과에서 모든 문서 내용 추출"""
        contexts = []
        if isinstance(rag_results, dict):
            documents = rag_results.get("documents", [])
            for doc in documents:
                # 다양한 문서 포맷 대응 (dict, Document 객체 등)
                content = None
                if isinstance(doc, dict):
                    content = doc.get("content") or doc.get("page_content")
                else:
                    content = getattr(doc, "page_content", None) or getattr(doc, "content", None)
                
                if content and isinstance(content, str) and content.strip():
                    contexts.append(content)
        return contexts

    def _extract_news_contexts(self, news_data: Any) -> List[str]:
        """뉴스 데이터에서 모든 기사 요약 추출"""
        contexts = []
        if isinstance(news_data, dict) and "news" in news_data:
            for entry in news_data["news"]:
                for article in entry.get("articles", []):
                    desc = article.get("description", "")
                    if desc and desc.strip():
                        contexts.append(desc)
        return contexts

    evaluate_report = execute