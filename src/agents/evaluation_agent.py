"""
Evaluation Agent (Standalone / Thread Isolated)
워크플로우 외부에서 독립적으로 실행되며, Ragas를 별도 스레드에서 구동합니다.
"""
import asyncio
import random
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
                request_timeout=300 
            )
        else:
            self.ragas_llm = llm

    async def execute(self, state: PipelineState) -> PipelineState:
        print(f"\n{'='*60}\n🔍 [EvaluationAgent] Starting Post-Process Evaluation\n{'='*60}")

        try:
            question = state.get("user_input", "")
            answer = state.get("final_report", "")
            
            # 1. 리포트 샘플링 (8000자 제한)
            if len(answer) > 8000:
                print(f"   ✂️ Report sampled to 8000 chars.")
                answer = self._sample_random_paragraphs(answer, max_chars=8000)

            rag_results = state.get("rag_results", {})
            contexts = self._extract_contexts(rag_results)
            
            # 2. 컨텍스트 샘플링 (최대 5개)
            if len(contexts) > 5:
                contexts = random.sample(contexts, 5)

            if not contexts:
                news_data = state.get("news_data", {})
                all_news = self._extract_news_contexts(news_data)
                if all_news:
                    contexts = random.sample(all_news, min(len(all_news), 5))

            if not contexts:
                print("   ❌ No data to evaluate.")
                state["evaluation_results"] = {"faithfulness": 0.0, "answer_relevancy": 0.0}
                return state

            # 데이터셋 준비
            data_dict = {"question": [question], "answer": [answer], "contexts": [contexts]}
            dataset = Dataset.from_dict(data_dict)
            
            print(f"   🚀 Offloading Ragas to separate thread...")

            # ---------------------------------------------------------
            # [핵심] Ragas 실행을 별도 스레드로 격리 (Thread Offloading)
            # 메인 루프와의 간섭을 피하기 위해 동기 함수를 스레드로 실행
            # ---------------------------------------------------------
            def run_ragas_sync():
                try:
                    return evaluate(
                        dataset=dataset,
                        metrics=[Faithfulness(llm=self.ragas_llm), AnswerRelevancy(embeddings=self.embeddings, llm=self.ragas_llm)],
                        llm=self.ragas_llm,
                        embeddings=self.embeddings,
                        raise_exceptions=True,
                        run_config=RunConfig(timeout=300, max_retries=1)
                    )
                except Exception as inner_e:
                    print(f"   ⚠️ Ragas Internal Error: {inner_e}")
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

    def _sample_random_paragraphs(self, text: str, max_chars: int) -> str:
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if len(paragraphs) < 3: paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        if not paragraphs: return text[:max_chars]
        random.shuffle(paragraphs)
        selected = []
        curr_len = 0
        for p in paragraphs:
            if curr_len + len(p) < max_chars:
                selected.append(p)
                curr_len += len(p)
            else: break
        return "\n\n".join(selected)

    def _extract_contexts(self, rag_results: Any) -> List[str]:
        contexts = []
        if isinstance(rag_results, dict):
            documents = rag_results.get("documents", [])
            for doc in documents:
                content = doc.get("content") or doc.get("page_content") if isinstance(doc, dict) else getattr(doc, "page_content", "")
                if content: contexts.append(content)
        return contexts

    def _extract_news_contexts(self, news_data: Any) -> List[str]:
        contexts = []
        if isinstance(news_data, dict) and "news" in news_data:
            for entry in news_data["news"]:
                for article in entry.get("articles", []):
                    contexts.append(article.get("description", ""))
        return contexts

    evaluate_report = execute