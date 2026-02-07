"""
Evaluation LLM with Ragas
"""
import asyncio
import traceback
from typing import List, Any

import nest_asyncio
nest_asyncio.apply()

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from src.graph.state import PipelineState
from src.core.settings import Settings

from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy
from langchain_community.embeddings import HuggingFaceEmbeddings
from datasets import Dataset
from ragas.run_config import RunConfig

class EvaluationLLM:
    def __init__(self, llm: BaseChatModel, settings: Settings = None):
        self.llm = llm
        self.settings = settings or Settings()
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.settings.EMBEDDING_MODEL,
            model_kwargs={'trust_remote_code': True, 'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        if isinstance(llm, ChatOpenAI):
            self.ragas_llm = ChatOpenAI(
                model=llm.model_name,
                temperature=0, 
                api_key=llm.openai_api_key,
                model_kwargs={"response_format": {"type": "json_object"}},
                request_timeout=600
            )
        else:
            self.ragas_llm = llm

    async def execute(self, state: PipelineState) -> PipelineState:
        print(f"\n{'='*60}\n🔍 [EvaluationLLM] Starting Full-Context Evaluation\n{'='*60}")

        try:
            question = state.get("user_input", "")
            answer = state.get("final_report", "")
            
            if not answer:
                print("   No report content to evaluate.")
                state["evaluation_results"] = {"faithfulness": 0.0, "answer_relevancy": 0.0}
                return state
            
            print(f"   📄 Evaluating Report Length: {len(answer)} chars")

            rag_results = state.get("rag_results", {})
            contexts = self._extract_contexts(rag_results)

            if not contexts:
                print("   No context data available for evaluation.")
                state["evaluation_results"] = {"faithfulness": 0.0, "answer_relevancy": 0.0}
                return state

            print(f"   📚 Using All Contexts: {len(contexts)} documents")

            data_dict = {"question": [question], "answer": [answer], "contexts": [contexts]}
            dataset = Dataset.from_dict(data_dict)
            
            print(f"   Offloading Ragas to separate thread (Full Data)...")


            def run_ragas_sync():
                try:
                    return evaluate(
                        dataset=dataset,
                        metrics=[Faithfulness(llm=self.ragas_llm), AnswerRelevancy(embeddings=self.embeddings, llm=self.ragas_llm)],
                        llm=self.ragas_llm,
                        embeddings=self.embeddings,
                        raise_exceptions=True,
                        run_config=RunConfig(timeout=600, max_retries=2) 
                    )
                except Exception as inner_e:
                    print(f"   ⚠️ Ragas Internal Error: {inner_e}")
                    return None

            results = await asyncio.to_thread(run_ragas_sync)

            scores = {}
            if results and hasattr(results, 'scores') and len(results.scores) > 0:
                scores = results.scores[0]
                print(f"   ✅ Success: F:{scores.get('faithfulness', 0):.2f}, R:{scores.get('answer_relevancy', 0):.2f}")
            else:
                print(f"   ⚠️ Empty results returned.")

            f_score = float(scores.get("faithfulness", 0.0) or 0.0)
            r_score = float(scores.get("answer_relevancy", 0.0) or 0.0)

            state["evaluation_results"] = {
                "faithfulness": f_score,
                "answer_relevancy": r_score,
                "details": str(scores)
            }

        except Exception as e:
            print(f"   Critical Evaluation Error: {e}")
            traceback.print_exc()
            state["evaluation_results"] = {"faithfulness": 0.0, "answer_relevancy": 0.0, "error": str(e)}

        return state

    def _extract_contexts(self, rag_results: Any) -> List[str]:
        """RAG 결과에서 모든 문서 내용 추출"""
        contexts = []
        if isinstance(rag_results, dict):
            documents = rag_results.get("documents", [])
            for doc in documents:
                content = None
                if isinstance(doc, dict):
                    content = doc.get("content") or doc.get("page_content")
                else:
                    content = getattr(doc, "page_content", None) or getattr(doc, "content", None)
                
                if content and isinstance(content, str) and content.strip():
                    contexts.append(content)
        return contexts


    evaluate_report = execute
