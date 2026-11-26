"""Edit the profiles here; keys are read only from the environment."""
from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    dataset: str
    model: str
    final_k: int
    summary_k: int
    dense_k: int
    bm25_k: int
    candidate_cap: int | None
    sessions_per_write: int = 1
    embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = 768
    thinking_budget: int = 0
    temperature: float = 0.0
    writer_tokens: int = 16384
    online_tokens: int = 8192
    reader_tokens: int = 4096
    qa_tokens: int = 2048
    online_fact_limit: int = 80
    online_view_limit: int = 10
    dense_budget_k: int = 50
    solver_seconds_per_stage: float = 5.0

PROFILES = {
    "locomo": Config("locomo", "gemini-2.5-flash", 20, 10, 20, 20, 120),
    "lme": Config("lme", "gemini-2.5-flash", 20, 20, 40, 40, 240,
                  sessions_per_write=3),
    # No model reranker for MemEye. Expand Top10 summary/view routes,
    # union Top20 dense and Top20 BM25, then mathematical selection <=30.
    "memeye": Config("memeye", "gemini-2.5-flash-lite", 30, 10, 20, 20,
                     None, embedding_dim=3072),
}
