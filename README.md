# STT→LLM→TTS Pipeline — Latency Comparison

## Non-Tool Query Latency (Utterance → First Audio)

| Scenario | Azure OpenAI | Groq | Saving |
|---|---|---|---|
| Cold (first utterance) | ~1.44s | ~0.86s | **-580ms** |
| Warm (context already built) | ~1.44s | ~0.24–0.37s | **-1.10s** |

## Tool Query Latency Breakdown

| Stage | Azure OpenAI | Groq | Saving |
|---|---|---|---|
| 1st LLM turn (tool detection) | ~1.10s | ~0.32–1.57s | variable |
| RAG round-trip (embed + pgvector) | ~1.45s | ~1.29–1.49s | ~0–160ms |
| 2nd LLM turn (synthesis) | ~1.27s | ~0.30s | **-970ms** |
| **Total tool latency** | **~3.83s** | **~2.12–3.16s** | **-0.7–1.7s** |

## Bug Fix Impact (Azure, before vs after)

| Metric | Before fixes | After fixes | Saving |
|---|---|---|---|
| RAG round-trip | ~1.70s | ~1.45s | **-250ms** |
| 2nd LLM turn | ~1.30s | ~1.27s | -30ms |
| **Total tool latency** | **~4.10s** | **~3.83s** | **-270ms** |