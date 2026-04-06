# lyca

On-device LLM chat — the [Lisette](https://github.com/AnswerDotAI/lisette) equivalent for local [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) models.

## Install

```bash
pip install lyca          # or: pip install -e '.[dev]'
```

## Quick Start

```python
import lyca

# Fastest path: auto-pick the best model, download, return Chat
c = lyca.quick_model(sp="Be terse.")
r = c("What is SQLite FTS5?")
r.content
```

### Step-by-step

```python
lyca.recommend()                              # what fits your machine?
path = lyca.download("gemma4-e4b")            # HF download with resume
c = lyca.Chat(path, sp="Be terse.")
r = c("What is SQLite FTS5?")
r.content
```

### One-liner: download + chat

```python
c = lyca.Chat.from_hf("gemma4-e4b", sp="Be terse.")
r = c("What is SQLite FTS5?")
```

### Multi-turn

```python
c = lyca.Chat(path)
c("My name is Karthik.")
r = c("What's my name?")
assert "Karthik" in r.content
```

### Tool Calling

```python
def search(query: str, top_k: int = 5) -> list:
    """Search the knowledge base.
    Args:
        query: The search string.
        top_k: Number of results.
    """
    return [{"content": f"Result for {query}"}]

c = lyca.Chat(path, tools=[search])
r = c("Search for 'RRF scoring'.")
```

### Async Streaming

```python
import asyncio, lyca

async def main():
    async with lyca.AsyncChat(model_path=path) as c:
        async for chunk in c("Write a haiku about SQLite."):
            if isinstance(chunk, str): print(chunk, end="", flush=True)

asyncio.run(main())
```

## Model Registry

```python
lyca.models()                        # all models
lyca.models(family="gemma4")         # filter by family
lyca.models(tag="reasoning")         # filter by tag
lyca.syscheck()                      # system info dict
```

## API

| Symbol | Kind | Description |
|---|---|---|
| `Chat(path, sp, tools, hist)` | class | Stateful chat client |
| `AsyncChat(model_path=path)` | class | Async streaming variant |
| `Chat.from_hf(model_id, **kw)` | classmethod | Download + Chat in one call |
| `Response` | dataclass | `.content`, `.tool_calls`, `.finish_reason` |
| `models(family, task, tag)` | function | List/filter model registry |
| `syscheck()` | function | Detect RAM, CPU, GPU (cached) |
| `recommend(min_tps, task)` | function | Rank models for this system |
| `quick_model(task, min_tps, sp)` | function | Recommend → download → Chat |
| `download(model_id)` | function | HF download with resume |
| `register_model(entry)` | function | Add custom model (rejects duplicates) |
| `MODEL_REGISTRY` | `list[dict]` | Curated model metadata |

## Testing

Tests live as nbdev test cells in the notebooks:

```bash
python -m nbdev.test   # run all notebook test cells
```

## Development

`nbs/` is the source of truth (nbdev workflow):

```bash
python -m nbdev.export   # regenerate lyca/*.py
python -m nbdev.test     # run notebook test cells
```
