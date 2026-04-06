# lyca

Stateful `Chat` / `AsyncChat` for on-device [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) models (Gemma 4, etc.) — the [Lisette](https://github.com/AnswerDotAI/lisette) equivalent for local inference.

## Install

```bash
pip install litert-lm-api-nightly toolslm msglm
pip install -e .
```

## Quick Start

```python
from lyca import Chat

# Download a model first:
# pip install huggingface_hub
# python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('litert-community/gemma-4-E2B-it-litert-lm', 'gemma-4-E2B-it.litertlm'))"

c = Chat("/path/to/gemma-4-E2B-it.litertlm", sp="Be terse.")
r = c("What is SQLite FTS5?")
print(r.content)
```

### Multi-turn

```python
c = Chat("/path/to/model.litertlm")
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
        top_k: Number of results to return.
    """
    return [{"content": f"Result for {query}"}]

c = Chat("/path/to/model.litertlm", tools=[search])
r = c("Search for 'RRF scoring' and summarise.")
```

### Async Streaming

```python
import asyncio
from lyca import AsyncChat

async def main():
    async with AsyncChat(model_path="/path/to/model.litertlm") as c:
        async for chunk in c("Write a haiku about SQLite."):
            if isinstance(chunk, str): print(chunk, end="", flush=True)

asyncio.run(main())
```

## Testing

```bash
# Pure function tests (no model needed)
pytest tests/ -v -k 'not model'

# Full tests with real model
LYCA_MODEL_PATH=/path/to/gemma-4-E2B-it.litertlm pytest tests/ -v

# Or let conftest.py auto-download the model
pip install huggingface_hub
pytest tests/ -v
```

## Development

Source lives in `nbs/00_core.ipynb`. After editing:

```bash
python -m nbdev.export   # regenerates lyca/core.py
pytest tests/ -v          # run tests
``` 
