# lyca Skills Reference

> For LLMs and AI coding harnesses — quick reference to lyca's API surface.

## What is lyca?

On-device LLM chat via LiteRT-LM. Stateful `Chat`/`AsyncChat` with tool calling, model registry, and HF download.

## Core API

```python
from lyca import Chat, AsyncChat, Response

# Sync chat
c = Chat(model_path, sp="", tools=[], hist=[], max_steps=5, max_tokens=1024)
r = c("prompt")           # → Response
r.content                  # str — assistant text
r.tool_calls               # list[dict] — parsed tool calls
r.finish_reason            # 'stop' | 'tool_calls'
r.model                    # str — model path

# History
c.h                        # message list (alias for c.hist)
c.reset()                  # clear history, keep engine
c.close()                  # release conversation

# Context manager
with Chat(path) as c: ...

# Async streaming
async with AsyncChat(model_path=path) as c:
    async for chunk in c("prompt"):
        if isinstance(chunk, str): print(chunk, end="")
        # final chunk is Response

# Convenience
r = await c.acall("prompt")  # collect all, return Response

# Download + chat in one call
c = Chat.from_hf("gemma4-e4b", sp="Be terse.")
c = AsyncChat.from_hf("gemma4-e4b")
```

## Model Hub API

```python
import lyca

lyca.models()                        # list all registered models (prints table)
lyca.models(family="gemma4")         # filter by family
lyca.models(tag="reasoning")         # filter by tag
lyca.models(task="chat")             # filter by task

lyca.syscheck()                      # → dict: platform, arch, cpu, ram_gb,
                                     #   free_ram_gb, gpu, gpu_vram_gb,
                                     #   apple_chip, apple_tier

lyca.recommend()                     # rank models that fit this machine
lyca.recommend(min_tps=10)           # minimum tokens/sec threshold

lyca.download("gemma4-e4b")          # → str: absolute path to .litertlm file
lyca.download("gemma4-e4b", dest="./models")

lyca.register_model({                # extend registry at runtime
    'id': 'my-model', 'repo': 'org/repo', 'file': 'model.litertlm',
    'size_gb': 2.0, 'min_ram_gb': 3.0, 'task': 'chat'
})
```

## Registry IDs

`gemma4-e2b`, `gemma4-e4b`, `gemma3-1b`, `phi4-mini`, `deepseek-r1-1.5b`, `qwen3.5-0.8b`, `qwen3.5-4b`, `qwen3.5-9b`, `smollm-135m`

## Tool Calling Pattern

Pass Python callables with type hints + Google-style docstrings:

```python
def search(query: str, top_k: int = 5) -> list:
    """Search the knowledge base.
    Args:
        query: The search string.
        top_k: Number of results.
    """
    return [{"content": f"Result for {query}"}]

c = Chat(path, tools=[search])
r = c("Search for 'RRF scoring'.")
```

LiteRT-LM reads type hints + docstrings to generate schemas and dispatches tool calls natively.

## Dependencies

`litert-lm-api-nightly`, `toolslm`, `msglm`, `huggingface_hub`, `fastcore` (transitive)

## Project Structure

```
nbs/00_core.ipynb    → lyca/core.py   (Chat, AsyncChat, Response, helpers)
nbs/01_fit.ipynb     → lyca/fit.py    (MODEL_REGISTRY, models, syscheck, recommend, download)
tests/test_core.py                     (pure function + model-dependent tests)
tests/test_fit.py                      (pure function tests for fit module)
```
