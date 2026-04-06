"""Tests for lyca — real data, no mocks.

Pure function tests run without a model.
Model-dependent tests require LYCA_MODEL_PATH env var pointing to a .litertlm file.

To run all tests locally:
    # Download model first
    python -c "
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo_id='litert-community/gemma-4-E2B-it-litert-lm', filename='gemma-4-E2B-it.litertlm')
    print(p)
    "
    # Then run with the path
    LYCA_MODEL_PATH=/path/to/gemma-4-E2B-it.litertlm pytest tests/ -v

To run pure function tests only (no model needed):
    pytest tests/test_core.py -v -k 'not model'
"""

import os
import pytest
import asyncio

from lyca.core import (
    get_text, extract_tool_calls, tool_schema, Response,
    Chat, AsyncChat, _get_engine, _engines
)

# ---------------------------------------------------------------------------
# Model path — set via env var; model-dependent tests skip if not available
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get('LYCA_MODEL_PATH', '')
needs_model = pytest.mark.skipif(
    not MODEL_PATH or not os.path.exists(MODEL_PATH),
    reason='LYCA_MODEL_PATH not set or model file not found'
)


# ===========================================================================
# Pure function tests — no model needed, real data structures
# ===========================================================================

class TestGetText:
    def test_simple(self):
        resp = {'content': [{'type': 'text', 'text': 'hello world'}]}
        assert get_text(resp) == 'hello world'

    def test_multimodal(self):
        resp = {'content': [{'type': 'audio', 'data': '...'}, {'type': 'text', 'text': 'description'}]}
        assert get_text(resp) == 'description'

    def test_empty_content(self):
        assert get_text({'content': []}) == ''

    def test_no_content_key(self):
        assert get_text({}) == ''

    def test_no_text_item(self):
        assert get_text({'content': [{'type': 'image', 'data': '...'}]}) == ''


class TestExtractToolCalls:
    def test_single_tool_call(self):
        raw = '<|tool_call>call:search{query:<|"|>hello<|"|>,top_k:5}<tool_call|>'
        tcs = extract_tool_calls(raw)
        assert len(tcs) == 1
        assert tcs[0]['name'] == 'search'
        assert tcs[0]['arguments'] == {'query': 'hello', 'top_k': 5}

    def test_no_tool_calls(self):
        assert extract_tool_calls('plain text with no tool calls') == []

    def test_multiple_tool_calls(self):
        raw = '<|tool_call>call:a{x:1}<tool_call|> some text <|tool_call>call:b{y:<|"|>hi<|"|>}<tool_call|>'
        tcs = extract_tool_calls(raw)
        assert len(tcs) == 2
        assert tcs[0]['name'] == 'a'
        assert tcs[1]['name'] == 'b'

    def test_mixed_types(self):
        raw = '<|tool_call>call:fn{a:42,b:3.14,c:true,d:<|"|>text<|"|>}<tool_call|>'
        tc = extract_tool_calls(raw)[0]['arguments']
        assert tc == {'a': 42, 'b': 3.14, 'c': True, 'd': 'text'}

    def test_false_bool(self):
        raw = '<|tool_call>call:fn{x:false}<tool_call|>'
        tc = extract_tool_calls(raw)[0]['arguments']
        assert tc == {'x': False}

    def test_empty_string(self):
        assert extract_tool_calls('') == []


class TestResponse:
    def test_basic_construction(self):
        r = Response('hello', [], 'stop', 'gemma-4-e2b')
        assert r.content == 'hello'
        assert r.finish_reason == 'stop'
        assert r.tool_calls == []
        assert r.model == 'gemma-4-e2b'

    def test_defaults(self):
        r = Response('hi')
        assert r.tool_calls == []
        assert r.finish_reason == 'stop'
        assert r.model == ''

    def test_repr_markdown_basic(self):
        r = Response('hello', [], 'stop', 'gemma-4-e2b')
        md = r._repr_markdown_()
        assert 'hello' in md
        assert 'gemma-4-e2b' in md
        assert 'stop' in md

    def test_repr_markdown_with_tools(self):
        r = Response('result', [{'name': 'search', 'arguments': {'q': 'test'}}], 'tool_calls', 'gemma-4-e2b')
        md = r._repr_markdown_()
        assert '🔧 search' in md
        assert 'tool_calls' in md

    def test_tool_calls_not_shared(self):
        """Verify default_factory prevents shared mutable default."""
        r1 = Response('a')
        r2 = Response('b')
        r1.tool_calls.append({'name': 'x'})
        assert r2.tool_calls == []


class TestToolSchema:
    def test_schema_generation(self):
        # tool_schema uses inspect.getsource, so the function must be defined in a file.
        # We test with a function defined at module level in this test file.
        s = tool_schema(_sample_add)
        assert s['name'] == '_sample_add'
        assert 'parameters' in s
        assert 'a' in s['parameters']['properties']
        assert 'b' in s['parameters']['properties']


def _sample_add(a: int, b: int) -> int:
    "Add two numbers."
    return a + b


# ===========================================================================
# Model-dependent tests — real model, no mocks
# ===========================================================================

class TestChatBasic:
    @needs_model
    def test_basic_response(self):
        with Chat(MODEL_PATH, sp='Be terse. Reply in one sentence.') as c:
            r = c('What is 2+2?')
            assert isinstance(r, Response)
            assert isinstance(r.content, str)
            assert len(r.content) > 0
            assert r.finish_reason == 'stop'
            assert r.model == MODEL_PATH

    @needs_model
    def test_multiturn_memory(self):
        with Chat(MODEL_PATH) as c:
            c('My name is Karthik.')
            r = c("What's my name?")
            assert 'Karthik' in r.content, f'Expected Karthik in response, got: {r.content}'

    @needs_model
    def test_history_tracking(self):
        with Chat(MODEL_PATH) as c:
            c('Hello')
            assert len(c.h) == 2  # user + assistant
            assert c.h[0]['role'] == 'user'
            assert c.h[1]['role'] == 'assistant'

    @needs_model
    def test_reset(self):
        with Chat(MODEL_PATH) as c:
            c('Hello')
            assert len(c.h) == 2
            c.reset()
            assert c.h == []

    @needs_model
    def test_prepopulated_history(self):
        with Chat(MODEL_PATH, hist=['What is 2+2?', '4', 'And 3+3?']) as c:
            assert len(c.hist) == 3
            r = c('And 4+4?')
            assert isinstance(r, Response)
            assert len(r.content) > 0

    @needs_model
    def test_context_manager(self):
        with Chat(MODEL_PATH) as c:
            r = c('Say hello')
            assert len(r.content) > 0


class TestChatTools:
    @needs_model
    def test_tool_calling(self):
        def add_numbers(a: float, b: float) -> float:
            """Adds two numbers.
            Args:
                a: The first number.
                b: The second number.
            """
            return a + b

        with Chat(MODEL_PATH, tools=[add_numbers]) as c:
            r = c('What is 123 + 456?')
            assert isinstance(r, Response)
            assert '579' in r.content, f'Expected 579 in response, got: {r.content}'


class TestEngineCache:
    @needs_model
    def test_same_path_same_engine(self):
        e1 = _get_engine(MODEL_PATH)
        e2 = _get_engine(MODEL_PATH)
        assert e1 is e2


class TestAsyncChat:
    @needs_model
    def test_streaming(self):
        async def _run():
            async with AsyncChat(model_path=MODEL_PATH) as c:
                chunks = []
                async for chunk in c('Say hello'):
                    chunks.append(chunk)
                assert isinstance(chunks[-1], Response)
                strs = [ch for ch in chunks if isinstance(ch, str)]
                assert len(strs) > 0
                assert chunks[-1].content == ''.join(strs)
        asyncio.run(_run())

    @needs_model
    def test_acall(self):
        async def _run():
            async with AsyncChat(model_path=MODEL_PATH) as c:
                r = await c.acall('What is Python?')
                assert isinstance(r, Response)
                assert len(r.content) > 0
        asyncio.run(_run())
