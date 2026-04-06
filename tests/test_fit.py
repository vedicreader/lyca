"""Tests for lyca.fit — pure function tests, no network, no model needed.

Run: pytest tests/test_fit.py -v
"""

import pytest

from lyca.fit import (
    MODEL_REGISTRY, register_model, _resolve_entry, _parse_apple_chip,
    syscheck, _gpu_usable, _fmt_table, models, recommend, download,
)
from lyca.core import Chat, AsyncChat


# ===========================================================================
# MODEL_REGISTRY
# ===========================================================================

class TestModelRegistry:
    def test_count(self):
        assert len(MODEL_REGISTRY) == 9

    def test_required_keys(self):
        required = {'id', 'repo', 'task', 'min_ram_gb'}
        for e in MODEL_REGISTRY:
            assert required <= set(e), f'{e["id"]} missing {required - set(e)}'

    def test_no_duplicate_ids(self):
        ids = [e['id'] for e in MODEL_REGISTRY]
        assert len(ids) == len(set(ids))

    def test_register_model_accepts_valid(self):
        entry = {'id': '_test', 'repo': 'x/y', 'file': 'a.litertlm',
                 'size_gb': 1.0, 'min_ram_gb': 2.0, 'task': 'chat'}
        register_model(entry)
        assert MODEL_REGISTRY[-1]['id'] == '_test'
        MODEL_REGISTRY.pop()  # cleanup

    def test_register_model_rejects_missing_keys(self):
        with pytest.raises(AssertionError):
            register_model({'id': 'bad'})


# ===========================================================================
# _resolve_entry
# ===========================================================================

class TestResolveEntry:
    def test_by_id(self):
        e = _resolve_entry('gemma4-e4b')
        assert e is not None and e['id'] == 'gemma4-e4b'

    def test_by_repo(self):
        e = _resolve_entry('litert-community/gemma-4-E4B-it-litert-lm')
        assert e is not None and e['id'] == 'gemma4-e4b'

    def test_unknown(self):
        assert _resolve_entry('nonexistent') is None


# ===========================================================================
# _parse_apple_chip
# ===========================================================================

class TestParseAppleChip:
    @pytest.mark.parametrize('brand,expected', [
        ('Apple M4 Max', ('M4', 3)),
        ('Apple M1', ('M1', 1)),
        ('Apple M2 Pro', ('M2', 2)),
        ('Apple M1 Ultra', ('M1', 4)),
        ('Apple M3', ('M3', 1)),
        ('Intel Core i9-12900K', (None, None)),
        ('', (None, None)),
    ])
    def test_parse(self, brand, expected):
        assert _parse_apple_chip(brand) == expected


# ===========================================================================
# syscheck
# ===========================================================================

class TestSyscheck:
    def test_returns_dict_with_expected_keys(self):
        spec = syscheck()
        expected = {'platform', 'arch', 'cpu', 'ram_gb', 'free_ram_gb',
                    'gpu', 'gpu_vram_gb', 'apple_chip', 'apple_tier'}
        assert expected == set(spec.keys())

    def test_ram_positive(self):
        assert syscheck()['ram_gb'] > 0

    def test_platform_known(self):
        assert syscheck()['platform'] in ('macOS', 'Linux', 'Windows')

    def test_cached(self):
        "syscheck() is lru_cached — same object on repeated calls."
        assert syscheck() is syscheck()


# ===========================================================================
# _gpu_usable
# ===========================================================================

class TestGpuUsable:
    def test_metal(self):  assert _gpu_usable({'gpu': 'metal'}) is True
    def test_cuda(self):   assert _gpu_usable({'gpu': 'cuda'}) is True
    def test_none(self):   assert _gpu_usable({'gpu': 'none'}) is False
    def test_empty(self):  assert _gpu_usable({}) is False


# ===========================================================================
# _fmt_table
# ===========================================================================

class TestFmtTable:
    def test_header_present(self):
        t = _fmt_table(MODEL_REGISTRY[:2])
        assert 'id' in t and 'params' in t and 'size' in t

    def test_entries_present(self):
        t = _fmt_table(MODEL_REGISTRY[:2], spec={'gpu': 'metal'})
        assert 'gemma4-e2b' in t
        assert 'gemma4-e4b' in t

    def test_empty_list(self):
        t = _fmt_table([])
        assert 'id' in t  # header still rendered


# ===========================================================================
# models
# ===========================================================================

class TestModels:
    def test_all(self, capsys):
        out = models()
        assert len(out) == 9

    def test_filter_family(self, capsys):
        out = models(family='gemma4')
        assert len(out) == 2
        assert all(e['family'] == 'gemma4' for e in out)

    def test_filter_tag(self, capsys):
        out = models(tag='reasoning')
        ids = {e['id'] for e in out}
        assert 'phi4-mini' in ids
        assert 'deepseek-r1-1.5b' in ids

    def test_filter_task(self, capsys):
        assert len(models(task='chat')) == 9


# ===========================================================================
# recommend
# ===========================================================================

class TestRecommend:
    def test_returns_list(self):
        recs = recommend(verbose=False)
        assert isinstance(recs, list)

    def test_entries_have_id(self):
        for r in recommend(verbose=False):
            assert 'id' in r

    def test_scoring_keys_cleaned(self):
        for r in recommend(verbose=False):
            assert '_score' not in r
            assert '_expected_tps' not in r

    def test_entries_fit_ram(self):
        spec = syscheck()
        for r in recommend(verbose=False):
            assert spec['free_ram_gb'] >= r['min_ram_gb'] * 1.15


# ===========================================================================
# download resolution (no network)
# ===========================================================================

class TestDownloadResolution:
    def test_resolve_gemma4(self):
        e = _resolve_entry('gemma4-e4b')
        assert e['repo'] == 'litert-community/gemma-4-E4B-it-litert-lm'
        assert e['file'] == 'gemma-4-E4B-it.litertlm'

    def test_resolve_smollm_file_none(self):
        e = _resolve_entry('smollm-135m')
        assert e['file'] is None

    def test_direct_repo_not_in_registry(self):
        assert _resolve_entry('some-org/some-model') is None


# ===========================================================================
# from_hf existence
# ===========================================================================

class TestFromHf:
    def test_chat_has_from_hf(self):
        assert hasattr(Chat, 'from_hf') and callable(Chat.from_hf)

    def test_asyncchat_has_from_hf(self):
        assert hasattr(AsyncChat, 'from_hf') and callable(AsyncChat.from_hf)
