"""Unit tests for component-aware KV cache event recording.

Covers the ``--enable-kv-events-component-types`` semantics of
``KVCacheEventRecorder``: full/SWA page chains carry their component label,
mamba checkpoints publish one leaf event anchored to the node's last page,
and coalescing never merges events across component boundaries.

Usage:
    python test_kv_events_component_types.py
"""

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")

import unittest

from sglang.srt.disaggregation.kv_events import (
    BlockRemoved,
    BlockRemovedWithComponentType,
    BlockStored,
    BlockStoredWithComponentType,
    StorageMedium,
)
from sglang.srt.mem_cache.events import KVCacheEventRecorder
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.test.test_utils import CustomTestCase

PAGE_SIZE = 4


class FakeKey:
    """Recorder-shaped key. Bigram keys expose ``logical_len`` positions but
    ``len(raw) == logical_len + 1`` overlapping tokens."""

    def __init__(self, token_ids, *, logical_len=None, is_bigram=False):
        self.token_ids = token_ids
        self.is_bigram = is_bigram
        self.cache_salt = None
        self._len = logical_len if logical_len is not None else len(token_ids)

    def __len__(self):
        return self._len


class FakeNode:
    """Minimal recorder-shaped node: key, parent, pre-set hash_value."""

    def __init__(self, key, hash_value, parent=None):
        self.key = key
        self.hash_value = hash_value
        self.parent = parent


def make_recorder(component_type=None):
    return KVCacheEventRecorder(
        enabled=True,
        page_size=PAGE_SIZE,
        component_types_enabled=True,
        default_component_type=component_type,
    )


class TestComponentAwareStoreEvents(CustomTestCase):
    def test_full_page_chain_carries_component_type(self):
        recorder = make_recorder(ComponentType.FULL)
        root = FakeNode(FakeKey([]), [])
        node = FakeNode(FakeKey(list(range(6))), ["aa", "bb"], parent=root)
        recorder.record_store(node)

        events = recorder.take()
        self.assertEqual(len(events), 2)
        for event in events:
            self.assertIsInstance(event, BlockStoredWithComponentType)
            self.assertEqual(event.component_type, "full")
        # Parent-linked page chain: the root contributes no link, and the
        # second page links to the first page's hash.
        self.assertIsNone(events[0].parent_block_hash)
        self.assertEqual(events[0].token_ids, [0, 1, 2, 3])
        self.assertEqual(events[0].block_size, 4)
        self.assertEqual(events[1].parent_block_hash, events[0].block_hashes[0])
        self.assertEqual(events[1].token_ids, [4, 5])

    def test_swa_component_label(self):
        recorder = make_recorder(ComponentType.SWA)
        node = FakeNode(FakeKey([1, 2]), ["aa"])
        recorder.record_store(node)

        events = recorder.take()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].component_type, "swa")

    def test_mamba_store_is_single_leaf_event(self):
        # A mamba checkpoint anchors to the node's last page hash and reports
        # the trailing page window of tokens — one event, no parent link.
        # Mamba key lengths are checkpoint-aligned, so the window is a page.
        recorder = make_recorder(ComponentType.MAMBA)
        node = FakeNode(FakeKey(list(range(8))), ["aa", "bb"])
        recorder.record_store(node)

        events = recorder.take()
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertIsInstance(event, BlockStoredWithComponentType)
        self.assertEqual(event.component_type, "mamba")
        self.assertIsNone(event.parent_block_hash)
        self.assertEqual(event.token_ids, [4, 5, 6, 7])
        self.assertEqual(event.block_size, 4)
        self.assertEqual(len(event.block_hashes), 1)

    def test_mamba_store_bigram_exposes_pairs(self):
        recorder = make_recorder(ComponentType.MAMBA)
        # 2 logical bigram positions backed by 3 overlapping raw tokens.
        node = FakeNode(
            FakeKey([10, 20, 30], logical_len=2, is_bigram=True), ["aa", "bb"]
        )
        recorder.record_store(node)

        events = recorder.take()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].token_ids, [(10, 20), (20, 30)])
        self.assertEqual(events[0].block_size, 2)

    def test_explicit_component_type_overrides_default(self):
        recorder = make_recorder(ComponentType.FULL)
        node = FakeNode(FakeKey([1, 2]), ["aa"])
        recorder.record_store(node, component_type=ComponentType.SWA)

        events = recorder.take()
        self.assertEqual(events[0].component_type, "swa")

    def test_medium_override_is_preserved(self):
        recorder = make_recorder(ComponentType.FULL)
        node = FakeNode(FakeKey([1, 2]), ["aa"])
        recorder.record_store(node, medium=StorageMedium.CPU)

        events = recorder.take()
        self.assertEqual(events[0].medium, StorageMedium.CPU)


class TestComponentAwareRemoveEvents(CustomTestCase):
    def test_full_remove_carries_all_page_hashes(self):
        recorder = make_recorder(ComponentType.FULL)
        node = FakeNode(FakeKey(list(range(6))), ["aa", "bb"])
        recorder.record_remove(node)

        events = recorder.take()
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertIsInstance(event, BlockRemovedWithComponentType)
        self.assertEqual(event.component_type, "full")
        self.assertEqual(len(event.block_hashes), 2)

    def test_mamba_remove_is_single_leaf_hash(self):
        recorder = make_recorder(ComponentType.MAMBA)
        node = FakeNode(FakeKey(list(range(6))), ["aa", "bb"])
        recorder.record_remove(node)

        events = recorder.take()
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.component_type, "mamba")
        self.assertEqual(len(event.block_hashes), 1)


class TestComponentCoalescingBoundaries(CustomTestCase):
    def test_same_component_chain_coalesces(self):
        recorder = make_recorder(ComponentType.FULL)
        root = FakeNode(FakeKey([]), [])
        first = FakeNode(FakeKey([0, 1, 2, 3]), ["aa"], parent=root)
        # Second node continues the chain: its parent's last hash equals the
        # first node's published hash, and both pages are full-sized.
        second = FakeNode(FakeKey([4, 5, 6, 7]), ["bb"], parent=first)
        recorder.record_store(first)
        recorder.record_store(second)

        events = recorder.take()
        self.assertEqual(len(events), 1)
        self.assertEqual(len(events[0].block_hashes), 2)

    def test_different_components_do_not_coalesce(self):
        recorder = make_recorder(ComponentType.FULL)
        node = FakeNode(FakeKey([0, 1, 2, 3]), ["aa"])
        recorder.record_store(node, component_type=ComponentType.FULL)
        recorder.record_store(node, component_type=ComponentType.SWA)

        events = recorder.take()
        self.assertEqual(len(events), 2)

    def test_removes_do_not_coalesce_across_components(self):
        recorder = make_recorder(ComponentType.FULL)
        node = FakeNode(FakeKey([0, 1]), ["aa"])
        recorder.record_remove(node, component_type=ComponentType.FULL)
        recorder.record_remove(node, component_type=ComponentType.MAMBA)

        events = recorder.take()
        self.assertEqual(len(events), 2)


class TestComponentTypesDisabledKeepsLegacy(CustomTestCase):
    def test_flag_off_emits_legacy_structs(self):
        recorder = KVCacheEventRecorder(
            enabled=True,
            page_size=PAGE_SIZE,
            component_types_enabled=False,
            default_component_type=ComponentType.FULL,
        )
        node = FakeNode(FakeKey([0, 1, 2, 3]), ["aa"])
        recorder.record_store(node)
        recorder.record_remove(node)

        events = recorder.take()
        self.assertIsInstance(events[0], BlockStored)
        self.assertNotIsInstance(events[0], BlockStoredWithComponentType)
        self.assertIsInstance(events[1], BlockRemoved)
        self.assertNotIsInstance(events[1], BlockRemovedWithComponentType)

    def test_no_default_component_emits_legacy_structs(self):
        # component_types_enabled but no component identity available (e.g. a
        # cache that never got wired) falls back to the legacy shape.
        recorder = KVCacheEventRecorder(
            enabled=True, page_size=PAGE_SIZE, component_types_enabled=True
        )
        node = FakeNode(FakeKey([0, 1]), ["aa"])
        recorder.record_store(node)

        events = recorder.take()
        self.assertIsInstance(events[0], BlockStored)
        self.assertNotIsInstance(events[0], BlockStoredWithComponentType)


if __name__ == "__main__":
    unittest.main()
