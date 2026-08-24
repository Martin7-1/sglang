# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""KV cache placement event recording.

Produces the ``BlockStored`` / ``BlockRemoved`` / ``AllBlocksCleared`` events
consumed by KV-aware routers (e.g. dynamo). A cache holds one recorder and calls
it; the recorder owns the queue and needs nothing back from its owner.
"""

from typing import Any, Optional

from sglang.srt.disaggregation.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockRemovedWithComponentType,
    BlockStored,
    BlockStoredMetadata,
    BlockStoredWithComponentType,
    BlockStoredWithComponentTypeAndMetadata,
    BlockStoredWithMetadata,
    StorageMedium,
)
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.utils import (
    compute_node_event_hash_values,
    compute_node_hash_values,
    hash_str_to_int64,
)


class KVCacheEventRecorder:
    """Collects KV placement events for one cache.

    ``enabled=False`` makes every ``record_*`` call a no-op and ``take`` return an
    empty list, so callers never have to guard.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        page_size: int,
        component_types_enabled: bool = False,
        default_component_type: Optional[ComponentType] = None,
    ):
        self.enabled = enabled
        self.page_size = page_size
        self.component_types_enabled = component_types_enabled
        self.default_component_type = default_component_type
        self._queue: list = []

    def enqueue(self, event) -> None:
        """Append an event, coalescing it with a compatible queue tail.

        KV event batches already support multiple block hashes.  Combining them
        here avoids emitting one event per page while preserving ordering and
        the parent-linked store chains consumers use to rebuild the cache tree.
        """
        if self._queue:
            tail = self._queue[-1]

            if isinstance(tail, BlockRemovedWithComponentType) and isinstance(
                event, BlockRemovedWithComponentType
            ):
                if (
                    tail.medium == event.medium
                    and tail.component_type == event.component_type
                ):
                    tail.block_hashes.extend(event.block_hashes)
                    return

            elif isinstance(tail, BlockStoredWithComponentType) and isinstance(
                event, BlockStoredWithComponentType
            ):
                tail_metadata = (
                    tail.metadata
                    if isinstance(tail, BlockStoredWithComponentTypeAndMetadata)
                    else None
                )
                event_metadata = (
                    event.metadata
                    if isinstance(event, BlockStoredWithComponentTypeAndMetadata)
                    else None
                )
                if (
                    tail.medium == event.medium
                    and tail.lora_id == event.lora_id
                    and tail.block_size == event.block_size
                    and tail.component_type == event.component_type
                    and tail_metadata == event_metadata
                    and tail.block_hashes
                    and event.parent_block_hash == tail.block_hashes[-1]
                ):
                    tail.block_hashes.extend(event.block_hashes)
                    tail.token_ids.extend(event.token_ids)
                    return

            elif isinstance(tail, BlockRemoved) and isinstance(event, BlockRemoved):
                if tail.medium == event.medium:
                    tail.block_hashes.extend(event.block_hashes)
                    return

            elif isinstance(tail, BlockStored) and isinstance(event, BlockStored):
                tail_metadata = (
                    tail.metadata if isinstance(tail, BlockStoredWithMetadata) else None
                )
                event_metadata = (
                    event.metadata
                    if isinstance(event, BlockStoredWithMetadata)
                    else None
                )
                if (
                    tail.medium == event.medium
                    and tail.lora_id == event.lora_id
                    and tail.block_size == event.block_size
                    and tail_metadata == event_metadata
                    and tail.block_hashes
                    and event.parent_block_hash == tail.block_hashes[-1]
                ):
                    tail.block_hashes.extend(event.block_hashes)
                    tail.token_ids.extend(event.token_ids)
                    return

        self._queue.append(event)

    def _node_event_hash_values(self, node: Any) -> list:
        """Hash values to publish for ``node``, computing them if not yet set."""
        if node.hash_value is None:
            node.hash_value = compute_node_hash_values(node, self.page_size)
        if node.key.cache_salt is not None:
            return compute_node_event_hash_values(node, self.page_size)
        return node.hash_value

    def _parent_block_hash(self, node: Any) -> Optional[int]:
        """The hash the first page of ``node`` links back to.

        ``None`` when the parent is the tree root: a root carries an empty
        ``hash_value`` and no event hash, so it contributes no link. Every other
        node on the path has a parent, which is what distinguishes the two.
        """
        parent = node.parent
        if parent is None or parent.parent is None:
            return None
        if node.key.cache_salt is not None:
            parent_hash_values = parent.event_hash_value
            assert parent_hash_values is not None
        else:
            parent_hash_values = parent.hash_value
        if not parent_hash_values:
            return None
        return hash_str_to_int64(parent_hash_values[-1])

    def _resolve_component_type(
        self, component_type: Optional[ComponentType]
    ) -> Optional[ComponentType]:
        """Component to publish for this recording, or ``None`` for legacy."""
        if not self.component_types_enabled:
            return None
        if component_type is None:
            component_type = self.default_component_type
        return component_type

    def _make_component_store_event(
        self,
        node: Any,
        *,
        block_hashes: list,
        parent_block_hash: Optional[int],
        token_ids: list,
        block_size: int,
        medium,
        component_type: str,
    ):
        event_args = {
            "block_hashes": block_hashes,
            "parent_block_hash": parent_block_hash,
            "token_ids": token_ids,
            "block_size": block_size,
            "lora_id": None,
            "component_type": component_type,
            "medium": medium,
        }
        if node.key.cache_salt is None:
            return BlockStoredWithComponentType(**event_args)
        return BlockStoredWithComponentTypeAndMetadata(
            **event_args,
            metadata=BlockStoredMetadata(cache_salt=node.key.cache_salt),
        )

    def _page_tokens(self, node: Any, start: int, end: int) -> list:
        raw = node.key.token_ids
        # Preserve historical event payload: bigram pages expose tuples.
        if node.key.is_bigram:
            return [(raw[j], raw[j + 1]) for j in range(start, end)]
        return list(raw[start:end])

    def record_store(self, node: Any, medium=None, component_type=None) -> None:
        # One BlockStored per ``page_size`` chunk.
        # ``medium`` defaults to StorageMedium.GPU but callers may override
        # for lower-tier insertions (e.g. StorageMedium.CPU for host/L2 cache).
        if not self.enabled:
            return
        if medium is None:
            medium = StorageMedium.GPU

        resolved_component = self._resolve_component_type(component_type)
        if resolved_component is not None:
            self._record_store_component(node, medium, resolved_component)
            return

        event_hash_values = self._node_event_hash_values(node)
        parent_block_hash = self._parent_block_hash(node)

        page_index = 0
        logical_len = len(node.key)
        is_bigram = node.key.is_bigram
        raw = node.key.token_ids
        for start in range(0, logical_len, self.page_size):
            end = min(start + self.page_size, logical_len)
            if end <= start:
                continue
            # Preserve historical event payload: bigram pages expose tuples.
            if is_bigram:
                page_tokens = [(raw[j], raw[j + 1]) for j in range(start, end)]
            else:
                page_tokens = list(raw[start:end])

            block_hash = hash_str_to_int64(event_hash_values[page_index])

            event_args = {
                "block_hashes": [block_hash],
                "parent_block_hash": parent_block_hash,
                "token_ids": page_tokens,
                "block_size": len(page_tokens),
                "lora_id": None,
                "medium": medium,
            }
            if node.key.cache_salt is None:
                event = BlockStored(**event_args)
            else:
                event = BlockStoredWithMetadata(
                    **event_args,
                    metadata=BlockStoredMetadata(cache_salt=node.key.cache_salt),
                )
            self.enqueue(event)

            parent_block_hash = block_hash
            page_index += 1

    def _record_store_component(self, node: Any, medium, component_type) -> None:
        """Record a Full/SWA page chain or a Mamba leaf checkpoint after commit."""
        event_hash_values = self._node_event_hash_values(node)
        label = str(component_type)

        if component_type.is_mamba:
            # The checkpoint anchors to the node's last page hash; report
            # exactly that page's tokens so hash and token range stay aligned.
            logical_len = len(node.key)
            start = max(0, logical_len - self.page_size)
            self.enqueue(
                self._make_component_store_event(
                    node,
                    block_hashes=[hash_str_to_int64(event_hash_values[-1])],
                    parent_block_hash=None,
                    token_ids=self._page_tokens(node, start, logical_len),
                    block_size=logical_len - start,
                    medium=medium,
                    component_type=label,
                )
            )
            return

        parent_block_hash = self._parent_block_hash(node)
        page_index = 0
        logical_len = len(node.key)
        for start in range(0, logical_len, self.page_size):
            end = min(start + self.page_size, logical_len)
            if end <= start:
                continue
            block_hash = hash_str_to_int64(event_hash_values[page_index])
            self.enqueue(
                self._make_component_store_event(
                    node,
                    block_hashes=[block_hash],
                    parent_block_hash=parent_block_hash,
                    token_ids=self._page_tokens(node, start, end),
                    block_size=end - start,
                    medium=medium,
                    component_type=label,
                )
            )
            parent_block_hash = block_hash
            page_index += 1

    def record_remove(self, node: Any, medium=None, component_type=None) -> None:
        # One BlockRemoved per radix node.
        # ``medium`` defaults to StorageMedium.GPU but callers may override for
        # lower-tier removals (e.g. StorageMedium.CPU when evicting from host).
        if not self.enabled:
            return
        if medium is None:
            medium = StorageMedium.GPU

        resolved_component = self._resolve_component_type(component_type)
        if resolved_component is not None:
            self._record_remove_component(node, medium, resolved_component)
            return

        # Hash values must match what was stored.
        event_hash_values = self._node_event_hash_values(node)

        block_hashes = []
        logical_len = len(node.key)
        page_index = 0
        for start in range(0, logical_len, self.page_size):
            end = min(start + self.page_size, logical_len)
            if end <= start:
                continue

            block_hashes.append(hash_str_to_int64(event_hash_values[page_index]))
            page_index += 1

        if block_hashes:
            self.enqueue(BlockRemoved(block_hashes=block_hashes, medium=medium))

    def _record_remove_component(self, node: Any, medium, component_type) -> None:
        """Record a component removal after the allocator has freed it."""
        event_hash_values = self._node_event_hash_values(node)
        label = str(component_type)

        if component_type.is_mamba:
            self.enqueue(
                BlockRemovedWithComponentType(
                    block_hashes=[hash_str_to_int64(event_hash_values[-1])],
                    component_type=label,
                    medium=medium,
                )
            )
            return

        block_hashes = []
        logical_len = len(node.key)
        page_index = 0
        for start in range(0, logical_len, self.page_size):
            end = min(start + self.page_size, logical_len)
            if end <= start:
                continue
            block_hashes.append(hash_str_to_int64(event_hash_values[page_index]))
            page_index += 1

        if block_hashes:
            self.enqueue(
                BlockRemovedWithComponentType(
                    block_hashes=block_hashes,
                    component_type=label,
                    medium=medium,
                )
            )

    def record_all_cleared(self) -> None:
        if not self.enabled:
            return
        self.enqueue(AllBlocksCleared())

    def take(self) -> list:
        """Atomically takes all events and clears the queue.

        Returns:
            A list of KV cache events.
        """
        if not self.enabled:
            return []
        events = self._queue
        self._queue = []
        return events
