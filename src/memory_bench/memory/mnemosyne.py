"""Mnemosyne memory provider — juanmackie/mnemosyne-hermes.

Drives the deployed `mnemosyne mcp` stdio server (the exact surface Hermes
uses) against a SCRATCH database under the harness store dir. The live store
(/opt/data/mnemosyne_data) is never touched.

Adaptation deltas (mirroring the dataset's own documented deltas):
1. recall() has no scope/session parameter on the MCP surface, so per-user
   isolation is enforced adapter-side: memories are ingested carrying their
   originating user in metadata, and retrieve() post-filters the ranked list
   by user_id before applying k. Upstream reference wrappers ignore scope in
   /search too; the cross-user leak fixture is still asserted against the
   filtered result set.
2. Returned Documents carry source_ids=[original doc id] and the original
   context, so the dataset's belief-ID resolver takes its strongest
   (source_id) path rather than falling back to lexical matching.
"""

import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

from ..models import Document
from .base import MemoryProvider

_MNEMOSYNE_BIN = os.environ.get("MNEMOSYNE_BIN") or shutil.which("mnemosyne") or "mnemosyne"
_MCP_TIMEOUT_S = 300
# Over-retrieval factor: post-filtering by user only removes cross-user noise
# (~1 of 35 beliefs on precisionmembench), so 2x is ample headroom.
_OVERFETCH = 2


class MnemosyneMemoryProvider(MemoryProvider):
    name = "mnemosyne"
    description = (
        "juanmackie/mnemosyne-hermes — Hermes memory provider driven over its "
        "`mnemosyne mcp` stdio server (hybrid vector + FTS + keyword recall, "
        "sentence-transformers MiniLM embeddings)."
    )
    kind = "local"
    link = "https://github.com/juanmackie/mnemosyne-hermes"
    concurrency = 1  # single stdio MCP connection; all calls serialized

    def __init__(self):
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._next_id = 0
        self._docs: dict[str, Document] = {}  # engine memory_id -> seeded Document

    # ── lifecycle ────────────────────────────────────────────────────────────

    def prepare(self, store_dir: Path, unit_ids: set[str] | None = None, reset: bool = True) -> None:
        data_dir = Path(store_dir) / "mnemosyne-db"
        if reset and data_dir.exists():
            shutil.rmtree(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        env = dict(os.environ)
        env["MNEMOSYNE_DATA_DIR"] = str(data_dir)
        env.setdefault("MNEMOSYNE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        env.setdefault("MNEMOSYNE_VEC_TYPE", "int8")
        env.setdefault("MNEMOSYNE_VEC_WEIGHT", "0.5")
        env.setdefault("MNEMOSYNE_FTS_WEIGHT", "0.3")
        env.setdefault("MNEMOSYNE_IMPORTANCE_WEIGHT", "0.2")
        env.setdefault("MNEMOSYNE_HOTNESS", "1")
        env.setdefault("MNEMOSYNE_ENHANCED_RECALL", "1")

        self._proc = subprocess.Popen(
            [_MNEMOSYNE_BIN, "mcp"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            env=env,
            cwd=os.environ.get("MNEMOSYNE_CWD", "/opt/data" if os.path.isdir("/opt/data") else None),
        )
        self._rpc(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "agent-memory-benchmark", "version": "1"},
            },
        )
        self._notify("notifications/initialized")
        self._docs.clear()

    def cleanup(self) -> None:
        if self._proc is not None:
            try:
                self._proc.stdin.close()
                self._proc.wait(timeout=15)
            except Exception:
                self._proc.kill()
            self._proc = None

    # ── MCP plumbing ─────────────────────────────────────────────────────────

    def _rpc(self, method: str, params: dict | None = None, timeout: int = _MCP_TIMEOUT_S) -> dict:
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                raise RuntimeError("mnemosyne mcp server is not running")
            self._next_id += 1
            mid = self._next_id
            self._proc.stdin.write(
                json.dumps({"jsonrpc": "2.0", "id": mid, "method": method, "params": params or {}}) + "\n"
            )
            self._proc.stdin.flush()
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                line = self._proc.stdout.readline()
                if not line:
                    raise RuntimeError("mnemosyne mcp closed stdout")
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if msg.get("id") != mid:
                    continue
                if "error" in msg:
                    raise RuntimeError(f"MCP error on {method}: {msg['error']}")
                return msg.get("result") or {}
        raise TimeoutError(f"MCP call {method} timed out after {timeout}s")

    def _notify(self, method: str) -> None:
        with self._lock:
            self._proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": method}) + "\n")
            self._proc.stdin.flush()

    def _tool(self, name: str, arguments: dict) -> dict:
        result = self._rpc("tools/call", {"name": name, "arguments": arguments})
        if result.get("isError"):
            raise RuntimeError(f"{name} returned isError: {result}")
        blocks = result.get("content") or []
        text = blocks[0].get("text", "") if blocks else ""
        data = json.loads(text)
        if isinstance(data, dict) and data.get("status") in ("error", "failed", "rejected"):
            raise RuntimeError(f"{name} failed: {data}")
        return data

    # ── provider interface ───────────────────────────────────────────────────

    def ingest(self, documents: list[Document]) -> None:
        for doc in documents:
            data = self._tool(
                "mnemosyne_remember",
                {
                    "content": doc.content,
                    "source": "amb",
                    "metadata": {
                        "doc_id": doc.id,
                        "user_id": doc.user_id or "",
                        "context": doc.context or "",
                        "tags": list(doc.tags or []),
                    },
                },
            )
            mem_id = data.get("memory_id")
            if mem_id:
                self._docs[mem_id] = doc

    def retrieve(
        self,
        query: str,
        k: int = 10,
        user_id: str | None = None,
        query_timestamp: str | None = None,
        filters: dict | None = None,
    ) -> tuple[list[Document], dict | None]:
        limit = max(k * _OVERFETCH, 20)
        data = self._tool("mnemosyne_recall", {"query": query, "limit": limit})
        out: list[Document] = []
        for r in data.get("results", []):
            doc = self._docs.get(r.get("id"))
            if doc is None:
                continue  # engine-derived memory, not a seeded document
            if user_id is not None and (doc.user_id or None) != user_id:
                continue  # adapter-side cross-user guard (see module docstring)
            out.append(
                Document(
                    id=doc.id,
                    content=r.get("content") or doc.content,
                    user_id=doc.user_id,
                    timestamp=doc.timestamp,
                    context=doc.context,
                    source_ids=[doc.id],
                )
            )
            if len(out) >= k:
                break
        return out, {"provider_raw": data}
