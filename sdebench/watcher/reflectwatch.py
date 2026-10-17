"""The watcher on reflect: one reflect call per moment, and Hindsight decides whether to speak.

The client only builds the question and the agent's situation, skips memories it already sent,
and checks the quote is really in the source. Recall, the decision and the note are reflect's.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass, field

import hsmemory as M
import watcher as W
from watcher import Moment, Verdict, clip_diff

HS = os.environ.get("WATCHER_HS_URL", "http://localhost:8899")

HEADSUP_Q = ("Before I start on this request: which project decision recorded in memory (an exact value, a rule, "
             "an exception) about exactly the code this request will change must my fix respect, that the request "
             "itself does not state? Tell me in at most 70 words, as 'Before you change <the code>: <every such rule, "
             "in its final form>', copying exact literals verbatim. If memory records no such decision about this "
             "code, reply with nothing.")
CONFLICT_Q = ("Does what I am doing right now (my plan, command or code change) contradict a project decision recorded "
              "in memory, miss one, or repeat an option recorded as rejected? If so, tell me in at most 50 words the "
              "recorded value or rule (exact literals verbatim) and what in my work disagrees. How I run, install or "
              "check tests is never a conflict. If nothing conflicts, reply with nothing.")


def situation(goal: str, m: Moment, sent: list[str]) -> str:
    doing = clip_diff(m.diff, 6000) if m.trigger == "edit" else m.text[:3000]
    parts = [f"MY REQUEST:\n{goal[:3000]}"]
    if m.trigger != "task":
        parts.append(f"WHAT I JUST DID ({m.trigger}):\n{doing}")
        if m.recent:
            parts.append("MY RECENT STEPS:\n" + "\n".join(m.recent[-8:]))
        if m.trigger != "edit" and m.diff.strip():
            parts.append(f"MY CODE CHANGE SO FAR:\n{clip_diff(m.diff, 4000)}")
    if sent:
        parts.append("ALREADY TOLD ME:\n" + "\n".join(sent))
    return "\n\n".join(parts)


def search_queries(m: Moment) -> list[str]:
    """What to look up, as v5 did: a request by its sentences naming code, then its paragraphs (as one
    query the harness boilerplate outweighed the request); an edit by its diff in words; else the text."""
    if m.trigger != "task":
        return [M.moment_query(m)]
    paras = [p.strip() for p in re.split(r"\n\s*\n", m.text) if len(p.strip()) >= 40]
    named = [x.strip() for x in re.split(r"(?<=[.!?])\s+", m.text)
             if re.search(r"\b[a-z]+_[a-z_]+\b|\w\(\)|`[^`]+`", x) and "test" not in x.lower()]
    return [q[:1500] for q in (named[:3] + paras)[:8]]


def ask(bank: str, goal: str, m: Moment, sent: list[str], extra: dict | None = None) -> dict:
    body = {"query": HEADSUP_Q if m.trigger == "task" else CONFLICT_Q, "context": situation(goal, m, sent),
            "abstain": True, "search_queries": search_queries(m),
            "budget": "low", "max_tokens": 600, "fact_types": ["observation"], "exclude_mental_models": True,
            "include": {"facts": {}, "tool_calls": {"output": False}}, **(extra or {})}
    req = urllib.request.Request(f"{HS}/v1/default/banks/{bank}/reflect", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def _readable(text: str) -> str:
    """A chat stored as JSON lines, as its messages' words: a quote must be text the agent can find."""
    out = []
    for line in text.splitlines():
        try:
            msg = json.loads(line)
        except ValueError:
            out.append(line)
            continue
        out.append(str(msg.get("content", "")) if isinstance(msg, dict) else line)
    return "\n".join(out)


def _source(bank: str, observation_id: str, note: str) -> tuple[str, str]:
    """Where the agent can read the decision, and its closest sentence there (verbatim by construction):
    the observation's newest source memory with a document, then that document's text."""
    obs = M._req("GET", f"{bank}/memories/{observation_id}", timeout=30, quiet=True)
    sources = []
    for sid in obs.get("source_memory_ids") or []:
        f = M._req("GET", f"{bank}/memories/{sid}", timeout=30, quiet=True)
        if f.get("document_id"):
            sources.append(f)
    sources.sort(key=lambda f: f.get("mentioned_at") or f.get("occurred_start") or "", reverse=True)
    for f in sources:
        text = M._doc_text(bank, f["document_id"])
        quote = W.best_sentence(_readable(text), note)
        if quote:
            return M._by_subject(M.where_for(f["document_id"], f.get("metadata") or {}), text), quote
    return "project memory", ""


@dataclass
class ReflectWatcher:
    """v5's watcher with reflect as its whole memory side: one reflect call per moment decides and
    writes. Kept here, as in v5: no memory is said twice (a heads-up does not count), at most
    MAX_MESSAGES notes, and the same situation is never judged twice."""
    bank: str
    goal: str
    sent: list[str] = field(default_factory=list)
    evidence_sent: set[str] = field(default_factory=set)
    judged: set[str] = field(default_factory=set)
    recall_ms: int = 0

    def review(self, m: Moment) -> tuple[Verdict, list]:
        key = hashlib.sha1(json.dumps([m.trigger, m.text if m.trigger != "edit" else "", m.diff]).encode()).hexdigest()
        if len(self.sent) >= W.MAX_MESSAGES or key in self.judged or not (m.diff if m.trigger == "edit" else m.text).strip():
            return Verdict(False), []
        self.judged.add(key)
        t0 = time.time()
        r = ask(self.bank, self.goal, m, self.sent)
        u = r.get("usage") or {}
        v = Verdict(False, ms=int((time.time() - t0) * 1000))
        v.decision = {"abstained": bool(r.get("abstained")), "llm_in": u.get("input_tokens"),
                      "llm_out": u.get("output_tokens"),
                      "steps": [c.get("scope") for c in ((r.get("trace") or {}).get("llm_calls") or [])]}
        W.CALLS.append({"ts": t0, "service": "hindsight", "name": "reflect", "ms": v.ms,
                        "request": json.dumps({"query": m.trigger, "search_queries": search_queries(m)})[:4000],
                        "response": json.dumps({k: r.get(k) for k in ("text", "abstained", "usage")})[:4000]})
        text = (r.get("text") or "").strip()
        used = [x["id"] for x in ((r.get("based_on") or {}).get("memories") or [])]
        if not text or (used and set(used) <= self.evidence_sent):
            return v, []
        v.where, v.quote = _source(self.bank, used[0], text) if used else ("project memory", "")
        v.speak, v.message, v.evidence = True, text, (used[0] if used else "")
        self.sent.append(text)
        if m.trigger != "task":  # a heads-up does not use up the memory: going against it later is still said
            self.evidence_sent.update(used)
        return v, []
