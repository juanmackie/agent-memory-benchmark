"""Proactive memory watcher (experiment).

A separate process beside a coding agent. It never answers questions: it watches the agent's
working tree, and when a change touches something its memory knows about, it decides whether one
short message would change what the agent does next. Only then does it speak.

    change (git diff) -> what to search (identifiers/values the diff touches) -> search memory
      -> judge: speak or stay silent -> one 1-2 line message into the agent's inbox

Memory here is plain files, no Hindsight: past developer chats (every task's `conversations`) and
the repo's git history. Stdlib only, so it runs anywhere the harness runs.

Live:    python watcher.py watch --repo DIR --task task.json --log watcher.jsonl
Replay:  python watcher.py replay --task task.json --patch-file p.diff
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

DATASETS = Path(__file__).resolve().parents[1] / "datasets"
MODEL = os.environ.get("WATCHER_MODEL", "gemini-3.8-flash")
sys.modules.setdefault("watcher", sys.modules[__name__])  # run as a script, hsmemory's `import watcher` gets this module
TOP_K = 3
CALLS: list[dict] = []  # outside calls not yet logged (Hindsight requests, judge prompts): the session viewer shows them
MAX_MESSAGES = 3

# Agent-side mailbox. Inside .git so it never shows up in the agent's diff (the graded patch).
INBOX = ".git/watcher-inbox.jsonl"
BUSY = ".git/watcher-busy"


# ---------------------------------------------------------------- memory


@dataclass
class Note:
    id: str
    source: str  # "chat" | "commit"
    text: str
    where: str  # how the AGENT can open it: a file path or a git command (unverifiable notes get ignored)
    sources: tuple = ()  # ((where, text), ...) the quote must come from, when not the note itself


def load_chats(task: dict) -> list[Note]:
    """This project's past developer chats — the same transcripts the harness puts in the agent's
    container (seed_transcript_files), so every note points at a file the agent can open.
    Only this project's: the dataset's sibling tasks are forks of it with different decisions, and
    citing them told the agent rules from a chat it could not find (it then distrusted everything)."""
    conv = task.get("conversations") or []
    if not conv:
        return []
    chats = conv if isinstance(conv[0], list) else [conv]
    return [Note(f"chat:{i + 1}", "chat", "\n".join(f"{c['role'].upper()}: {c['text']}" for c in chat),
                 f"/root/project-history/session-{i + 1:02d}.md")
            for i, chat in enumerate(chats) if chat]


def load_commits(repo: Path, limit: int = 200) -> list[Note]:
    """The repo's own history: subject, body and the changed lines of each commit."""
    out = subprocess.run(
        ["git", "-C", str(repo), "log", f"-n{limit}", "-p", "--format=\x1e%h%n%s%n%b", "--", ".", ":(exclude)tests"],
        capture_output=True, text=True, errors="replace",
    ).stdout
    notes = []
    for block in out.split("\x1e")[1:]:
        sha, _, rest = block.partition("\n")
        lines = rest.splitlines()
        changed = [l for l in lines if l[:1] in "+-" and not l.startswith(("+++", "---"))][:40]
        msg = "\n".join(l for l in lines if l and l[:1] not in "+-@" and not l.startswith(("diff ", "index ")))
        notes.append(Note(f"commit:{sha}", "commit", (msg[:1500] + "\n" + "\n".join(changed)).strip(),
                          f"git show {sha}"))
    return notes


# ---------------------------------------------------------------- search

_STOP = set(
    "the a an and or of to in is it that this for on with as be are was we our you i at by from "
    "not but if then else return def self none true false import class pass elif while try except "
    "raise lambda yield with assert del global nonlocal int str float bool list dict set tuple len "
    "range print can will should would just so do does did has have had what why how".split()
)


def tokens(text: str) -> list[str]:
    """Words, identifiers split into their parts (MAX_ATTEMPTS -> max_attempts, max, attempts;
    camelCase too), and numbers. Numbers matter: a changed value is often the whole story."""
    out = []
    for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?", text):
        lw = w.lower()
        parts = [p.lower() for p in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", w)]
        for t in {lw, *parts}:
            if len(t) > 1 and t not in _STOP:
                out.append(t)
    return out


class Index:
    """BM25 over whole notes. Chats are short (a handful of turns), so a whole chat is the unit."""

    def __init__(self, notes: list[Note]):
        self.notes = notes
        self.docs = [Counter(tokens(n.text)) for n in notes]
        self.len = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.len) / max(1, len(self.len))
        df = Counter(t for d in self.docs for t in d)
        n = len(notes)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def search(self, query: list[str], k: int = TOP_K) -> list[tuple[float, Note]]:
        q = Counter(query)
        scored = []
        for note, d, ln in zip(self.notes, self.docs, self.len):
            s = 0.0
            for t, qw in q.items():
                f = d.get(t)
                if f:
                    s += qw * self.idf[t] * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * ln / self.avg))
            if s > 0:
                scored.append((s, note))
        scored.sort(key=lambda x: -x[0])
        return scored[:k]


def query_from_diff(diff: str) -> list[str]:
    """What to search for: the words on the changed lines (weighted x2, they ARE the change),
    the functions the hunks sit in, and the files touched."""
    q: list[str] = []
    for line in diff.splitlines():
        if line.startswith(("+++", "---")):
            q += tokens(Path(line[4:]).stem)
        elif line.startswith("@@"):
            q += tokens(line.split("@@")[-1])
        elif line[:1] in "+-":
            q += tokens(line[1:]) * 2
    return q


# ---------------------------------------------------------------- judge

JUDGE = """You are a memory watcher sitting beside a coding agent. The agent cannot see you. You see
what the agent is doing right now and a few entries from the project's memory (past developer sessions).

Decide whether to send the agent ONE short message. The only test: would the agent do something
different after reading it? Send only when memory states something specific (a value, a rule, an
exception, a past failure) about exactly what the agent is working on, and what the agent is doing
(its plan, its command, or its code change) contradicts it, misses it, or repeats something recorded
as tried and rejected. Early is better: a plan that is about to break a recorded rule is worth a
message before any code is written. At the task start, speak only if the request itself asks for
something memory rejected, or memory records a requirement for exactly this code that the request
does not mention and a careful engineer would get wrong.

Stay silent when: no memory is about this exact code or step; the agent already agrees with the
memory; the agent has not yet shown what it will do (it is only reading or searching); the memory
is generic; you would only be confirming, praising or giving general advice; you already said it
(see ALREADY SENT). Running tests or builds is never a problem. When a chat changes its mind, only
its LAST word counts. A CURRENT BELIEF line is the memory's up-to-date view and wins over older
wording in its source; copy the quote from the SOURCE text, choosing words that support it.

USER REQUEST TO THE AGENT:
{goal}

WHAT JUST HAPPENED ({trigger}):
{text}

AGENT'S RECENT STEPS (oldest first):
{recent}

AGENT'S CODE CHANGE SO FAR (git diff):
{diff}

MEMORY EXCERPTS:
{excerpts}

ALREADY SENT:
{sent}

First, for each excerpt about what the agent is working on, write down the exact value or rule it
records (the LAST word if the chat changed its mind), what the agent plans or does instead, and
whether they conflict. Then decide.

Reply with JSON only:
{{"checks": [{{"evidence": "<id>", "recorded": "<value/rule in memory>", "agent_does": "<what the agent plans or does>", "conflict": true|false}}],
  "speak": true|false, "evidence": "<id of the conflicting excerpt, or empty>",
  "quote": "<if speak: the decisive words copied EXACTLY from that excerpt, at most 25 words>",
  "message": "<if speak: at most 50 words, written TO the agent as 'you'. State the recorded value or rule itself (e.g. 'X must be exactly 7') and what in your plan or change disagrees. Plain words, no preamble.>"}}"""


@dataclass
class Moment:
    """What the watcher reviews: something the agent just did, with what it did before."""
    trigger: str  # task | plan | command | edit | failure
    text: str  # the request, the agent's words, the command, the diff, or the failure output
    recent: list[str] = field(default_factory=list)  # the agent's last steps, one line each
    diff: str = ""  # its code change so far


@dataclass
class Verdict:
    speak: bool
    evidence: str = ""
    message: str = ""
    quote: str = ""
    where: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    ms: int = 0
    decision: dict = field(default_factory=dict)  # Jev's pick and probabilities, when it decided


def gemini(prompt: str, model: str | None = None, thinking: str | None = None) -> tuple[dict, int, int]:
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/{model or MODEL}:generateContent"
           f"?key={os.environ['GEMINI_API_KEY']}")
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0,
                                 **({"thinkingConfig": {"thinkingLevel": thinking}} if thinking else {})}}
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                d = json.load(r)
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
    text = d["candidates"][0]["content"]["parts"][0]["text"]
    u = d.get("usageMetadata", {})
    CALLS.append({"ts": t0, "service": "llm", "name": model or MODEL, "request": prompt[:30000],
                  "response": text[:8000], "ms": int((time.time() - t0) * 1000)})
    return json.loads(text), u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0)


def clip_diff(diff: str, cap: int = 6000) -> str:
    return diff if len(diff) <= cap else diff[:cap] + "\n…(diff truncated)"


def judge(goal: str, m: Moment, hits: list[tuple[float, Note]], sent: list[str]) -> Verdict:
    if not hits:
        return Verdict(False)
    excerpts = "\n\n".join(f"[{n.id}]\n{n.text[:12000]}" for _, n in hits if not n.sources)
    shared = {w: t for _, n in hits for w, t in n.sources}  # observations: each source printed once
    if shared:
        excerpts += "\n\n".join(f"[{n.id}] {n.text} (sources: {'; '.join(w for w, _ in n.sources)})"
                                  for _, n in hits if n.sources)
        budget = 24000  # escalations read up to 63k tokens; the decisive passage is rarely past this
        # chats first (decisions live there; the longest is ~10k chars), then commits
        for w, t in sorted(shared.items(), key=lambda kv: "session-" not in kv[0]):
            if budget <= 0:
                break
            excerpts += f"\n\nSOURCE {w}:\n{t[:min(10000, budget)]}"
            budget -= min(10000, len(t))
    prompt = JUDGE.format(goal=goal or "(unknown)", trigger=m.trigger,
                          text=clip_diff(m.text, 3000) if m.trigger != "edit" else "the agent changed code (see the diff below)",
                          recent="\n".join(m.recent) or "(nothing yet)",
                          diff=clip_diff(m.diff) if m.diff.strip() else "(no code changed yet)", excerpts=excerpts,
                          sent="\n".join(f"- {s}" for s in sent) or "(nothing)")
    t0 = time.time()
    # default thinking took 14-43s and the note landed after the agent's turn ended; low takes ~2s
    out, ti, to = gemini(prompt, thinking=os.environ.get("WATCHER_JUDGE_THINKING", "low") or None)
    v = Verdict(bool(out.get("speak")), str(out.get("evidence") or ""), str(out.get("message") or "").strip(),
                str(out.get("quote") or "").strip(), "", ti, to, int((time.time() - t0) * 1000))
    # Speak only with proof the agent can check: the quote must really be in the cited note.
    note = next((n for _, n in hits if n.id == v.evidence), None)
    found = [w for w, t in (note.sources or ((note.where, note.text),))
             if squash(v.quote) in squash(t)] if v.speak and v.message and note and v.quote else []
    v.speak = bool(found)
    if v.speak:
        v.where = found[0]
    return v


# ---------------------------------------------------------------- decide with Jev, write with a small LLM

JEV_URL = os.environ.get("JEV_URL", "https://api.typesafe.ai/v1/systemone")
# Jev alone speaks only when sure; 0.45-0.47 calls on test failures were junk (cited unrelated commits)
SPEAK_AT = float(os.environ.get("WATCHER_SPEAK_AT", "0.3"))  # speak when Jev's p("no conflict") is below this
WRITER = os.environ.get("WATCHER_WRITER", "gemini-3.5-flash-lite")
# between SPEAK_AT and this, Jev is unsure (live misses sat at 0.52-0.61): ask the LLM judge instead
ESCALATE_BELOW = float(os.environ.get("WATCHER_ESCALATE_BELOW", "0.9"))

DECIDE = ("Does what the agent is doing right now (its plan, command or code change) contradict a recorded "
          "project decision, miss one, or repeat an option recorded as rejected? Pick that memory. Pick 'none' "
          "when the agent agrees with memory, is only reading or exploring, or no memory is about exactly this code. "
          "How the agent runs, installs or checks tests is never a conflict.")

# At the task start nothing is done yet, so ask something else: is there a recorded rule this fix must
# respect that the request does not say? Claude otherwise makes the mistake first and pays a fix round.
HEADSUP = ("Before any work starts: which memory records a specific decision (a value, a rule, an exception) "
           "about exactly the code this request will change, one that the request itself does not state and a "
           "careful engineer fixing the request could easily get wrong? Pick 'none' when no memory is clearly "
           "about this code, or the memory is generic.")

HEADSUP_WRITE = """A memory watcher is giving a coding agent a heads-up before it starts on a request.

REQUEST: {doing}

MEMORY (current belief): {memory}

SOURCE TEXT (where the decision was recorded):
{source}

Reply JSON only: {{"message": "<at most 70 words, written TO the agent as 'you': 'Before you change <the code>: <every rule the SOURCE TEXT records for this code, in its final form>.' Include each rule that a fix could get wrong, not only the MEMORY line. Copy exact literals (values, formats, examples such as ="00042" or 30s) verbatim from the source, never paraphrase them. Plain words, no preamble, no advice beyond the rule.>",
  "quote": "<the decisive words copied EXACTLY from the SOURCE TEXT, at most 25 words>"}}"""

WRITE = """A memory watcher caught a coding agent going against a recorded project decision. Write the note.

MEMORY (current belief): {memory}

SOURCE TEXT (where the decision was recorded):
{source}

WHAT THE AGENT IS DOING ({trigger}):
{doing}

Reply JSON only: {{"message": "<at most 50 words, written TO the agent as 'you' (never 'the agent'): state the recorded value or rule itself (copy exact literals such as ="00042" or 30s verbatim from the source) and what in your plan or change disagrees. Plain words, no preamble.>",
  "quote": "<the decisive words copied EXACTLY from the SOURCE TEXT, at most 25 words>"}}"""


def jev(state: str, questions: dict) -> dict:
    body = {"model": "jev-latest", "state": state, "questions": questions}
    req = urllib.request.Request(JEV_URL, json.dumps(body).encode(), {
        "Authorization": f"Bearer {os.environ['JEV_KEY']}", "Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=15) as r:
        out = json.load(r)
    CALLS.append({"ts": t0, "service": "decision", "name": out.get("model", "jev"),
                  "request": json.dumps(body, indent=1)[:20000], "response": json.dumps(out, indent=1),
                  "ms": int((time.time() - t0) * 1000)})
    return out


def best_sentence(text: str, like: str) -> str:
    """Fallback quote: the source sentence sharing the most words with the memory (verbatim by construction)."""
    want = set(tokens(like))
    sents = [x.strip() for x in re.split(r"(?<=[.!?])\s+|\n+", text) if 20 <= len(x.strip()) <= 300]
    return max(sents, key=lambda x: len(want & set(tokens(x))), default="")


def decide_and_write(goal: str, m: Moment, hits: list[tuple[float, Note]], sent: list[str]) -> Verdict:
    """Jev decides (one choice question, ~0.5s); only when it picks a memory does an LLM write the note."""
    if not hits:
        return Verdict(False)
    t0 = time.time()
    doing = clip_diff(m.diff, 6000) if m.trigger == "edit" else m.text[:3000]
    state = (f"USER REQUEST TO THE AGENT:\n{goal[:2000]}\n\nWHAT JUST HAPPENED ({m.trigger}):\n{doing}\n\n"
             f"AGENT'S RECENT STEPS:\n" + "\n".join(m.recent[-8:]) +
             (f"\n\nCODE CHANGE SO FAR:\n{clip_diff(m.diff, 4000)}" if m.trigger != "edit" and m.diff.strip() else "") +
             "\n\nPROJECT MEMORY:\n" + "\n".join(f"[{i + 1}] {n.text[:800]}" for i, (_, n) in enumerate(hits)) +
             ("\n\nALREADY TOLD THE AGENT:\n" + "\n".join(sent) if sent else ""))
    crit = {"none": "none: no conflict with any memory"}
    crit.update({f"m{i + 1}": f"memory [{i + 1}]: {n.text[:300]}" for i, (_, n) in enumerate(hits)})
    task = m.trigger == "task"
    ans = jev(state, {"conflict": {"type": "choice", "instructions": HEADSUP if task else DECIDE,
                                   "criteria": crit}})["answers"]["conflict"]
    probs = {k: float(v) for k, v in (ans.get("probabilities") or {}).items()}
    pick = max(probs, key=probs.get) if probs else "none"
    v = Verdict(False, ms=int((time.time() - t0) * 1000))
    v.decision = {"choice": pick, "p": round(probs.get(pick, 0), 2), "none": round(probs.get("none", 0), 2)}
    # a conflict can be split over two memories saying the same thing: speak when "none" is unlikely
    if probs.get("none", 1.0) >= SPEAK_AT:
        if probs.get("none", 1.0) < ESCALATE_BELOW and not task:  # Jev is unsure: the stronger judge decides
            j = judge(goal, m, hits, sent)
            j.decision = {**v.decision, "escalated": True}
            j.ms += v.ms
            return j
        return v
    pick = max((k for k in probs if k != "none"), key=probs.get)
    note = hits[int(pick[1:]) - 1][1]
    sources = note.sources or ((note.where, note.text),)
    out, ti, to = gemini((HEADSUP_WRITE if task else WRITE).format(memory=note.text, trigger=m.trigger, doing=doing,
                                      source="\n\n".join(f"[{w}]\n{t[:8000]}" for w, t in sources)), model=WRITER)
    quote = str(out.get("quote") or "").strip()
    found = [w for w, t in sources if quote and squash(quote) in squash(t)]
    if not found:  # the writer paraphrased: quote the closest real sentence instead
        where, text = sources[0]
        quote, found = best_sentence(text, note.text), [where]
    v.speak = bool(quote and out.get("message"))
    v.evidence, v.message, v.quote, v.where = note.id, str(out.get("message") or "").strip(), quote, found[0]
    v.tokens_in, v.tokens_out, v.ms = ti, to, int((time.time() - t0) * 1000)
    return v


def squash(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def render(v: Verdict) -> str:
    """The text the agent sees: the claim, then the exact words and where to read them."""
    return f'{v.message} Source: {v.where} — "{v.quote}"' if v.quote else f"{v.message} Source: {v.where}"


# ---------------------------------------------------------------- the watcher


@dataclass
class Watcher:
    goal: str
    search: "Callable[[Moment], list[tuple[float, Note]]]"  # moment -> ranked notes
    sent: list[str] = field(default_factory=list)
    recall_ms: int = 0
    evidence_sent: set[str] = field(default_factory=set)
    judged: set[str] = field(default_factory=set)

    def review(self, m: "Moment | str") -> tuple[Verdict, list[tuple[float, Note]]]:
        if isinstance(m, str):  # a bare diff (replay, diff-polling mode)
            m = Moment("edit", m, diff=m)
        t0 = time.time()
        hits = self.search(m) if (m.diff if m.trigger == "edit" else m.text).strip() else []
        self.recall_ms = int((time.time() - t0) * 1000)
        # the same memories against the same situation were already judged: nothing new to say
        key = hashlib.sha1(json.dumps([m.trigger, m.text if m.trigger != "edit" else "", m.diff,
                                       sorted(n.id for _, n in hits)]).encode()).hexdigest()
        if len(self.sent) >= MAX_MESSAGES or key in self.judged:
            return Verdict(False), hits
        self.judged.add(key)
        v = (decide_and_write if os.environ.get("JEV_KEY") and os.environ.get("WATCHER_DECIDER", "jev") == "jev"
             else judge)(self.goal, m, hits, self.sent)
        # Never repeat a source: once said, the agent has it.
        if v.speak and v.evidence and v.evidence in self.evidence_sent:
            v.speak = False
        if v.speak:
            self.sent.append(v.message)
            # a heads-up does not use up the memory: if the agent then goes against it, say so
            if v.evidence and m.trigger != "task":
                self.evidence_sent.add(v.evidence)
        return v, hits


def source_diff(repo: Path) -> str:
    """The agent's change to source: tracked edits plus new files (git diff alone skips those)."""
    spec = ["--", ".", ":(exclude)tests/**", ":(exclude)*.pyc"]
    git = lambda *a: subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, errors="replace").stdout
    out = git("diff", "HEAD", *spec)
    for f in git("ls-files", "--others", "--exclude-standard", *spec).split():
        out += git("diff", "--no-index", "/dev/null", f)
    return out


def bm25(notes: list[Note]):
    idx = Index(notes)
    return lambda m: idx.search(query_from_diff(m.diff) if m.trigger == "edit" else tokens(m.text))


def build_watcher(task: dict, repo: Path) -> Watcher:
    """Memory backend: Hindsight recall when WATCHER_BANK names a bank (ingested by the harness),
    else the local BM25 over this project's chats and commits."""
    bank = os.environ.get("WATCHER_BANK")
    if bank and os.environ.get("WATCHER_DECIDER") == "reflect":
        import reflectwatch
        return reflectwatch.ReflectWatcher(bank, task.get("bug_report", ""))
    if bank:
        import hsmemory
        return Watcher(task.get("bug_report", ""), lambda m: hsmemory.search(bank, m))
    return Watcher(task.get("bug_report", ""), bm25(load_chats(task) + load_commits(repo)))


def log_event(log: Path, **kw) -> None:
    with log.open("a") as f:
        f.write(json.dumps({"ts": time.time(), **kw}) + "\n")


def watch(repo: Path, task: dict, log: Path, poll: float = 2.0, settle: float = 4.0) -> None:
    """Poll the working tree. When the source diff changes and then holds still for `settle`
    seconds (the agent finished an edit burst), review it once. Runs until killed."""
    w = build_watcher(task, repo)
    busy, inbox = repo / BUSY, repo / INBOX
    last_seen, last_judged, changed_at = "", "", 0.0
    cursor, delivered = repo / ".git/watcher-delivered", "0"
    log_event(log, event="start", bank=os.environ.get("WATCHER_BANK"), model=MODEL)
    while True:
        now_delivered = cursor.read_text().strip() if cursor.exists() else "0"
        if now_delivered != delivered:  # the agent (hook) or the harness (end of turn) took messages
            log_event(log, event="delivered", count=int(now_delivered or 0) - int(delivered or 0))
            delivered = now_delivered
        diff = source_diff(repo)
        h = hashlib.sha1(diff.encode()).hexdigest()
        now = time.time()
        if h != last_seen:
            last_seen, changed_at = h, now
            if h != last_judged:
                busy.touch()  # tell the agent's stop hook a review is pending
        elif not diff.strip():
            last_judged = h  # nothing changed yet: nothing to review
            busy.unlink(missing_ok=True)
        elif h != last_judged and now - changed_at >= settle:
            last_judged = h
            try:
                CALLS.clear()
                v, hits = w.review(diff)
                for c in CALLS:
                    log_event(log, event="call", **c)
                log_event(log, event="review", diff_bytes=len(diff), recall_ms=w.recall_ms,
                          hits=[(round(s, 2), n.id) for s, n in hits], diff=diff[:6000],
                          memories=[n.text[:400] for _, n in hits],
                          **asdict(v))
                if v.speak:
                    with inbox.open("a") as f:
                        f.write(json.dumps({"message": render(v), "evidence": v.evidence}) + "\n")
            except Exception as e:  # a watcher failure must never touch the agent
                log_event(log, event="error", error=repr(e)[:300])
            finally:
                if last_seen == last_judged:
                    busy.unlink(missing_ok=True)
        time.sleep(poll)


# ---------------------------------------------------------------- following the agent

EDIT_TOOLS = {"edit", "write", "multiedit", "notebookedit"}
LOOK_TOOLS = {"read", "grep", "glob", "ls", "todowrite", "webfetch", "websearch", "task"}
READ_ONLY_CMD = re.compile(
    r"^\s*(cd [^;&|]+(&&|;)\s*)?(cat|ls|find|grep|rg|head|tail|wc|tree|file|which|pwd|echo|sed -n|awk|"
    r"git (status|diff|log|show|blame)|python3? -m pytest|pytest|python3? -c|python3? -I -c|pip3? install)\b")
ENV_TROUBLE = re.compile(r"command not found|No module named (pytest|pip)")  # the sandbox, not the project
FAILED = re.compile(r"\b\d+ failed\b|Traceback \(most recent call last\)|\bFAILED\b|\bError:")
PRIORITY = {"task": 0, "plan": 1, "failure": 2, "command": 3, "edit": 3}


def claude_events(e: dict) -> list[dict]:
    """One Claude Code message (stream-json line or session-transcript line, same shape) as the
    watcher's steps: what it says, the tools it calls, what they return."""
    out = []
    msg = e.get("message")
    content = msg.get("content") if isinstance(msg, dict) else None  # error events carry a plain string
    if e.get("type") == "assistant" and isinstance(content, list):
        for blk in content:
            if blk.get("type") == "text" and blk.get("text"):
                out.append({"src": "agent", "kind": "say", "text": blk["text"][:3000]})
            elif blk.get("type") == "tool_use":
                out.append({"src": "agent", "kind": "tool", "id": blk.get("id"), "tool": blk.get("name"),
                            "input": json.dumps(blk.get("input") or {})[:4000]})
    elif e.get("type") == "user":
        if isinstance(content, str):  # a prompt typed into the session
            out.append({"src": "user", "kind": "round", "role": "initial", "text": content[:4000]})
        for blk in content if isinstance(content, list) else []:
            if isinstance(blk, dict) and blk.get("type") == "tool_result":
                c = blk.get("content")
                text = c if isinstance(c, str) else "\n".join(x.get("text", "") for x in c or [] if isinstance(x, dict))
                out.append({"src": "agent", "kind": "result", "id": blk.get("tool_use_id"),
                            "error": bool(blk.get("is_error")), "text": text[:3000]})
    return out


def read_step(e: dict, tools: dict) -> tuple[str, str | None, str]:
    """(one line for the recent-steps list, trigger or None, the trigger's text)."""
    kind = e.get("kind")
    if kind == "round":
        role = e.get("role", "")
        if role.startswith("watcher"):
            return "the agent was given the watcher's note", None, ""
        if role.startswith("intervention"):
            return f"the user reported: {e['text'][:200]}", "failure", e["text"]
        return f"the user asked: {e['text'][:200]}", "task", e["text"]
    if kind == "say":
        line = f"agent said: {e['text'][:300]}"
        return line, ("plan" if len(e["text"]) >= 60 else None), e["text"]
    if kind == "tool":
        tool = str(e.get("tool") or "").lower()
        try:
            inp = json.loads(e.get("input") or "{}")
        except ValueError:
            inp = {}
        tools[e.get("id")] = (tool, inp)
        what = inp.get("command") or inp.get("file_path") or inp.get("pattern") or inp.get("path") or ""
        if tool == "bash" and not READ_ONLY_CMD.match(what):
            return f"agent runs: {what[:300]}", "command", what
        return f"agent {tool}: {str(what)[:200]}", None, ""
    if kind == "result":
        tool, inp = tools.get(e.get("id"), ("", {}))
        if "Note from this project's memory" in e.get("text", ""):  # our own note, held-back step
            return "the agent was given the watcher's note", None, ""
        if tool in EDIT_TOOLS:
            return f"agent edited {inp.get('file_path', '')}", "edit", ""
        if (e.get("error") or FAILED.search(e.get("text", ""))) and not ENV_TROUBLE.search(e.get("text", "")):
            return f"it failed: {e.get('text', '')[-300:]}", "failure", e.get("text", "")[-2000:]
    return "", None, ""


def follow(feed: Path, repo: Path, task: dict, log: Path, quiet: float = 1.0, max_wait: float = 3.0) -> None:
    """Follow the agent itself: every step it takes, as it takes it (the harness's events.jsonl, or a
    Claude Code session transcript). A task, a plan, a command that changes things, an edit or a
    failure is a moment to review; reads and searches are only context. When steps arrive while a
    review runs, the next review takes the latest moment with everything that happened before it."""
    w = build_watcher(task, repo)
    busy, inbox = repo / BUSY, repo / INBOX
    cursor, delivered = repo / ".git/watcher-delivered", "0"
    pos, buf, tools = 0, "", {}
    recent: list[str] = []
    pending: tuple[str, str] | None = None
    first_at = last_at = 0.0
    log_event(log, event="start", bank=os.environ.get("WATCHER_BANK"), model=MODEL, mode="follow")
    while True:
        now_delivered = cursor.read_text().strip() if cursor.exists() else "0"
        if now_delivered != delivered:
            log_event(log, event="delivered", count=int(now_delivered or 0) - int(delivered or 0))
            delivered = now_delivered
        if feed.exists():
            with feed.open() as f:
                f.seek(pos)
                buf += f.read()
                pos = f.tell()
        *lines, buf = buf.split("\n")  # keep a half-written last line for next time
        now = time.time()
        for line in lines:
            try:
                e = json.loads(line)
            except ValueError:
                continue
            for step in ([e] if "kind" in e else claude_events(e)):
                said, trigger, text = read_step(step, tools)
                if said:
                    recent = (recent + [said])[-12:]
                if trigger and (pending is None or PRIORITY[trigger] >= PRIORITY[pending[0]]):
                    if pending is None:
                        first_at = now
                    pending = (trigger, text)
                    busy.touch()  # tell the harness's end-of-turn flush a review is coming
                if said:
                    last_at = now
        if pending and (now - last_at >= quiet or now - first_at >= max_wait):
            m = Moment(pending[0], pending[1], recent[:], source_diff(repo))
            pending = None
            try:
                CALLS.clear()
                v, hits = w.review(m)
                for c in CALLS:
                    log_event(log, event="call", **c)
                log_event(log, event="review", trigger=m.trigger, trigger_text=m.text[:1500],
                          diff_bytes=len(m.diff), recall_ms=w.recall_ms, judged=bool(v.ms),
                          hits=[(round(s, 2), n.id) for s, n in hits], diff=m.diff[:6000],
                          memories=[n.text[:400] for _, n in hits], **asdict(v))
                if v.speak:
                    with inbox.open("a") as f:
                        f.write(json.dumps({"message": render(v), "evidence": v.evidence}) + "\n")
            except Exception as ex:  # a watcher failure must never touch the agent
                log_event(log, event="error", error=repr(ex)[:300])
            finally:
                if pending is None:
                    busy.unlink(missing_ok=True)
        time.sleep(0.3)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("watch")
    a.add_argument("--repo", required=True)
    a.add_argument("--task", required=True)
    a.add_argument("--log", required=True)
    f = sub.add_parser("follow")
    f.add_argument("--feed", required=True, help="the agent's step feed: harness events.jsonl or a Claude Code transcript")
    f.add_argument("--repo", required=True)
    f.add_argument("--task", required=True)
    f.add_argument("--log", required=True)
    b = sub.add_parser("replay")
    b.add_argument("--task", required=True)
    b.add_argument("--patch-file", required=True)
    b.add_argument("--repo", help="repo for commit memory (default: none)")
    args = ap.parse_args()
    task = json.loads(Path(args.task).read_text())
    if args.cmd == "watch":
        watch(Path(args.repo), task, Path(args.log))
    elif args.cmd == "follow":
        follow(Path(args.feed), Path(args.repo), task, Path(args.log))
    else:
        w = build_watcher(task, Path(args.repo)) if args.repo else Watcher(task.get("bug_report", ""), bm25(load_chats(task)))
        v, hits = w.review(Path(args.patch_file).read_text())
        print(json.dumps({"hits": [(round(s, 2), n.id) for s, n in hits], **asdict(v)}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
