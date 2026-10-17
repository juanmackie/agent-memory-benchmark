"""Hand watcher messages to the agent exactly once. Shared by the Claude hook (inside the agent
container, repo at /work) and the harness's end-of-turn flush (on the host).

The watcher appends to .git/watcher-inbox.jsonl; .git/watcher-delivered holds how many lines the
agent has already been given. Both live in .git so they never enter the graded diff."""

import json
import sys
from pathlib import Path

LEAD = "🧠 Note from this project's memory (past sessions and commits):"


def take_new(repo: Path) -> list[str]:
    inbox, cursor = repo / ".git/watcher-inbox.jsonl", repo / ".git/watcher-delivered"
    if not inbox.exists():
        return []
    lines = [l for l in inbox.read_text().splitlines() if l.strip()]
    done = int(cursor.read_text() or 0) if cursor.exists() else 0
    cursor.write_text(str(len(lines)))
    return [json.loads(l)["message"] for l in lines[done:]]


def render(messages: list[str]) -> str:
    return "\n".join(f"{LEAD} {m}" for m in messages)


if __name__ == "__main__":
    # Claude Code hook. It only hands over notes already written: the agent never waits on the watcher.
    #   PostToolUse / UserPromptSubmit: the note rides next to the tool result or the new message.
    #   (No PreToolUse deny: claude shows it as "hook error" and read a real decision as an injection.)
    #   Stop: a note that arrived late keeps the agent going instead of being lost.
    try:
        event = json.load(sys.stdin)
    except ValueError:
        event = {}
    name = event.get("hook_event_name", "PostToolUse")
    msgs = take_new(Path("/work"))
    if not msgs:
        sys.exit(0)
    if name == "Stop":
        out = {"decision": "block", "reason": render(msgs)}
    else:
        out = {"hookSpecificOutput": {"hookEventName": name, "additionalContext": render(msgs)}}
    print(json.dumps(out))
    sys.exit(0)
