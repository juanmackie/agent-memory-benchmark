// opencode delivery for the memory watcher: append pending messages to the next tool result.
// Same inbox/cursor contract as deliver.py (that file is the reference).
import { existsSync, readFileSync, writeFileSync } from "node:fs";

const LEAD = "🧠 Note from this project's memory (past sessions and commits):";

function takeNew(dir) {
  const inbox = `${dir}/.git/watcher-inbox.jsonl`, cursor = `${dir}/.git/watcher-delivered`;
  if (!existsSync(inbox)) return [];
  const lines = readFileSync(inbox, "utf8").split("\n").filter((l) => l.trim());
  const done = existsSync(cursor) ? Number(readFileSync(cursor, "utf8") || 0) : 0;
  writeFileSync(cursor, String(lines.length));
  return lines.slice(done).map((l) => JSON.parse(l).message);
}

export const WatcherDelivery = async () => ({
  "tool.execute.after": async (_input, output) => {
    const msgs = takeNew("/work");
    if (msgs.length) output.output = `${output.output ?? ""}\n\n${msgs.map((m) => `${LEAD} ${m}`).join("\n")}`;
  },
});
