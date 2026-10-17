import json,sys,statistics as st
rows=[json.loads(l) for f in sys.argv[1:] for l in open(f)]
for arm in ("watcher","reflect"):
    rs=[r for r in rows if r["arm"]==arm]; ft=[r for r in rs if r["solved"] and r["corrections"]==0]
    c=[r["agent_cost"] for r in rs]; w=[r["agent_wall_s"] for r in rs]
    print(arm,len(rs),"ft",len(ft),"cost mean %.3f med %.3f"%(st.mean(c),st.median(c)),"wall mean %.1f med %.1f"%(st.mean(w),st.median(w)),
      "| ft-only cost %.3f wall %.1f"%(st.mean(r["agent_cost"] for r in ft),st.mean(r["agent_wall_s"] for r in ft)),
      "mem %.4f"%st.mean(r["memory_cost"] for r in rs))
    print("  misses:",[(r["task"],r["run"]) for r in rs if not(r["solved"] and r["corrections"]==0)])
# paired per task means
from collections import defaultdict
d=defaultdict(lambda: defaultdict(list))
for r in rows: d[r["task"]][r["arm"]].append(r)
dc=[];dw=[]
for t,a in d.items():
    if "watcher" in a and "reflect" in a:
        dc.append(st.mean(x["agent_cost"] for x in a["reflect"])-st.mean(x["agent_cost"] for x in a["watcher"]))
        dw.append(st.mean(x["agent_wall_s"] for x in a["reflect"])-st.mean(x["agent_wall_s"] for x in a["watcher"]))
print("paired task diff reflect-watcher: cost mean %+.4f med %+.4f cheaper in %d/%d; wall mean %+.1f med %+.1f faster in %d"%(st.mean(dc),st.median(dc),sum(x<0 for x in dc),len(dc),st.mean(dw),st.median(dw),sum(x<0 for x in dw)))
