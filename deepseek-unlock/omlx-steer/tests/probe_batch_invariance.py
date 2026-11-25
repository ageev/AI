"""Is oMLX itself batch-invariant? Compare a request alone vs inside N concurrent ones."""
import json, sys, threading, urllib.request
URL, MODELS = sys.argv[1], sys.argv[2].split(",")
P = "Write three short facts about rivers."
def chat(model):
    body = {"model": model, "messages": [{"role": "user", "content": P}], "temperature": 0,
            "max_tokens": 24, "logprobs": True, "top_logprobs": 5}
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=600).read())
    c = r["choices"][0]
    return json.dumps(c["message"], sort_keys=True), [round(t["logprob"], 6) for t in (c.get("logprobs") or {}).get("content") or []]
alone = {m: chat(m) for m in set(MODELS)}
for trial in range(2):
    out = {}
    th = [threading.Thread(target=lambda i=i, m=m: out.__setitem__(i, chat(m))) for i, m in enumerate(MODELS)]
    [t.start() for t in th]; [t.join() for t in th]
    for i, m in enumerate(MODELS):
        same_tok = out[i][0] == alone[m][0]
        same_lp = out[i][1] == alone[m][1]
        d = max((abs(a - b) for a, b in zip(out[i][1], alone[m][1])), default=0)
        print(f"trial {trial} row {i} {m}: tokens {'same' if same_tok else 'DIFF'}, logprobs {'same' if same_lp else f'DIFF max {d:.2e}'}")
