"""Greedy reference outputs of a live server, to prove a deployment changed nothing.

    golden.py record  URL KEYFILE OUT.json [--model deepseek]
    golden.py compare URL KEYFILE REF.json [--model deepseek]

Requests run one at a time (oMLX is not batch-invariant, so concurrent traffic
can still perturb a request; run compare twice before calling a diff real).
"""
import json
import sys
import urllib.request

CASES = [
    ("chat", "Explain in two sentences why the sky is blue."),
    ("chat", "List five prime numbers greater than 100, comma separated."),
    ("chat", "Translate to German: The train leaves at seven and arrives at noon."),
    ("chat", "Write a haiku about a lighthouse."),
    ("chat", "What is 17 * 23? Answer with the number only."),
    ("chat", "Summarise the plot of Hamlet in one paragraph."),
    ("chat", "Give a Python one-liner that reverses a string s."),
    ("chat", "Name three uses of baking soda at home."),
    ("chat", "Rewrite more formally: gonna be late, traffic is nuts."),
    ("chat", "Was ist die Hauptstadt von Australien? Antworte kurz."),
    ("think", "A bat and a ball cost 1.10 in total; the bat costs 1.00 more than the ball. How much is the ball?"),
    ("think", "Is 221 a prime number? Explain briefly."),
    ("think", "Plan a three-step morning routine for someone who works night shifts."),
    ("think", "Which is larger, 2^30 or 10^9? Show the reasoning."),
    ("think", "Write a short limerick about a cat who learns to code."),
    ("think", "If all bloops are razzies and some razzies are lazzies, are some bloops lazzies?"),
    ("tool", "What's the weather in Zurich right now?"),
    ("long", None),
]
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]


def body(kind, text, model):
    if kind == "long":
        text = "Below is a log. Report the last line number.\n" + "\n".join(
            f"line {i}: sensor {i % 13} reads {(i * 37) % 101} units" for i in range(2400))
    b = {"model": model, "messages": [{"role": "user", "content": text}], "temperature": 0, "max_tokens": 256,
         "logprobs": True, "top_logprobs": 20, "stream": False,
         "chat_template_kwargs": {"enable_thinking": kind == "think"}}
    if kind == "think":
        b["reasoning_effort"] = 50
    if kind == "tool":
        b["tools"] = TOOLS
    return b


def run(url, key, model):
    out = []
    for kind, text in CASES:
        req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body(kind, text, model)).encode(),
                                     {"content-type": "application/json", "authorization": f"Bearer {key}"})
        r = json.loads(urllib.request.urlopen(req, timeout=1800).read())
        c = r["choices"][0]
        for call in c["message"].get("tool_calls") or []:
            call["id"] = "<random>"  # oMLX draws a fresh call id per response
        out.append({"kind": kind, "message": c["message"], "finish": c.get("finish_reason"), "logprobs": c.get("logprobs")})
        print(kind, c.get("finish_reason"), flush=True)
    return out


def main():
    mode, url, keyfile, path = sys.argv[1:5]
    model = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else "deepseek"
    key = open(keyfile).read().strip()
    res = run(url, key, model)
    if mode == "record":
        json.dump(res, open(path, "w"))
        print(f"recorded {len(res)} cases to {path}")
        return
    ref = json.load(open(path))
    for case in ref:
        for call in case["message"].get("tool_calls") or []:
            call["id"] = "<random>"
    diffs = 0
    for i, (a, b) in enumerate(zip(res, ref)):
        same = json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
        diffs += not same
        print(f"case {i:2d} {a['kind']:5s} {'IDENTICAL' if same else 'DIFFERENT'}")
    print("RESULT", "all identical" if not diffs else f"{diffs} of {len(ref)} differ")


if __name__ == "__main__":
    main()
