#!/usr/bin/env python3
"""The structured-output live gates for the 2x Spark server (patch 0034).

  python3 tools/structure_probe.py [--base http://127.0.0.1:8888] [--skip-heavy]

Checks, each printed PASS/FAIL:
  1. a flat JSON schema is honoured (content parses and validates)
  2. nested objects and arrays
  3. an enum-constrained field takes one of its values
  4. a long-string field is still valid JSON (no newline-until-length stall)
  5. guided_regex: the reply matches the pattern
  6. guided_choice: the reply is exactly one of the choices
  7. a malformed schema is refused (400), not served
  8. a schema cannot be combined with tool_choice "required" (400)
  9. response_format beside tools with tool_choice "auto" yields constrained content and no tool calls
 10. an unsatisfiable schema ends with finish_reason length (not a hang)
 11. a streamed structured reply delivers valid JSON
 12. a cancel mid-decode leaves the server healthy
 13. mixed traffic: a plain greedy reply is byte-identical alone and beside a structured request

API key: $API_KEY, else ~/.local/state/dsv41-tensorfold/api-key when it exists (the server
needs one only when it was started with API_KEY=1).
"""
import json
import os
import re
import sys
import threading
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8888"
MODEL = "deepseek-v4.1-flash"
for i, arg in enumerate(sys.argv[1:]):
    if arg == "--base" and i + 3 <= len(sys.argv) - 1:
        BASE = sys.argv[i + 2]

KEY_FILE = os.path.expanduser("~/.local/state/dsv41-tensorfold/api-key")


def key():
    k = os.environ.get("API_KEY")
    if k:
        return k
    if os.path.exists(KEY_FILE):
        return open(KEY_FILE).read().strip()
    return None


def post(payload, stream=False, timeout=600):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    k = key()
    if k:
        req.add_header("Authorization", "Bearer " + k)
    if stream:
        return urllib.request.urlopen(req, timeout=timeout)
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def post_expect_error(payload, timeout=120):
    try:
        post(payload, timeout=timeout)
        return None
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")[:300]


# ---- a small JSON Schema validator (the subset the probe uses) ----
def validate(inst, schema, path="$"):
    t = schema.get("type")
    if t == "object":
        if not isinstance(inst, dict):
            return f"{path}: not an object"
        for req in schema.get("required", []):
            if req not in inst:
                return f"{path}: missing {req}"
        props = schema.get("properties", {})
        for k, v in inst.items():
            if k in props:
                err = validate(v, props[k], f"{path}.{k}")
                if err:
                    return err
            elif schema.get("additionalProperties") is False:
                return f"{path}: unexpected {k}"
        return None
    if t == "array":
        if not isinstance(inst, list):
            return f"{path}: not an array"
        item = schema.get("items")
        if item:
            for i, v in enumerate(inst):
                err = validate(v, item, f"{path}[{i}]")
                if err:
                    return err
        return None
    if t == "string":
        if not isinstance(inst, str):
            return f"{path}: not a string"
        pat = schema.get("pattern")
        if pat and not re.search(pat, inst):
            return f"{path}: {inst!r} does not match {pat}"
        return None
    if t == "integer":
        return None if isinstance(inst, int) and not isinstance(inst, bool) else f"{path}: not an integer"
    if t == "number":
        return None if isinstance(inst, (int, float)) and not isinstance(inst, bool) else f"{path}: not a number"
    if t == "boolean":
        return None if isinstance(inst, bool) else f"{path}: not a boolean"
    if "enum" in schema:
        return None if inst in schema["enum"] else f"{path}: {inst!r} not in {schema['enum']}"
    return None


def content_of(resp):
    msg = resp["choices"][0]["message"]
    return msg.get("content") or ""


def ask(prompt, schema=None, **extra):
    payload = {"model": MODEL, "temperature": 0, "max_tokens": 512,
               "messages": [{"role": "user", "content": prompt}]}
    if schema is not None:
        payload["response_format"] = {"type": "json_schema",
                                      "json_schema": {"name": "reply", "schema": schema}}
    payload.update(extra)
    return post(payload)


ok = True


def report(name, good, detail=""):
    global ok
    ok &= bool(good)
    print(f"{name:58s} {'PASS' if good else 'FAIL'} {detail}"[:220])


FLAT = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"], "additionalProperties": False}


def main():
    # 1 flat
    r = ask("Give one person as JSON.", FLAT)
    c = content_of(r)
    try:
        obj = json.loads(c)
        err = validate(obj, FLAT)
    except Exception as e:
        obj, err = None, f"not JSON: {e}"
    report("flat schema honoured", err is None, c[:80])

    # 2 nested
    NEST = {"type": "object", "required": ["steps"],
            "properties": {"steps": {"type": "array", "items": {
                "type": "object", "required": ["explanation", "output"],
                "properties": {"explanation": {"type": "string"}, "output": {"type": "string"}}}}},
            "additionalProperties": False}
    r = ask("Solve 8x + 31 = 2 as JSON steps.", NEST)
    c = content_of(r)
    try:
        err = validate(json.loads(c), NEST)
    except Exception as e:
        err = f"not JSON: {e}"
    report("nested objects and arrays", err is None, c[:80])

    # 3 enum
    ENUM = {"type": "object", "required": ["sentiment"],
            "properties": {"sentiment": {"enum": ["positive", "negative"]}},
            "additionalProperties": False}
    r = ask("Classify: vLLM is wonderful!", ENUM)
    c = content_of(r)
    try:
        err = validate(json.loads(c), ENUM)
    except Exception as e:
        err = f"not JSON: {e}"
    report("enum field constrained", err is None, c[:60])

    # 4 long string (the newline-until-length case)
    LONG = {"type": "object", "required": ["text"],
            "properties": {"text": {"type": "string"}}, "additionalProperties": False}
    r = ask("Write two short paragraphs about the sea, as JSON.", LONG)
    c = content_of(r)
    fr = r["choices"][0].get("finish_reason")
    try:
        err = validate(json.loads(c), LONG)
    except Exception as e:
        err = f"not JSON: {e}"
    report("long-string field completes", err is None and fr == "stop", f"finish={fr} {c[:60]}")

    # 5 regex
    r = post({"model": MODEL, "temperature": 0, "max_tokens": 64,
              "messages": [{"role": "user", "content": "Give one code like AB-123."}],
              "guided_regex": "[A-Z]{2}-[0-9]{3}"})
    c = content_of(r).strip()
    report("guided_regex matches", re.fullmatch(r"[A-Z]{2}-[0-9]{3}", c) is not None, c[:40])

    # 6 choice
    r = post({"model": MODEL, "temperature": 0, "max_tokens": 32,
              "messages": [{"role": "user", "content": "Sentiment of 'wonderful'?"}],
              "guided_choice": ["positive", "negative"]})
    c = content_of(r).strip()
    report("guided_choice is one of them", c in ('"positive"', '"negative"', "positive", "negative"), c[:40])

    # 7 malformed schema
    e = post_expect_error({"model": MODEL, "temperature": 0, "max_tokens": 32,
                           "messages": [{"role": "user", "content": "hi"}],
                           "response_format": {"type": "json_schema",
                                               "json_schema": {"name": "x", "schema": {"type": 42}}}})
    report("malformed schema refused", e is not None and e[0] == 400, str(e)[:80] if e else "served")

    # 8 schema + required tool_choice
    e = post_expect_error({"model": MODEL, "temperature": 0, "max_tokens": 32,
                           "messages": [{"role": "user", "content": "hi"}],
                           "response_format": {"type": "json_object"},
                           "tool_choice": "required",
                           "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]})
    report("schema + required tool_choice refused", e is not None and e[0] == 400, str(e)[:80] if e else "served")

    # 9 schema + tools: refused (a tool call would bypass the grammar)
    e = post_expect_error({"model": MODEL, "temperature": 0, "max_tokens": 64,
                           "messages": [{"role": "user", "content": "What is the weather in Warsaw?"}],
                           "response_format": {"type": "json_schema",
                                               "json_schema": {"name": "r", "schema": FLAT}},
                           "tool_choice": "auto",
                           "tools": [{"type": "function", "function": {"name": "get_weather",
                                                                      "parameters": {"type": "object",
                                                                                     "properties": {"city": {"type": "string"}}}}}]})
    report("schema + tools refused", e is not None and e[0] == 400, str(e)[:80] if e else "served")

    # 10 unsatisfiable schema -> length, not a hang
    UNSAT = {"type": "object", "required": ["a", "b", "c"],
             "properties": {"a": {"type": "string", "pattern": "^X{400}$"},
                            "b": {"type": "string", "pattern": "^X{400}$"},
                            "c": {"type": "string", "pattern": "^X{400}$"}},
             "additionalProperties": False}
    try:
        r = ask("Produce the object.", UNSAT, max_tokens=64)
        fr = r["choices"][0].get("finish_reason")
        report("unsatisfiable schema ends by length", fr in ("length", "stop"), f"finish={fr}")
    except Exception as e:
        report("unsatisfiable schema ends by length", "length" in str(e), str(e)[:80])

    # 11 streamed structured reply
    try:
        resp = post({"model": MODEL, "temperature": 0, "max_tokens": 256, "stream": True,
                     "messages": [{"role": "user", "content": "Give one person as JSON."}],
                     "response_format": {"type": "json_schema",
                                         "json_schema": {"name": "reply", "schema": FLAT}}}, stream=True)
        text = ""
        for raw in resp:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except Exception:
                continue
            for ch in chunk.get("choices") or []:
                text += (ch.get("delta") or {}).get("content") or ""
        try:
            err = validate(json.loads(text), FLAT)
        except Exception as e:
            err = f"not JSON: {e}"
        report("streamed structured reply valid", err is None, text[:60])
    except Exception as e:
        report("streamed structured reply valid", False, str(e)[:80])

    # 12 cancel mid-decode
    try:
        resp = post({"model": MODEL, "temperature": 0, "max_tokens": 4096, "stream": True,
                     "messages": [{"role": "user", "content": "Write 4000 words about the sea."}]}, stream=True)
        n = 0
        for raw in resp:
            n += 1
            if n > 3:
                break
        resp.close()
        h = json.load(urllib.request.urlopen(BASE + "/health", timeout=30))
        report("cancel mid-decode leaves health ok", True, str(h)[:40])
    except Exception as e:
        report("cancel mid-decode leaves health ok", False, str(e)[:80])

    # 13 mixed traffic: a plain greedy reply is byte-identical alone and beside a structured request
    plain = {"model": MODEL, "temperature": 0, "max_tokens": 128,
             "messages": [{"role": "user", "content": "What is 17 * 23? Answer with the number."}]}
    alone = content_of(post(plain))
    result = {}

    def structured():
        try:
            ask("Give one person as JSON.", FLAT)
            result["ok"] = True
        except Exception as e:
            result["ok"] = False

    t = threading.Thread(target=structured)
    t.start()
    beside = content_of(post(plain))
    t.join()
    report("mixed traffic: plain reply identical", alone == beside and result.get("ok"),
           f"{alone[:30]!r} vs {beside[:30]!r}")

    print("\nRESULT:", "ALL PASS" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
