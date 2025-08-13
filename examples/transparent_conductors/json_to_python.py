# json2py.py
import json, re, sys, pathlib

def extract_code(d: dict) -> str | None:
    if isinstance(d.get("code"), str) and d["code"].strip():
        return d["code"]

    # fallback: first ```python fenced block in first response
    resp = (d.get("responses") or [])
    if resp and isinstance(resp[0], str):
        m = re.search(r"```python\s*(.*?)\s*```", resp[0], flags=re.S)
        if m:
            return m.group(1)

    # deeper fallback: search any prompt strings for a fenced block
    prompts = d.get("prompts") or {}
    for v in prompts.values():
        if isinstance(v, dict):
            for s in v.values():
                if isinstance(s, str):
                    m = re.search(r"```python\s*(.*?)\s*```", s, flags=re.S)
                    if m:
                        return m.group(1)
    return None

def main():
    if len(sys.argv) < 3:
        print("Usage: python json_to_python.py <openevolve_result.json> <out.py>")
        print("Usage: python json_to_python.py b0517537-d969-4f75-a698-ef4ab294c2de.json model2.py")

        raise SystemExit(1)
    in_path = pathlib.Path(sys.argv[1])
    out_path = pathlib.Path(sys.argv[2])

    data = json.loads(in_path.read_text(encoding="utf-8"))
    code = extract_code(data)
    if not code:
        raise SystemExit("No Python code found in JSON.")

    out_path.write_text(code, encoding="utf-8")
    print(f"Wrote {out_path} ({len(code)} bytes)")

if __name__ == "__main__":
    main()
