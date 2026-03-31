import urllib.request

urls = [
    "https://www.overleaf.com/latex/templates/thesis-template-tau-tel-aviv-university/thfzpfyhhxpy.zip",
    "https://www.overleaf.com/project/thfzpfyhhxpy/download/zip",
    "https://www.overleaf.com/read/thfzpfyhhxpy",
]

for url in urls:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            data = r.read()
            ct = r.headers.get("content-type", "")
            print(f"OK {url}")
            print(f"  {len(data)} bytes, content-type={ct}")
            if b"zip" in ct.encode() or data[:2] == b"PK":
                with open("tau_template.zip", "wb") as f:
                    f.write(data)
                print("  Saved as tau_template.zip")
            else:
                print("  Preview:", data[:300])
    except Exception as e:
        print(f"FAIL {url}: {e}")