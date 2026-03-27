"""Replace both mermaid diagrams with a compact two-row LR layout:
  Row 1 (TX path): Source -> BPSK -> Hop1 Channel -> Relay
  Row 2 (RX path): Relay -> Hop2 MIMO Channel -> Equalizer -> Destination
Same node size as original single-row LR, but split across two lines."""
import re, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

NEW_DIAGRAM = """\
```mermaid
flowchart LR
    %% ── TX path (top row) ──────────────────────────────────────
    SRC(["Source\ntx bits"])
    MOD["BPSK\nMod"]
    CH1["Hop 1 Channel\nAWGN/Rayleigh/Rician"]
    REL["Relay\nNeural Net"]

    %% ── RX path (bottom row) ───────────────────────────────────
    REL2["Relay\nout x_R"]
    CH2["Hop 2 MIMO\ny=Hx_R+n"]
    EQ["Equalizer\nZF/MMSE/SIC"]
    DST(["Destination\nrx bits"])

    %% ── TX row connections ─────────────────────────────────────
    SRC --> MOD --> CH1 -->|"noisy y_R"| REL

    %% ── drop from TX row to RX row ────────────────────────────
    REL -->|"clean x_R"| REL2

    %% ── RX row connections ─────────────────────────────────────
    REL2 --> CH2 --> EQ --> DST

    style SRC  fill:#4A90D9,color:#fff,stroke:#2c5f8a
    style DST  fill:#4A90D9,color:#fff,stroke:#2c5f8a
    style MOD  fill:#7B68EE,color:#fff,stroke:#4a3fa0
    style CH1  fill:#E8A838,color:#fff,stroke:#b07820
    style REL  fill:#2ECC71,color:#fff,stroke:#1a8a4a
    style REL2 fill:#2ECC71,color:#fff,stroke:#1a8a4a
    style CH2  fill:#E8A838,color:#fff,stroke:#b07820
    style EQ   fill:#9B59B6,color:#fff,stroke:#6c3483
```"""

with open("thesis.md", encoding="utf-8") as f:
    content = f.read()

blocks = list(re.finditer(r"```mermaid\n.*?```", content, re.DOTALL))
print(f"Found {len(blocks)} mermaid block(s)")

# Replace from last to first to keep positions valid
for b in reversed(blocks):
    line_no = content[: b.start()].count("\n") + 1
    content = content[: b.start()] + NEW_DIAGRAM + content[b.end() :]
    print(f"  Replaced block originally at line {line_no}")

with open("thesis.md", "w", encoding="utf-8") as f:
    f.write(content)

print("Done — thesis.md updated")
