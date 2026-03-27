"""Replace both LR mermaid diagrams with a TB subgraph version that gives
each node full width so text is not cramped."""
import re, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

NEW_DIAGRAM = """\
```mermaid
flowchart TB
    SRC([\"**Source**\\ntx bits\"])

    subgraph HOP1 [\" Hop 1 \"]
        direction TB
        MOD[\"**BPSK Modulator**\"]
        CH1[\"**Channel**\\nAWGN / Rayleigh / Rician\"]
        MOD --> CH1
    end

    subgraph RELAY_BOX [\" Relay Node \"]
        direction TB
        REL[\"**Neural Net Relay**\\ndenoise Hop 1 noise\"]
    end

    subgraph HOP2 [\" Hop 2 — 2x2 MIMO \"]
        direction TB
        CH2[\"**MIMO Channel**\\n4 Rayleigh links\\ny = H·x_R + n\\nH = h11 h12 / h21 h22\"]
        EQ[\"**Equalizer**\\nZF / MMSE / SIC\\ncancel inter-stream interference\"]
        CH2 --> EQ
    end

    DST([\"**Destination**\\nrecovered bits\"])

    SRC        --> HOP1
    HOP1       -->|\"noisy y_R\"| RELAY_BOX
    RELAY_BOX  -->|\"clean x_R\"| HOP2
    HOP2       --> DST

    style SRC        fill:#4A90D9,color:#fff,stroke:#2c5f8a
    style DST        fill:#4A90D9,color:#fff,stroke:#2c5f8a
    style MOD        fill:#7B68EE,color:#fff,stroke:#4a3fa0
    style CH1        fill:#E8A838,color:#fff,stroke:#b07820
    style REL        fill:#2ECC71,color:#fff,stroke:#1a8a4a
    style CH2        fill:#E8A838,color:#fff,stroke:#b07820
    style EQ         fill:#9B59B6,color:#fff,stroke:#6c3483
    style HOP1       fill:#fff8ee,stroke:#E8A838,stroke-width:2px
    style RELAY_BOX  fill:#f0fff4,stroke:#2ECC71,stroke-width:2px
    style HOP2       fill:#fff8ee,stroke:#E8A838,stroke-width:2px
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
