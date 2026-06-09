"""Apply C4 (time-index + half-duplex listening/transmission phases) to ch04 §4.1.

Plain-string replacements only; no regex, no editor.
"""
import io

PATH = 'chapters/ch04_methods.tex'

with io.open(PATH, 'r', encoding='utf-8') as f:
    text = f.read()

original = text

# ---- Replacement 1: Section 4.1 opener ---------------------------------
old1 = (
    "\\section{System Model}\\label{sec:system-model-1}\n"
    "\n"
    "The system under study is a two-hop relay network with a single relay node:\n"
)
new1 = (
    "\\section{System Model}\\label{sec:system-model-1}\n"
    "\n"
    "The system under study is the two-hop relay network introduced in "
    "Section~\\ref{sec:two-hop-relay-model}: a \\textbf{half-duplex}, two-hop "
    "link with a single relay and \\textbf{no direct source--destination path}. "
    "\\emph{Half-duplex} means that, at every channel use, the relay is either "
    "receiving from the source or transmitting to the destination, but not both. "
    "Without loss of generality, the relay operation therefore splits in time "
    "into a \\textbf{listening phase} (Hop~1: the relay collects $y_R[\\cdot]$) "
    "and a \\textbf{transmission phase} (Hop~2: the relay emits $x_R[\\cdot]$). "
    "At time index $i$, the listening phase produces "
    "$y_R[i] = h_1\\!\\left(x[i]\\right) + n_1[i]$ and the transmission phase "
    "produces $\\mathbf{y}[i] = \\mathbf{H}\\,\\mathbf{x}_R[i] + \\mathbf{n}[i]$ "
    "at the destination, where $\\mathbf{H}\\in\\mathbb{C}^{2\\times 2}$ is the "
    "i.i.d.\\ Rayleigh fast-fading channel of Hop~2 (a fresh realization per "
    "channel use). The relay's neural network output is computed during the "
    "transmission phase from a window of listening-phase observations:\n"
    "\\[\n"
    "x_R[i] \\;=\\; f_\\theta\\!\\left(y_R[i-w:i+w]\\right),\n"
    "\\]\n"
    "where $w$ is the half-window size of the relay (memoryless for the "
    "classical AF/DF baselines, $w>0$ for the AI relays). For the explicit "
    "diagram of the end-to-end signal flow see below.\n"
)
assert old1 in text, "REPL1 anchor not found"
text = text.replace(old1, new1, 1)

# ---- Replacement 2: \"Hop Model\" paragraph ---------------------------------
old2 = (
    "\\textbf{Hop Model.} Each hop applies a channel function followed by optional equalization:\n"
    "\n"
    "The received signal follows the channel models defined in Chapter~\\ref{sec:introduction}.\n"
    "\n"
    "where \\(h(\\cdot)\\) depends on the specific channel type (AWGN, fading, or MIMO).\n"
)
new2 = (
    "\\textbf{Hop Model.} Each hop applies a channel function followed by optional "
    "equalization. With the time index made explicit, the listening phase (Hop~1) "
    "gives $y_R[i] = h_1\\!\\left(x[i]\\right) + n_1[i]$ for SISO channels (AWGN, "
    "Rayleigh, or Rician), and the transmission phase (Hop~2) gives "
    "$\\mathbf{y}[i] = \\mathbf{H}\\,\\mathbf{x}_R[i] + \\mathbf{n}[i]$ for the "
    "$2\\times 2$ MIMO Rayleigh fast-fading channel; the destination then applies a "
    "linear or non-linear equalizer (ZF, MMSE, or SIC) to recover the symbol "
    "estimates. The detailed channel models are those defined in "
    "Chapter~\\ref{sec:introduction}, where $h_1(\\cdot)$ depends on the SISO Hop-1 "
    "channel type (AWGN, Rayleigh, or Rician) and $\\mathbf{H}$ is the per-use "
    "Hop-2 MIMO matrix.\n"
)
assert old2 in text, "REPL2 anchor not found"
text = text.replace(old2, new2, 1)

# ---- Replacement 3: relay neural-network paragraph ------------------------
old3 = (
    "\\textbf{The relay's neural network} operates on Hop 1 and solves a "
    "\\textbf{denoising} problem. Each antenna at the relay receives:\n"
    "\n"
    "The signal received at the relay is defined as in Chapter~\\ref{sec:introduction}.\n"
    "\n"
    "The neural network processes a sliding window of received samples and "
    "outputs a cleaner estimate \\(\\hat{x}_R = f_\\theta(y_{R,i-w:i+w})\\). "
    "This is purely a noise-removal task --- there is no inter-stream "
    "interference at this stage.\n"
)
new3 = (
    "\\textbf{The relay's neural network} operates on Hop~1 (during the "
    "listening phase) and solves a \\textbf{denoising} problem. The signal "
    "received at the relay is the time-indexed Hop~1 model from "
    "Chapter~\\ref{sec:introduction}, "
    "$y_R[i] = h_1\\!\\left(x[i]\\right) + n_1[i]$. The neural network processes "
    "a sliding window of received samples and outputs a cleaner estimate "
    "$\\hat{x}_R[i] = f_\\theta\\!\\left(y_R[i-w:i+w]\\right)$, which is then "
    "power-normalised and re-emitted during the transmission phase as "
    "$x_R[i]$. This is purely a noise-removal task --- there is no inter-stream "
    "interference at this stage; inter-stream interference only arises on Hop~2.\n"
)
assert old3 in text, "REPL3 anchor not found"
text = text.replace(old3, new3, 1)

# ---- Write back ----------------------------------------------------------
if text == original:
    print("ERROR: no changes applied")
    raise SystemExit(1)

with io.open(PATH, 'w', encoding='utf-8') as f:
    f.write(text)

print("OK: 3 replacements applied to", PATH)
print("Length before:", len(original), "after:", len(text), "delta:", len(text) - len(original))