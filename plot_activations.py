"""
Plot three relay activation functions in one figure:
  f(z) = tanh(z)
  g(z) = z          (linear / identity)
  h(z) = clip(z, -3/sqrt(10), +3/sqrt(10))   (hard-tanh / scaled clip)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── z range ──────────────────────────────────────────────────────────────────
z = np.linspace(-3, 3, 2000)

# ── functions ─────────────────────────────────────────────────────────────────
clip_val = 3 / np.sqrt(10)          # ≈ 0.9487

f = np.tanh(z)                      # tanh
g = z                               # linear (identity)
h = np.clip(z, -clip_val, clip_val) # hard clip

# ── figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(z, f, color="#2ECC71", linewidth=2.5, label=r"$f(z)=\tanh(z)$")
ax.plot(z, g, color="#4A90D9", linewidth=2.5, linestyle="--",
        label=r"$g(z)=z$  (linear)")
ax.plot(z, h, color="#E74C3C", linewidth=2.5, linestyle="-.",
        label=r"$h(z)=\mathrm{clip}(z,\,-3/\sqrt{10},\,+3/\sqrt{10})$")

# ── clip boundary markers ─────────────────────────────────────────────────────
for sign in (-1, +1):
    ax.axhline(sign * clip_val, color="#E74C3C", linewidth=0.8,
               linestyle=":", alpha=0.6)
    ax.axvline(sign * clip_val, color="#E74C3C", linewidth=0.8,
               linestyle=":", alpha=0.6)

ax.annotate(
    rf"$\pm\,3/\sqrt{{10}}\approx\pm{clip_val:.3f}$",
    xy=(clip_val, clip_val), xytext=(1.4, 0.55),
    arrowprops=dict(arrowstyle="->", color="#E74C3C", lw=1.2),
    fontsize=9, color="#E74C3C",
)

# ── axes & grid ───────────────────────────────────────────────────────────────
ax.axhline(0, color="black", linewidth=0.8)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlim(-3, 3)
ax.set_ylim(-3.2, 3.2)
ax.set_xlabel(r"$z$", fontsize=13)
ax.set_ylabel(r"Activation output", fontsize=12)
ax.set_title("Relay Activation Functions", fontsize=14, fontweight="bold")
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(fontsize=10, loc="upper left", framealpha=0.9)

# ── saturation region shading for tanh ───────────────────────────────────────
ax.fill_between(z, f, where=(z > 2),  alpha=0.08, color="#2ECC71",
                label="_nolegend_")
ax.fill_between(z, f, where=(z < -2), alpha=0.08, color="#2ECC71",
                label="_nolegend_")

# ── tick marks at clip boundaries ─────────────────────────────────────────────
ax.set_xticks([-3, -2, -1, -clip_val, 0, clip_val, 1, 2, 3])
ax.set_xticklabels(
    ["-3", "-2", "-1", r"$-\frac{3}{\sqrt{10}}$",
     "0",  r"$+\frac{3}{\sqrt{10}}$", "1", "2", "3"],
    fontsize=8,
)

plt.tight_layout()
plt.savefig("activation_functions.png", dpi=180, bbox_inches="tight")
plt.savefig("activation_functions.pdf", bbox_inches="tight")
print("Saved: activation_functions.png  /  activation_functions.pdf")
print(f"Clip boundaries: ±{clip_val:.6f}")
