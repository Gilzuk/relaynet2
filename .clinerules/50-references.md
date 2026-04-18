# References Rules

## Citation Syntax
- Use LaTeX `\cite{key}` syntax exclusively — never inline `[7]` or `{[}7{]}`.
- Do not invent references.
- When unsure about a source, ask before citing.

## BibTeX File
- File: `references.bib` (32 entries)
- Bibliography style: `IEEEtran` via `\bibliographystyle{IEEEtran}` in `ch08_references.tex`
- Citation key format: `authorYYYYkeyword` (e.g., `laneman2004cooperative`, `vaswani2017attention`, `gu2024mamba`)
- Preserve existing citation keys unless explicitly asked to refactor.

## Canonical Sources
Prefer canonical sources by topic:
| Topic | Preferred Source |
|-------|-----------------|
| Digital communications theory | Proakis & Salehi (`proakis2008digital`), Tse & Viswanath (`tse2005fundamentals`) |
| Fading channels / BER | Simon & Alouini (`simon2005digital`), Sklar (`sklar2001digital`) |
| MIMO / equalization | Telatar (`telatar1999capacity`), Foschini (`foschini1996layered`) |
| Relay networks | Laneman et al. (`laneman2004cooperative`), Cover & El Gamal (`cover1979capacity`) |
| Deep learning theory | Goodfellow et al. (`goodfellow2016deep`) |
| Channel estimation / detection | Ye et al. (`ye2018power`), Samuel et al. (`samuel2019learning`) |
| End-to-end learning | Dorner et al. (`dorner2018deep`) |
| Transformers | Vaswani et al. (`vaswani2017attention`) |
| Mamba S6 | Gu & Dao (`gu2024mamba`) |
| Mamba-2 / SSD | Dao & Gu (`dao2024transformers`) |
| VAE | Kingma & Welling (`kingma2014auto`) |
| GAN / WGAN-GP | Goodfellow et al. (`goodfellow2014generative`), Gulrajani et al. (`gulrajani2017improved`) |

## Equation Citations
- Every equation must cite its canonical source using `\cite{}`.
- If the equation is the author's own derivation, mark it explicitly (e.g., "derived from \cite{...}").
- Source citations appear as `\par{\small\textit{Source: \cite{...}}}` after the equation environment.

## Reference Chapter
- `ch08_references.tex` contains only `\bibliographystyle{IEEEtran}` and `\bibliography{references}`.
- Do not add inline reference lists to this chapter.