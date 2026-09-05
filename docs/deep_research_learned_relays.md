# Learned Relay Processing in Two-Hop Links: A Bounded Evidence Map

**Mode:** deep-research `full` (6-phase pipeline) · **Date:** 2026-09-05 · **Revision 1** (after a five-seat simulated peer-review panel; decision: major revision, all roadmap items applied)
**Prepared for:** *Deep Learning Architectures for Two-Hop Relay Communication* (M.Sc. thesis, this repository)

---

## Abstract

This report maps what a GitHub-verifiable subset of the 2016-2025 literature establishes about neural processing at the relay node of a two-hop (source -> relay -> destination) link, and audits the gaps that this repository's thesis claims against that subset. It is a bounded evidence map and gap audit, not a comprehensive novelty search: the producing environment could reach only GitHub, the candidate list was partly seeded from the thesis's own bibliography, and work without released code or GitHub-visible citations is invisible to this method. Eleven sources passed the verification rule and were admitted; every unverifiable candidate was excluded, and one relay-side journal item was demoted to a noted-but-not-admitted project record because its year could not be verified. The admitted record splits into three strands: classical relay theory (Cover & El Gamal, 1979; Laneman et al., 2004), learned physical-layer processing at the *receiver* (Ye et al., 2018; Samuel et al., 2019; Farsad & Goldsmith, 2018; Gruber et al., 2017; Nachmani et al., 2016), and a small, recent strand that trains a network *at the relay itself* (Lu et al., 2020; Bian et al., 2025). The relay-side strand frames the problem as end-to-end autoencoding or joint source-channel coding and benchmarks against amplify-and-forward only. Within this corpus, no work performs a controlled cross-architecture comparison of symbol-level neural relays against both AF and decode-and-forward under a fixed canonical channel, characterizes the parameter-count floor of the relay denoising task, or benchmarks a learned relay against pilot-based estimate-then-MLSE on ISI channels. Those are the thesis's Gaps 1-2 and hypotheses H1-H5. The audit finds the thesis's gap claims not contradicted by this corpus; it cannot confirm them against the literature at large, and it identifies one line of work (process-and-forward) and several adjacent works in the thesis's own bibliography that the thesis should engage more directly (Sections 4.4 and 3.4).

---

## 1. Introduction

### 1.1 Research question

*What does a GitHub-verifiable subset of the literature from roughly 2016 to 2025 establish about deep-learning-based relay processing in two-hop links (which architectures, channel models, and training regimes have been studied), and does that evidence contradict the gap claims that the thesis's experiments (E1-E6, hypotheses H1-H5) rest on?*

Sub-questions:

- **SQ1.** Which admitted works place a trained neural network *at the relay node*, as opposed to at the receiver or across the whole link?
- **SQ2.** Against which classical baselines (AF, DF, MLSE/Viterbi) has learned relaying been evaluated in the admitted works, and under which channel models?
- **SQ3.** Has the complexity-performance relationship (minimum viable model size) been characterized for the relay task in the admitted works?

### 1.2 Scope

**In scope:** three-node two-hop relaying; neural networks trained to transform or detect the relayed signal; classical AF/DF/MLSE baselines; AWGN, Rayleigh fading, and ISI channels; deep unfolding and sequence detectors as receiver-side precedents.

**Out of scope:** relay *selection* and resource allocation (learning which relay to use rather than what the relay computes); multi-antenna beamforming relays; reconfigurable intelligent surfaces; semantic communication beyond the one verified relay-side JSCC work. These areas optimize network or resource behavior around the relay rather than the learned signal transformation at the relay, which is the thesis's object of study; excluding them narrows the map without cutting into the thesis's comparison set. Named out-of-scope context the thesis already cites: deep-learning relay *selection* (Akdemir et al., in the thesis bibliography as `Akdemir2024DLRelaySelection`) and the machine-learning-in-wireless survey of Gündüz et al. (`Gunduz2019MLAir`).

---

## 2. Method

The deep-research pipeline ran in six phases: scoping, investigation, synthesis, composition, editorial/ethics/devil's-advocate review, and revision. The investigation phase had one unusual constraint: the execution environment blocks all outbound web access except GitHub. Publisher databases (IEEE Xplore, arXiv, Crossref, Semantic Scholar, OpenAlex) were unreachable: every attempt failed at DNS resolution.

Source verification therefore proceeded through GitHub evidence: official author code repositories, `.bib` files in reputable third-party repositories (including NVIDIA's Sionna library), and README citations. The admission rule was strict: a source enters the admitted corpus only if concrete GitHub evidence confirms its title, authors, and year; anything short of that was excluded or, where title and authors were verified but the year was not, demoted to a noted-but-not-admitted record (Section 5.3). Metadata not visible in the evidence is reported with its evidence tier in Appendix B rather than asserted as publisher-grade fact. Eleven sources passed.

Two design facts bound what this method can conclude. First, the candidate list of key works was partly assembled from the thesis's own bibliography and the surveyor's domain knowledge, which makes this a *gap audit for this thesis* rather than a field-level novelty search: it checks whether the thesis's claimed gaps survive contact with a corpus the thesis itself helped seed, plus open search. Second, GitHub visibility is a selection filter, not just a verification channel: paywalled and code-less literature, common in communications engineering, is structurally invisible here. Negative findings in this report therefore mean "not found by this protocol," never "does not exist."

Search strategy: (a) verification of nine candidate key works; (b) open GitHub repository and code search for five relay-specific terms: "learned relay", "neural relay", "relay autoencoder", "deep learning amplify-and-forward", and "DNN decode-and-forward"; (c) annotation of each verified source by problem addressed, method, and finding. The full search log, including per-term outcomes for all five terms, is in Appendix A.

---

## 3. Verified evidence base

The table below summarizes the admitted corpus before the per-strand discussion.

| Work | Year | Network location | Task | Channel | Baselines | Metric |
|---|---|---|---|---|---|---|
| Cover & El Gamal | 1979 | none (theory) | relay capacity | discrete memoryless | cut-set bound | rate |
| Laneman, Tse, Wornell | 2004 | none (protocols) | AF/DF relaying | Rayleigh fading | direct transmission | outage, diversity |
| O'Shea & Hoydis | 2017 | end-to-end (TX+RX) | autoencoder link | AWGN, Rayleigh | classical modulation | BLER |
| Gruber et al. | 2017 | receiver (decoder) | NN channel decoding | AWGN | MAP decoding | BER vs block length |
| Nachmani et al. | 2016 | receiver (decoder) | neural BP decoding | AWGN | plain BP | BER |
| Ye, Li, Juang | 2018 | receiver | joint est.+detection (OFDM) | multipath, clipping | LS/MMSE | BER |
| Dörner et al. | 2018 | end-to-end (TX+RX) | over-the-air autoencoder | measured SDR | classical link | BLER |
| Farsad & Goldsmith | 2018 | receiver | sequence detection (SBRNN) | ISI, molecular/optical | Viterbi (matched and mismatched) | BER |
| Samuel, Diskin, Wiesel | 2019 | receiver | unfolded MIMO detection | MIMO fading | ML/sphere decoding | BER, complexity |
| Lu et al. (ICASSP) | 2020 | **relay** (in autoencoder) | cooperative link design | AWGN | AF-style baselines | BLER |
| Bian et al. | 2025 | **relay** (JSCC modules) | process-and-forward | half/full-duplex relay | AF | image quality |

### 3.1 Classical relay theory (the baselines)

**Cover and El Gamal (1979)** established the information-theoretic treatment of the three-node relay channel, introducing the coding strategies now called decode-and-forward and compress-and-forward together with the cut-set upper bound. This defines the ceiling against which any relay processing, classical or learned, is measured.

**Laneman, Tse, and Wornell (2004)** defined the practical fixed and adaptive relaying protocols, amplify-and-forward and decode-and-forward, and derived their outage behavior and diversity order in fading. These are the AF/DF baselines every learned-relay study inherits.

### 3.2 Learned physical-layer processing at the receiver (the precedents)

Five admitted works establish that neural networks can replace or augment classical receiver blocks. None of them touches the relay node, but each supplies a design principle the relay problem borrows.

- **O'Shea and Hoydis (2017)** introduced the channel-autoencoder paradigm: transmitter, channel, and receiver as one network trained end-to-end, with learned constellations matching or beating classical baselines on AWGN and Rayleigh channels. Conceptually, a learned relay is a middle segment of such an autoencoder.
- **Dörner, Cammerer, Hoydis, and ten Brink (2018)** showed that an autoencoder transceiver trained on a synthetic channel, then fine-tuned on measured data, operates over the air on SDR hardware. This is evidence that learned transceivers survive the sim-to-real gap, and its two-phase training methodology applies to any deployed learned relay.
- **Ye, Li, and Juang (2018)** trained a DNN to perform joint channel estimation and detection in OFDM, beating LS/MMSE precisely where the classical chain is weakest: few pilots, short cyclic prefix, clipping nonlinearity. This is the canonical demonstration that a neural detector can *absorb* channel estimation, the same argument made when a neural relay replaces explicit estimation plus equalization.
- **Samuel, Diskin, and Wiesel (2019)** unfolded projected gradient descent for maximum-likelihood MIMO detection into DetNet, reaching near-ML accuracy at a fraction of sphere-decoding complexity. It anchors the model-based end of the design axis (unfolded classical algorithm vs. generic network) that also runs through relay design.
- **Farsad and Goldsmith (2018)** proposed the sliding bidirectional RNN detector for channels with memory. It approaches Viterbi performance when the Viterbi detector has the correct channel model, and it beats a Viterbi detector that is running on a mismatched or incorrect model; it makes no claim against matched, CSI-aware MLSE. Of all receiver-side work, this is the strongest precedent for the thesis's chapter on unknown and mismatched channels: it shows a trained sequence detector displacing a *mis-specified* MLSE.

Two decoding-side works complete the receiver picture. **Gruber, Cammerer, Hoydis, and ten Brink (2017)** trained plain feed-forward networks to decode short polar and random codes and quantified the exponential training cost in block length, a hard limit any naively coded neural relay would face. **Nachmani, Be'ery, and Burshtein (2016)** instead augmented classical belief propagation with learnable edge weights, improving BCH decoding without replacing the algorithm, the augmentation option available to a DF relay's decoder.

### 3.3 Neural processing at the relay node (the direct comparators)

Two admitted works train a network at the relay itself, and both are recent.

- **Lu et al. (2020, ICASSP)** designed the cooperative (relay) communication system as coupled neural encoders and decoders. It is the earliest GitHub-verifiable learned-relay paper with author-provided code. A journal-length extension by the same group (Lu, Cheng, Chen, Li, Mow, and Vucetic, "Deep autoencoder learning for relay-assisted cooperative communication systems") is visible in the author's repository, which names the title and all six authors and describes a two-stage training procedure, adopted because end-to-end backpropagation through the relay is not directly applicable. Neither its venue nor its year is shown in that evidence, so under the admission rule it is recorded as a noted project record of the same line, not counted as an admitted source (Section 5.3, Appendix B). The repository remains one of very few open implementations of a neural relay node.
- **Bian, Shao, Wu, Ozfatura, and Gündüz (2025)** proposed "process-and-forward": transformer and CNN modules at the relay that learn what to forward in a deep joint source-channel coding (image transmission) setting, in half- and full-duplex three-node networks, benchmarked explicitly against amplify-and-forward. The venue and volume reported in Appendix B come from the author repository's README bibliography; the publisher record was not reachable from this environment and has not been verified.

### 3.4 Adjacent relay-side learning already in the thesis's bibliography

The thesis's own bibliography contains relay-relevant learned-processing works that sit at the boundary of this report's scope and were not part of the admitted corpus (they were not re-verified externally; they are cited here as entries of the thesis's `references.bib`, which is itself in this repository). They matter because they qualify any "only N relay-side works" statement:

- **Bergel (2023, arXiv) and Bergel (ICASSP 2024), "relays as neurons":** treat AF relay nodes as neurons with nonlinear transfer functions and use deep-learning optimization to tune relay gains, exploiting hardware nonlinearities. This *is* relay-side learned processing, but of analog gains, not of a symbol-level detection/regeneration mapping; the thesis already positions it that way in Chapter 2.
- **Shlezinger et al. (2020), ViterbiNet:** a learned Viterbi algorithm for symbol detection that replaces the channel-model-dependent metric with a learned one. Receiver-side, but it bears directly on the thesis's estimate-then-MLSE benchmark discipline: it is the model-based-learning alternative in exactly the regime (ISI, imperfect channel knowledge) where the thesis deploys a generic MLP.
- **Park, Jang, Simeone, and Kang (2021), meta-learned demodulation from few pilots:** receiver-side meta-learning for the low-pilot regime, adjacent to the thesis's pilot-budget crossover layer.

These works sharpen rather than close the thesis's gaps: none of them evaluates a symbol-level neural relay against AF, DF, and MLSE together. But a reader of Section 4 should know they exist, and the thesis's related-work chapter already cites all of them.

### 3.5 What the searches did not find

Open code and repository search for all five terms of Section 2 ("learned relay", "neural relay", "relay autoencoder", "deep learning amplify-and-forward", "DNN decode-and-forward") returned no bibliography hits outside this repository's own code; the two productive routes were repository search for cooperative-communication phrasings, which surfaced the Lu et al. repositories, and author-repository discovery, which surfaced Bian et al. (Appendix A). Within the admitted corpus, no work implements a *symbol-level* neural relay evaluated against both AF and DF, and none benchmarks a learned relay against pilot-based estimate-then-MLSE under ISI. Absence of GitHub evidence is not proof of absence in the literature (see Limitations); relevant work may also exist under other names this search did not use, such as denoise-and-forward, compress-and-forward learning, or neural forwarding.

---

## 4. Synthesis

All statements in this section are claims about the admitted corpus of Section 3, under the visibility limits of Section 2, not about the literature at large.

### 4.1 Three strands that do not meet

The admitted record has a clear shape. Classical theory (Strand 1) defines the relay's operating points and baselines but predates learning. Receiver-side deep learning (Strand 2) is well developed in this corpus: detection, estimation, decoding, and sequence detection each have verified results, including the mismatched-channel result of Farsad and Goldsmith that most closely anticipates the thesis's central hypothesis. Relay-side learning (Strand 3) exists but is young, small, and framed differently: both verified groups treat the relay inside an end-to-end autoencoder or JSCC objective, benchmark against AF only, and report link-level (block error or image quality) metrics rather than symbol-level BER against the full classical repertoire.

Within this corpus the three strands do not meet at the point the thesis occupies. Strand 3 never adopts Strand 1's full baseline set (DF is absent as a comparator in the verified relay-side work), and it does not import Strand 2's hard questions (model mismatch, channels with memory, complexity floors) into the relay setting. One alternative explanation deserves stating: relay-side work may adopt autoencoder/JSCC framing for sound technical reasons, because relay optimization is naturally end-to-end and DF/MLSE are not meaningful comparators for image or semantic objectives. That would make the absence of symbol-level baselines a difference in problem formulation rather than an oversight, which narrows, without erasing, the space the thesis occupies.

### 4.2 Mapping to the thesis's stated gaps

**Gap 1 (no cross-paradigm relay comparison).** Not contradicted by this corpus. The verified relay-side works each study one architecture family (autoencoder MLPs; transformer/CNN JSCC). None compares supervised, generative, adversarial, and sequence architectures on one relay task under controlled conditions. Within this corpus, the thesis's eight-relay canonical comparison has no precedent; whether one exists in the invisible (paywalled, code-less) literature this method cannot say.

**Gap 2 (complexity-performance uncharacterized for the relay task).** Not contradicted by this corpus, and the corpus adds a detail: the verified relay-side works do not report parameter-normalized comparisons at all. The closest verified result is Gruber et al.'s scaling analysis for decoding, which concerns a different task. The same visibility caveat applies, and so does an alternative framing the report should not hide: practical relay deployments may care more about latency, FLOPs, memory, or CSI overhead than about a parameter-count floor, and none of those axes is settled by this corpus either.

**H5 territory (unknown and mismatched channels).** Farsad and Goldsmith verified that a trained sequence detector beats a mis-specified Viterbi detector, at the *receiver*. ViterbiNet (thesis bibliography, Section 3.4) attacks the same regime with model-based learning. No admitted work poses this at the relay, where the node must re-transmit rather than decide, and none runs the pilot-budget crossover or blind-regime layers of the thesis's chapter on unknown channels. The thesis's estimate-then-MLSE benchmark discipline (genie-CSI Viterbi as the matched bound) has no counterpart in the admitted relay-side work.

### 4.3 Contradictions and tensions in the evidence

The record contains one genuine tension worth carrying into the thesis's discussion. Strand 2 offers two competing design philosophies: model-based unfolding (DetNet; neural BP; ViterbiNet in the thesis's own bibliography), which embeds the classical algorithm and learns residuals, versus model-free learning (SBRNN; the OFDM DNN), which discards the classical structure. The verified evidence does not settle which philosophy wins at the relay; nobody in this corpus has run that comparison there. The thesis's finding that a minimal generic MLP suffices for the memoryless task, while the ISI task rewards windowed input rather than architectural sophistication, is a data point in that open argument, not a resolution of it.

### 4.4 Two caveats on novelty

First, the Bian et al. process-and-forward line is closer to the thesis than any source in the thesis's own bibliography: it trains networks at the relay and benchmarks against AF in a three-node network. That the closest comparator surfaced through this audit rather than through the thesis's own related-work search is itself a finding, and it weakens Chapter 2's claim that relay-side learning attention has gone mainly to relay *selection*. The thesis remains distinct (uncoded/coded symbol-level BER versus a JSCC image objective, DF and MLSE baselines versus AF only, a complexity study versus none), but the related-work chapter should engage this line directly once it is verifiable through a channel the university library trusts. Second, the two-stage training difficulty Lu et al. report (backpropagation through the relay) is a methodological point the thesis sidesteps by training the relay in isolation on a local denoising loss; that design choice deserves the one-sentence justification it gets, and no more, but it is a difference in problem formulation as well as in evaluation discipline.

---

## 5. Discussion

### 5.1 Findings against the sub-questions

- **SQ1:** Two admitted works place a trained network at the relay (Lu et al. 2020 and Bian et al. 2025), plus one noted-but-not-admitted journal record of the Lu line; all are autoencoder/JSCC-framed. The Bergel "relays as neurons" line (thesis bibliography, not re-verified here) is relay-side learned gain optimization at the boundary of scope. The remainder of the learned-PHY corpus operates at the receiver.
- **SQ2:** Admitted relay-side baselines are AF only. No admitted relay-side work compares against DF or MLSE. Channel models in the admitted relay-side work are AWGN (Lu et al.) and the JSCC setting of Bian et al.; none uses the ISI or biased-nonlinearity families of the thesis's chapter on unknown channels.
- **SQ3:** No. The complexity floor of relay denoising is uncharacterized in the admitted corpus.

### 5.2 Implications for the thesis

Within the admitted corpus, no row of the thesis's positioning table (Chapter 3, research positioning) is contradicted, and the audit suggests one row the table could state more explicitly: *evaluation against the full classical repertoire* (AF **and** DF **and** estimate-then-MLSE) is absent from the admitted relay-side work, beyond the missing cross-architecture breadth. Because the audit is bounded and partly self-seeded (Section 2), this is provisional support pending conventional database verification, not confirmation. Concretely, before any of this reaches the thesis or a defense:

**Re-verification checklist (to run with library access):**
1. Verify Bian et al. against IEEE Xplore/Crossref: exact title, DOI, volume, issue, pages; confirm the author repository matches the published paper.
2. Resolve the Lu et al. journal version: find its venue and year in a publisher record; decide whether to cite it, the ICASSP version, or both.
3. Run forward-citation search (Google Scholar / Semantic Scholar) on Cover & El Gamal, Laneman et al., and Bian et al. filtered by "relay" + "neural/learning" to catch code-less relay-side work this method could not see.
4. Repeat the five Section 2 search terms, plus "denoise-and-forward", "compress-and-forward learning", and "neural cooperative communication", on Scholar/IEEE.
5. Update `thesis/chapters/references.bib` only with publisher-verified entries; date and log each search.
6. Re-read Chapter 2's relay-selection claim against whatever step 3-4 return, and revise or keep it deliberately.

### 5.3 Excluded sources and admitted sources with omitted metadata

Two different things were previously listed together; they are now separated.

**Excluded or demoted sources (not in the admitted corpus):**
- Nachmani et al.'s 2018 JSTSP journal extension of the 2016 Allerton paper: not found in any reachable GitHub bibliography; only the 2016 version is admitted.
- Lu et al., "Deep autoencoder learning for relay-assisted cooperative communication systems": title and authors verified from the author repository, but no venue or year visible, which fails the title+authors+year admission rule. Recorded as a noted project record of the admitted Lu et al. (2020) line (Section 3.3), not as an admitted source.

**Admitted sources with metadata omitted or tiered (see Appendix B):**
- Farsad & Goldsmith: journal and year verified; the GitHub bib shows an early-access volume ("PP"), so final volume/issue/pages are not asserted.
- Ye et al.: venue, year, and pages verified; volume/issue not visible in the evidence and not asserted.
- Bian et al.: venue/volume/issue reported from the author README bibliography only; publisher record unverified.

### 5.4 Limitations

1. **Single-channel verification.** All verification ran through GitHub. Author repositories and third-party bibliography files are good evidence of existence and basic metadata, but they are not publisher records, and the search cannot see paywalled or code-less literature at all. The relay-side strand in particular is probably undercounted: work without released code is invisible to this method. This limitation is structural, which is why the report's conclusions are stated as corpus-bounded throughout.
2. **Partly self-seeded candidate list.** The candidate key works came in part from the thesis's own bibliography, so the audit checks the thesis's framing against a corpus the thesis helped select. The open searches mitigate but do not remove this.
3. **No citation-graph traversal.** Without Crossref/Semantic Scholar access, forward and backward citation chasing was impossible; coverage relies on keyword search and prior knowledge of the field.
4. **Recency skew.** GitHub evidence favors post-2015 work with code releases; classical literature is verified only where third-party bibliographies happen to cite it.
5. **One surveyor pass.** The synthesis reflects a single pipeline run with internal review checkpoints (Appendix C), not independent replication.

---

## 6. Conclusion

Within the admitted, GitHub-verifiable corpus, receiver-side deep learning is well developed, classical relay theory is settled, and relay-side learning is a thin, recent, differently-framed strand. No work in this corpus compares neural relay architectures against each other, against DF, or against estimate-then-MLSE under a fixed protocol, and none measures how small the relay network can be. The thesis's gap claims are therefore not contradicted by this audit, and they gain provisional support, bounded by the visibility limits and the partly self-seeded candidate list documented in Section 2 and Section 5.4. Confirmation requires the library-access re-verification of Section 5.2. The audit returns two action items to the thesis: engage the process-and-forward line directly rather than leave it outside the related-work perimeter, and connect the ViterbiNet/model-based-learning strand (already in the thesis bibliography) to the unknown-channel chapter's positioning.

---

## References

Admitted corpus (11 sources). Per-field evidence provenance is in Appendix B; fields whose only evidence is a GitHub bibliography or README are marked there and should be treated as reported, not publisher-verified.

- Bian, C., Shao, Y., Wu, H., Ozfatura, E., & Gündüz, D. (2025). Process-and-forward: Deep joint source-channel coding over cooperative relay networks. Reported venue: *IEEE Journal on Selected Areas in Communications, 43*(5) [README bib; publisher record unverified].
- Cover, T. M., & El Gamal, A. A. (1979). Capacity theorems for the relay channel. *IEEE Transactions on Information Theory, 25*, 572-584.
- Dörner, S., Cammerer, S., Hoydis, J., & ten Brink, S. (2018). Deep learning based communication over the air. *IEEE Journal of Selected Topics in Signal Processing, 12*, 132-143. https://doi.org/10.1109/JSTSP.2017.2784180
- Farsad, N., & Goldsmith, A. (2018). Neural network detection of data sequences in communication systems. *IEEE Transactions on Signal Processing*.
- Gruber, T., Cammerer, S., Hoydis, J., & ten Brink, S. (2017). On deep learning-based channel decoding. *51st Annual Conference on Information Sciences and Systems (CISS)*, 1-6.
- Laneman, J. N., Tse, D. N. C., & Wornell, G. W. (2004). Cooperative diversity in wireless networks: Efficient protocols and outage behavior. *IEEE Transactions on Information Theory, 50*, 3062-3080.
- Lu, Y., et al. (2020). A learning approach to cooperative communication system design. *IEEE ICASSP 2020*. https://doi.org/10.1109/ICASSP40776.2020.9054093
- Nachmani, E., Be'ery, Y., & Burshtein, D. (2016). Learning to decode linear codes using deep learning. *IEEE Annual Allerton Conference on Communication, Control, and Computing*.
- O'Shea, T., & Hoydis, J. (2017). An introduction to deep learning for the physical layer. *IEEE Transactions on Cognitive Communications and Networking, 3*(4), 563-575.
- Samuel, N., Diskin, T., & Wiesel, A. (2019). Learning to detect. *IEEE Transactions on Signal Processing, 67*(10), 2554-2564. https://doi.org/10.1109/TSP.2019.2899805
- Ye, H., Li, G. Y., & Juang, B.-H. (2018). Power of deep learning for channel estimation and signal detection in OFDM systems. *IEEE Wireless Communications Letters*, 114-117.

Noted, not admitted: Lu, Y., Cheng, P., Chen, Z., Li, Y., Mow, W. H., & Vucetic, B. Deep autoencoder learning for relay-assisted cooperative communication systems. [Title/authors verified from the author repository github.com/ylubg/Autoencoder-relay-AWGN-ROB; venue and year not shown in that evidence.]

Adjacent works cited from the thesis's own `references.bib` (not re-verified by this protocol): Bergel (2023, arXiv:2306.14253); Bergel (ICASSP 2024); Shlezinger, Farsad, Eldar, & Goldsmith (2020, ViterbiNet, IEEE TWC); Park, Jang, Simeone, & Kang (2021, IEEE TSP); Akdemir, Karabulut, & Ilhan (2024); Gündüz et al. (2019).

---

## Appendix A: Search log

Run date: 2026-09-05, from a container whose only reachable domain was github.com. Connectivity to export.arxiv.org, api.semanticscholar.org, api.crossref.org, api.openalex.org, doi.org, scholar.google.com, arxiv.org, and en.wikipedia.org was tested first; every request failed at DNS resolution (evidence: curl exit with no response, HTTP code 000).

**Candidate-verification searches (GitHub code/repo search over `.bib` files and READMEs).** Nine candidates checked; outcomes and evidence URLs are in Appendix B. One candidate (Nachmani et al. 2018 JSTSP) failed verification.

**Open relay-term searches (GitHub code search, exact phrases, in `.bib` files and code):**

| # | Query | Result |
|---|---|---|
| 1 | "learned relay" | 0 relevant hits outside Gilzuk/relaynet2 repositories |
| 2 | "neural relay" | 0 relevant hits outside Gilzuk/relaynet2 repositories |
| 3 | "relay autoencoder" | led (with query 5 below) to ylubg/Autoencoder-relay-AWGN-ROB and ylubg/Autoencoder-relay-ICASSP |
| 4 | "deep learning amplify-and-forward" | 0 relevant bibliography hits |
| 5 | "DNN decode-and-forward" | 0 relevant bibliography hits |

**Open repository searches (GitHub repo search):** "relay autoencoder communication", "deep learning relay network cooperative communication", "relay deep learning decode-and-forward", "neural network relay selection OR forwarding physical layer". Productive results: the two ylubg repositories and aprilbian/Process-and-Forward plus aprilbian/Relay_JSCC. Hit counts per query were not recorded by the search agent; that is a logging gap, noted here rather than reconstructed.

**Screening rule:** a hit was screened in if its README or `.bib` identified a paper about neural processing in a two-hop/cooperative relay link; relay-selection, RIS, and beamforming repositories were screened out per Section 1.2.

## Appendix B: Per-reference provenance

Evidence types: **AR** = official author repository; **3B** = third-party `.bib` file; **RB** = README bibliography in the author repository. "Verified" = field visible in the cited evidence. Publisher records were reachable for none of these.

| Source | Evidence (GitHub) | Verified fields | Reported-only / omitted fields |
|---|---|---|---|
| O'Shea & Hoydis 2017 | 3B: laurabrink13/ML-Receiver `tex/references.bib`; 3B: patrickvonplaten/course_work | title, authors, year, venue, vol/no/pages | none |
| Dörner et al. 2018 | 3B: patrickvonplaten/course_work `references.bib` (JSTSP entry with DOI) | title, authors, year, venue, vol/pages, DOI | none |
| Ye et al. 2018 | 3B: laurabrink13/ML-Receiver `tex/references.bib` | title, authors, year, venue, pages | vol/issue omitted |
| Samuel et al. 2019 | 3B: dfigueroa11/...DetNet `Report/references.bib`; AR: neevsamuel/LearningToDetect | title, authors, year, venue, vol/no/pages, DOI | none (third author corrected to Wiesel per evidence) |
| Farsad & Goldsmith 2018 | 3B: laurabrink13/ML-Receiver `tex/references.bib` | title, authors, year, venue | vol/issue/pages omitted (evidence shows early-access "PP") |
| Laneman et al. 2004 | 3B: wu-victor/2008-ieee-jcn-ofdm-coop-networks `mybib.bib` | title, authors, year, venue, vol/pages | none |
| Cover & El Gamal 1979 | 3B: marcusmueller/iscml `localtexmf/bibtex/relay.bib` | title, authors, year, venue, vol/pages | none |
| Nachmani et al. 2016 | 3B: NVlabs/sionna `doc/source/phy/phy.bib` | title, authors, year, venue | pages omitted |
| Gruber et al. 2017 | 3B: vinhhuy15/stt-mram-slnn `report_paper/refs.bib`; AR: gruberto/DL-ChannelDecoding | title, authors, year, venue, pages | none |
| Lu et al. 2020 (ICASSP) | AR: ylubg/Autoencoder-relay-ICASSP (README with venue + DOI) | title, authors, year, venue, DOI | none |
| Bian et al. 2025 | AR: aprilbian/Process-and-Forward (RB with bibtex); AR: aprilbian/Relay_JSCC | title, authors, year | venue "IEEE JSAC", vol 43, no 5: **reported from RB only, publisher-unverified** |
| Lu et al. journal (noted, not admitted) | AR: ylubg/Autoencoder-relay-AWGN-ROB (README) | title, authors | venue, year not shown; fails admission rule |

## Appendix C: Internal process note (not audit evidence)

This report was produced and revised by an AI pipeline with internal checkpoints; this appendix documents that process. It is self-reported process disclosure, not independent validation.

First pass (composition): internal devil's-advocate checkpoints raised cherry-picking (process-and-forward proximity), the absence-of-evidence fallacy, and the so-what test; all three were addressed before the first commit.

Revision 1: a five-seat simulated peer-review panel (journal-fit, methodology, domain, integrity-perspective, devil's advocate seats, role-separated but same model family) returned a major-revision decision with two validated CRITICAL findings (novelty language exceeding the evidence tier; circular "every row checked" claim) and a six-item roadmap. All six items are applied in this revision: bounded corpus-relative claims throughout; this search log (Appendix A); the Lu admission-rule resolution and 11-source count (Sections 3.3, 5.3); the provenance table (Appendix B); the adjacent-works section and SBRNN correction (Sections 3.4, 3.2); the taxonomy table, this appendix, the re-verification checklist (Section 5.2), and the defense-use disclosure below. The panel artifacts exist only in the producing session's transcript; they were not committed, which is why this note is labeled process disclosure rather than audit evidence.

---

## Disclosure

This report was produced with AI-assisted research tooling (a multi-agent deep-research pipeline running in GitHub Copilot), then revised once after a simulated peer-review panel. Source discovery and verification used GitHub repository and code search exclusively, because the execution environment blocked all other network access. Every admitted reference was verified against the GitHub evidence listed in Appendix B; unverifiable candidates were excluded or demoted, and reported-only metadata is labeled as such. **This document is not suitable as final thesis-defense evidence until the publisher/database re-verification of Section 5.2 is completed**: it is a bounded, corpus-relative gap audit, not a comprehensive literature review. Search terms, admission rules, exclusions, and provenance are documented in Section 2, Section 5.3, and Appendices A-B for reproducibility.
