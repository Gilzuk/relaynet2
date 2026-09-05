# Learned Relay Processing in Two-Hop Links: A Deep-Research Report

**Mode:** deep-research `full` (6-phase pipeline) · **Date:** 2026-09-05
**Prepared for:** *Deep Learning Architectures for Two-Hop Relay Communication* (M.Sc. thesis, this repository)

---

## Abstract

This report surveys what the 2016-2025 literature establishes about neural processing at the relay node of a two-hop (source -> relay -> destination) link, and locates the gaps that this repository's thesis addresses. Twelve sources were verified and admitted; every unverifiable candidate was excluded. The verified record splits into three strands: classical relay theory (Cover & El Gamal, 1979; Laneman et al., 2004), learned physical-layer processing at the *receiver* (Ye et al., 2018; Samuel et al., 2019; Farsad & Goldsmith, 2018; Gruber et al., 2017; Nachmani et al., 2016), and a small, recent strand that trains a network *at the relay itself* (Lu et al., 2020; Bian et al., 2025). The relay-side strand frames the problem as end-to-end autoencoding or joint source-channel coding and benchmarks against amplify-and-forward only. No verified work performs a controlled cross-architecture comparison of symbol-level neural relays against both AF and decode-and-forward under a fixed canonical channel, characterizes the parameter-count floor of the relay denoising task, or benchmarks a learned relay against pilot-based estimate-then-MLSE on ISI channels. Those are precisely the thesis's Gaps 1-2 and hypotheses H1-H5; the survey supports the thesis's novelty claims, with two caveats noted in the Discussion. A verification constraint applies throughout: this run could reach only GitHub, so sources were verified through author code repositories and bibliography files rather than publisher records (see Limitations).

---

## 1. Introduction

### 1.1 Research question

*What does the literature from roughly 2016 to 2025 establish about deep-learning-based relay processing in two-hop links (which architectures, channel models, and training regimes have been studied), and where does the evidence leave gaps that the thesis's experiments (E1-E6, hypotheses H1-H5) address?*

Sub-questions:

- **SQ1.** Which works place a trained neural network *at the relay node*, as opposed to at the receiver or across the whole link?
- **SQ2.** Against which classical baselines (AF, DF, MLSE/Viterbi) has learned relaying been evaluated, and under which channel models?
- **SQ3.** Has the complexity-performance relationship (minimum viable model size) been characterized for the relay task?

### 1.2 Scope

**In scope:** three-node two-hop relaying; neural networks trained to transform or detect the relayed signal; classical AF/DF/MLSE baselines; AWGN, Rayleigh fading, and ISI channels; deep unfolding and sequence detectors as receiver-side precedents.

**Out of scope:** relay *selection* and resource allocation (learning which relay to use rather than what the relay computes); multi-antenna beamforming relays; reconfigurable intelligent surfaces; semantic communication beyond the one verified relay-side JSCC work.

---

## 2. Method

The deep-research pipeline ran in six phases: scoping, investigation, synthesis, composition, editorial/ethics/devil's-advocate review, and revision. The investigation phase had one unusual constraint: the execution environment blocks all outbound web access except GitHub. Publisher databases (IEEE Xplore, arXiv, Crossref, Semantic Scholar, OpenAlex) were unreachable: every attempt failed at DNS resolution.

Source verification therefore proceeded through GitHub evidence: official author code repositories, `.bib` files in reputable third-party repositories (including NVIDIA's Sionna library), and README citations. The admission rule was strict, following the pipeline's iron rule: a source enters the report only if concrete GitHub evidence confirms its title, authors, and year; anything short of that was excluded, and metadata not visible in the evidence (some volume/issue numbers, one venue) is left unstated rather than filled in from memory. Twelve sources passed; the exclusions are listed in Section 5.3.

Search strategy: (a) verification of nine candidate key works assembled from the thesis's own bibliography and the surveyor's domain knowledge; (b) open GitHub repository and code search for relay-specific terms ("learned relay", "neural relay", "relay autoencoder", "deep learning amplify-and-forward", "DNN decode-and-forward"); (c) annotation of each verified source by problem addressed, method, and finding.

---

## 3. Verified evidence base

### 3.1 Classical relay theory (the baselines)

**Cover and El Gamal (1979)** established the information-theoretic treatment of the three-node relay channel, introducing the coding strategies now called decode-and-forward and compress-and-forward together with the cut-set upper bound. This defines the ceiling against which any relay processing, classical or learned, is measured.

**Laneman, Tse, and Wornell (2004)** defined the practical fixed and adaptive relaying protocols, amplify-and-forward and decode-and-forward, and derived their outage behavior and diversity order in fading. These are the AF/DF baselines every learned-relay study inherits.

### 3.2 Learned physical-layer processing at the receiver (the precedents)

Five verified works establish that neural networks can replace or augment classical receiver blocks. None of them touches the relay node, but each supplies a design principle the relay problem borrows.

- **O'Shea and Hoydis (2017)** introduced the channel-autoencoder paradigm: transmitter, channel, and receiver as one network trained end-to-end, with learned constellations matching or beating classical baselines on AWGN and Rayleigh channels. Conceptually, a learned relay is a middle segment of such an autoencoder.
- **Dörner, Cammerer, Hoydis, and ten Brink (2018)** showed that an autoencoder transceiver trained on a synthetic channel, then fine-tuned on measured data, operates over the air on SDR hardware. This is evidence that learned transceivers survive the sim-to-real gap, and its two-phase training methodology applies to any deployed learned relay.
- **Ye, Li, and Juang (2018)** trained a DNN to perform joint channel estimation and detection in OFDM, beating LS/MMSE precisely where the classical chain is weakest: few pilots, short cyclic prefix, clipping nonlinearity. This is the canonical demonstration that a neural detector can *absorb* channel estimation, the same argument made when a neural relay replaces explicit estimation plus equalization.
- **Samuel, Diskin, and Wiesel (2019)** unfolded projected gradient descent for maximum-likelihood MIMO detection into DetNet, reaching near-ML accuracy at a fraction of sphere-decoding complexity. It anchors the model-based end of the design axis (unfolded classical algorithm vs. generic network) that also runs through relay design.
- **Farsad and Goldsmith (2018)** proposed the sliding bidirectional RNN detector for channels with memory, approaching Viterbi performance when the channel model is correct and beating it when the model is mismatched. Of all receiver-side work, this is the strongest precedent for the thesis's chapter on unknown and mismatched channels: it shows a trained sequence detector displacing MLSE exactly when the model assumption breaks.

Two decoding-side works complete the receiver picture. **Gruber, Cammerer, Hoydis, and ten Brink (2017)** trained plain feed-forward networks to decode short polar and random codes and quantified the exponential training cost in block length, a hard limit any naively coded neural relay would face. **Nachmani, Be'ery, and Burshtein (2016)** instead augmented classical belief propagation with learnable edge weights, improving BCH decoding without replacing the algorithm, the augmentation option available to a DF relay's decoder.

### 3.3 Neural processing at the relay node (the direct comparators)

Only three verified works train a network at the relay itself, and all are recent.

- **Lu et al. (2020, ICASSP)** designed the cooperative (relay) communication system as coupled neural encoders and decoders. It is the earliest GitHub-verifiable learned-relay paper with author-provided code.
- **Lu, Cheng, Chen, Li, Mow, and Vucetic** (journal version; venue not shown in the verified evidence and therefore not asserted here) modeled the full source-relay-destination link as a single autoencoder under AWGN, with a two-stage training procedure because end-to-end backpropagation through the relay is not directly applicable. The author repository is one of very few open implementations of a neural relay node.
- **Bian, Shao, Wu, Ozfatura, and Gündüz (2025, IEEE JSAC)** proposed "process-and-forward": transformer and CNN modules at the relay that learn what to forward in a deep joint source-channel coding (image transmission) setting, in half- and full-duplex three-node networks, benchmarked explicitly against amplify-and-forward.

### 3.4 What the searches did not find

Open code and repository search for "learned relay", "neural relay", and "DNN decode-and-forward" returned zero bibliography hits outside this repository's own code. No verified work implements a *symbol-level* neural relay evaluated against both AF and DF, and none benchmarks a learned relay against pilot-based estimate-then-MLSE under ISI. Absence of GitHub evidence is not proof of absence in the literature (see Limitations), but the search was direct and the terms exact.

---

## 4. Synthesis

### 4.1 Three strands that do not meet

The verified record has a clear shape. Classical theory (Strand 1) defines the relay's operating points and baselines but predates learning. Receiver-side deep learning (Strand 2) is mature: detection, estimation, decoding, and sequence detection each have strong verified results, including the mismatched-channel result of Farsad and Goldsmith that most closely anticipates the thesis's central hypothesis. Relay-side learning (Strand 3) exists but is young, small, and framed differently: both verified groups treat the relay inside an end-to-end autoencoder or JSCC objective, benchmark against AF only, and report link-level (block error or image quality) metrics rather than symbol-level BER against the full classical repertoire.

The three strands therefore do not meet at the point the thesis occupies. Strand 3 never adopts Strand 1's full baseline set (DF is absent as a comparator in the verified relay-side work), and it does not import Strand 2's hard questions (model mismatch, channels with memory, complexity floors) into the relay setting.

### 4.2 Mapping to the thesis's stated gaps

**Gap 1 (no cross-paradigm relay comparison).** Supported. The verified relay-side works each study one architecture family (autoencoder MLPs; transformer/CNN JSCC). None compares supervised, generative, adversarial, and sequence architectures on one relay task under controlled conditions. The thesis's eight-relay canonical comparison has no verified precedent.

**Gap 2 (complexity-performance uncharacterized for the relay task).** Supported, and sharper than the thesis states it. There is no minimum-size study for relay denoising, and the verified relay-side works do not report parameter-normalized comparisons at all. The closest verified result is Gruber et al.'s scaling analysis for decoding, which concerns a different task.

**H5 territory (unknown and mismatched channels).** Farsad and Goldsmith verified that a trained sequence detector beats Viterbi under model mismatch, at the *receiver*. No verified work poses this at the relay, where the node must re-transmit rather than decide, and none runs the pilot-budget crossover or blind-regime layers of the thesis's chapter on unknown channels. The thesis's estimate-then-MLSE benchmark discipline (genie-CSI Viterbi as the matched bound) likewise has no verified relay-side counterpart.

### 4.3 Contradictions and tensions in the evidence

The record contains one genuine tension worth carrying into the thesis's discussion. Strand 2 offers two competing design philosophies: model-based unfolding (DetNet; neural BP), which embeds the classical algorithm and learns residuals, versus model-free learning (SBRNN; the OFDM DNN), which discards the classical structure. The verified evidence does not settle which philosophy wins at the relay; nobody has run that comparison there. The thesis's finding that a minimal generic MLP suffices for the memoryless task, while the ISI task rewards windowed input rather than architectural sophistication, is a data point in that open argument, not a resolution of it.

### 4.4 Two caveats on novelty

First, the Bian et al. process-and-forward line is closer to the thesis than any source in the thesis's own bibliography: it trains networks at the relay and benchmarks against AF in a three-node network. The thesis remains distinct (uncoded/coded symbol-level BER versus a JSCC image objective, DF and MLSE baselines versus AF only, a complexity study versus none), but Chapter 2's claim that relay-side learning attention has gone mainly to relay *selection* is now weaker than when written, and the thesis would be safer citing this line if it can be verified through a channel the university library trusts. Second, the two-stage training difficulty Lu et al. report (backpropagation through the relay) is a methodological point the thesis sidesteps by training the relay in isolation on a local denoising loss; that design choice deserves the one-sentence justification it gets, and no more, but it is a difference in problem formulation as well as in evaluation discipline.

---

## 5. Discussion

### 5.1 Findings against the sub-questions

- **SQ1:** Three verified works place a trained network at the relay (Lu et al. x2, Bian et al.); all are autoencoder/JSCC-framed. The remainder of the learned-PHY literature verified here operates at the receiver.
- **SQ2:** Verified relay-side baselines are AF only. No verified relay-side work compares against DF or MLSE. Channel models in the verified relay-side work are AWGN (Lu et al.) and the JSCC setting of Bian et al.; none uses the ISI or biased-nonlinearity families of the thesis's chapter on unknown channels.
- **SQ3:** No. The complexity floor of relay denoising is uncharacterized in the verified record.

### 5.2 Implications for the thesis

The survey strengthens the thesis's positioning table (Chapter 3, research positioning) on every row checked, and it adds one row the table could state more forcefully: *evaluation against the full classical repertoire* (AF **and** DF **and** estimate-then-MLSE) is itself absent from prior relay-side work, beyond the missing cross-architecture breadth. Conversely, Section 4.4's first caveat suggests the related-work chapter acknowledge the process-and-forward line explicitly once it is verifiable through a trusted channel, so the "to the best of our knowledge" hedges rest on a documented search rather than silence.

### 5.3 Excluded material

Per the pipeline's verification rule, the following were considered and excluded: Nachmani et al.'s 2018 JSTSP journal extension (only the 2016 Allerton version verified); final volume/issue for Farsad and Goldsmith (early-access entry only) and for Ye et al. (venue and pages verified, volume/issue not); the journal venue of the Lu et al. autoencoder-relay paper (title and authors verified from the author repository, venue not shown). These works are real with high likelihood; the specific unverified metadata is simply not asserted.

### 5.4 Limitations

1. **Single-channel verification.** All verification ran through GitHub. Author repositories and third-party bibliography files are good evidence of existence and metadata, but they are not publisher records, and the search cannot see paywalled or code-less literature at all. The relay-side strand in particular is probably undercounted: work without released code is invisible to this method.
2. **No citation-graph traversal.** Without Crossref/Semantic Scholar access, forward and backward citation chasing was impossible; coverage relies on keyword search and prior knowledge of the field.
3. **Recency skew.** GitHub evidence favors post-2015 work with code releases; classical literature is verified only where third-party bibliographies happen to cite it.
4. **One surveyor pass.** The synthesis reflects a single pipeline run with internal review checkpoints, not independent replication.

A follow-up run on a machine with library access should re-verify Section 3.3's metadata against publisher records before any of it enters the thesis's bibliography.

### 5.5 Review-checkpoint record

The devil's-advocate checkpoints raised, and the revision addressed, three challenges. First, cherry-picking: the report initially omitted the process-and-forward line's proximity to the thesis, and Section 4.4 now states it plainly. Second, the absence-of-evidence fallacy: Section 3.4 and Limitation 1 now separate "not found on GitHub" from "does not exist". Third, the so-what test: Section 5.2 states what the thesis should do differently, not just that its gaps survive. The editorial pass verdict was accept-with-minor-revisions (applied); the ethics review cleared the report with the AI-disclosure and attribution statements below.

---

## 6. Conclusion

The verified literature contains mature receiver-side deep learning, settled classical relay theory, and a thin, recent, differently-framed strand of relay-side learning. Nobody in the verified record compares neural relay architectures against each other, against DF, or against estimate-then-MLSE under a fixed protocol; nobody measures how small the relay network can be. Those are the thesis's claims to novelty, and this survey supports them within the stated limits of its single verification channel. The one action item it returns to the thesis is to engage the process-and-forward line directly rather than leave it outside the related-work perimeter.

---

## References

All references verified via GitHub evidence as described in Section 2; metadata not visible in that evidence is omitted rather than reconstructed.

- Bian, C., Shao, Y., Wu, H., Ozfatura, E., & Gündüz, D. (2025). Process-and-forward: Deep joint source-channel coding over cooperative relay networks. *IEEE Journal on Selected Areas in Communications, 43*(5).
- Cover, T. M., & El Gamal, A. A. (1979). Capacity theorems for the relay channel. *IEEE Transactions on Information Theory, 25*, 572-584.
- Dörner, S., Cammerer, S., Hoydis, J., & ten Brink, S. (2018). Deep learning based communication over the air. *IEEE Journal of Selected Topics in Signal Processing, 12*, 132-143. https://doi.org/10.1109/JSTSP.2017.2784180
- Farsad, N., & Goldsmith, A. (2018). Neural network detection of data sequences in communication systems. *IEEE Transactions on Signal Processing*.
- Gruber, T., Cammerer, S., Hoydis, J., & ten Brink, S. (2017). On deep learning-based channel decoding. *51st Annual Conference on Information Sciences and Systems (CISS)*, 1-6.
- Laneman, J. N., Tse, D. N. C., & Wornell, G. W. (2004). Cooperative diversity in wireless networks: Efficient protocols and outage behavior. *IEEE Transactions on Information Theory, 50*, 3062-3080.
- Lu, Y., Cheng, P., Chen, Z., Li, Y., Mow, W. H., & Vucetic, B. (n.d.). Deep autoencoder learning for relay-assisted cooperative communication systems. [Venue not verified; author repository: github.com/ylubg/Autoencoder-relay-AWGN-ROB]
- Lu, Y., et al. (2020). A learning approach to cooperative communication system design. *IEEE ICASSP 2020*. https://doi.org/10.1109/ICASSP40776.2020.9054093
- Nachmani, E., Be'ery, Y., & Burshtein, D. (2016). Learning to decode linear codes using deep learning. *IEEE Annual Allerton Conference on Communication, Control, and Computing*.
- O'Shea, T., & Hoydis, J. (2017). An introduction to deep learning for the physical layer. *IEEE Transactions on Cognitive Communications and Networking, 3*(4), 563-575.
- Samuel, N., Diskin, T., & Wiesel, A. (2019). Learning to detect. *IEEE Transactions on Signal Processing, 67*(10), 2554-2564. https://doi.org/10.1109/TSP.2019.2899805
- Ye, H., Li, G. Y., & Juang, B.-H. (2018). Power of deep learning for channel estimation and signal detection in OFDM systems. *IEEE Wireless Communications Letters*, 114-117.

---

## Disclosure

This report was produced with AI-assisted research tooling (a multi-agent deep-research pipeline running in GitHub Copilot). Source discovery and verification used GitHub repository and code search exclusively, because the execution environment blocked all other network access. Every admitted reference was independently verified against GitHub evidence; unverifiable candidates and metadata were excluded rather than approximated. Search terms, admission rules, and exclusions are documented in Sections 2 and 5.3 for reproducibility.
