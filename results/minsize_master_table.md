# Smallest relay per dB budget, against each scenario's own published relay

Cell = smallest configuration meeting the budget, as `<parameters>p w<window>`.
Dash = no smaller size stays within that budget.

| channel | taps | mod | reference | ≤0 dB | ≤0.1 dB | ≤0.25 dB | ≤0.5 dB | ≤1 dB | ≤2 dB | vs classical at ≤0.5 dB cell |
|---|---|---|---|---|---|---|---|---|---|---|
| awgn | 1 | bpsk | MLP-169 | 4p w1 | 4p w1 | 4p w1 | 4p w1 | 4p w1 | 4p w1 | +0.14 dB |
| rayleigh | 1 | qpsk | MLP-169 | 4p w1 | 4p w1 | 4p w1 | 4p w1 | 4p w1 | 4p w1 | +0.03 dB |
| flat_gain | 1 | bpsk | MLP-169 | 21p w3 | 21p w3 | 21p w3 | 4p w1 | 4p w1 | 4p w1 | +0.89 dB |
| branch_asym | 1 | bpsk | MLP-169 | 4p w1 | 4p w1 | 4p w1 | 4p w1 | 4p w1 | 4p w1 | +0.09 dB |
| nlbias | 1 | bpsk | MLP-169 | 29p w5 | 4p w1 | 4p w1 | 4p w1 | 4p w1 | 4p w1 | -1.63 dB |
| isi | 3 | bpsk | MLP-169 | 73p w7 | 73p w7 | 57p w5 | 57p w5 | 37p w7 | 21p w3 | +2.99 dB |
| isi_complex | 3 | qpsk | MLP-169 | 73p w7 | 73p w7 | 73p w7 | 73p w7 | 57p w5 | 21p w3 | +4.84 dB |
| isi_rayleigh | 3 | qpsk | MLP-169 | 145p w7 | 73p w7 | 73p w7 | 73p w7 | 37p w7 | 21p w3 | -0.12 dB |
| composite | 3 | bpsk | MLP-169 | 145p w7 | 73p w7 | 57p w5 | 21p w3 | 21p w3 | 6p w3 | -1.82 dB |
| coded | code | qpsk | MLP-756 | 372p w9 | 244p w5 | 244p w5 | 18p w1 | 18p w1 | 18p w1 | +1.86 dB |
