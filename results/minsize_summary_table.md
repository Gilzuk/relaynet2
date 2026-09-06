| channel | baseline | validity | best config | dB penalty | BER@4dB base | BER@4dB relay | BER@16dB base | BER@16dB relay |
|---|---|---|---|---|---|---|---|---|
| awgn | DF | near-optimal | w=1 h1-16p (49p) | +0.06 | 0.02489 | 0.02520 | 0.00000 | 0.00000 |
| rayleigh | DF | ok | w=1 h1-1p (4p) | +0.03 | 0.22185 | 0.21975 | 0.02400 | 0.02421 |
| flat_gain | DF | ok | w=1 h1-16p (49p) | +0.07 | 0.09798 | 0.09797 | 0.00610 | 0.00616 |
| branch_asym | DF | ok | w=1 h1-16p (49p) | -0.01 | 0.09477 | 0.09242 | 0.00603 | 0.00600 |
| nlbias | DF | near-optimal | w=1 h1-2p (7p) | -2.88 | 0.10138 | 0.03351 | 0.00008 | 0.00000 |
| isi | MLSE | near-optimal | w=7 h1-48p (433p) | +1.67 | 0.05101 | 0.06482 | 0.00000 | 0.00000 |
| isi_complex | MLSE | near-optimal | w=7 h1-48p (433p) | +3.03 | 0.15947 | 0.16195 | 0.00000 | 0.00016 |
| isi_rayleigh | MLSE | ok | w=7 h1-48p (433p) | -0.90 | 0.22864 | 0.21609 | 0.06694 | 0.04079 |
| composite | MLSE | ok | w=7 h1-32p (289p) | -2.26 | 0.25760 | 0.23077 | 0.05011 | 0.00688 |
