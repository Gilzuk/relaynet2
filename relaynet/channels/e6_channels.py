"""Chapter 7 (E6) experiment-specific channels for relaynet."""

import numpy as np


class ISIChannel:
    """Inter-Symbol Interference channel with configurable taps.

    Parameters
    ----------
    taps : array-like
        Channel impulse response coefficients. Will be normalized to unit energy.
    seed : int, optional
        Random seed for reproducibility.
    rng : numpy.random.Generator, optional
        Random generator (overrides seed if provided).
    """

    def __init__(self, taps, seed=None, rng=None):
        self.taps = np.asarray(taps, dtype=float)
        self.taps /= np.linalg.norm(self.taps)
        if rng is None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = rng

    def __call__(self, signal, snr_db):
        """Apply ISI channel and AWGN.

        Parameters
        ----------
        signal : numpy.ndarray
            Input signal.
        snr_db : float
            SNR in dB (using thesis convention: gamma = 1/sigma^2).

        Returns
        -------
        output : numpy.ndarray
            Noisy channel output.
        """
        # Apply ISI via convolution
        isi_output = np.convolve(signal, self.taps)[:signal.size]

        # Add AWGN
        sigma = 10 ** (-snr_db / 20.0)
        noise = sigma * self.rng.standard_normal(signal.size)

        return isi_output + noise


class ComplexISIChannel:
    """Inter-Symbol Interference channel for complex baseband signals (QPSK/QAM).

    Same physical model as :class:`ISIChannel` (real-valued unknown taps,
    convolution, thesis SNR convention gamma = 1/sigma^2) but adds circularly
    symmetric complex AWGN so it is correct for higher-order modulations
    where the transmitted symbols themselves are complex.

    Parameters
    ----------
    taps : array-like
        Channel impulse response coefficients. Will be normalized to unit energy.
    seed : int, optional
        Random seed for reproducibility.
    rng : numpy.random.Generator, optional
        Random generator (overrides seed if provided).
    """

    def __init__(self, taps, seed=None, rng=None):
        self.taps = np.asarray(taps, dtype=float)
        self.taps /= np.linalg.norm(self.taps)
        if rng is None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = rng

    def __call__(self, signal, snr_db):
        """Apply ISI channel and complex AWGN.

        Parameters
        ----------
        signal : numpy.ndarray
            Input signal (complex).
        snr_db : float
            SNR in dB (using thesis convention: gamma = 1/sigma^2).

        Returns
        -------
        output : numpy.ndarray
            Noisy channel output (complex).
        """
        isi_output = np.convolve(signal, self.taps)[:signal.size]

        sigma = 10 ** (-snr_db / 20.0)
        noise = sigma * (
            self.rng.standard_normal(signal.size) +
            1j * self.rng.standard_normal(signal.size)
        ) / np.sqrt(2)

        return isi_output + noise


class ComplexAWGNChannel:
    """Plain AWGN channel for complex baseband signals, thesis SNR convention.

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility.
    rng : numpy.random.Generator, optional
        Random generator (overrides seed if provided).
    """

    def __init__(self, seed=None, rng=None):
        if rng is None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = rng

    def __call__(self, signal, snr_db):
        """Add complex AWGN.

        Parameters
        ----------
        signal : numpy.ndarray
            Input signal (complex or real).
        snr_db : float
            SNR in dB (using thesis convention: gamma = 1/sigma^2).

        Returns
        -------
        output : numpy.ndarray
            Noisy channel output.
        """
        sigma = 10 ** (-snr_db / 20.0)
        if np.iscomplexobj(signal):
            noise = sigma * (
                self.rng.standard_normal(signal.size) +
                1j * self.rng.standard_normal(signal.size)
            ) / np.sqrt(2)
        else:
            noise = sigma * self.rng.standard_normal(signal.size)
        return signal + noise


class ISIRayleighChannel:
    """Combined unknown ISI + coherently-compensated Rayleigh fading + AWGN.

    y = |h| * conv(x, taps) + n, real-valued (BPSK). Intended to be used
    identically on BOTH hops of a two-hop link (same taps, independent
    per-hop RNG streams) so that neither hop is artificially easier than
    the other -- isolating relay-architecture differences rather than
    channel-asymmetry differences.

    Parameters
    ----------
    taps : array-like
        Channel impulse response coefficients. Will be normalized to unit energy.
    seed : int, optional
    rng : numpy.random.Generator, optional
    """

    def __init__(self, taps, seed=None, rng=None):
        self.taps = np.asarray(taps, dtype=float)
        self.taps /= np.linalg.norm(self.taps)
        self.rng = np.random.default_rng(seed) if rng is None else rng

    def __call__(self, signal, snr_db):
        isi_output = np.convolve(signal, self.taps)[:signal.size]
        h = np.abs(
            (self.rng.standard_normal(signal.size) +
             1j * self.rng.standard_normal(signal.size)) / np.sqrt(2)
        )
        faded = h * isi_output
        sigma = 10 ** (-snr_db / 20.0)
        noise = sigma * self.rng.standard_normal(signal.size)
        return faded + noise


class ComplexISIRayleighChannel:
    """Combined unknown ISI + coherently-compensated Rayleigh fading + AWGN,
    for complex baseband signals (QPSK/QAM). See :class:`ISIRayleighChannel`.
    """

    def __init__(self, taps, seed=None, rng=None):
        self.taps = np.asarray(taps, dtype=float)
        self.taps /= np.linalg.norm(self.taps)
        self.rng = np.random.default_rng(seed) if rng is None else rng

    def __call__(self, signal, snr_db):
        isi_output = np.convolve(signal, self.taps)[:signal.size]
        h = np.abs(
            (self.rng.standard_normal(signal.size) +
             1j * self.rng.standard_normal(signal.size)) / np.sqrt(2)
        )
        faded = h * isi_output
        sigma = 10 ** (-snr_db / 20.0)
        noise = sigma * (
            self.rng.standard_normal(signal.size) +
            1j * self.rng.standard_normal(signal.size)
        ) / np.sqrt(2)
        return faded + noise


class NonlinearBiasChannel:
    """Nonlinear saturation with DC bias channel.

    Models a saturating amplifier: y = tanh(1.5 * x) + 0.5 + n.

    Parameters
    ----------
    saturation : float, optional
        Saturation parameter (default 1.5).
    dc_bias : float, optional
        DC bias term (default 0.5).
    seed : int, optional
        Random seed.
    rng : numpy.random.Generator, optional
        Random generator.
    """

    def __init__(self, saturation=1.5, dc_bias=0.5, seed=None, rng=None):
        self.saturation = saturation
        self.dc_bias = dc_bias
        if rng is None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = rng

    def __call__(self, signal, snr_db):
        """Apply nonlinear bias and AWGN.

        Parameters
        ----------
        signal : numpy.ndarray
        snr_db : float

        Returns
        -------
        output : numpy.ndarray
        """
        nonlinear = np.tanh(self.saturation * signal) + self.dc_bias
        sigma = 10 ** (-snr_db / 20.0)
        noise = sigma * self.rng.standard_normal(signal.size)
        return nonlinear + noise


class RayleighChannel:
    """Coherently-compensated Rayleigh fading channel.

    Output: y = |h| * x + n, where |h| is the magnitude of complex Rayleigh fading
    and the phase is perfectly compensated (receiver knows channel perfectly).

    Parameters
    ----------
    seed : int, optional
    rng : numpy.random.Generator, optional
    """

    def __init__(self, seed=None, rng=None):
        if rng is None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = rng

    def __call__(self, signal, snr_db):
        """Apply Rayleigh fading and AWGN.

        Parameters
        ----------
        signal : numpy.ndarray
        snr_db : float

        Returns
        -------
        output : numpy.ndarray
        """
        # Complex Rayleigh fading, magnitude compensation (coherent)
        h = np.abs(
            (self.rng.standard_normal(signal.size) +
             1j * self.rng.standard_normal(signal.size)) / np.sqrt(2)
        )
        faded = h * signal

        # Add AWGN
        sigma = 10 ** (-snr_db / 20.0)
        noise = sigma * self.rng.standard_normal(signal.size)

        return faded + noise


class FlatPhaseChannel:
    """Unknown phase channel (DBPSK scenario).

    Output: y = exp(j*theta) * s + n, where theta ~ U[0, 2π) is constant per block.

    Parameters
    ----------
    seed : int, optional
    rng : numpy.random.Generator, optional
    """

    def __init__(self, seed=None, rng=None):
        if rng is None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = rng

    def __call__(self, signal, snr_db):
        """Apply unknown phase rotation and AWGN.

        Parameters
        ----------
        signal : numpy.ndarray
            Complex-valued signal (DBPSK constellation).
        snr_db : float

        Returns
        -------
        output : numpy.ndarray
            Complex output.
        """
        # Random phase per block (constant over entire signal)
        theta = self.rng.uniform(0, 2 * np.pi)
        phase_rotation = np.exp(1j * theta)
        phased = phase_rotation * signal

        # Add AWGN (complex)
        sigma = 10 ** (-snr_db / 20.0)
        noise = sigma * (self.rng.standard_normal(signal.size) +
                        1j * self.rng.standard_normal(signal.size)) / np.sqrt(2)

        return phased + noise


class FlatGainChannel:
    """Unknown flat gain channel.

    Output: y = g * x + n, where g ~ U[0.3, 2.0] is constant per block.

    Parameters
    ----------
    gain_min : float, optional
        Minimum gain (default 0.3).
    gain_max : float, optional
        Maximum gain (default 2.0).
    seed : int, optional
    rng : numpy.random.Generator, optional
    """

    def __init__(self, gain_min=0.3, gain_max=2.0, seed=None, rng=None):
        self.gain_min = gain_min
        self.gain_max = gain_max
        if rng is None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = rng

    def __call__(self, signal, snr_db):
        """Apply unknown gain and AWGN.

        Parameters
        ----------
        signal : numpy.ndarray
        snr_db : float

        Returns
        -------
        output : numpy.ndarray
        """
        # Random gain per block
        g = self.rng.uniform(self.gain_min, self.gain_max)
        gained = g * signal

        # Add AWGN
        sigma = 10 ** (-snr_db / 20.0)
        noise = sigma * self.rng.standard_normal(signal.size)

        return gained + noise


class BranchAsymmetryChannel:
    """Branch asymmetry channel (I/Q imbalance).

    For DBPSK input x, output y = a+ if x > 0 else -a-, where a+/a- ~ U[0.6, 1.4].

    Parameters
    ----------
    seed : int, optional
    rng : numpy.random.Generator, optional
    """

    def __init__(self, seed=None, rng=None):
        if rng is None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = rng

    def __call__(self, signal, snr_db):
        """Apply branch asymmetry and AWGN.

        Parameters
        ----------
        signal : numpy.ndarray
            Real-valued BPSK signal.
        snr_db : float

        Returns
        -------
        output : numpy.ndarray
        """
        # Random asymmetry amplitudes per block
        a_plus = self.rng.uniform(0.6, 1.4)
        a_minus = self.rng.uniform(0.6, 1.4)

        # Apply asymmetric gain
        asym = np.where(signal > 0, a_plus, -a_minus)

        # Add AWGN
        sigma = 10 ** (-snr_db / 20.0)
        noise = sigma * self.rng.standard_normal(signal.size)

        return asym + noise


class PowerAmplifierChannel:
    """Rapp-like soft-limiter power amplifier.

    Soft limiting: g = a / (1 + (a/sat)^2)^0.5, preserves phase.

    Parameters
    ----------
    saturation : float, optional
        Saturation level (default 1.2).
    seed : int, optional
    rng : numpy.random.Generator, optional
    """

    def __init__(self, saturation=1.2, seed=None, rng=None):
        self.saturation = saturation
        if rng is None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = rng

    def __call__(self, signal, snr_db=None):
        """Apply soft-limiter PA.

        Note: This is a non-noisy channel (noise added elsewhere in composite).

        Parameters
        ----------
        signal : numpy.ndarray
        snr_db : float, optional
            Unused (for API compatibility).

        Returns
        -------
        output : numpy.ndarray
        """
        a = np.abs(signal)
        g = a / np.sqrt(1 + (a / self.saturation) ** 2)
        phase = np.angle(signal) if np.iscomplexobj(signal) else np.sign(signal)

        if np.iscomplexobj(signal):
            return g * np.exp(1j * phase)
        else:
            return g * np.sign(signal + 1e-12)  # Preserve sign for real signals


class CompositeChannel:
    """Composite cascade of multiple channel effects.

    Chains ISI -> Power Amplifier -> Phase rotation -> AWGN.

    Parameters
    ----------
    isi_taps : array-like, optional
        ISI channel taps (default [1, 0.6, 0.4]).
    pa_sat : float, optional
        PA saturation level (default 1.2).
    include_phase : bool, optional
        Include random phase rotation (default True).
    seed : int, optional
    rng : numpy.random.Generator, optional
    """

    def __init__(self, isi_taps=None, pa_sat=1.2, include_phase=True,
                 seed=None, rng=None):
        if isi_taps is None:
            isi_taps = [1.0, 0.6, 0.4]

        self.isi_taps = np.asarray(isi_taps, dtype=float)
        self.isi_taps /= np.linalg.norm(self.isi_taps)
        self.pa_sat = pa_sat
        self.include_phase = include_phase

        if rng is None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = rng

    def __call__(self, signal, snr_db):
        """Apply composite channel cascade.

        Parameters
        ----------
        signal : numpy.ndarray
        snr_db : float

        Returns
        -------
        output : numpy.ndarray
        """
        # Step 1: ISI
        isi_out = np.convolve(signal, self.isi_taps)[:signal.size]

        # Step 2: Power Amplifier (soft limiter)
        a = np.abs(isi_out)
        pa_out = a / np.sqrt(1 + (a / self.pa_sat) ** 2) * np.sign(isi_out + 1e-12)

        # Step 3: Phase rotation (if complex signal becomes needed)
        if self.include_phase and np.iscomplexobj(signal):
            theta = self.rng.uniform(0, 2 * np.pi)
            pa_out = pa_out * np.exp(1j * theta)

        # Step 4: AWGN
        sigma = 10 ** (-snr_db / 20.0)
        if np.iscomplexobj(pa_out):
            noise = sigma * (self.rng.standard_normal(signal.size) +
                           1j * self.rng.standard_normal(signal.size)) / np.sqrt(2)
        else:
            noise = sigma * self.rng.standard_normal(signal.size)

        return pa_out + noise
