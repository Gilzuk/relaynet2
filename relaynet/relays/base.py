"""Base Relay class."""


class Relay:
    """Abstract base class for all relay strategies."""

    def process(self, received_signal):
        """Process the received signal and return the forwarded signal.

        Parameters
        ----------
        received_signal : numpy.ndarray
            Signal received from the source hop.

        Returns
        -------
        forwarded_signal : numpy.ndarray
            Processed signal to forward to the destination hop.
        """
        raise NotImplementedError("Subclasses must implement process()")
