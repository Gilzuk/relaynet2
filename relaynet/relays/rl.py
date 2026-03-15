"""Q-Learning Reinforcement Learning relay."""

import numpy as np

from .base import Relay
from relaynet.modulation.bpsk import bpsk_modulate, bpsk_demodulate
from relaynet.channels.awgn import awgn_channel


class RLRelay(Relay):
    """Reinforcement Learning relay using tabular Q-learning.

    States: discretised received-signal amplitude.
    Actions: 5 processing strategies (scale factors + decode-forward).
    Uses only NumPy — no external ML framework dependency.
    """

    _ACTIONS = [0.5, 1.0, 1.5, 2.0, "df"]  # 4 gain factors + DF action

    def __init__(self, target_power=1.0, num_states=20):
        self.target_power = target_power
        self.num_states = num_states
        self.num_actions = len(self._ACTIONS)
        self.Q = np.zeros((num_states, self.num_actions))
        self.lr = 0.1
        self.gamma = 0.95
        self.epsilon = 0.1
        self.is_trained = False

    def _discretize(self, value):
        norm = (np.clip(value, -3, 3) + 3.0) / 6.0
        return int(norm * (self.num_states - 1))

    def _apply_action(self, signal, action_idx):
        action = self._ACTIONS[action_idx]
        if action == "df":
            bits = bpsk_demodulate(signal)
            clean = bpsk_modulate(bits)
            pwr = np.mean(np.abs(clean) ** 2)
            if pwr > 0:
                clean *= np.sqrt(self.target_power / pwr)
            return clean
        scaled = signal * action
        pwr = np.mean(np.abs(scaled) ** 2)
        if pwr > 0:
            scaled *= np.sqrt(self.target_power / pwr)
        return scaled

    def train(self, training_snrs=None, num_episodes=500, bits_per_episode=1000, seed=None):
        """Train the Q-table via simulated episodes.

        Parameters
        ----------
        training_snrs : list of float, optional
            SNR values used during training. Defaults to [5, 10, 15].
        num_episodes : int
        bits_per_episode : int
        seed : int, optional
        """
        if training_snrs is None:
            training_snrs = [5, 10, 15]
        if seed is not None:
            np.random.seed(seed)

        for episode in range(num_episodes):
            snr = training_snrs[episode % len(training_snrs)]
            bits = np.random.randint(0, 2, bits_per_episode)
            tx = bpsk_modulate(bits)
            rx_relay = awgn_channel(tx, snr)

            for sym_idx in range(len(rx_relay)):
                state = self._discretize(rx_relay[sym_idx])
                # epsilon-greedy
                if np.random.rand() < self.epsilon:
                    a = np.random.randint(self.num_actions)
                else:
                    a = np.argmax(self.Q[state])

                processed = self._apply_action(rx_relay[sym_idx: sym_idx + 1], a)
                rx_dest = awgn_channel(processed, snr)
                rx_bit = bpsk_demodulate(rx_dest)
                reward = 1.0 if rx_bit[0] == bits[sym_idx] else -1.0

                next_state = self._discretize(processed[0])
                self.Q[state, a] += self.lr * (
                    reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state, a]
                )

        self.is_trained = True

    def process(self, received_signal):
        processed = np.zeros_like(received_signal)
        for i, sym in enumerate(received_signal):
            state = self._discretize(sym)
            a = np.argmax(self.Q[state])
            processed[i] = self._apply_action(np.array([sym]), a)[0]

        pwr = np.mean(np.abs(processed) ** 2)
        if pwr > 0:
            processed *= np.sqrt(self.target_power / pwr)
        return processed

    # ------------------------------------------------------------------
    # Weight persistence
    # ------------------------------------------------------------------

    def save_weights(self, path):
        """Save trained Q-table to *path*."""
        from relaynet.utils.torch_compat import save_state
        save_state({
            "type": "RLRelay",
            "Q": self.Q,
            "config": {"num_states": self.num_states},
        }, path)

    def load_weights(self, path):
        """Load Q-table from *path* and mark the relay as trained."""
        from relaynet.utils.torch_compat import load_state
        state = load_state(path)
        self.Q = state["Q"]
        self.is_trained = True
