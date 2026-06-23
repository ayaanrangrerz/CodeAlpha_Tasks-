try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
except Exception:
    Sequential = None
    LSTM = None
    Dense = None

import sys
try:
    import numpy as np
except Exception:
    np = None
import random
try:
    import winsound
except Exception:
    winsound = None


class MockModel:
    """A tiny mock model that simulates a Keras predict() for demos without TensorFlow."""
    def __init__(self, output_dim=50, seed=0):
        self.output_dim = output_dim
        if np is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = random.Random(seed)

    def predict(self, x):
        # x: (batch, seq_len, features) -> return (batch, output_dim) probabilities
        batch = 1
        if np is not None:
            logits = self.rng.standard_normal((batch, self.output_dim))
            exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exps / np.sum(exps, axis=1, keepdims=True)
            return probs
        else:
            # use python random to create a probability vector
            probs = []
            for _ in range(batch):
                vals = [self.rng.gauss(0, 1) for _ in range(self.output_dim)]
                maxv = max(vals)
                exps = [__import__('math').exp(v - maxv) for v in vals]
                s = sum(exps)
                probs.append([e / s for e in exps])
            return probs


def build_model(seq_length=100, feature_dim=1, lstm_units=128, dense_units=128, output_dim=50):
    if Sequential is None:
        # return a lightweight mock model so script can run without TensorFlow
        return MockModel(output_dim=output_dim)

    model = Sequential()
    model.add(LSTM(lstm_units, input_shape=(seq_length, feature_dim)))
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dense(output_dim, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam'
    )

    return model


def demo_generate(model, steps=8, seq_length=100, feature_dim=1):
    # create a fake input and run predict to show it's working
    if np is not None:
        x = np.zeros((1, seq_length, feature_dim), dtype=float)
    else:
        x = None
    probs = model.predict(x)
    # pick top tokens as a toy 'melody'
    if np is not None:
        top_idx = np.argsort(probs[0])[-steps:][::-1]
        print('Generated token indices (toy melody):', top_idx.tolist())
    else:
        # probs is a Python list of lists
        top_idx = sorted(range(len(probs[0])), key=lambda i: probs[0][i], reverse=True)[:steps]
        print('Generated token indices (toy melody):', top_idx)
    return top_idx


def play_melody(indices, base_freq=220, duration_ms=300):
    """Simple playback: map token index to semitone steps from base_freq and play using winsound.Beep on Windows."""
    freqs = [int(base_freq * (2 ** (i / 12.0))) for i in indices]
    if winsound is not None:
        for f in freqs:
            winsound.Beep(f, duration_ms)
    else:
        print('winsound not available — would play frequencies:', freqs)


if __name__ == '__main__':
    model = build_model()
    if isinstance(model, MockModel):
        print('TensorFlow not found — using lightweight mock model for demo.')
    else:
        print('Built real Keras model.')
    indices = demo_generate(model)
    if '--play' in sys.argv:
        play_melody(indices)