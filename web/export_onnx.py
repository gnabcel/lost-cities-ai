"""Exports a checkpoint to web/model.onnx for the browser version and checks that
onnxruntime matches PyTorch on real game states.

    .venv/bin/pip install onnx onnxruntime
    .venv/bin/python web/export_onnx.py [models/champion.pt]
"""

import os
import random
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lost_cities.engine import GameState  # noqa: E402
from lost_cities.ppo import PolicyBot, features_batch  # noqa: E402


class Export(torch.nn.Module):
    """Network without the legal-move mask (applied in JavaScript): obs -> logits, value."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, obs):
        h = self.net.body(obs)
        return self.net.pi(h), self.net.v(h).squeeze(-1)


def main():
    import onnxruntime as ort

    ckpt = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "models", "champion.pt")
    out = os.path.join(ROOT, "web", "model.onnx")
    bot = PolicyBot.load(ckpt, device="cpu")
    assert bot.net.feat == 2, "web/engine.js computes the version-2 features only"
    model = Export(bot.net).eval()
    torch.onnx.export(model, (torch.zeros(1, 429),), out, input_names=["obs"], output_names=["logits", "value"],
                      dynamic_axes={"obs": {0: "batch"}, "logits": {0: "batch"}, "value": {0: "batch"}},
                      opset_version=17, dynamo=False)

    rng, states = random.Random(0), []
    sampler = PolicyBot(bot.net, "cpu", greedy=False)
    for g in range(30):
        s = GameState.new_game(rng, starting_player=g % 2)
        while not s.done:
            states.append(s.clone())
            s.step(sampler.act(s, rng))
    x = features_batch(states, [s.current for s in states], 2)
    logits, value = ort.InferenceSession(out).run(None, {"obs": x})
    with torch.no_grad():
        lt, vt = model(torch.from_numpy(x))
    print(f"{ckpt} -> {out} ({os.path.getsize(out) // 1024} KB) · {len(states)} states · "
          f"max diff logits {np.abs(logits - lt.numpy()).max():.1e}, value {np.abs(value - vt.numpy()).max():.1e}")


if __name__ == "__main__":
    main()
