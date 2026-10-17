# live_detect.py -- drone/no-drone on a live mic feed. records one second
# at a time (same length the model was trained on, no overlap, so every
# second gets judged exactly once), runs it through the saved checkpoint
# from save_model.py, applies the recalibration correction, prints the call.

import sys

import librosa
import numpy as np
import sounddevice as sd
import torch

from features import SR, spectrogram_from_array
from train import DroneCNN

CAPTURE_SR = 48000  # almost every mic supports this natively, 16khz often doesnt


def load_checkpoint(path):
    ckpt = torch.load(path, weights_only=False)
    model = DroneCNN()
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["mean"], ckpt["std"], ckpt["calib_coef"], ckpt["calib_intercept"]


# same correction recalibrate.py fits with sklearn, written out by hand so
# this file doesnt need sklearn installed just to run a live demo
def calibrate(raw_prob, coef, intercept):
    logit = coef * raw_prob + intercept
    return 1 / (1 + np.exp(-logit))


@torch.no_grad()
def predict(model, chunk, mean, std):
    spec = spectrogram_from_array(chunk)
    spec = (spec.astype(np.float32) - mean) / std
    spec = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0)  # batch dim, channel dim
    return torch.sigmoid(model(spec)).item()


def main(checkpoint_path, threshold):
    model, mean, std, coef, intercept = load_checkpoint(checkpoint_path)
    print(f"listening on {sd.query_devices(kind='input')['name']}, ctrl-c to stop")

    try:
        while True:
            chunk = sd.rec(CAPTURE_SR, samplerate=CAPTURE_SR, channels=1, dtype="float32")
            sd.wait()
            chunk = librosa.resample(chunk[:, 0], orig_sr=CAPTURE_SR, target_sr=SR)

            raw_prob = predict(model, chunk, mean, std)
            prob = calibrate(raw_prob, coef, intercept)
            call = "DRONE" if prob >= threshold else "no drone"
            print(f"  {call:10s}  confidence {prob:.1%}  (raw model output {raw_prob:.1%})")
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "model.pt"
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    main(checkpoint_path, threshold)
