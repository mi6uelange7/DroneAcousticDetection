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


# usb device indices arent stable between runs, portaudio can re-number them
# depending on whats plugged in that moment. name matching doesnt have that
# problem, "snowball" always means the snowball
def find_device(name_substring):
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and name_substring.lower() in d["name"].lower():
            return i
    sys.exit(f"no input device matching {name_substring!r}, run sd.query_devices() to see whats available")


def main(checkpoint_path, threshold, device):
    model, mean, std, coef, intercept = load_checkpoint(checkpoint_path)
    sd.default.device = (device, None)
    device_info = sd.query_devices(device)
    # capture at whatever rate the device actually natively runs, not a
    # hardcoded guess. a usb mic over raw alsa is often 44100, the same mic
    # routed through the system audio layer is often 48000, found that out
    # the hard way testing this on a few different devices
    capture_sr = int(device_info["default_samplerate"])
    print(f"listening on {device_info['name']} at {capture_sr}hz, ctrl-c to stop")

    try:
        while True:
            # a transient audio-subsystem hiccup shouldnt kill a live demo
            # over one bad second, log it and keep going instead
            try:
                chunk = sd.rec(capture_sr, samplerate=capture_sr, channels=1, dtype="float32")
                sd.wait()
            except sd.PortAudioError as e:
                print(f"  (mic hiccup, skipping this second: {e})")
                continue

            chunk = librosa.resample(chunk[:, 0], orig_sr=capture_sr, target_sr=SR)

            raw_prob = predict(model, chunk, mean, std)
            prob = calibrate(raw_prob, coef, intercept)
            call = "DRONE" if prob >= threshold else "no drone"
            print(f"  {call:10s}  confidence {prob:.1%}  (raw model output {raw_prob:.1%})")
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "model.pt"
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    # a bare number still works as a raw index, anything else is a name to
    # search for, e.g. "snowball"
    device_arg = sys.argv[3] if len(sys.argv) > 3 else None
    if device_arg is None:
        device = sd.default.device[0]
    elif device_arg.isdigit():
        device = int(device_arg)
    else:
        device = find_device(device_arg)
    main(checkpoint_path, threshold, device)
