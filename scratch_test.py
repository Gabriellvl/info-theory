import copy
import math
import wave
import numpy as np
import csv
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from src.AudioCD import AudioCD

def run_test(scratch_length, cd_bits_original, cd, interpolation_shape):
    cd.cd_bits = copy.deepcopy(cd_bits_original)
    cd.number_of_errors = 0

    T_scratch = 600000
    for i in range(math.floor(cd_bits_original.size / T_scratch)):
        cd.scratchCd(scratch_length, 30000 + i * T_scratch)

    out, interpolation_flags = cd.readCd()

    n_erasure = int(np.sum(interpolation_flags != 0))
    n_failed = int(np.sum(interpolation_flags == -1))
    n_undetected = int(np.sum(
        out[interpolation_flags == 0] != cd.scaled_quantized_padded_original[interpolation_flags == 0]
    ))

    return n_erasure, n_failed, n_undetected


def main():
    print("Loading WAV file...")
    wave_object = wave.open("Hallelujah.wav", "rb")
    Fs = wave_object.getframerate()
    nch = wave_object.getnchannels()
    depth = wave_object.getsampwidth()
    wave_object.setpos(0)
    sdata = wave_object.readframes(wave_object.getnframes())
    typ = {1: np.int8, 2: np.int16, 4: np.int32}.get(depth)
    data = np.frombuffer(sdata, dtype=typ) / (2**15)
    ch_1 = data[0::nch]
    ch_2 = data[1::nch]
    audiofile = np.transpose(np.vstack((ch_1, ch_2)))

    print("Writing CD (done once)...")
    cd = AudioCD(Fs, 1, 8)
    cd.writeCd(audiofile)
    cd_bits_original = copy.deepcopy(cd.cd_bits)

    # Test scratch lengths: coarse pass 1000-6000 step 100, fine pass 3000-4000 step 25
    coarse = list(range(1000, 6001, 100))
    fine = list(range(3000, 4001, 25))
    scratch_lengths = sorted(set(coarse + fine))

    results = []
    total = len(scratch_lengths)
    print(f"Running {total} scratch length tests...")

    for idx, sl in enumerate(scratch_lengths):
        n_erasure, n_failed, n_undetected = run_test(sl, cd_bits_original, cd, None)
        results.append((sl, n_erasure, n_failed, n_undetected))
        print(f"[{idx+1}/{total}] scratch={sl}: erasures={n_erasure}, failed={n_failed}, undetected={n_undetected}")

    out_path = "scratch_test_results.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scratch_length_bits", "samples_with_erasure_flags", "samples_with_failed_interpolation", "undetected_errors"])
        writer.writerows(results)

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
