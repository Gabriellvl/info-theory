import copy
import math
import os
import sys
import wave

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from src.AudioCD import AudioCD

WAV_FILE = os.path.join(os.path.dirname(__file__), "Hallelujah.wav")
T_SCRATCH = 600000
SCRATCH_OFFSET = 30000
MAX_INTERP = 8
SCRATCH_WIDTHS = [100, 3000, 10000]
CONFIGS_ALL = [0, 1, 2, 3]
CONFIGS_BER = [1, 2, 3]
ERROR_PROBS = np.logspace(-1 - math.log10(2), -3, 10)

CONFIG_LABELS = {
    0: "Config 0\n(No EC)",
    1: "Config 1\n(Full CIRC)",
    2: "Config 2\n(Concat RS,\nno interleave)",
    3: "Config 3\n(Single RS32,24)",
}
CONFIG_COLORS = {0: "#999999", 1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c"}


def load_audiofile(wav_path):
    wave_object = wave.open(wav_path, "rb")
    Fs = wave_object.getframerate()
    nch = wave_object.getnchannels()
    depth = wave_object.getsampwidth()
    wave_object.setpos(0)
    sdata = wave_object.readframes(wave_object.getnframes())
    typ = {1: np.int8, 2: np.int16, 4: np.int32}[depth]
    data = np.frombuffer(sdata, dtype=typ) / (2**15)
    ch_1 = data[0::nch]
    ch_2 = data[1::nch]
    return np.transpose(np.vstack((ch_1, ch_2))), Fs


def build_cd(config, audiofile, Fs):
    cd = AudioCD(Fs, config, MAX_INTERP)
    cd.writeCd(audiofile)
    return cd


def run_scratch_trial(cd, scratch_width):
    cd.cd_bits = copy.deepcopy(cd.cd_bits_original)
    n_bits = cd.cd_bits_original.size
    for i in range(math.floor(n_bits / T_SCRATCH)):
        cd.scratchCd(scratch_width, SCRATCH_OFFSET + i * T_SCRATCH)
    _, flags = cd.readCd()
    total = flags.size
    p_erasure = np.sum(flags != 0) / total
    p_fail = np.sum(flags == -1) / total
    return p_erasure, p_fail


def run_biterror_trial(cd, p):
    cd.cd_bits = copy.deepcopy(cd.cd_bits_original)
    # bitErrorsCd has a bug (np.random.rand receives a tuple), so we inline it:
    noise = np.random.random(cd.cd_bits.shape) < p
    cd.cd_bits = np.bitwise_xor(cd.cd_bits, noise.astype(int))
    _, flags = cd.readCd()
    total = flags.size
    p_erasure = np.sum(flags != 0) / total
    p_fail = np.sum(flags == -1) / total
    return p_erasure, p_fail


def part_a_scratch(audiofile, Fs):
    print("Building CDs for all 4 configurations...")
    cds = {}
    for cfg in CONFIGS_ALL:
        print(f"  Encoding config {cfg}...")
        cds[cfg] = build_cd(cfg, audiofile, Fs)

    results = {cfg: {} for cfg in CONFIGS_ALL}
    for cfg in CONFIGS_ALL:
        for width in SCRATCH_WIDTHS:
            print(f"  [config={cfg}, scratch={width:>6} bits]", end="", flush=True)
            p_e, p_f = run_scratch_trial(cds[cfg], width)
            results[cfg][width] = (p_e, p_f)
            print(f"  P(erasure)={p_e:.4f}, P(fail)={p_f:.4f}")

    # Grouped bar chart
    n_widths = len(SCRATCH_WIDTHS)
    n_configs = len(CONFIGS_ALL)
    x = np.arange(n_widths)
    bar_width = 0.18
    offsets = np.linspace(-(n_configs - 1) / 2, (n_configs - 1) / 2, n_configs) * bar_width

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Part (a): Scratch Performance — Periodic scratch, period=600,000 bits", fontsize=13)

    FLOOR = 1e-5  # sentinel for zero values on log scale

    for ax_idx, (metric_key, title, ylabel) in enumerate([
        (0, "P(erasure flag raised before interpolation)", "Probability of erasure"),
        (1, "P(failed interpolation)", "Probability of failed interpolation"),
    ]):
        ax = axes[ax_idx]
        for i, cfg in enumerate(CONFIGS_ALL):
            raw = [results[cfg][w][metric_key] for w in SCRATCH_WIDTHS]
            # Replace exact zeros with FLOOR so they appear on log scale; mark with hatching
            plot_vals = [v if v > 0 else FLOOR for v in raw]
            hatches = ["////" if v == 0 else "" for v in raw]
            for j, (xpos, h, hatch) in enumerate(zip(x + offsets[i], plot_vals, hatches)):
                ax.bar(xpos, h, bar_width, label=CONFIG_LABELS[cfg] if j == 0 else "",
                       color=CONFIG_COLORS[cfg], edgecolor="white", hatch=hatch, alpha=0.85)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Scratch width (bits)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([str(w) for w in SCRATCH_WIDTHS])
        ax.set_yscale("log")
        ax.set_ylim(FLOOR * 0.5, 1.0)
        ax.legend(fontsize=8)
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.text(0.01, 0.01, "Hatched bars = zero (no errors detected/reported)",
                transform=ax.transAxes, fontsize=7, color="grey", va="bottom")

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "q5a_scratch_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


def part_b_biterrors(audiofile, Fs):
    print("Building CDs for configs 1, 2, 3...")
    cds = {}
    for cfg in CONFIGS_BER:
        print(f"  Encoding config {cfg}...")
        cds[cfg] = build_cd(cfg, audiofile, Fs)

    results_erasure = {cfg: [] for cfg in CONFIGS_BER}
    results_fail = {cfg: [] for cfg in CONFIGS_BER}

    for cfg in CONFIGS_BER:
        print(f"  Config {cfg}:")
        for p in ERROR_PROBS:
            print(f"    p={p:.5f}", end="", flush=True)
            p_e, p_f = run_biterror_trial(cds[cfg], p)
            results_erasure[cfg].append(p_e)
            results_fail[cfg].append(p_f)
            print(f"  P(erasure)={p_e:.4f}, P(fail)={p_f:.4f}")

    markers = {1: "o", 2: "s", 3: "^"}
    linestyles = {1: "-", 2: "--", 3: "-."}
    cfg_display = {1: "Config 1 (Full CIRC)", 2: "Config 2 (Concat RS, no interleave)", 3: "Config 3 (Single RS32,24)"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Part (b): Random Bit Error Performance", fontsize=13)

    for ax_idx, (data_dict, title, ylabel) in enumerate([
        (results_erasure, "P(erasure flagged before interpolation)", "P(erasure)"),
        (results_fail, "P(failed interpolation)", "P(failed interpolation)"),
    ]):
        ax = axes[ax_idx]
        for cfg in CONFIGS_BER:
            vals = np.array(data_dict[cfg])
            # Points with probability exactly 0 are dropped silently on log scale
            ax.plot(ERROR_PROBS, vals,
                    marker=markers[cfg], linestyle=linestyles[cfg],
                    color=CONFIG_COLORS[cfg], label=cfg_display[cfg], linewidth=1.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Bit error probability p")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        ax.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "q5b_biterror_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


def main():
    print("Loading WAV file...")
    audiofile, Fs = load_audiofile(WAV_FILE)

    print("\n=== Part (a): Scratch test ===")
    part_a_scratch(audiofile, Fs)

    print("\n=== Part (b): Random bit error sweep ===")
    part_b_biterrors(audiofile, Fs)

    print("\nDone.")


if __name__ == "__main__":
    main()
