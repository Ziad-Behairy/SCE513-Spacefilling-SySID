"""
score.py  --  original background score for the cinematic video, synthesised in numpy
(no samples, no third-party audio). A minor, slow cinematic build that follows the cut:

  0-16   dark drone + pad (the problem)
  16-27  pad + soft heartbeat pulse (random data, tension)
  27-40  arpeggio enters, chords brighten (the paper's idea), riser into 40
  40-45  impact, then near-silence with shimmer (the question)
  45-61  build: pad + arp + soft kick (extension E1), hit at the reveal
  61-73  pulses (the learning loop)
  73-82  lighter texture (practical limits)
  82-102 full: pad, bass, kick, arp (robot gimbal), riser + hit at the result
  102-110 resolve on A minor add9, fade out
Output: results/video/score.wav (48 kHz stereo, normalised later by ffmpeg loudnorm)
"""
import os
import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve
from scipy.io import wavfile

SR = 48000
DUR = 110.0
NS = int(SR * DUR)
BPM = 96.0
BEAT = 60.0 / BPM
rng = np.random.default_rng(11)
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "results", "video", "score.wav")


def hz(note):
    names = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}
    n, o = note[:-1], int(note[-1])
    return 440.0 * 2 ** ((names[n] + 12 * (o + 1) - 69) / 12)


def lp(x, fc, order=2):
    return sosfilt(butter(order, fc, "low", fs=SR, output="sos"), x, axis=-1)


def hp(x, fc, order=2):
    return sosfilt(butter(order, fc, "high", fs=SR, output="sos"), x, axis=-1)


def env(n, a, r, sustain=1.0):
    e = np.full(n, sustain)
    na, nr = int(a * SR), int(r * SR)
    if na: e[:na] = np.linspace(0, sustain, na) ** 2
    if nr: e[-nr:] *= np.linspace(1, 0, nr) ** 2
    return e


def place(buf, x, t0, gain=1.0):
    i = int(t0 * SR)
    if i < 0:
        x, i = x[..., -i:], 0
    if i >= buf.shape[-1]:
        return
    n = min(x.shape[-1], buf.shape[-1] - i)
    buf[..., i:i + n] += gain * x[..., :n]


def warm(f, n, harm=10, detune=0.0, phase=0.0):
    """additive saw-like tone (band-limited, 1/k^1.3 roll-off = softer than a saw)"""
    t = np.arange(n) / SR
    y = np.zeros(n)
    for k in range(1, harm + 1):
        if f * k > 9000:
            break
        y += np.sin(2 * np.pi * f * k * (1 + detune) * t + phase * k) / k ** 1.3
    return y


def pad(chord, t0, dur, gain, bright=1600):
    n = int(dur * SR)
    L, R = np.zeros(n), np.zeros(n)
    for note in chord:
        f = hz(note)
        for d, side in [(-0.004, 0), (0.0, 1), (0.004, 2)]:
            y = warm(f, n, 8, d, rng.uniform(0, 6.28))
            if side == 0: L += y
            elif side == 2: R += y
            else: L += 0.7 * y; R += 0.7 * y
    lfo = 1 + 0.08 * np.sin(2 * np.pi * 0.17 * np.arange(n) / SR)
    e = env(n, min(2.5, dur / 3), min(2.5, dur / 3)) * lfo
    st = np.stack([lp(L, bright), lp(R, bright)]) * e
    return gain * st / (3 * len(chord))


def pluck(f, dur=1.2, bright=3500):
    n = int(dur * SR)
    t = np.arange(n) / SR
    y = (np.sin(2 * np.pi * f * t) + 0.35 * np.sin(4 * np.pi * f * t) + 0.12 * np.sin(6 * np.pi * f * t))
    y *= np.exp(-t * 4.0) * (1 - np.exp(-t * 400))
    return lp(y, bright)


def kick(gain=1.0, f0=110, f1=42, dur=0.5):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = f1 + (f0 - f1) * np.exp(-t * 28)
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 7)
    return gain * y


def impact(gain=1.0):
    n = int(3.5 * SR)
    t = np.arange(n) / SR
    f = 28 + 70 * np.exp(-t * 9)
    boom = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 1.6)
    nz = lp(rng.standard_normal(n), 900) * np.exp(-t * 6) * 0.35
    return gain * (boom + nz)


def riser(dur, gain=1.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    k = t / dur
    f = 180 * (12 ** k)
    tone = np.sin(2 * np.pi * np.cumsum(f) / SR) * 0.25
    nz = hp(rng.standard_normal(n), 2500) * 0.25
    nz = lp(nz, 9000)
    e = k ** 2.2
    return gain * (tone + nz) * e


def shimmer(t0, dur, gain):
    n = int(dur * SR)
    t = np.arange(n) / SR
    y = np.zeros((2, n))
    for i, note in enumerate(["A5", "E6", "B5", "C6"]):
        trem = 0.5 + 0.5 * np.sin(2 * np.pi * (0.21 + 0.07 * i) * t + i)
        s = np.sin(2 * np.pi * hz(note) * t) * trem
        y[i % 2] += s
    return gain * y * env(n, 1.5, 2.0) / 4


def reverb_ir(dur=3.8, pre=0.025):
    n = int(dur * SR)
    t = np.arange(n) / SR
    ir = rng.standard_normal((2, n)) * np.exp(-t * 3.0 / dur * 2.3)
    ir = lp(ir, 5500)
    ir[:, :int(pre * SR)] = 0
    return ir / np.sqrt((ir ** 2).sum(axis=1, keepdims=True))


def delay(x, d, fb=0.38, n=5):
    y = x.copy()
    for k in range(1, n):
        s = int(d * k * SR)
        g = fb ** k
        y[0, s:] += g * x[1 if k % 2 else 0, :-s]
        y[1, s:] += g * x[0 if k % 2 else 1, :-s]
    return y


def main():
    dry = np.zeros((2, NS))       # goes to reverb
    direct = np.zeros((2, NS))    # stays dry (low end, kicks)

    # ---------------- drone (A1 + A2), whole piece, swells with the story
    t = np.arange(NS) / SR
    drone_env = np.interp(t, [0, 5, 16, 40, 41, 45, 82, 100, 104, 110], [0, .35, .5, .6, .25, .45, .7, .8, .5, 0])
    drone = (np.sin(2 * np.pi * hz("A1") * t) + 0.3 * np.sin(2 * np.pi * hz("A2") * t + 1)) * drone_env
    direct += 0.16 * np.stack([drone, drone])

    # ---------------- pads (chord, start, duration, gain, brightness)
    prog = [(["A2", "E3", "A3", "C4"], 0.5, 16.5, 0.28, 800),
            (["F2", "C3", "A3", "C4"], 16, 6, 0.55, 1100), (["A2", "E3", "A3", "C4"], 21.5, 6, 0.55, 1100),
            (["C3", "G3", "C4", "E4"], 27, 6.5, 0.6, 1700), (["G2", "D3", "G3", "B3"], 33, 7.5, 0.6, 2000),
            (["A2", "E3", "B3", "C4"], 41, 5, 0.35, 800),
            (["F2", "C3", "A3", "E4"], 45, 4.2, 0.6, 1600), (["C3", "G3", "C4", "E4"], 49, 4.2, 0.6, 1900),
            (["G2", "D3", "B3", "D4"], 53, 4.2, 0.65, 2200), (["A2", "E3", "C4", "E4"], 57, 4.5, 0.7, 2500),
            (["D3", "A3", "D4", "F4"], 61, 6.2, 0.6, 1800), (["F2", "C3", "A3", "C4"], 67, 6.2, 0.6, 1900),
            (["C3", "G3", "C4", "E4"], 73, 4.7, 0.5, 2200), (["G2", "D3", "G3", "B3"], 77.5, 5, 0.5, 2200),
            (["A2", "E3", "A3", "C4"], 82, 5.2, 0.7, 2000), (["F2", "C3", "A3", "C4"], 87, 5.2, 0.7, 2300),
            (["C3", "G3", "C4", "E4"], 92, 5.2, 0.75, 2600), (["G2", "D3", "B3", "D4"], 97, 5.2, 0.8, 3000),
            (["A2", "E3", "B3", "C4", "E4"], 102, 8, 0.7, 2000)]
    for ch, t0, d, g, b in prog:
        place(dry, pad(ch, t0, d + 1.5, g, b), t0)

    # ---------------- bass under the build sections
    for ch, t0, d, g, b in prog:
        if 45 <= t0 < 102 and t0 != 41:
            n = int(d * SR)
            f = hz(ch[0]) / 2
            y = lp(warm(f, n, 4), 300) * env(n, 0.3, 0.8)
            place(direct, np.stack([y, y]) * 0.16, t0)

    # ---------------- heartbeat pulse 16-27 (tension)
    for k in range(int((27 - 16.5) / (BEAT * 2))):
        tb = 16.5 + k * BEAT * 2
        place(direct, np.stack([kick(0.22, 80, 38, 0.6)] * 2), tb)
        place(direct, np.stack([kick(0.12, 70, 36, 0.5)] * 2), tb + 0.28)

    # ---------------- arpeggios (8th notes) with ping-pong delay
    arp = np.zeros((2, NS))
    def arpeggio(notes, t0, t1, gain, step=BEAT / 2):
        k, tt = 0, t0
        while tt < t1:
            f = hz(notes[k % len(notes)])
            y = pluck(f)
            pan = 0.5 + 0.35 * np.sin(k * 1.3)
            place(arp, np.stack([y * (1 - pan), y * pan]) * gain, tt)
            k += 1; tt += step
    arpeggio(["A3", "C4", "E4", "B4", "C5", "E4"], 28.0, 33.0, 0.22)
    arpeggio(["G3", "B3", "D4", "A4", "B4", "D4"], 33.0, 39.6, 0.24)
    arpeggio(["F3", "A3", "C4", "E4", "G4", "C4"], 45.0, 49.0, 0.2)
    arpeggio(["C4", "E4", "G4", "B4", "C5", "G4"], 49.0, 53.0, 0.22)
    arpeggio(["G3", "B3", "D4", "A4", "B4", "D4"], 53.0, 57.0, 0.24)
    arpeggio(["A3", "C4", "E4", "B4", "C5", "E4"], 57.0, 61.0, 0.26)
    arpeggio(["D4", "F4", "A4", "C5"], 61.0, 73.0, 0.16, BEAT)
    arpeggio(["C4", "E4", "G4", "B4"], 73.0, 82.0, 0.14, BEAT / 2)
    arpeggio(["A3", "C4", "E4", "B4", "C5", "E4"], 82.0, 87.0, 0.24)
    arpeggio(["F3", "A3", "C4", "E4", "G4", "C4"], 87.0, 92.0, 0.25)
    arpeggio(["C4", "E4", "G4", "B4", "C5", "G4"], 92.0, 97.0, 0.27)
    arpeggio(["G3", "B3", "D4", "A4", "B4", "D4"], 97.0, 101.5, 0.28)
    dry += delay(arp, BEAT * 0.75)

    # ---------------- soft kicks on the beat in the build sections
    for a, b, g in [(49.0, 61.0, 0.2), (87.0, 101.6, 0.26)]:
        tb = a
        while tb < b:
            place(direct, np.stack([kick(g)] * 2), tb)
            tb += BEAT

    # ---------------- risers, impacts, shimmer
    place(dry, np.stack([riser(3.5, 0.18)] * 2), 36.5)
    place(direct, np.stack([impact(0.55)] * 2), 40.0)
    place(dry, np.stack([impact(0.25)] * 2), 40.0)
    place(dry, shimmer(40.5, 5.0, 0.10), 40.5)
    place(dry, np.stack([riser(2.0, 0.12)] * 2), 54.2)
    place(direct, np.stack([impact(0.35)] * 2), 56.2)                 # E1 reveal (8.7x)
    place(direct, np.stack([impact(0.3)] * 2), 70.6)                   # E2 reveal (6.7x)
    place(dry, np.stack([riser(3.0, 0.16)] * 2), 95.0)
    place(direct, np.stack([impact(0.5)] * 2), 98.0)                   # gimbal result
    place(dry, np.stack([impact(0.2)] * 2), 98.0)
    place(dry, shimmer(102.0, 8.0, 0.12), 102.0)
    for tt in [0.5, 6.0, 16.0, 27.0, 45.0, 61.0, 73.0, 82.0, 102.0]:  # gentle whoosh at each cut
        n = int(1.2 * SR)
        w = lp(hp(rng.standard_normal(n), 400), 3000) * np.sin(np.linspace(0, np.pi, n)) ** 2 * 0.03
        place(dry, np.stack([w, w[::-1]]), tt - 0.6)

    # ---------------- mix
    wet = np.stack([fftconvolve(dry[i], reverb_ir()[i])[:NS] for i in range(2)])
    mix = 0.55 * dry + 0.45 * wet + direct
    mix = hp(mix, 28)
    fade = np.interp(np.arange(NS) / SR, [0, 0.4, DUR - 6, DUR], [0, 1, 1, 0])
    mix = np.tanh(1.2 * mix * fade) / 1.2
    mix /= np.abs(mix).max() / 0.9
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    wavfile.write(OUT, SR, (mix.T * 32767).astype(np.int16))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
