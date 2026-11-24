import numpy as np, wave
SR = 48000
def compose(OUTF, BPM, CH, ROOT, BARS=16, SEED=7, PAT=(0,1,2,1,2,0,1,2,0,2,1,2,0,1,2,1), PLUCK=0.18, STABS=(0,1.5,3), PADL=0.06, OCT=(), HATS=1.0):
    BEAT = 60 / BPM; BAR = 4 * BEAT
    N = int(SR * (BARS * BAR + 2.5))
    L = np.zeros(N); R = np.zeros(N); DL = np.zeros(N); DR = np.zeros(N)
    rng = np.random.default_rng(SEED)
    def midi(n): return 440 * 2 ** ((n - 69) / 12)
    def add(sig, t, pan=0.0, gain=1.0, drum=False):
        i = int(t * SR); j = min(N, i + len(sig)); s = sig[:j - i] * gain
        A, B = (DL, DR) if drum else (L, R)
        A[i:j] += s * (1 - max(0, pan)); B[i:j] += s * (1 + min(0, pan))
    def lp(x, a):  # one-pole lowpass
        y = np.empty_like(x); acc = 0.0
        for k in range(len(x)): acc += a * (x[k] - acc); y[k] = acc
        return y
    def saw(f, dur, det=0.0):
        t = np.arange(int(dur * SR)) / SR; f = f * (1 + det)
        K = int(min(24, 9000 / f)); ph = 2 * np.pi * f * t
        return sum(np.sin(k * ph) / k for k in range(1, K + 1)) * 0.6
    def env(n, a, d, s=0.0, r=0.05, dur=None):
        t = np.arange(n) / SR; e = np.minimum(1, t / max(a, 1e-4))
        e *= np.where(t > a, s + (1 - s) * np.exp(-(t - a) / d), 1)
        rel = int(r * SR); e[-rel:] *= np.linspace(1, 0, rel); return e
    # kick
    kt = np.arange(int(0.45 * SR)) / SR
    kick = np.sin(2 * np.pi * (45 * kt + (150 - 45) * 0.035 * (1 - np.exp(-kt / 0.035)))) * np.exp(-kt / 0.18)
    kick += 0.3 * np.exp(-kt / 0.004) * rng.standard_normal(len(kt)) * 0.3
    # clap
    cn = rng.standard_normal(int(0.25 * SR)); clap = lp(cn, 0.35) - lp(cn, 0.05)
    ce = np.zeros(len(cn))
    for o in (0, 0.011, 0.022): i = int(o * SR); ce[i:] += np.exp(-np.arange(len(cn) - i) / SR / (0.012 if o < 0.02 else 0.09))
    clap *= ce
    # hats
    hn = rng.standard_normal(int(0.2 * SR)); hat = hn - lp(hn, 0.6)
    chat = hat[:int(0.05 * SR)] * np.exp(-np.arange(int(0.05 * SR)) / SR / 0.012)
    ohat = hat * np.exp(-np.arange(len(hat)) / SR / 0.07)
    duck = np.ones(N)  # sidechain curve
    for bar in range(BARS):
        t0 = bar * BAR; c = bar % 4; intro = bar == 0
        # pad (every bar)
        pad = sum(lp(saw(midi(n), BAR + 0.3, d), 0.04) for n in CH[c] for d in (-0.004, 0.004))
        pad *= env(len(pad), 0.05 if not intro else 1.2, 9, 0.8, 0.3)
        add(pad, t0, -0.2, PADL); add(pad, t0, 0.2, PADL)
        for b in range(4):
            tb = t0 + b * BEAT
            if not intro:
                add(kick, tb, 0, 0.9, True)
                i = int(tb * SR); k = min(N, i + int(0.22 * SR)); duck[i:k] = np.minimum(duck[i:k], 0.35 + 0.65 * np.linspace(0, 1, k - i) ** 0.7)
                if b in (1, 3): add(clap, tb, 0, 0.30, True)
                add(ohat, tb + BEAT / 2, 0.15, 0.045 * HATS, True)
            for h in range(2): add(chat, tb + h * BEAT / 2, -0.25, (0.05 if not intro else 0.03) * HATS, True)
            if not intro:  # offbeat bass
                for h in (0.5,):
                    f = midi(ROOT[c]); bs = lp(saw(f, BEAT * 0.45) + 0.5 * saw(f / 2, BEAT * 0.45), 0.08)
                    add(bs * env(len(bs), 0.005, 0.12, 0.3, 0.03), tb + h * BEAT, 0, 0.35)
                add(lp(np.sin(2 * np.pi * midi(ROOT[c]) / 2 * np.arange(int(BEAT * SR)) / SR), 0.5) * env(int(BEAT * SR), 0.005, 0.2, 0.2, 0.05), tb, 0, 0.25)
        # 16th pluck arpeggio
        pat = PAT
        for s in range(16):
            if intro and s < 8: continue
            n = CH[c][pat[s]] + (24 if OCT and s % 8 in OCT else 12)
            p = lp(saw(midi(n), 0.22) + 0.6 * saw(midi(n), 0.22, 0.006), PLUCK)
            add(p * env(len(p), 0.002, 0.07, 0.0, 0.02), t0 + s * BEAT / 4, 0.35 if s % 2 else -0.35, 0.10)
        # piano-ish stabs on 1 and the "and" of 2
        if not intro:
            for off in [o * BEAT for o in STABS]:
                st = sum(np.sin(2 * np.pi * midi(n + 12) * np.arange(int(0.6 * SR)) / SR) * (1 + 0.3 * np.sin(2 * np.pi * midi(n + 24) * np.arange(int(0.6 * SR)) / SR)) for n in CH[c])
                add(st * env(len(st), 0.003, 0.18, 0.0, 0.05), t0 + off, 0, 0.07)
    # riser into bar 1
    rs = rng.standard_normal(int(BAR * SR)); rs = lp(rs, 0.25) * np.linspace(0, 1, len(rs)) ** 2
    add(rs, 0, 0, 0.06, True)
    # final ring chord
    endt = BARS * BAR
    ring = sum(lp(saw(midi(n), 2.5, d), 0.05) for n in CH[2] + [CH[2][0] + 12] for d in (-0.004, 0.004)) * env(int(2.5 * SR), 0.01, 0.9, 0, 0.3)
    add(ring, endt, 0, 0.09); add(kick, endt, 0, 0.9, True); add(clap, endt, 0, 0.3, True)
    # sidechain on everything except drums is approximated by ducking the whole mix lightly
    mix = np.stack([L, R], 1) * (0.45 + 0.55 * duck)[:, None] + np.stack([DL, DR], 1)
    mix = np.tanh(mix * 1.4) / np.tanh(1.4)
    mix /= np.abs(mix).max() / 0.89
    with wave.open(OUTF, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((mix * 32767).astype("<i2").tobytes())

