"""Finite-law projection using only public counts, plant status and game clocks."""
import numpy as np

MARKS = np.arange(1, 512)
HORIZON = 20
END = 256
SLACK = .02


def permitted(public):
    planted = public[:, 3, None] > .5
    defenders = np.rint(5 * public[:, 1, None]).astype(int)
    attackers = np.rint(5 * public[:, 2, None]).astype(int)
    plant = (MARKS & 48) != 0
    attack_death = (MARKS & 64) != 0
    defense_death = (MARKS & 128) != 0
    terminal = (MARKS & END) != 0
    conflict = ((attackers < 2) & ((MARKS & 15) != 0)) | (planted & plant)
    conflict |= (attackers == 0) & (attack_death | plant)
    conflict |= (defenders == 0) & defense_death
    conflict |= ~terminal & ((defense_death & (defenders == 1)) |
                             (attack_death & (attackers == 1) & ~planted & ~plant))
    return ~conflict


def duration_model(rounds):
    samples = [[], []]
    for r in rounds:
        if r['events'][-1, 1] != END:
            continue
        now, u = r['time'][-1], r['pub'][-1]
        armed = bool(u[3] > .5)
        time_left = 40. - 45. * u[4] if armed else 115. - 120. * u[0]
        dt = r['event_seconds'][-1] - (float(now) + time_left)
        if abs(dt) <= 2.5:
            samples[int(armed)].append(round(float(dt) * 64) / 64)
    return [np.unique(v if v else [0.], return_counts=True) for v in samples], list(map(len, samples))


def clock_kernel(public, fitted):
    """Row stochastic over the 20 time bins and tail: entries give deadline probability."""
    armed = public[:, 3] > .5
    remaining = np.where(armed, 40. - public[:, 4].astype(float)*45.,
                         115. - public[:, 0].astype(float)*120.)
    clock = np.zeros((len(public), HORIZON + 1))
    for state, (offsets, counts) in enumerate(fitted):
        rows = np.flatnonzero(armed == bool(state))
        for offset, count in zip(offsets, counts):
            delay = remaining[rows] + offset
            # Clock reconstruction uses float32 public values; exact integer boundaries
            # in the native 64-tick demo should stay in the right-closed bin.
            bins = np.ceil(delay - 1.e-5).astype(int) - 1
            bins[(bins < 0) | (bins >= HORIZON)] = HORIZON
            clock[rows, bins] += count / counts.sum()
    return clock


def project(law, tail, public, fitted, rules=True, clock=True):
    """law[n,t,m-1] and tail[n] are normalized. Keep a full-support escape component."""
    source = law
    out = law.copy()
    end = tail.copy()
    if rules:
        allowed = permitted(public)
        out *= allowed[:, None, :]
        mass = out.sum(axis=2)
        if np.any(mass <= 0):
            raise ValueError('The base law requires positive support at every time')
        out *= (source.sum(axis=2) / mass)[:, :, None]
    if clock:
        kernel = clock_kernel(public, fitted)
        # At equality retain the original mark (including a possible native-tick tie).
        unexpired = 1 - np.cumsum(kernel[:, :-1], axis=1) + kernel[:, :-1]
        time_mass = out.sum(axis=2)
        after = np.cumsum(time_mass[:, ::-1], axis=1)[:, ::-1] - time_mass + tail[:, None]
        out *= unexpired[:, :, None]
        out[:, :, END - 1] += kernel[:, :-1] * after
        end *= kernel[:, -1]
    out = SLACK * source + (1.-SLACK) * out
    end = SLACK * tail + (1.-SLACK) * end
    return out, end
