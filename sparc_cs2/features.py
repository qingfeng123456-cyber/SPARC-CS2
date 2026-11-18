"""Causal feature and cutoff transforms from the frozen implementation."""
import numpy as np

def public_features(pub, players):
    """All transforms read only the current emitted CT observations."""
    u = np.asarray(pub)
    x = np.asarray(players)
    elapsed, planted = 120*u[:, 0], u[:, 3]
    age = 45*u[:, 4]
    alive_ct, alive_t = 5*u[:, 1], 5*u[:, 2]
    exposed = np.maximum(0, 5*u[:, 18])
    ct, foe = x[:, :5, :], x[:, 5:, :]
    dxy = ct[:, :, None, :2] - foe[:, None, :, :2]
    dist = np.sqrt(np.square(dxy).sum(-1)) * 4096
    known = (ct[:, :, None, 3] > .5) & (foe[:, None, :, 31] > .5)
    exposure = known & (foe[:, None, :, 4] > .5)
    close = np.exp(-dist / 850)
    fresh = np.exp(-30*foe[:, None, :, 5] / 5)
    features = [
        (1-planted)*np.minimum(elapsed, 100)/35,
        (1-planted)*np.exp(-np.maximum(115-elapsed, 0)/9)*1.6,
        planted*3, planted*age/11, planted*np.exp(-np.maximum(40-age, 0)/6),
        .45*alive_ct, .45*alive_t, .85*(alive_ct==1), .85*(alive_t==1),
        1.5*np.exp(-30*u[:,5]/4), 1.5*np.exp(-30*u[:,6]/4),
        (exposed>0)*2., np.minimum(exposed, 3)*.32,
        1.2*(close*exposure).max(axis=(1,2)), .65*(close*known*fresh).max(axis=(1,2)),
        .55*u[:,19], .8*(u[:,1]-u[:,15]), .30*u[:,14],
        .4*u[:,8], .4*u[:,9], .45*u[:,20], .45*u[:,24],
    ]
    return np.ascontiguousarray(np.column_stack(features), dtype=np.float32)


def frame_indices(row):
    return np.flatnonzero(row["cutoff_valid"] & (row["target_bin"] >= 0))

