"""Synthetic stimulus set: 8 categories (4 shapes x 2 colour families), 5 exemplars each.

``make_stimuli(seed, n_categories, per_category, size)`` returns
``(images, labels)``: images float32 in [0, 1] of shape ``[N, 3, size, size]``
ordered category-major (all exemplars of category 0 first), labels ``[N]``.
Exemplars vary in position, size, rotation, brightness and background noise.
"""
import numpy as np

SHAPES = ("disc", "square", "bar", "ring")
FAMILIES = ((0.9, 0.3, 0.2), (0.2, 0.4, 0.9))  # warm, cool


def _mask(shape, size, cx, cy, r, theta):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    x = xx - cx
    y = yy - cy
    xr = np.cos(theta) * x + np.sin(theta) * y
    yr = -np.sin(theta) * x + np.cos(theta) * y
    if shape == "disc":
        return (xr ** 2 + yr ** 2) <= r ** 2
    if shape == "square":
        return (np.abs(xr) <= r) & (np.abs(yr) <= r)
    if shape == "bar":
        return (np.abs(xr) <= r) & (np.abs(yr) <= r / 3.5)
    if shape == "ring":
        d2 = xr ** 2 + yr ** 2
        return (d2 <= r ** 2) & (d2 >= (0.6 * r) ** 2)
    raise ValueError(shape)


def make_stimuli(seed, n_categories=8, per_category=5, size=224):
    rng = np.random.default_rng(seed)
    images = []
    labels = []
    for c in range(n_categories):
        shape = SHAPES[c % len(SHAPES)]
        fam = np.array(FAMILIES[c // len(SHAPES)])
        for _ in range(per_category):
            bg = 0.45 + 0.1 * rng.standard_normal((1, size, size))
            img = np.repeat(bg, 3, axis=0) + 0.02 * rng.standard_normal((3, size, size))
            cx, cy = rng.uniform(0.3, 0.7, size=2) * size
            r = rng.uniform(0.15, 0.28) * size
            theta = rng.uniform(0, np.pi)
            m = _mask(shape, size, cx, cy, r, theta)
            colour = np.clip(fam + 0.08 * rng.standard_normal(3), 0, 1) * rng.uniform(0.8, 1.0)
            img[:, m] = colour[:, None]
            images.append(np.clip(img, 0, 1))
            labels.append(c)
    return np.stack(images).astype(np.float32), np.array(labels)
