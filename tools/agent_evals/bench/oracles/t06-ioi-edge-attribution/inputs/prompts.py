"""Seeded ABBA/BABA examples and matched ABC corruptions."""
import numpy as np


def make_prompts(seed, n_prompts, names):
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(n_prompts):
        io, subject, other = rng.choice(names, size=3, replace=False).tolist()
        first, second = (io, subject) if index % 2 == 0 else (subject, io)
        beginning = f"When {first} and {second} went to the store, "
        rows.append({"clean": beginning + f"{subject} gave a drink to",
                     "corrupt": beginning + f"{other} gave a drink to",
                     "positive_token": " " + io, "negative_token": " " + subject,
                     "template": "ABBA" if index % 2 == 0 else "BABA"})
    return rows
