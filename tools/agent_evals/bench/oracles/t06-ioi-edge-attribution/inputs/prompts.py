"""Seeded ABBA/BABA examples and matched ABC corruptions."""

import numpy as np

# {first} and {second} open each sentence; {subject} repeats one of them.
TEMPLATES = [
    "When {first} and {second} went to the {place}, {subject} gave a {object} to",
    "After {first} and {second} went to the {place}, {subject} gave a {object} to",
    "When {first} and {second} arrived at the {place}, {subject} handed a {object} to",
    "While {first} and {second} were at the {place}, {subject} gave a {object} to",
    "Once {first} and {second} got to the {place}, {subject} gave a {object} to",
    "When {first} and {second} came to the {place}, {subject} brought a {object} to",
    "After {first} and {second} walked to the {place}, {subject} gave a {object} to",
    "When {first} and {second} met at the {place}, {subject} offered a {object} to",
    "Since {first} and {second} were at the {place}, {subject} gave a {object} to",
]
PLACES = [
    "store",
    "park",
    "school",
    "office",
    "station",
    "restaurant",
    "garden",
    "beach",
    "library",
    "museum",
    "market",
    "house",
    "hospital",
    "airport",
    "church",
    "hotel",
    "bank",
    "farm",
    "zoo",
    "club",
]
OBJECTS = [
    "drink",
    "ring",
    "book",
    "bag",
    "gift",
    "ball",
    "card",
    "letter",
    "key",
    "phone",
    "cake",
    "hat",
    "box",
    "pen",
    "flower",
    "bottle",
    "computer",
    "basket",
    "necklace",
    "snack",
]


def make_prompts(seed, n_prompts, names):
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(n_prompts):
        io, subject, other = rng.choice(names, size=3, replace=False).tolist()
        template = TEMPLATES[rng.integers(len(TEMPLATES))]
        place = PLACES[rng.integers(len(PLACES))]
        thing = OBJECTS[rng.integers(len(OBJECTS))]
        first, second = (io, subject) if index % 2 == 0 else (subject, io)
        fill = {"first": first, "second": second, "place": place, "object": thing}
        rows.append(
            {
                "clean": template.format(subject=subject, **fill),
                "corrupt": template.format(subject=other, **fill),
                "positive_token": " " + io,
                "negative_token": " " + subject,
                "template": "ABBA" if index % 2 == 0 else "BABA",
            }
        )
    return rows
