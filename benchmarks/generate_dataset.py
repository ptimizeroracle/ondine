"""Generate a synthetic Amazon-reviews-style dataset for repositioning benchmarks.

Produces a CSV of ``N`` rows with realistic-but-synthetic review text plus
ground-truth columns the benchmark prompt can target (``sentiment``,
``category``). Purely deterministic (fixed seed) so two runs compare like
for like. No external data is downloaded.

Usage::

    python benchmarks/generate_dataset.py --rows 100000 --out benchmarks/data/amazon_reviews_100k.csv

The dataset is the shared input for all three benchmark arms in
``repositioning.py``.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

# Deterministic seed so every arm sees byte-identical input.
SEED = 42

PRODUCT_CATEGORIES = [
    "Electronics",
    "Books",
    "Home & Kitchen",
    "Clothing",
    "Toys",
    "Sports",
    "Beauty",
    "Grocery",
    "Tools",
    "Pet Supplies",
]

# Review bodies are templated so they look like real reviews but stay
# fully reproducible. Each template carries an embedded sentiment cue
# the LLM can actually classify.
POSITIVE_TEMPLATES = [
    "Absolutely love this {product}! Works exactly as described and arrived two days early. Would buy again.",
    "Best {product} I have owned in years. Solid build quality and the price was unbeatable.",
    "Five stars for this {product}. Exceeded my expectations and the packaging was excellent.",
    "Fantastic {product}. My family uses it daily and it has held up perfectly for six months.",
    "Great value {product}. Does what it promises and the instructions were clear and helpful.",
]
NEGATIVE_TEMPLATES = [
    "Disappointed with this {product}. It stopped working after one week and customer service ignored me.",
    "Would not recommend this {product}. Cheap materials and the description was misleading.",
    "Worst {product} purchase I have made. Arrived broken and the return process was a nightmare.",
    "One star. This {product} is nothing like the photos and broke within days of normal use.",
    "Regret buying this {product}. Overpriced for the quality and the warranty is useless.",
]
NEUTRAL_TEMPLATES = [
    "This {product} is okay. Nothing special but it does the job for the price I paid.",
    "Average {product}. Some features work well, others feel rushed. Mixed feelings overall.",
    "Decent {product} for the price. Not amazing, not terrible. I have no strong opinion.",
    "The {product} works as expected. Nothing stood out as either great or poor.",
    "Middle of the road {product}. Three stars feels fair given the competition at this price.",
]

PRODUCT_WORDS = {
    "Electronics": [
        "USB-C cable",
        "wireless mouse",
        "Bluetooth speaker",
        "noise-cancelling headphones",
        "smart plug",
    ],
    "Books": [
        "cookbook",
        "mystery novel",
        "biography",
        "programming guide",
        "coffee-table book",
    ],
    "Home & Kitchen": [
        "knife set",
        "air fryer",
        "cast-iron pan",
        "bedsheet set",
        "coffee grinder",
    ],
    "Clothing": [
        "winter jacket",
        "running shoes",
        "cotton t-shirt",
        "denim jeans",
        "wool socks",
    ],
    "Toys": [
        "building blocks",
        "remote-control car",
        "puzzle set",
        "action figure",
        "board game",
    ],
    "Sports": ["yoga mat", "dumbbell set", "bicycle pump", "jump rope", "water bottle"],
    "Beauty": [
        "face moisturizer",
        "shampoo",
        "nail polish set",
        "sunscreen",
        "hairbrush",
    ],
    "Grocery": [
        "coffee beans",
        "olive oil",
        "protein bars",
        "spice rack",
        "loose-leaf tea",
    ],
    "Tools": [
        "cordless drill",
        "tape measure",
        "screwdriver set",
        "utility knife",
        "level",
    ],
    "Pet Supplies": [
        "dog leash",
        "cat scratching post",
        "fish tank filter",
        "bird seed",
        "hamster bedding",
    ],
}


def make_rows(n: int):
    """Yield ``n`` deterministic review dicts with ground-truth labels."""
    rng = random.Random(SEED)
    for i in range(n):
        sentiment_pool = rng.choices(
            ["positive", "negative", "neutral"], weights=[5, 3, 2], k=1
        )
        sentiment = sentiment_pool[0]
        category = rng.choice(PRODUCT_CATEGORIES)
        product = rng.choice(PRODUCT_WORDS[category])
        star = {
            "positive": rng.randint(4, 5),
            "negative": rng.randint(1, 2),
            "neutral": 3,
        }[sentiment]
        if sentiment == "positive":
            body = rng.choice(POSITIVE_TEMPLATES).format(product=product)
        elif sentiment == "negative":
            body = rng.choice(NEGATIVE_TEMPLATES).format(product=product)
        else:
            body = rng.choice(NEUTRAL_TEMPLATES).format(product=product)
        title = body.split(".")[0][:60]
        yield {
            "review_id": i,
            "title": title,
            "review": body,
            "category": category,
            "ground_truth_sentiment": sentiment,
            "ground_truth_stars": star,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument(
        "--out", type=Path, default=Path("benchmarks/data/amazon_reviews_100k.csv")
    )
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "review_id",
        "title",
        "review",
        "category",
        "ground_truth_sentiment",
        "ground_truth_stars",
    ]
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in make_rows(args.rows):
            writer.writerow(row)
    print(
        f"Wrote {args.rows:,} rows to {args.out} ({args.out.stat().st_size / 1_048_576:.1f} MiB)"
    )


if __name__ == "__main__":
    main()
