"""
sili_peridot/model/train_texts.py
────────────────────────────────────
Training corpus for model.train_online's online recovery probe.

TRAIN_TEXTS below is the SANITY-tier corpus: same no-external-dataset
convention as eval_pruning.EVAL_TEXTS (plain declarative English,
hand-written, no download), ~50 short sentences -- enough to smoke-test
that the online training mechanism actually runs and moves the loss,
NOT enough to conclude anything about whether training genuinely
recovers accuracy on a 1B-parameter model.

load_real_tier_train_texts() below is the REAL-tier corpus: real text
(WikiText-2, Wikipedia-derived prose), thousands of paragraphs, for an
overnight-scale run where the sanity corpus would just repeat forever
without adding real diversity. Requires the `datasets` package
(installed on demand, not a hard import at module load -- this file
stays importable without it for anyone only using the sanity tier).

Both stay disjoint in topic/domain from EVAL_TEXTS/EVAL_TEXTS_HELDOUT
(hand-written factual sentences, used unchanged so accuracy numbers
stay comparable to every baseline already recorded in sili_v_torch.md/
JOURNAL.md) -- training and evaluation never touch the same text.
"""
from __future__ import annotations

from typing import List

TRAIN_TEXTS: List[str] = [
    "The old lighthouse stood at the edge of the cliff, its light "
    "sweeping across the dark water every few seconds.",

    "Copper conducts electricity very well, which is why it is used "
    "in most household wiring and electrical cables.",

    "He tightened the last bolt on the bicycle wheel and gave it a "
    "spin to make sure it turned freely.",

    "The recipe called for two cups of flour, a teaspoon of salt, "
    "and just enough water to form a soft dough.",

    "Glaciers move slowly downhill under their own weight, carving "
    "valleys into the rock over thousands of years.",

    "She planted rows of carrots and lettuce along the back fence, "
    "leaving a narrow path between the beds.",

    "The violinist tuned each string carefully before the orchestra "
    "began to play the opening movement.",

    "Sound travels faster through water than it does through air, "
    "because water molecules are packed more tightly together.",

    "The mechanic listened to the engine idle, then adjusted a valve "
    "until the noise smoothed out.",

    "A thick fog rolled in over the harbor just after sunrise, "
    "hiding the fishing boats from view.",

    "The librarian stacked the returned books onto a cart and wheeled "
    "them back toward the history section.",

    "Bees communicate the location of flowers to each other through "
    "a series of movements known as the waggle dance.",

    "He folded the letter twice, sealed it in an envelope, and "
    "walked down to the corner mailbox.",

    "The bridge was built from steel cables strong enough to support "
    "the weight of hundreds of cars at once.",

    "Rain fell steadily through the night, filling the barrels lined "
    "up beneath the gutter.",

    "The chess player studied the board for a long moment before "
    "moving her knight into position.",

    "Volcanic soil is often very fertile, which is why farmers "
    "sometimes settle near active volcanoes.",

    "The children built a small dam across the stream using rocks "
    "and handfuls of wet mud.",

    "A compass needle points toward magnetic north because it aligns "
    "itself with the Earth's magnetic field.",

    "The baker pulled the tray from the oven and set the loaves on "
    "a rack to cool.",

    "Wind turbines convert the kinetic energy of moving air into "
    "electricity through a spinning generator.",

    "The old dog stretched out in a patch of sunlight on the kitchen "
    "floor and fell asleep.",

    "She measured the length of the shelf twice before cutting the "
    "wood to size.",

    "The comet's tail always points away from the sun, pushed back "
    "by solar wind and radiation.",

    "A small crowd gathered on the platform, waiting for the next "
    "train to arrive.",

    "The potter shaped the clay on the wheel, pressing her thumbs "
    "gently into the center to open it up.",

    "Owls can rotate their heads nearly all the way around, which "
    "helps them track prey without moving their bodies.",

    "The carpenter sanded the tabletop until the surface felt smooth "
    "under his hand.",

    "Lightning heats the surrounding air so quickly that it creates "
    "the shockwave we hear as thunder.",

    "The hikers reached the summit just before noon and stopped to "
    "eat lunch overlooking the valley.",

    "A thin layer of ice had formed on the pond overnight, cracking "
    "slightly under the weight of a thrown stone.",

    "The tailor pinned the fabric along the seam and marked where "
    "the next cut should go.",

    "Coral reefs support an enormous variety of marine life despite "
    "covering only a small fraction of the ocean floor.",

    "The postman rode his bicycle along the same route every "
    "morning, dropping letters into each mailbox.",

    "The engineer checked the pressure gauge twice before opening "
    "the valve.",

    "Migrating birds often fly in a V formation, which reduces wind "
    "resistance for the birds behind the leader.",

    "The old clock in the hallway chimed twice, and the house fell "
    "quiet again.",

    "He replaced the bicycle chain and oiled the gears before riding "
    "off down the gravel road.",

    "Salt lowers the freezing point of water, which is why it is "
    "spread on icy roads in winter.",

    "The seamstress threaded the needle and began stitching the hem "
    "of the dress.",

    "A narrow trail wound through the pine forest, disappearing "
    "behind a ridge in the distance.",

    "The astronomer adjusted the telescope slightly and waited for "
    "the clouds to clear.",

    "Yeast produces carbon dioxide as it ferments, which is what "
    "causes bread dough to rise.",

    "The blacksmith heated the iron rod in the forge until it "
    "glowed a dull orange.",

    "The ferry crossed the strait twice a day, carrying passengers "
    "and a handful of cars.",

    "Spiders spin silk threads that are, ounce for ounce, stronger "
    "than steel.",

    "The gardener trimmed the hedges into a neat, even line along "
    "the front walk.",

    "The river widened as it approached the delta, slowing before "
    "it emptied into the sea.",

    "A single beehive can produce several jars of honey over the "
    "course of a summer.",

    "The night sky was clear enough to see the faint band of the "
    "Milky Way stretching overhead.",
]


def load_real_tier_train_texts(
    max_texts: int = 4000, min_words: int = 8, max_words: int = 60,
) -> List[str]:
    """WikiText-2 (Salesforce/wikitext, wikitext-2-raw-v1 config, train
    split) filtered down to plain prose paragraphs: drops empty lines
    and section-heading lines (WikiText's raw format marks these with
    leading '='), keeps lines with min_words..max_words words -- short
    enough to keep the frozen-prefix cost per text bounded (see
    train_online.py's memoization cache), long enough to carry real
    sentence structure. ~3,500 lines survive this filter out of the
    full ~36,700-line split; max_texts caps how many are returned
    (first N after filtering, deterministic).

    Requires the `datasets` package -- imported here, not at module
    load, so this file stays importable for sanity-tier-only use
    without it. First call downloads/caches the dataset (~4-5MB) via
    HuggingFace Hub -- needs network access once, then reads from the
    local HF cache.
    """
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    texts: List[str] = []
    for row in ds:
        t = row["text"].strip()
        if not t or t.startswith("="):
            continue
        n_words = len(t.split())
        if min_words <= n_words <= max_words:
            texts.append(t)
            if len(texts) >= max_texts:
                break
    return texts
