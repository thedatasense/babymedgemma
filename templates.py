"""Paraphrase bank for the scaled index.

The previous canonical index used 8 templates reused across every finding and every
dataset, which collapsed the vocabulary to 22 content words. A model memorises 9
sentence patterns immediately, so any low flip rate measured on it reports template
count rather than robustness -- and the mechanism under study is a register/formality
gate, which needs linguistic room to vary in.

PARAPHRASES below are strictly MEANING-PRESERVING: every one asks "is {f} present in
this chest radiograph?" and must carry the SAME yes/no answer. They span the four
operator-preserving phenomena used by the validated experiments.

OPERATOR_VARIANTS are NOT paraphrases. They invert the answer, so they are kept in a
separate field and never enter a paraphrase cluster. Treating negation as a paraphrase
is precisely the mixed-question defect flagged in the PSF-Med committee review.
"""

from __future__ import annotations

PARAPHRASES = {
    # swap the content words, keep the structure
    "lexical_substitution": [
        "is there evidence of {f}?",
        "is there radiographic evidence of {f}?",
        "are there signs of {f}?",
        "are there findings of {f}?",
        "is there an indication of {f}?",
        "does this radiograph demonstrate {f}?",
        "does this film reveal {f}?",
        "are there features of {f}?",
        "is {f} demonstrated?",
        "is {f} apparent?",
        "does this examination show {f}?",
        "is there anything suggesting {f}?",
    ],
    # keep the words, change the grammar
    "syntactic_restructuring": [
        "can {f} be identified?",
        "does the image show {f}?",
        "is {f} seen?",
        "do you see {f}?",
        "can you identify {f}?",
        "has {f} been identified?",
        "would you report {f}?",
        "is it correct that {f} is present?",
        "{f} is present, correct?",
        "am i seeing {f} here?",
        "is what we see {f}?",
        "could this be {f}?",
    ],
    # widen or narrow the quantifier
    "scope_quantification": [
        "is there any {f}?",
        "is there any sign of {f}?",
        "is there any evidence of {f}?",
        "is there some {f}?",
        "is any degree of {f} present?",
        "is there definite {f}?",
        "is there clear {f}?",
        "is there at least some {f}?",
        "is there obvious {f}?",
        "is there frank {f}?",
        "is even mild {f} present?",
        "is there a suggestion of {f}?",
    ],
    # move the clinical register up and down
    "specificity_modulation": [
        "is {f} visible on this chest radiograph?",
        "is {f} present on this study?",
        "is {f} noted on this examination?",
        "is {f} observed in this image?",
        "is {f} present on the current film?",
        "on this chest x-ray, is {f} present?",
        "radiographically, is {f} present?",
        "in your assessment, is {f} present?",
        "is {f} evident on this frontal chest radiograph?",
        "is {f} detected on imaging?",
        "looking at this picture, is {f} there?",
        "anything like {f} on here?",
    ],
}

# These INVERT the answer. Never place them in a paraphrase cluster.
OPERATOR_VARIANTS = {
    "negation_pattern": [
        "is there no {f}?",
        "is {f} absent?",
        "is the chest clear of {f}?",
        "is {f} ruled out?",
    ],
}


def paraphrases_for(phrase: str) -> list[dict]:
    """Meaning-preserving paraphrases, answer unchanged."""
    return [{"text": t.format(f=phrase), "phenomenon": phen}
            for phen, ts in PARAPHRASES.items() for t in ts]


def operator_variants_for(phrase: str) -> list[dict]:
    """Answer-inverting rewrites, tagged and kept out of the flip clusters."""
    return [{"text": t.format(f=phrase), "phenomenon": phen, "inverts_answer": True}
            for phen, ts in OPERATOR_VARIANTS.items() for t in ts]


def n_paraphrases() -> int:
    return sum(len(v) for v in PARAPHRASES.values())


if __name__ == "__main__":
    import re
    ex = paraphrases_for("pleural effusion")
    words = set()
    for p in ex:
        words |= set(re.findall(r"[a-z0-9]+", p["text"].lower()))
    print(f"paraphrases per question: {n_paraphrases()}")
    print(f"phenomena: {list(PARAPHRASES)}")
    print(f"distinct words in one cluster: {len(words)}")
    print(f"operator variants (excluded from clusters): {len(operator_variants_for('x'))}")
