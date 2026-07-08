"""
Cross-platform campaign taxonomy unifier.

Google and Bing ship a clean campaign-type column; Meta does not — its type is
embedded in the campaign name (e.g. "Prospecting_DPA_Campaign_04"). This module
parses those names into a normalized taxonomy with a rule-based parser, and
flags anything that doesn't match a rule so it can be routed to the LLM
fallback in the demo layer (never in the scored pipeline — no network there).

Normalized taxonomy:
  primary type:  Prospecting | Remarketing | Generic | Advantage+
  sub-tag:       Brand | DPA | (none)
"""

import re

# Order matters: first match wins.
PRIMARY_RULES = [
    (re.compile(r"prospect", re.I), "Prospecting"),
    (re.compile(r"remarket|retarget", re.I), "Remarketing"),
    (re.compile(r"adv[_ ]?plus|advantage", re.I), "Advantage+"),
    (re.compile(r"generic", re.I), "Generic"),
]

SUBTAG_RULES = [
    (re.compile(r"\bdpa\b|_dpa_|_dpa$", re.I), "DPA"),
    (re.compile(r"\bbrand\b|_brand_|_brand$", re.I), "Brand"),
]


def parse_meta_campaign_type(campaign_name):
    """Return (campaign_type, subtag, matched) for a Meta campaign name.

    campaign_type folds the sub-tag in (e.g. "Prospecting - DPA") so it can be
    used directly as the normalized type column; `matched` is False when no
    rule fired and the name needs manual/LLM review.
    """
    name = str(campaign_name)

    primary = None
    for pattern, label in PRIMARY_RULES:
        if pattern.search(name):
            primary = label
            break

    subtag = None
    for pattern, label in SUBTAG_RULES:
        if pattern.search(name):
            subtag = label
            break

    # "Prospecting_Adv_Plus_..." is both Prospecting and Advantage+; keep the
    # buying-objective (Prospecting) as primary and record Advantage+ as subtag.
    if primary == "Prospecting" and re.search(r"adv[_ ]?plus|advantage", name, re.I):
        subtag = "Advantage+" if subtag is None else f"Advantage+/{subtag}"

    if primary is None:
        return "Unclassified", subtag, False

    campaign_type = f"{primary} - {subtag}" if subtag else primary
    return campaign_type, subtag, True


def classify_meta_campaigns(names):
    """Classify an iterable of Meta campaign names.

    Returns dict name -> {campaign_type, primary, subtag, matched}.
    """
    result = {}
    for name in names:
        campaign_type, subtag, matched = parse_meta_campaign_type(name)
        result[name] = {
            "campaign_type": campaign_type,
            "subtag": subtag,
            "matched": matched,
        }
    return result


# Google / Bing types are already clean — just normalize casing/spelling so
# the same concept reads the same across platforms.
GOOGLE_TYPE_MAP = {
    "SEARCH": "Search",
    "PERFORMANCE_MAX": "Performance Max",
    "DISPLAY": "Display",
    "VIDEO": "Video",
    "DEMAND_GEN": "Demand Gen",
    "SHOPPING": "Shopping",
}

BING_TYPE_MAP = {
    "Search": "Search",
    "PerformanceMax": "Performance Max",
    "Audience": "Audience",
    "Shopping": "Shopping",
}


def normalize_google_type(raw):
    return GOOGLE_TYPE_MAP.get(str(raw), str(raw).replace("_", " ").title())


def normalize_bing_type(raw):
    return BING_TYPE_MAP.get(str(raw), str(raw))


if __name__ == "__main__":
    samples = [
        "Generic_Campaign_02",
        "Prospecting_DPA_Campaign_04",
        "Remarketing_Brand_Campaign_03",
        "Prospecting_Adv_Plus_Campaign_02",
        "Generic_Brand_Campaign_01",
        "Weird_Unknown_Name_99",
    ]
    for name, info in classify_meta_campaigns(samples).items():
        print(f"{name:40s} -> {info}")
