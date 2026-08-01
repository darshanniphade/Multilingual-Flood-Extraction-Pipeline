"""Self-test for the flood matcher. Run it directly; exits non-zero on failure.

    C:\\darsh\\AI_MODELS\\translator_env\\Scripts\\python.exe test_lexicon.py
"""

from __future__ import annotations

import sys

from flood_lexicon import is_flood_title

# (title, should_match)
CASES = [
    # ---- must match: core flood events -------------------------------------
    ("Flash floods kill 18 in Assam", True),
    ("Heavy rainfall causes severe flooding in Kerala", True),
    ("Torrential rain lashes Mumbai, streets submerged", True),
    ("Waterlogging cripples Delhi after overnight downpour", True),
    ("Cloudburst triggers flash flood in Uttarakhand", True),
    ("Brahmaputra in spate, thousands marooned", True),
    ("Embankment breach floods 20 villages in Bihar", True),
    ("Water released from Mettur dam as inflow rises", True),
    ("Cyclone Tauktae makes landfall in Gujarat", True),
    ("Storm surge inundates coastal Odisha", True),
    ("Rain-triggered landslide buries five in Idukki", True),
    ("Villages submerged as Kosi swells", True),
    ("Low-lying areas inundated in Chennai", True),
    ("Water enters homes in Nashik after incessant rain", True),
    ("Rescue operations underway as rains flood Pune", True),
    ("NDRF teams deployed in flood-hit districts", True),
    ("Monsoon fury: 12 dead in Himachal", True),
    ("Death toll rises to 30 in Maharashtra floods", True),
    ("Waist-deep water in Hyderabad colonies", True),
    ("Spillway gates opened at Krishnarajasagar dam", True),
    ("Red alert issued as heavy rain batters Kerala", True),
    ("River bursts its banks, submerging farmland", True),
    ("Typhoon In-Fa dumps record rainfall on Zhejiang", True),
    ("Crops damaged as floodwaters recede in Punjab", True),

    # ---- must NOT match: metaphors and homonyms -----------------------------
    ("Flood of migrants at the southern border", False),
    ("Biden faces a flood of criticism over withdrawal", False),
    ("Company floods the market with cheap imports", False),
    ("Opening the floodgates for private investment", False),
    ("Floodlights installed at the new stadium", False),
    ("Curt Flood and the fight for free agency", False),
    ("Modi wins by a landslide in state polls", False),
    ("Landslide victory for the ruling party", False),
    ("Iowa State Cyclones win season opener", False),
    ("Cyclones beat Longhorns 24-21 in Ames", False),
    ("Treasury unveils $2bn rescue package for airlines", False),
    ("Rescue dog finds forever home in Ohio", False),
    ("Animal rescue shelter seeks volunteers", False),
    ("The Great Depression and its lessons for today", False),
    ("Study links social media to teen depression", False),
    ("Battling depression during the pandemic", False),
    ("Perfect storm of inflation and supply chain woes", False),
    ("Protesters storm the Capitol building", False),
    ("Monsoon Session of Parliament begins Monday", False),
    ("Monsoon Wedding returns to Broadway", False),
    ("Surge in COVID cases across Delhi", False),
    ("Hospital inundated with calls after announcement", False),
    ("Deluge of applications for the new scheme", False),
    ("Brainstorming session on climate policy", False),

    # ---- must NOT match: unrelated news -------------------------------------
    ("11 Billion Dollars", False),
    ("US to modify H-1B visa selection process", False),
    ("Tone-deaf lawmaker on another fool's errand", False),
    ("No Cookies | The Courier Mail", False),
    ("Stocks rally as Fed holds rates steady", False),
    ("Manchester United sign new striker", False),
    ("Former Tulsa police officer's double-jeopardy claims denied", False),
    ("Netflix announces third season of hit series", False),

    # ---- regressions found by auditing real corpus output -------------------
    # Solid compound, no word boundary before "flood".
    ("DSWD DROMIC Report #4 on the Flashflood Incident in Northern Mindanao", True),
    ("Mpumalanga braces for the worst as major dam threatens to overflow", True),
    ("The rains continue to disrupt the entrance, in Poya and La Foa", True),
    # Winter weather is a storm but not a flood event.
    ("A winter storm caused Texas's power outage", False),
    ("Power failure: How a winter storm pushed Texas into crisis", False),
    ("Weather Alert: End of cold wave this Sunday", False),
    ("Snowfall has closed new highways in Ukraine", False),
    ("Blizzard leaves thousands without power in Minnesota", False),
    # Loose context words that used to leak unrelated news in.
    ("Wave of COVID cases overwhelms hospitals", False),
    ("Wet market linked to virus outbreak", False),
    ("Qingdao Bay sea star capture brings economic losses to fishermen", False),
    ("Rainwater harvesting system and drought in Turkey", False),
    ("It's raining criticism of the security guard in charge of the Capitol", False),
    ("Perseverance lands on Mars and unleashes a meme rain on Earth", False),
    ("BMKG predicts peak rainy season in late January 2021", False),
    # Substring traps that word boundaries must reject.
    ("How COVID-19 Attacks The Brain And May Cause Lasting Damage", False),
    ("Training of Angolan senior judges at the University of Coimbra", False),
    ("Bahrain announces new investment fund", False),
    ("RTE viewers left in floods by Let The Rest Of The World Go By show", False),
    ("Gun shops in the U.S. flooded with buyers for Biden's inauguration", False),
    ("Warrington address flooded with water up to eight inches deep", True),

    # ---- weak-without-context: flooding not the primary subject -------------
    ("Landslide buries five in Idukki", False),
    ("Rescue mission launched after building collapse", False),
    ("Death toll rises in factory fire", False),
    ("Evacuation ordered as wildfire spreads", False),
    ("Power outage hits thousands in Texas", False),
    ("Bridge collapses in Genoa, killing 20", False),
    ("School closed after bomb threat", False),
]


def main() -> int:
    failures = []
    for title, expected in CASES:
        got, reason = is_flood_title(title, explain=True)
        if got != expected:
            failures.append((title, expected, got, reason))

    passed = len(CASES) - len(failures)
    print("%d/%d passed" % (passed, len(CASES)))

    if failures:
        print("\nFAILURES")
        print("-" * 70)
        for title, expected, got, reason in failures:
            print("  expected %-5s got %-5s | %s" % (expected, got, title))
            print("      reason: %s" % reason)
        return 1

    print("all good")
    return 0


if __name__ == "__main__":
    sys.exit(main())
