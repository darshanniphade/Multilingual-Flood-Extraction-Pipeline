"""Flood-event detection over translated article titles.

The matcher runs in three stages on a normalised copy of the title:

  1. GUARDS  - metaphorical / homonymous uses ("flood of migrants", "landslide
               victory", "Iowa State Cyclones") are masked out of the text
               before any matching happens. Masking rather than rejecting means
               a title keeps its real signal: "Flood of donations as floods hit
               Assam" loses only the metaphor and still matches on "floods".
  2. STRONG  - terms that denote a flood/hydro-met event on their own.
  3. WEAK    - impact/response terms that are common in unrelated news
               ("rescue", "death toll", "red alert"). These only count when the
               title also carries an independent CONTEXT word (rain, river,
               dam, ...) that is not itself part of the weak match.

Stage 3 is what implements "flooding must be the primary subject": a landslide
or an evacuation is only in scope when water or rain is named alongside it.
"""

from __future__ import annotations

import re
import unicodedata

# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize(title: str) -> str:
    """Lowercase, strip punctuation to spaces, collapse runs of whitespace.

    Hyphens and curly quotes become spaces, so "flood-hit", "rain-triggered"
    and mojibake like "fool<E2><80><99>s" all reduce to plain word sequences.
    The result is padded with single spaces so \\b anchors behave at the edges.
    """
    folded = unicodedata.normalize("NFKC", title).lower()
    return " %s " % " ".join(_NON_ALNUM.sub(" ", folded).split())


def _compile(patterns: list[str]) -> re.Pattern:
    return re.compile("|".join("(?:%s)" % p for p in patterns))


# --------------------------------------------------------------------------
# Stage 1 - guards (masked out before matching)
# --------------------------------------------------------------------------

_METAPHOR_NOUNS = (
    r"migrant|refugee|immigrant|asylum|memor|tear|call|message|text|request|"
    r"tribute|application|complaint|order|enquir|inquir|quer|emotion|money|"
    r"cash|fund|donation|light|information|data|meme|tweet|email|support|"
    r"wish|greeting|question|resume|entr|offer|import|good|product|drug|"
    r"weapon|gun|counterfeit|fake|misinformation|rumou?r|claim|lawsuit|"
    r"visitor|tourist|customer|applicant|candidate|nostalgia|praise|abuse|"
    r"criticism|compliment|comment|review|response|reply|bid|listing|advert|"
    r"buyer|shopper|booking|patient|worker|staff|volunteer|player|fan|student|"
    r"voter|passenger|guest|traveller|traveler|user|subscriber|vaccine|dose"
)

GUARDS = [
    # "a flood of X" / "flooded with X" used figuratively.
    r"\bflood(?:s|ed|ing)?\s+(?:of|with)\s+(?:the\s+|a\s+|new\s+|cheap\s+|more\s+)?"
    r"(?:%s)\w*" % _METAPHOR_NOUNS,
    r"\bdeluge[ds]?\s+(?:of|with)\s+(?:%s)\w*" % _METAPHOR_NOUNS,
    r"\binundat\w+\s+(?:of|with|by)\s+(?:%s)\w*" % _METAPHOR_NOUNS,
    r"\bswamped\s+with\s+(?:%s)\w*" % _METAPHOR_NOUNS,
    # Market / media metaphors.
    r"\bflood(?:s|ed|ing)?\s+(?:the|into|onto)\s+"
    r"(?:market|zone|airwave|internet|media|social|street|shelve|store|economy)\w*",
    r"\bfloodgate\w*",
    r"\bflood\s?light\w*",
    # British idiom: "left in floods (of tears)".
    r"\bfloods?\s+of\s+tears\b",
    r"\b(?:left|was|were|burst|reduced|had\s+\w+)\s+in\s+floods?\b",
    r"\bflood\s+(?:plain|zone)\s+(?:map|zoning|ordinance|insurance|policy)\w*",
    # "Flood" as a personal surname.
    r"\b(?:curt|toby|tobi|matt|matthew|kevin|mary|patrick|gerald|joe|john|"
    r"michael|dan|daniel|chris|sean|shane|david|paddy)\s+flood\b",
    # Landslide as an election margin - very common, must not match.
    r"\blandslide\s+(?:victor|win|won|elect|majorit|vote|defeat|margin|poll|"
    r"result|mandate|triumph|sweep)\w*",
    r"\b(?:win|wins|won|winning|elected|re\s?elected|victory|defeat|lead|leads|"
    r"sweep|sweeps|swept|romp|romps|cruise|cruises)\s+(?:to\s+|in\s+|by\s+|with\s+)?"
    r"(?:a\s+|an\s+)?landslide\b",
    r"\blandslide\b(?=\s+(?:for|against)\b)",
    # "Cyclones" as a team nickname (Iowa State et al).
    r"\b(?:iowa\s+state|ames|isu)\s+cyclones?\b",
    r"\bcyclones?\s+(?:win|wins|won|beat|beats|lose|loses|lost|defeat|top|tops|"
    r"fall|falls|rally|rallies|host|hosts|face|faces|edge|edges|rout|routs|"
    r"upset|upsets|game|games|coach|basketball|football|volleyball|wrestling|"
    r"score|scores|roster|recruit|season|player|athletics|stadium|arena|"
    r"quarterback|lineup|schedule|softball|baseball)\w*",
    r"\b(?:men|women|mens|womens)\s+cyclones?\b",
    # "Rescue" in finance / animal-shelter senses.
    r"\brescue\s+(?:package|packages|plan|plans|deal|deals|bid|bids|fund|funds|"
    r"funding|loan|loans|financ|money|cash|capital|investor|takeover|merger|"
    r"buyout|act|bill|programme|program|stimulus|aid\s+package)\w*",
    r"\b(?:animal|dog|cat|pet|puppy|kitten|horse|wildlife|donkey|bird|greyhound|"
    r"husky|terrier|koala|panda|elephant)\s+rescue\w*",
    r"\brescue\s+(?:dog|dogs|cat|cats|pet|pets|puppy|puppies|kitten|kittens|"
    r"horse|horses|animal|animals|shelter|shelters|centre|center|centres|"
    r"centers|group|groups|charity|organisation|organization|foundation|"
    r"kitty|farm|sanctuary)\w*",
    r"\bbail\s?out\w*",
    # "Depression" in economic / clinical senses.
    r"\b(?:great|economic|financial|global|prolonged|era\s+of)\s+depression\b",
    r"\b(?:clinical|postpartum|post\s?natal|manic|bipolar|severe|major|teen|"
    r"seasonal|chronic)\s+depression\b",
    r"\bdepression\s+(?:era|symptom|treatment|medication|therapy|screening|"
    r"diagnos|drug|pill|rate|risk|help|support|awareness|stigma|study)\w*",
    r"\b(?:battl|fight|fighting|suffer|suffering|struggl|deal|dealing|cope|"
    r"coping|tackl|overcome|overcoming|treat|treating|diagnos)\w*\s+"
    r"(?:with\s+)?depression\b",
    r"\bdepression\s+and\s+(?:anxiety|stress|ptsd|insomnia)\b",
    r"\b(?:anxiety|stress|ptsd|insomnia|suicide|mental\s+health)\s+and\s+depression\b",
    # "Storm" as political/social metaphor.
    r"\b(?:perfect|political|media|twitter|social|racism|racist|sexism)\s+storm\b",
    r"\bstorm\s+in\s+a\s+tea\s?cup\b",
    r"\bstorm\s+of\s+(?:protest|criticism|controvers|outrage|abuse|anger)\w*",
    r"\bstorm(?:s|ed|ing)?\s+(?:the|into|out\s+of|off|past|back|through)\b",
    r"\bbrain\s?storm\w*",
    r"\bdesert\s+storm\b",
    # Monsoon in non-weather senses.
    r"\bmonsoon\s+(?:session|wedding|sale|collection|fashion|recipe|special|"
    r"menu|offer|discount|skincare|hair|health\s+tip)\w*",
    # Winter weather. These are storms, but not flood events - masking the
    # phrase removes the "storm"/"wave" context word so that a bare impact term
    # ("power outage", "rescue") can no longer promote to a match.
    r"\b(?:winter|snow|ice|icy|hail|hail\s?stone|dust|sand|wind|thunder|"
    r"electrical|solar|geomagnetic|fire)\s?storm\w*",
    r"\bcold\s+(?:wave|snap|spell|front)\w*",
    r"\bheat\s?wave\w*",
    r"\b(?:polar\s+vortex|blizzard|frost|freeze|freezing|snowfall|snowstorm|"
    r"avalanche|hailstorm|sleet)\w*",
    r"\bwave\s+of\s+(?:covid|case|infection|crime|violence|protest|attack|"
    r"layoff|resignation|migrant|refugee|nostalgia|support|arrest)\w*",
    r"\b(?:crime|heat|cold|covid|third|second|fourth|omicron|delta|infection)\s+wave\w*",
    # Misc homonyms.
    r"\bsurge\s+(?:in|of)\s+(?:case|covid|infection|price|demand|sale|inflation|"
    r"crime|violence|unemployment|share|stock)\w*",
    r"\bwet\s+market\w*",
    r"\brain\s?water\s+harvest\w*",
    r"\brainy\s+(?:season\s+(?:start|begin|end|forecast|predict)|day\s+fund)\w*",
    r"\bit\s+s\s+raining\s+(?:criticism|money|men|cat)\w*",
    r"\b(?:meme|money|cash|goal|gold|fire|arrow|bullet)\s+rain\w*",
    r"\bwashed\s+away\s+(?:the\s+)?(?:memor|sin|doubt|pain|stain|fear)\w*",
    r"\bwater\s?gate\b",
]

GUARD_RE = _compile(GUARDS)


# --------------------------------------------------------------------------
# Stage 2 - strong terms (a hit here is sufficient)
# --------------------------------------------------------------------------

STRONG = [
    # Flood family. Guards above have already removed the figurative uses, so a
    # bare flood stem is safe here and covers flooding / floodwaters /
    # flood-hit / flood-ravaged / flash flood / urban flood / floodplain.
    # The leading \w* catches solid compounds that news wires write without a
    # space or hyphen - "flashflood", "rainflood".
    r"\b\w*flood\w*",
    r"\bdeluge[ds]?\b",
    r"\binundat\w+",
    r"\bwater\s?log\w+",
    r"\bsubmerg\w+",
    r"\bwater\s+level\w*\s+(?:rise|rising|rose|surge|cross|breach)\w*",
    # Rain that is qualified or destructive. Bare "rain" is context only.
    r"\b(?:heavy|very\s+heavy|extremely\s+heavy|torrential|incessant|continuous|"
    r"persistent|relentless|intense|record|record\s?breaking|unprecedented|"
    r"unseasonal|unseasonable|excessive|nonstop|non\s+stop|days\s+of|high|"
    r"widespread)\s+rain(?:s|fall|storm)?\b",
    # Rain followed by a destructive verb, allowing a couple of intervening
    # words ("rains continue to disrupt", "rain has paralysed").
    r"\brain(?:s|fall|storm|storms)?\b(?:\s+\w+){0,3}?\s+(?:lash|batter|pound|"
    r"hammer|wreak|trigger|induce|caus|flood|inundat|submerg|swamp|drench|"
    r"paralys|paralyz|disrupt|devastat|ravage|pummel|thrash|maroon|cripple|"
    r"halt|kill|claim|displace|destroy|damag|sweep|wash|slam)\w*",
    r"\brain\s+(?:triggered|induced|hit|battered|lashed|swept|soaked|related|"
    r"ravaged|affected|damaged|fed)\b",
    r"\b(?:rain|monsoon|flood|water|storm|cyclone)\w*\s+"
    r"(?:fury|havoc|mayhem|misery|chaos|carnage|devastation)\b",
    r"\bcloud\s?burst\w*",
    r"\bdownpour\w*",
    r"\brain\s?storm\w*",
    r"\b(?:mm|cm|inches|inch)\s+of\s+rain\w*",
    r"\brainfall\s+(?:record|warning|alert|deficit\s+ends)\w*",
    # Rivers and water bodies behaving as a hazard.
    r"\bin\s+spate\b",
    r"\bswollen\s+(?:river|stream|creek|nullah|nala|canal|water)\w*",
    r"\briver\w*\s+(?:overflow|swell|swoll|breach|burst|rise|rising|rose|"
    r"spate|flood|submerg|inundat|erod|erosion)\w*",
    r"\boverflow\w*\s+(?:river|stream|canal|lake|dam|reservoir|drain|nullah|nala)\w*",
    r"\b(?:river|water)\s+bursts?\s+(?:its\s+|their\s+)?banks?\b",
    r"\bbank\s+erosion\b",
    r"\briver\s?bank\s+(?:erosion|collapse)\w*",
    # Dams, reservoirs, embankments failing or discharging.
    r"\b(?:embankment|levee|dam|dyke|dike|bund|barrage|weir)\s+"
    r"(?:breach|burst|collapse|fail|overflow|overtop)\w*",
    r"\bbreach\w*\s+(?:in\s+)?(?:the\s+)?(?:embankment|levee|bund|dam|dyke|dike)\b",
    r"\b(?:spillway|spill\s+gate|flood\s?gate\s+of\s+dam)\w*",
    # "water released from <Name> dam" - allow a reservoir name in between.
    r"\bwater\s+(?:is\s+|being\s+|to\s+be\s+)?released?\s+from\s+"
    r"(?:the\s+)?(?:\w+\s+){0,3}?(?:dam|reservoir|barrage|project|weir)\w*",
    r"\brelease\s+of\s+water\s+from\s+"
    r"(?:the\s+)?(?:\w+\s+){0,3}?(?:dam|reservoir|barrage|project|weir)\w*",
    r"\b(?:dam|reservoir|barrage)\s+(?:release|releases|released|discharge|"
    r"discharges|discharging|overflow)\w*",
    # Inundation imagery.
    r"\b(?:waist|knee|chest|neck|hip|thigh)\s+deep\s+water\b",
    r"\bwater\s+(?:enter|entered|enters|entering|gush|gushed|gushing|seep|"
    r"seeped|flow|flowed|flowing)\s+(?:into\s+)?(?:home|house|shop|premise|"
    r"building|villag|colon|apartment|basement|localit|societ)\w*",
    r"\b(?:street|road|house|home|villag|farmland|field|localit|area|hamlet|"
    r"colon|crop|market|lane|highway|neighbourhood|neighborhood)\w*\s+"
    r"(?:submerg|inundat|under\s?water|flooded)\w*",
    r"\blow\s+lying\s+(?:area|localit|region|colon|villag|neighbourhood)\w*",
    # Named storm systems (guarded against the sports sense above).
    r"\bcyclon\w+",
    r"\btyphoon\w*",
    r"\b(?:tropical|severe|super|extremely\s+severe|very\s+severe)\s+"
    r"(?:cyclonic\s+)?storm\b",
    r"\bstorm\s+surge\w*",
    r"\bhurricane\w*",
    r"\bmakes?\s+landfall\b",
    r"\bmade\s+landfall\b",
]

STRONG_RE = _compile(STRONG)


# --------------------------------------------------------------------------
# Stage 3 - weak terms (need independent context) and the context vocabulary
# --------------------------------------------------------------------------

WEAK = [
    # Response and casualties.
    r"\brescue\w*",
    r"\bevacuat\w+",
    r"\brelief\s+(?:camp|operation|material|work|effort|fund|distribution)\w*",
    r"\b(?:ndrf|sdrf|nrdf|imd|ndma)\b",
    r"\bsearch\s+and\s+rescue\b",
    r"\bdisaster\s+(?:response|management|relief|declaration|zone|area)\w*",
    r"\bemergency\s+(?:response|declaration|service|worker|order|evacuation)\w*",
    r"\bmarooned?\b",
    r"\bstranded\b",
    r"\bdisplaced\b",
    r"\bhomeless\b",
    r"\bdeath\s+toll\b",
    r"\b(?:casualt|fatalit)\w+",
    r"\bmissing\s+(?:person|people|villager|resident|fisherman|after)\w*",
    r"\baid\s+distribution\b",
    r"\btrapped\b",
    # Mass movement (guarded against "landslide victory").
    r"\bland\s?slid\w+",
    r"\bland\s?slip\w*",
    r"\bmud\s?slid\w+",
    r"\bdebris\s+flow\w*",
    r"\bslope\s+(?:failure|collapse)\w*",
    r"\bhill\s+collapse\w*",
    r"\bboulder\w*",
    # Damage.
    r"\bwashed\s+away\b",
    r"\bswept\s+away\b",
    r"\bwreak\w*\s+havoc\b",
    r"\bcaus\w*\s+havoc\b",
    r"\bhavoc\b",
    r"\bcut\s+off\b",
    r"\btrail\s+of\s+destruction\b",
    r"\bcrop\s+(?:damage|loss|destroy|ruin)\w*",
    r"\bfarmland\w*",
    r"\blivestock\b",
    r"\b(?:propert|house|home|infrastructure)\w*\s+"
    r"(?:damag|destroy|collaps|ruin)\w*",
    r"\bbridge\w*\s+(?:collaps|wash|swept|damag)\w*",
    r"\broad\w*\s+(?:wash|swept|cave|caved|collaps|block|damag)\w*",
    r"\b(?:highway|road|airport|school|college|office)\w*\s+(?:closed|shut|blocked)\b",
    r"\b(?:railway|train|flight|transport|traffic|bus)\w*\s+"
    r"(?:disrupt|cancel|suspend|halt|delay|divert)\w*",
    r"\bpower\s+(?:outage|cut|failure|supply\s+hit)\w*",
    r"\belectricity\s+(?:cut|supply\s+hit|disrupt)\w*",
    r"\bcommunication\w*\s+(?:disrupt|snap|cut)\w*",
    r"\beconomic\s+loss\w*",
    # Hydrology that is only meaningful with water context.
    r"\bwater\s+level\w*",
    r"\brising\s+water\w*",
    r"\bhigh\s+water\s+level\w*",
    r"\boverflow\w*",
    r"\bswollen\b",
    r"\bbreach\w*",
    r"\bgates?\s+(?:opened|open|lifted|raised)\b",
    r"\breservoir\w*",
    r"\bdischarge\w*",
    r"\bcatchment\w*",
    r"\bunder\s?water\b",
    r"\bdrown\w+",
    # Weather systems / warnings that are ambiguous alone.
    r"\bmonsoon\w*",
    r"\bdepression\b",
    r"\blow\s?\s*pressure\b",
    r"\bweather\s+(?:system|alert|warning|office|department|bureau)\w*",
    r"\b(?:red|orange|yellow|amber)\s+alert\b",
    r"\bhigh\s+alert\b",
    r"\bwarning\s+issued\b",
    r"\balert\s+issued\b",
    r"\bwet\s+weather\b",
    r"\bseasonal\s+rain\w*",
    r"\bshower\w*",
    r"\bdrizzl\w+",
]

WEAK_RE = _compile(WEAK)

# Words that establish a genuine water / rain / hydro-met setting. A weak term
# only promotes to a match when one of these appears *outside* the weak span.
CONTEXT = [
    r"\brain\w*",
    r"\bmonsoon\w*",
    r"\bwater\w*",
    r"\briver\w*",
    r"\bflood\w*",
    r"\bdam\b",
    r"\bdams\b",
    r"\breservoir\w*",
    r"\blake\w*",
    r"\bcanal\w*",
    r"\bstream\w*",
    r"\bnullah\w*|\bnala\w*|\bnadi\w*",
    r"\bdrain\w*",
    r"\btide\w*|\btidal\b",
    r"\bstorm\w*",
    r"\bcyclon\w*",
    r"\btyphoon\w*",
    r"\bhurricane\w*",
    r"\bdownpour\w*",
    r"\bdeluge\w*",
    r"\bcloud\s?burst\w*",
    r"\bembankment\w*",
    r"\blevee\w*",
    r"\bbarrage\w*",
    r"\bspillway\w*",
    r"\bwater\s?log\w*",
    r"\binundat\w*",
    r"\bsubmerg\w*",
    r"\bspate\b",
    r"\btorrential\w*",
    r"\bshower\w*",
    r"\bmonsoonal\b",
    r"\bbay\s+of\s+bengal\b",
    r"\barabian\s+sea\b",
    r"\bcreek\w*",
    r"\bcoast\w*",
    r"\bsurge\w*",
    r"\bdrown\w*",
    r"\bimd\b",
    r"\bsilt\w*",
    r"\bsluice\w*",
    r"\bponding\b",
    r"\bwaterway\w*",
    r"\bglacier\w*|\bglacial\b",
    r"\bsnow\s?melt\w*",
    # Deliberately NOT context: bare "sea", "wet", "wave(s)", "mud". Each let
    # unrelated news in - "wet market", "wave of COVID cases", "sea star
    # economic loss" - without adding flood recall the terms above miss.
]

CONTEXT_RE = _compile(CONTEXT)


# --------------------------------------------------------------------------
# Matcher
# --------------------------------------------------------------------------


def _overlaps(span, spans) -> bool:
    start, end = span
    return any(start < e and s < end for s, e in spans)


def is_flood_title(title: str, explain: bool = False):
    """Return True when `title` is about a real flood / hydro-met event.

    With explain=True, return a (bool, reason) tuple instead.
    """
    if not title:
        return (False, "empty") if explain else False

    text = GUARD_RE.sub(" ", normalize(title))

    strong = STRONG_RE.search(text)
    if strong:
        return (True, "strong:%s" % strong.group(0).strip()) if explain else True

    weak_spans = [m.span() for m in WEAK_RE.finditer(text)]
    if not weak_spans:
        return (False, "no-signal") if explain else False

    for ctx in CONTEXT_RE.finditer(text):
        if not _overlaps(ctx.span(), weak_spans):
            if explain:
                weak_term = WEAK_RE.search(text).group(0).strip()
                return True, "weak:%s + context:%s" % (weak_term, ctx.group(0).strip())
            return True

    return (False, "weak-without-context") if explain else False
