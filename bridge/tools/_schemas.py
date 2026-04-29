"""Tool JSON Schema definitions (Ollama tools format).

Each entry includes routing_patterns for the pre-classifier.
The classifier reads them at runtime from this list.
"""
from __future__ import annotations

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Current date/time. Use for: 'time', 'what time', 'what date', 'today'.",
            "routing_patterns": [
                r"\bwhat('?s|\s+is)\s+(the\s+)?(time|date)\b",
                r"\bwhat\s+is\s+the\s+time\b",
                r"\b(current\s+time|time\s+now|what\s+day|today'?s\s+date)\b",
                r"\bwhat\s+(time|date)\b",
            ],
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "IANA zone like 'Asia/Tokyo'. Omit for local.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Current weather (ALWAYS use this for ANY weather question — rain, temperature, wind, forecast). Use for: 'weather', 'is it raining', 'how hot', 'what to wear'. Empty location = user's IP-detected city.",
            "routing_patterns": [
                r"\b(weather|forecast|temperature|humidity)\b",
                r"\b(is\s+it\s+(raining|snowing|hot|cold|windy|sunny|cloudy|chilly|humid))\b",
                r"\bwhat\s+(should\s+i\s+wear|to\s+wear)\b",
                r"\b(rain|snow|storm)\s+today\b",
                r"\bhow\s+(hot|cold|warm)\b",
                r"\b(need|bring)\s+(an?\s+)?(umbrella|jacket|coat|sunscreen)\b",
                # Wrapped-intent weather queries (iter-5 Phase 0):
                # "Can I BBQ outside today?", "Should I BBQ tomorrow?",
                # "Will it be nice for a run?" etc.
                r"\b(can|should|will)\s+i\s+(bbq|barbecue|grill|picnic|hike|run|swim|bike|cycle|walk|exercise)\s+(outside|today|tomorrow|this\s+(morning|afternoon|evening|weekend))\b",
                r"\bshaping\s+up\s+(for|to)\b",
                r"\bgood\s+day\s+(for|to)\s+(a|an|go|be)\b",
                r"\bwill\s+it\s+(rain|snow|be\s+(hot|cold|warm))\b",
                # Paraphrase expansion (Round 3 eval uncovered these):
                r"\bchilly\b",
                r"\bgrill\s+(outside|outdoors?|out\s+on)\b",
                r"\bthermometer\b",
                r"\bhow'?s\s+the\s+(sky|weather|air)\b",
            ],
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City, nickname ('Philly'), or 'lat,lon'. Empty = user's location.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_news",
            "description": "Recent news headlines (ALWAYS use this for ANY news question — NOT web_search). Use for: 'news', 'headlines', 'what's happening'. Optional topic filter.",
            "routing_patterns": [
                r"\b(news|headlines?|latest\s+stories|breaking(\s+news)?)\b",
                r"\bwhat'?s\s+happening\b",
                r"\bany\s+news\b",
                # Paraphrase expansion:
                r"\btop\s+stories?\b",
                r"\b(current|recent)\s+events?\b",
                r"\b(big|major)\s+stories?\b",
                r"\banything\s+(new|big|happening)\s+(with|in|about)\b",
            ],
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Optional keyword. Omit for top story.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_alerts",
            "description": "Active alerts (market shocks, travel advisories, watchlist news). Use for: 'alerts', 'advisories', 'anything I should know'.",
            "routing_patterns": [
                r"\b(alerts?|warnings?|advisor(?:y|ies)|emergenc(?:y|ies))\b",
                r"\banything\s+i\s+should\s+know\b",
                # Paraphrase expansion — specific alert types:
                r"\b(tornado|hurricane|flood|flash\s+flood|storm)\s+(watch|warning)\b",
                r"\bsevere\s+weather\b",
                r"\bair\s+quality\b",
                r"\banything\s+dangerous\b",
            ],
            "parameters": {
                "type": "object",
                "properties": {
                    "max_results": {
                        "type": "integer",
                        "description": "Cap on alerts. Default 5.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_digest",
            "description": "Today's digest — top confirmed news, active alerts, notable tracker moves. Use for: 'catch me up', 'what's today's digest', 'summary of today', 'what's new'.",
            "routing_patterns": [
                r"\b(digest|daily\s+brief|morning\s+brief|my\s+brief)\b",
                r"\bwhat\s+do\s+i\s+(care\s+about|need\s+to\s+know)\b",
                r"\b(catch\s+me\s+up|brief\s+me|what'?s\s+new)\b",
                # Paraphrase expansion:
                r"\bmorning\s+(summary|briefing|roundup)\b",
                r"\btoday'?s\s+(summary|roundup|highlights)\b",
                r"\bwhile\s+i\s+was\s+away\b",
                r"\bwhat'?s\s+important\s+(right\s+now|today)\b",
                # Paraphrase expansion (iter-5 Phase 0):
                r"\b(on\s+my\s+)?agenda\s+(today|this\s+morning|this\s+afternoon)\b",
                r"\brun\s+through\s+(what'?s\s+on|today'?s|the)\b",
            ],
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tracker",
            "description": "Market/FX/commodity values (ALWAYS use this for ANY price question). Use for: 'gold', 'silver', 'oil', 'USD/JPY', 'gasoline', 'food CPI', 'gold price'. Empty id = grid. Set period for long-window stats ('gold over last month').",
            "routing_patterns": [
                r"\b(price|quote|value|rate|cost)\s+(of|for)\s+(gold|silver|oil|gas|copper|platinum|btc|bitcoin|eth)\b",
                r"\b(gold|silver|oil|copper|platinum|bitcoin|btc|ethereum|eth)\s+(price|prices|quote|value|today)\b",
                r"\b(gas\s+price|fuel\s+price|oil\s+price)\b",
                r"\b(usd|eur|jpy|gbp|cny|hkd|aud|cad|chf)\s*/\s*(usd|eur|jpy|gbp|cny|hkd|aud|cad|chf)\b",
                r"\bexchange\s+rate\b",
                r"\bhow\s+much\s+is\s+(gold|silver|oil|bitcoin)\b",
                # Comparative (iter-5 Phase 0): "which costs more, gold or silver?"
                r"\bwhich\s+(costs?|is)\s+(more|higher|bigger|worth\s+more)\b.*(gold|silver|oil|btc|bitcoin|eth|ethereum|copper|platinum|jpy|usd|eur)",
                r"\b(gold|silver|oil|bitcoin|gasoline|food)\s+(index|cpi)\b",
                r"\b(gold|silver|oil|bitcoin|eth)\s+(trading|worth|going\s+for)\b",
                r"\b(stock|market|dow|s&p|nasdaq|nikkei)\b",
                # Paraphrase expansion:
                r"\bhow'?s\s+(gold|silver|oil|bitcoin|btc|eth|ethereum|the\s+market)\s+doing\b",
                r"\b(bitcoin|btc|eth|ethereum|crypto)\b",
            ],
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Tracker id or alias: 'gold', 'usd_cny', 'rbob_gasoline', 'food'. Empty = grid.",
                    },
                    "period": {
                        "type": "string",
                        "description": "Optional long-window span: '1w', '1m', '3m', '6m', '1y'. Empty = latest snapshot only.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "math",
            "description": (
                "Symbolic + numeric math. Pick an 'op' exactly matching "
                "one in the enum. Matrix input: '[[1,2],[3,4]]'. Vector "
                "input: '[1,2,3]'. Two-vector ops (dot, mse, mae, "
                "cross_entropy) use '[1,2,3] * [4,5,6]'. Use this tool "
                "rather than guessing — don't invent softmax/norm/etc. "
                "values yourself."
            ),
            "routing_patterns": [
                r"\b(integrate|differentiate|derivative|solve\s+for|factor|simplify|expand)\b",
                # "Solve <expr> = <expr> (for <var>)" — catches equation-style
                # prompts like "Solve 2x + 5 = 11 for x." that the
                # "solve for" pattern above misses (solve isn't adjacent
                # to "for"). Requires an '=' so we don't over-match
                # casual "solve this".
                r"\bsolve\b[^.\n]*=",
                r"\bsolve\b[^.\n]*\bequals?\s+\d",   # "Solve ... equals 0" (word form)
                r"\b(eigenvalues?|eigenvectors?|determinant|inverse\s+of|transpose)\b",
                r"\b(laplace|fourier|fft|ifft)\s+(transform)?\b",
                # Calculus — series, limits, optimization.
                r"\b(taylor|maclaurin)\s+(series|expansion)?\b",
                r"\bseries\s+(expansion|of)\b",
                r"\blim(it)?\s+(of\s+|as\b)",
                r"\b(critical\s+points?|local\s+(min|max|extrema)|optimize|minimize|maximize)\b",
                r"\b(maximum|minimum)\s+(of\b|value)",
                # ML-oriented ops — dot/norm/softmax/activations/stats.
                r"\b(dot\s+product|inner\s+product)\b",
                r"\b(l1|l2|l\\u221e|l-infinity|norm)\s+(of\s+)?\[",
                r"\b(softmax|sigmoid|tanh|relu)\s+(of\s+)?",
                r"\b(mean|variance|std(ev)?|standard\s+deviation)\s+(of\s+)?\[",
                r"\b(mse|mean\s+squared\s+error|cross[-\s]entropy)\b",
                r"\b(shannon\s+)?entropy\s+(of\b|\[)",
                r"\b(kl|kullback[-\s]?leibler)\s+divergence\b",
                r"\b(js|jensen[-\s]?shannon)\s+divergence\b",
                r"\bkl\s+(between|of)\s+(two\s+)?(normal|gaussian|beta|exp|bernoulli)",
                r"\bentropy\s+of\s+(a\s+|the\s+)?(normal|gaussian|beta|exp|bernoulli|uniform)",
                # Bare "N(0,1) || N(1,2)" / "Beta(2,5) vs Beta(3,4)" shorthand.
                # Requires two Name(args) calls joined by ||/vs/between+and, so
                # we don't match arbitrary parenthesised expressions.
                r"\b(?:N|Normal|Gaussian|Exp|Exponential|Beta|Bern|Bernoulli|Uniform|U)\s*\([^)]*\)\s*(?:\|\||vs\.?|against)\s*(?:N|Normal|Gaussian|Exp|Exponential|Beta|Bern|Bernoulli|Uniform|U)\s*\(",
                r"\bH\s*\(\s*(?:N|Normal|Gaussian|Exp|Exponential|Beta|Bern|Bernoulli|Uniform|U)\s*\(",
                r"(?<![=^\w])\d+(\.\d+)?\s*(times|x|\*|plus|\+|minus|-|divided\s+by|/)\s*\d+(\.\d+)?\b",
                r"\b\d+(\.\d+)?\s*(%|percent)\s+of\s+\d+(\.\d+)?\b",
                r"\b(what'?s|what\s+is|compute|calculate)\s+\d",
            ],
            "parameters": {
                "type": "object",
                "properties": {
                    "op": {
                        "type": "string",
                        "enum": [
                            "evaluate", "solve", "simplify", "factor",
                            "expand", "integrate", "differentiate",
                            # Calculus — series, limits, optimisation.
                            # For taylor: lower=center, upper=order.
                            # For limit: lower=target (supports 'inf').
                            # For extrema: lower/upper optional interval.
                            "taylor", "series", "maclaurin",
                            "limit",
                            "extrema", "optimize", "minimize", "maximize",
                            "rref", "det", "inverse", "transpose",
                            "eigenvalues", "eigenvectors", "multiply",
                            "laplace", "inverse_laplace",
                            "fourier", "inverse_fourier",
                            "fft", "ifft",
                            # ML / numeric vector ops
                            "dot", "norm", "softmax",
                            "sigmoid", "tanh", "relu",
                            "mean", "variance", "std",
                            "mse", "mae", "cross_entropy",
                            # Information theory (discrete — vector input)
                            "entropy",
                            "kl_divergence", "kl",
                            "js_divergence", "js",
                            # Distribution-aware (closed-form) variants.
                            # Accepts Normal(N), Exponential(Exp), Beta,
                            # Bernoulli(Bern), Uniform. Input examples:
                            #   kl_dist:     "N(0, 1) || N(1, 2)"
                            #   entropy_dist: "Beta(2, 5)"
                            "kl_dist", "entropy_dist",
                        ],
                    },
                    "expression": {
                        "type": "string",
                        "description": (
                            "Expression / equation / matrix / matrix pair "
                            "/ vector / two-vector pair. Single vector: "
                            "'[1,2,3]'. Two-vector ops (dot, mse, mae, "
                            "cross_entropy, kl, js): pass BOTH vectors in "
                            "ONE string, e.g. expression='[0.9, 0.1] * "
                            "[0.5, 0.5]'. Separators '*', 'and', ',' all "
                            "work, but never omit either vector."
                        ),
                    },
                    "variable": {
                        "type": "string",
                        "description": (
                            "Variable name. Default 'x'. For Laplace use "
                            "'t'; for Fourier use 'x'. For op='norm' pass "
                            "'1', '2' (default), or 'inf' as the Lp order."
                        ),
                    },
                    "transform_var": {
                        "type": "string",
                        "description": (
                            "Target variable for transforms. Default 's' "
                            "for Laplace, 'k' for Fourier."
                        ),
                    },
                    "lower": {
                        "type": "string",
                        "description": (
                            "Multi-purpose numeric slot. "
                            "integrate: lower bound. "
                            "taylor: expansion center (default 0). "
                            "limit: target value ('inf' / '-inf' allowed). "
                            "extrema: optional interval lower bound."
                        ),
                    },
                    "upper": {
                        "type": "string",
                        "description": (
                            "Multi-purpose numeric slot. "
                            "integrate: upper bound. "
                            "taylor: number of terms / order (default 6). "
                            "limit: direction '+' or '-' for one-sided. "
                            "extrema: optional interval upper bound."
                        ),
                    },
                },
                "required": ["op", "expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_places",
            "description": "Places nearby. Use for: 'pizza near me', 'best X', 'coffee shops'. Empty location = user's city.",
            "routing_patterns": [
                r"\bnear\s+me\b",
                r"\b(nearby|around\s+here|close\s+by)\b",
                r"\bwhere\s+can\s+i\s+(find|get|buy)\b",
            ],
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to find: 'pizza', 'ramen', 'bookstore'. No location.",
                    },
                    "location": {
                        "type": "string",
                        "description": "Optional city. Empty = user's location.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Open-web search (DuckDuckGo). ALWAYS use this (not wiki) "
                "for fix-it / how-to / shopping / review queries. "
                "Concrete triggers: 'how do I fix X', 'how to Y', "
                "'best X 2026', 'top rated Y', 'review of Z', "
                "'DIY X plans', 'X tutorial', 'X recipe'. These are "
                "NEVER wiki queries. Also use for any niche lookup that "
                "isn't an encyclopedia-style fact. Do NOT use for: "
                "weather (get_weather), news (get_news), prices (tracker), "
                "or 'who is X' / 'when did Y happen' / 'history of Z' "
                "(wiki)."
            ),
            "routing_patterns": [
                r"\b(recipe|how\s+to|review[s]?|tutorial)\b",
                r"\bsearch\s+(for|the\s+web|online)\b(?!.*\bwiki)",
                r"\bgoogle\b",
                # Paraphrase of "how to" — imperative form "how do I X"
                # hit none of the above before and was under-firing.
                r"\bhow\s+do\s+i\s+\w+",
            ],
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keywords.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wiki",
            "description": (
                "Wikipedia lookup. Use for ENCYCLOPEDIA-STYLE FACTS: "
                "'who is X', 'what is Y', 'when did Z happen', historical "
                "events, biographies, scientific concepts, geographical "
                "facts. Do NOT use for how-to instructions, DIY, product "
                "recommendations, or niche guides — those go to "
                "web_search. Empty query = random article."
            ),
            "routing_patterns": [
                r"\b(who\s+(is|was|were)|what\s+(is|was|are|were))\s+(?!the\s|a\s|an\s|it\s|my\s|your\s|this\s|gold\s|silver\s|oil\s|bitcoin\s|eth\s)[a-z]\w*",
                r"\btell\s+me\s+about\s+(?!yourself|you\b|my\s)\w",
                r"\brandom\s+fact\b",
                r"\bwikipedia\b",
                r"\blook\s+up\b",
                r"\bhow\s+old\s+(is|was|are|were)\b",
                r"\bwhen\s+was\s+\w+\s+(born|died)\b",
                r"\b(birth|death)\s*(day|date)\s+(of|for)\b",
                r"\bwhat\s+year\s+(was|did)\s+\w+\s+(born|die)\b",
                r"\bwho\s+(invented|discovered|created|founded|built|wrote|composed|designed|developed)\b",
                r"\bwhere\s+(is|was|are|were)\s+(?!the\s+(weather|time|date|news))\w",
                r"\bwhen\s+did\s+\w+",
                r"\b(history|origin|meaning)\s+of\b",
                # Definition queries: "what does X mean", "define X",
                # "definition of X". Routes to wiki rather than
                # web_search for encyclopedic word/concept lookups.
                r"\bwhat\s+does\s+[\"']?\w+[\"']?\s+mean\b",
                r"\bdefine\s+[\"']?\w+",
                r"\bdefinition\s+of\s+\w+",
                # Historical era questions: "in 1970 america", "in 1980
                # europe". REMOVED money|dollars from this group — those
                # now route to the `inflation` tool which has actual data.
                r"\bin\s+\d{4}s?\s+(value|terms|america|europe)\b",
                # "how much was X in YYYY" — only when X is NOT money-shaped
                # AND not a known consumer item. Money-shaped or item-shaped
                # variants ("how much was a dollar in 1970", "how much was
                # gas in 1985") match inflation patterns and route there.
                r"\bhow\s+much\s+(was|were|did)\s+(?!.*\b(?:cost|worth|dollars?|money|\$|gasoline|gas|petrol|bread|eggs|milk|beef|chicken|bacon|bananas?|tomatoes?|coffee|sugar|electricity|kwh|therm)\b).+\s+in\s+\d{4}s?\b",
                # Historical-cost phrasings: "how much did a Model T cost
                # when it was new". Item-level queries (no specific year)
                # for things wiki actually has articles on. Negative
                # lookahead filters BLS item words AND money words
                # (dollar/buck) so they reach inflation instead.
                r"\bhow\s+much\s+(was|were|did)\s+(a|an|the|my)\s+(?!(?:gallon|loaf|pound|lb|dozen|carton|pint|quart|stick|bottle)\s+of\s+\w+|gasoline|gas|petrol|bread|eggs|milk|beef|chicken|bacon|bananas?|tomatoes?|coffee|sugar|electricity|kwh|therm|natural\s+gas|dollar|bucks?).+?\s+(cost|priced|worth)\b",
                r"\bwhat\s+was\s+the\s+price\s+of\s+(?!(?:gas|gasoline|petrol|bread|eggs|milk|beef|chicken|bacon|bananas?|tomatoes?|coffee|sugar|electricity|natural\s+gas)\b)\w+",
                # Construction / founding / invention events: "when was
                # the Berlin Wall built", "when was X founded".
                r"\bwhen\s+was\s+.+\s+(built|founded|established|created|invented|discovered|published|opened|written|composed)\b",
                # Death/birth years of multi-word subjects: "what year
                # did Queen Elizabeth II die?" (prior pattern only
                # matched single-token subjects between ``did`` and the
                # verb).
                r"\bwhat\s+year\s+did\s+.+?\s+(die|born|happen|launch|start|end|begin|occur)\b",
                # Current-X title queries: "who's the current prime minister
                # of the UK", "who's the president right now". Wiki is more
                # up-to-date than pretraining on political / leadership
                # holders of named offices. Guards against the LoRA stating
                # stale parametric facts as current (iter-7 Theme 5 case 26).
                r"\b(current|latest|today'?s|present)\s+(prime\s+minister|president|chancellor|ceo|king|queen|monarch|pope|emperor|empress|head\s+of\s+(state|government))\b",
                r"\bwho(?:'s|\s+is)\s+(?:the\s+)?(?:current|latest|today'?s|present)\s+\w+",
                r"\b(president|prime\s+minister|ceo|king|queen|monarch|chancellor)\s+(right\s+now|today|these\s+days|currently)\b",
            ],
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Topic (person, place, event). Empty = random article.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "convert",
            "description": "Unit conversion. Use for: 'X mi to km', 'C to F', 'oz to g'.",
            "routing_patterns": [
                r"\b\d+(\.\d+)?\s*(miles?|km|kg|lbs?|pounds?|feet|meters?|inches?|cm|celsius|fahrenheit|gallons?|liters?|oz|ounces?|cups?|tbsp|tsp)\s+(to|in)\s+\w+\b",
                r"\b\d+(\.\d+)?\s*(usd|eur|jpy|gbp|cny|hkd|aud|cad|chf|krw|twd)\s+(to|in)\s+(usd|eur|jpy|gbp|cny|hkd|aud|cad|chf|krw|twd)\b",
                r"\bconvert\s+\d",
                r"\bhow\s+(many|much)\s+\w+\s+(is|are|in)\s+\d",
            ],
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "string", "description": "Numeric value."},
                    "from_unit": {"type": "string", "description": "'mile', 'celsius', 'mi', etc."},
                    "to_unit": {"type": "string", "description": "Target unit."},
                },
                "required": ["value", "from_unit", "to_unit"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "graph",
            "description": "Plot a 1-var function. Use for: 'plot', 'graph', 'draw'. Multiple curves via ';' ('sin(x); cos(x)').",
            "routing_patterns": [
                r"\b(plot|graph|draw|chart)\s+\w",
                r"\by\s*=\s*",
                r"\bf\s*\(\s*x\s*\)\s*=",
                # Paraphrase expansion (iter-5 Phase 0):
                r"\bchart\s+(the\s+)?(last|next|past)\s+\d",
                r"\bshow\s+(me\s+)?(a\s+)?(graph|chart|plot)\s+of\b",
            ],
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Expr(s) to plot. ';' separates curves."},
                    "variable": {"type": "string", "description": "Default 'x'."},
                    "x_min": {"type": "number", "description": "Default -10."},
                    "x_max": {"type": "number", "description": "Default 10."},
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "circuit",
            "routing_patterns": [
                r"\b(voltage\s+divider|rc\s+circuit|rl\s+circuit|rlc|impedance)\b",
                r"\b(ohm'?s\s+law)\b",
                # Op-verb patterns (Round 2 eval fixes — classifier was
                # abstaining on these and the model wouldn't route from
                # description alone):
                r"\b(series|parallel)\s+(resistance|capacitance|inductance|resistor)\b",
                r"\b(rc|rl|lc)\s+(time\s+constant|cutoff(\s+frequency)?|resonance)\b",
                r"\btime\s+constant\s+(for|of)\b",
                r"\btruth\s+table\b",
                r"\blogic\s+(eval|gate|expression)\b",
            ],
            "description": (
                "Circuit calc (analog + digital). Ops: "
                "resistance_parallel, resistance_series, rc_time_constant, "
                "rc_cutoff, rl_cutoff, lc_resonance, impedance, "
                "voltage_divider, logic_eval, truth_table, synthesize. "
                "SI prefixes OK (k/M/m/u/n/p)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "op": {
                        "type": "string",
                        "enum": [
                            "resistance_parallel", "resistance_series",
                            "rc_time_constant", "rc_cutoff", "rl_cutoff",
                            "lc_resonance", "impedance", "voltage_divider",
                            "logic_eval", "truth_table", "synthesize",
                        ],
                    },
                    "values": {
                        "type": "string",
                        "description": "Comma-separated list (resistance_parallel/series).",
                    },
                    "R": {"type": "string", "description": "Resistance (ohms, SI prefixes OK)."},
                    "L": {"type": "string", "description": "Inductance (H)."},
                    "C": {"type": "string", "description": "Capacitance (F)."},
                    "R1": {"type": "string", "description": "Voltage-divider top resistor."},
                    "R2": {"type": "string", "description": "Voltage-divider bottom resistor."},
                    "Vin": {"type": "string", "description": "Voltage-divider input voltage."},
                    "component": {
                        "type": "string",
                        "enum": ["R", "L", "C"],
                        "description": "Component kind for impedance.",
                    },
                    "value": {"type": "string", "description": "Component value for impedance."},
                    "frequency": {"type": "string", "description": "Frequency in Hz."},
                    "expression": {
                        "type": "string",
                        "description": "Boolean expression (logic_eval / truth_table).",
                    },
                    "inputs": {
                        "type": "string",
                        "description": "Inputs like 'A=1,B=0' (logic_eval).",
                    },
                    "variables": {
                        "type": "string",
                        "description": "synthesize: variable names, e.g. 'A,B,C'.",
                    },
                    "minterms": {
                        "type": "string",
                        "description": (
                            "synthesize: comma-separated minterm indices "
                            "where the output is 1, e.g. '1,3,5,7'."
                        ),
                    },
                    "dontcares": {
                        "type": "string",
                        "description": "synthesize: optional don't-care indices.",
                    },
                },
                "required": ["op"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_reminder",
            "routing_patterns": [
                r"\bremind\s+me\b",
                r"\bset\s+(a\s+|an\s+)?(reminder|alarm|timer|alert)\s+(for|at|in)\b",
                r"\bdon'?t\s+(let\s+me\s+)?forget\b",
                # Paraphrase expansion (Round 2 of the round-3 eval fixes):
                r"\balarm\s+(for|at)\s+\d",  # "alarm for 6am"
                r"\b(wake|nudge)\s+me\s+(up\s+)?(at|in)\b",  # "wake me up at 7am", "nudge me in 15 min"
                # Conditional-trigger reminders (iter-5 Phase 0): "when I
                # get home, remind me to X", "if traffic's bad, ping me",
                # "once dinner's done, nudge me about pills". These have
                # a conditional clause + a reminder verb.
                r"\b(when|once|after|as\s+soon\s+as)\s+.+,?\s+(remind|ping|nudge|wake|alert|tell)\s+(me|us)\b",
                r"\bif\s+.+,?\s+(remind|ping|nudge|wake|alert|tell)\s+(me|us)\b",
            ],
            "description": (
                "Schedule a future reminder, timer, alarm, or nudge. Use "
                "for: 'remind me to X at 5pm', 'set a timer for 10 minutes', "
                "'alarm for 6am', 'wake me up at 7am', 'nudge me in 15 min', "
                "'don't let me forget the dentist Wednesday'. Do NOT "
                "call this for 'what time is it'. For raw facts the user "
                "wants you to remember about THEM (names, preferences), "
                "use update_memory instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "trigger_at": {
                        "type": "string",
                        "description": (
                            "ISO 8601 UTC timestamp when the reminder "
                            "should fire. Example: '2026-04-16T22:00:00Z'. "
                            "Must be in the future, within one year."
                        ),
                    },
                    "message": {
                        "type": "string",
                        "description": (
                            "Short description of what to remind about, "
                            "like 'call mom' or 'doctor appointment'."
                        ),
                    },
                },
                "required": ["trigger_at", "message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_memory",
            "description": (
                "Save a durable, actionable fact about the USER (name, "
                "occupation, allergies, location, canonical preferences). "
                "Call only when the user is explicitly telling you about "
                "themselves in a way that would affect future replies "
                "('my name is X', 'I'm allergic to Y', 'I study Z'). Do "
                "NOT call for casual emotional expressions ('I love jazz', "
                "'I hate Mondays'), reminiscing ('remember that time...'), "
                "or conversational asides ('call me when you're free')."
            ),
            "routing_patterns": [
                # "remember that MY/I X" — anchors the reminisce vs fact split
                r"\bremember\s+that\s+(my|i['\s])",
                # Canonical fact patterns — tight anchors with object shape
                r"\bmy\s+(name|birthday|gym|favorite|allergy|occupation|preference|job|school|hometown)\s+(is|are)\b",
                # Identity/habit statements — restricted verb list with copula
                r"\bi\s+(am|work|study|live)\s+\w",
                r"\bi'?m\s+allergic\s+to\b",
                # "call me X from now on" — name setter; the "from now on"
                # disambiguates from "call me when" / "call me back"
                r"\bcall\s+me\s+\w+\s+(from\s+now\s+on|instead)\b",
            ],
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {
                        "type": "string",
                        "description": "A concise fact about the user, like 'prefers short answers' or 'studies EE at Penn State'.",
                    },
                },
                "required": ["fact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "say",
            "description": (
                "Repeat a phrase verbatim back to the user. Call this "
                "ONLY when the user explicitly asks to be echoed — "
                "phrasings like 'repeat after me X', 'say X', 'echo "
                "this back: X', 'parrot these words: X'. Set the `text` "
                "argument to the EXACT phrase the user wants repeated, "
                "stripping framing words ('after me', 'this back') and "
                "any surrounding quotes. Do NOT use this for normal "
                "conversation, definition lookups, or pronunciation "
                "questions ('how do you say X' goes to wiki / web search)."
            ),
            # Conservative anchors — every pattern requires an imperative
            # framing word that natural language rarely has ("repeat after
            # me", "echo this", a quoted phrase right after "say"). The
            # L2 regex hint nudges the LoRA when its description-only
            # routing misses; the schema description still does most of
            # the work.
            "routing_patterns": [
                # "repeat after me ...", "say after me ..." — strongest signal
                r"\b(?:repeat|say|echo|parrot)\s+after\s+me\b",
                # "repeat this:", "echo this back", "say the following"
                r"\b(?:repeat|echo|parrot)\s+(?:this|these|the\s+following)\b",
                # `say "..."` / `repeat '...'` — quoted-phrase imperative
                r"\b(?:repeat|say|echo)\s+['\"]",
                # "say exactly", "repeat exactly" — emphatic verbatim cue
                r"\b(?:repeat|say)\s+exactly\b",
                # `say <text>` at the start of the message. Anchored to ^
                # so mid-sentence "did you say...", "I'd say...", "how do
                # you say..." don't fire — the imperative form requires
                # `say` as the very first non-whitespace word. Added
                # 2026-04-28 alongside the schema-leak scrub: as the tool
                # count reached 21, plain "say hello world" stopped
                # description-only routing reliably and the LoRA started
                # narrating the catalog instead.
                r"^\s*say\s+\S",
            ],
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The exact phrase to repeat, with no framing or quotes.",
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inflation",
            "description": (
                "Calculate purchasing-power equivalence between two years "
                "using authoritative BLS data. Examples: 'how much is $1 "
                "in 1970 worth today', 'what would a 1965 dollar be "
                "today', 'inflation between 1980 and 2020', 'how have "
                "wages kept up with inflation since 1970', 'how much was "
                "a gallon of gas in 1985', 'eggs in 1990'. US-only in "
                "v2 (international planned). Returns a structured JSON "
                "with the equivalent amount + source + confidence label "
                "+ caveats — paraphrase those, do not invent numbers. "
                "Set measure='wages' for hourly-wage equivalence (BLS "
                "AHETPI, 1964+) or measure='both' for CPI + wages + "
                "real-wage delta. Set item=<key> for a concrete consumer "
                "item price (BLS AP series, 1980+): gasoline, bread, "
                "eggs, milk_gallon, ground_beef, chicken, bacon, bananas, "
                "tomatoes, coffee, sugar, electricity, natural_gas."
            ),
            "routing_patterns": [
                # "$1 in 1970", "$50 in 1965 dollars"
                r"\$\d[\d,.]*\s+(?:in|from)\s+\d{4}\b",
                # "1970 dollars", "1970 money", "1970 prices/wages"
                r"\b\d{4}\s+(?:dollars?|money|prices?|wages?)\b",
                # "in 1970 money", "in 1965 dollars"
                r"\bin\s+\d{4}\s+(?:money|dollars?|prices?|wages?)\b",
                # "a/the dollar in 1865", "dollar in 1900" — common
                # phrasing that doesn't quote a $-amount but is clearly
                # about purchasing power, not a Wikipedia article.
                r"\b(?:a|the|one|each|that)?\s*dollar\s+(?:in|from|of|back\s+in)\s+\d{4}\b",
                # "inflation since/from/between/over"
                r"\binflation\s+(?:from|since|between|in|over)\s+",
                # "purchasing power"
                r"\bpurchasing\s+power\b",
                # "worth today/now/in today's money" — but only when a
                # year or money cue is also present nearby. Without that
                # gate the pattern swallows generic "X worth today" spot-
                # price queries like "gold worth today" (which is a
                # tracker, not an inflation, intent).
                r"(?:\$|\bdollars?\b|\bbucks?\b|\b\d{4}\b|\bsalary\b|\bwages?\b|\bpay(?:check)?\b|\bsavings?\b|\bmoney\b).{0,40}\bworth\s+(?:today|now|in\s+today)",
                r"\bworth\s+(?:today|now|in\s+today).{0,40}(?:\$|\bdollars?\b|\bbucks?\b|\b\d{4}\b|\bsalary\b|\bwages?\b)",
                # "how much did X cost in YYYY"
                r"\bhow\s+much\s+(?:did|was|were)\s+.+(?:cost|worth)\s+in\s+\d{4}\b",
                # "how much was X worth today" — covers "how much was a
                # dollar in 1865 worth today" where YYYY is BEFORE
                # "worth today" (different from cost-in-YYYY pattern).
                r"\bhow\s+much\s+(?:was|were|did)\s+.+\s+in\s+\d{4}\b.{0,20}\bworth\b",
                # "wage/salary/paycheck/earnings ... in YYYY" (catches
                # "what was the average wage in 1980", "salary in 1970",
                # "earnings vs today" — without this the LoRA
                # fabricates wage numbers from training memory).
                r"\b(?:wages?|salar(?:y|ies)|paycheck|earnings?|hourly\s+(?:pay|rate|wage))\b.{0,40}\b(?:in|from|since|of|back\s+in)\s+\d{4}\b",
                # "wages/salary kept up with inflation", "real wages"
                r"\b(?:real\s+wages?|wages?\s+(?:keep|kept|keeping|outpace[ds]?|lag(?:ged|ging)?))\b",
                r"\b(?:keep|kept|keeping)\s+(?:up\s+)?with\s+inflation\b",
                # Cross-region comparison phrasings.
                r"\bcompare\s+inflation\b",
                r"\binflation\s+(?:between|across|in\s+(?:the\s+)?(?:us|usa|united\s+states|japan|china|korea|hong\s*kong))\s+(?:and|vs\.?|versus|to)\b",
                # Hong Kong SAR, China (v3 region) — "in HK", "Hong
                # Kong dollar", "HKD in YYYY". Catches both "X in YYYY
                # in Hong Kong" and "Hong Kong CPI in YYYY" shapes.
                r"\bhong\s*kong\b.{0,40}\b(?:in|from|since|of|cpi|inflation|dollar|hkd)\b",
                r"\b(?:in|from|since|of)\s+\d{4}.{0,40}\bhong\s*kong\b",
                r"\bhkd\b|\bhk\$",
                # China Mainland — "in mainland China", "Chinese yuan",
                # "RMB", "1 yuan in YYYY".
                r"\b(?:mainland\s+china|china\s+mainland|prc)\b.{0,40}\b(?:in|from|since|of|cpi|inflation|yuan|rmb|cny)\b",
                r"\b(?:in|from|since|of)\s+\d{4}.{0,40}\b(?:mainland\s+china|china\s+mainland)\b",
                r"\b(?:cny|rmb|renminbi|chinese\s+yuan)\b",
                r"\b\d+\s+yuan\b",
                # Japan — "yen in YYYY", "Japanese inflation"
                r"\bjapan(?:ese)?\b.{0,40}\b(?:in|from|since|of|cpi|inflation|yen|jpy)\b",
                r"\b(?:in|from|since|of)\s+\d{4}.{0,40}\bjapan(?:ese)?\b",
                r"\b(?:jpy|japanese\s+yen)\b|\b\d+\s+yen\b",
                # South Korea — "Korean won in YYYY", "South Korea inflation"
                r"\b(?:south\s+korea(?:n)?|rok)\b.{0,40}\b(?:in|from|since|of|cpi|inflation|won|krw)\b",
                r"\b(?:in|from|since|of)\s+\d{4}.{0,40}\b(?:south\s+korea(?:n)?|rok)\b",
                r"\b(?:krw|korean\s+won)\b|\b\d+\s+won\b",
                # Item queries: "<item> in YYYY", "<item> cost in YYYY",
                # "how much was/did <item> cost in YYYY". The item word
                # alone disambiguates from generic CPI queries — the
                # extractor's _ITEM_PHRASE_MAP fills the `item` arg.
                r"\b(?:gasoline|gas|petrol|bread|eggs|milk|beef|chicken|bacon|bananas?|tomatoes?|coffee|sugar|electricity|kwh|natural\s+gas)\b.{0,40}\b(?:in|from|since|of|back\s+in)\s+\d{4}\b",
                r"\b(?:cost|price|worth|value)\s+of\s+(?:a\s+|the\s+|some\s+)?(?:gallon\s+of\s+|loaf\s+of\s+|pound\s+of\s+|lb\s+of\s+|dozen\s+|carton\s+of\s+)?(?:gasoline|gas|petrol|bread|eggs|milk|beef|chicken|bacon|bananas?|tomatoes?|coffee|sugar|electricity|natural\s+gas)\b",
                r"\bhow\s+much\s+(?:was|were|did)\s+(?:a\s+|an\s+|the\s+)?(?:gallon\s+of\s+|loaf\s+of\s+|pound\s+of\s+|dozen\s+)?(?:gasoline|gas|petrol|bread|eggs|milk|beef|chicken|bacon|bananas?|tomatoes?|coffee|sugar|electricity|natural\s+gas)\b",
            ],
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {
                        "type": "number",
                        "description": "Dollar amount in source-year terms.",
                    },
                    "from_year": {
                        "type": "integer",
                        "description": "Source year (1860 - current). 1860-1912 uses NBER historical estimates rebased to BLS CPI-U at 1913 (confidence 'low'); 1913-1946 early BLS ('medium'); 1947+ modern BLS ('high'). Pre-1860 not supported.",
                    },
                    "to_year": {
                        "type": "integer",
                        "description": "Target year (defaults to most recent year in dataset).",
                    },
                    "measure": {
                        "type": "string",
                        "enum": ["cpi", "wages", "both"],
                        "description": "Use 'cpi' (default) for ordinary price/dollar-equivalence questions. Use 'wages' ONLY when the user explicitly mentions wages, salary, paycheck, hourly pay, or earnings (1964+ only). Use 'both' when the user is comparing wages to inflation or asking about real wages.",
                    },
                    "item": {
                        "type": "string",
                        "description": "Optional consumer item key. Set this when the user asks about a specific item (gas, bread, eggs, milk, beef, chicken, bacon, bananas, tomatoes, coffee, sugar, electricity, natural gas). US only — BLS AP series, 1980+.",
                    },
                    "region": {
                        "type": "string",
                        "enum": ["us", "hk_sar", "cn_mainland", "japan", "south_korea"],
                        "description": "Country/region. Default 'us' (BLS CPI-U, 1860+). 'hk_sar' = Hong Kong SAR, China (HKD, 1981+). 'cn_mainland' = China Mainland (CNY, 1986+). 'japan' = Japan (JPY, 1960+). 'south_korea' = South Korea (KRW, 1960+). Wages and items are US-only — pick 'us' (or omit) for those.",
                    },
                    "regions": {
                        "type": "string",
                        "description": "OPTIONAL CSV of region keys for cross-region comparison (e.g. 'us,japan' or 'us,hk_sar,cn_mainland'). When set, returns a comparison block with per-region CPI values + an overlay-friendly per-region series. Use this for 'compare inflation in X and Y' / 'X vs Y inflation' queries.",
                    },
                },
                "required": ["amount", "from_year"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "population",
            "description": (
                "Look up population for a country/region at a given year, "
                "or compute the change between two years. Examples: 'how "
                "many people lived in Japan in 1970', 'world population in "
                "1985', 'US population from 1980 to 2020', 'population of "
                "Hong Kong'. Returns a structured JSON with the population "
                "+ source — paraphrase, do not invent. Data: World Bank "
                "SP.POP.TOTL midyear estimates, 1960-current."
            ),
            "routing_patterns": [
                # "world population", "global population", "earth population"
                r"\b(?:world|global|earth|planet|worldwide)\s+population\b",
                # "population of <region>", "population in <region>"
                r"\bpopulation\s+(?:of|in)\s+(?:the\s+)?\w+",
                # "X population in YYYY"
                r"\b(?:us|united\s+states|hong\s*kong|china|japan|korea|world)\s+population\b",
                # "how many people..."
                r"\bhow\s+many\s+people\s+(?:are|were|live[ds]?|lived|are\s+there)\b",
                # "population growth/change/decline"
                r"\bpopulation\s+(?:growth|change|decline|increase|decrease)\b",
                # Rank queries: "most populous", "biggest country", "top N populous"
                r"\bmost\s+populous\b",
                r"\b(?:biggest|largest)\s+(?:countr(?:y|ies)|nations?)\b.{0,30}\bpopulation\b",
                r"\bpopulation\s+rank(?:ing)?s?\b",
                r"\btop\s+\d+\s+(?:countr(?:y|ies)|most\s+populous)\b",
                r"\bwhere\s+does\s+\w+\s+rank\b",
            ],
            "parameters": {
                "type": "object",
                "properties": {
                    "region": {
                        "type": "string",
                        "enum": ["world", "us", "hk_sar", "cn_mainland", "japan", "south_korea"],
                        "description": "Country/region. Default 'world'. Same region keys as the inflation tool plus 'world' for the global aggregate.",
                    },
                    "year": {
                        "type": "integer",
                        "description": "Single year of interest (1960-current). Use this for 'population in YEAR' queries.",
                    },
                    "from_year": {
                        "type": "integer",
                        "description": "Range query — start year. Use with to_year for 'population from X to Y' / 'how much did the population grow' queries.",
                    },
                    "to_year": {
                        "type": "integer",
                        "description": "Range query — end year (defaults to latest data point if from_year is set without to_year).",
                    },
                    "metric": {
                        "type": "string",
                        "enum": ["value", "rank"],
                        "description": "'value' (default): the population number for the year. 'rank': top-N most populous countries for the year, plus the region's position in the global list.",
                    },
                    "top": {
                        "type": "integer",
                        "description": "When metric='rank', how many countries to return (default 10, clamped 1-50).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "facts",
            "description": (
                "Curated 'YEAR in numbers' aggregator — packages "
                "population (world + region), inflation baseline ($1 "
                "in YEAR vs today), CPI ratio, and source citations "
                "into a single response. Use when the user asks "
                "open-ended questions about a year ('tell me about "
                "1985', 'what was 1965 like', 'fun facts about 1990') "
                "rather than a specific dollar/population question. "
                "Reuses the same region keys as inflation/population "
                "(us, hk_sar, cn_mainland, japan, south_korea) — "
                "default region: world for population + US for inflation."
            ),
            "routing_patterns": [
                # "tell me about 1985", "what was 1965 like", "fun facts about 1990"
                r"\btell\s+me\s+about\s+(?:the\s+year\s+)?\d{4}\b",
                r"\bwhat\s+was\s+\d{4}\s+like\b",
                r"\b(?:fun|interesting|cool)\s+facts?\s+(?:about|from|of)\s+(?:the\s+year\s+)?\d{4}\b",
                # "year YYYY in numbers" / "summary of YYYY"
                r"\b(?:year\s+)?\d{4}\s+in\s+numbers\b",
                r"\bsummary\s+of\s+(?:the\s+year\s+)?\d{4}\b",
                # "snapshot of YYYY", "overview of YYYY"
                r"\b(?:snapshot|overview|context)\s+of\s+(?:the\s+year\s+)?\d{4}\b",
            ],
            "parameters": {
                "type": "object",
                "properties": {
                    "year": {
                        "type": "integer",
                        "description": "The year to summarize (1960-current — CPI is broader but population caps the joint range).",
                    },
                    "region": {
                        "type": "string",
                        "enum": ["world", "us", "hk_sar", "cn_mainland", "japan", "south_korea"],
                        "description": "Optional region for the regional-population + regional-inflation breakdowns. World population + US inflation always included as the universal anchor.",
                    },
                },
                "required": ["year"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze",
            "description": (
                "Statistical analysis over a cached economic series "
                "(CPI, wages, BLS AP items, World Bank population). Use "
                "when the user asks about peaks ('what was the highest "
                "inflation year'), troughs, trends ('how fast are wages "
                "growing'), volatility, percentile rank ('how unusual "
                "was 2022'), z-score, real (inflation-adjusted) values, "
                "or correlations between two series. Pure-numeric, "
                "no LLM math — paraphrase the returned interpretation."
            ),
            "routing_patterns": [
                r"\b(?:peak|highest|maximum|max|record\s+high|all[-\s]time\s+high)\s+(?:cpi|inflation|wage|wages|population|gas(?:oline)?|price|prices|electricity)\b",
                r"\b(?:trough|lowest|minimum|min|record\s+low)\s+(?:cpi|inflation|wage|wages|population|gas(?:oline)?|price|prices)\b",
                r"\b(?:what\s+was\s+the\s+(?:highest|lowest|peak)\s+(?:year\s+for\s+)?)(?:cpi|inflation|wage|wages|gas(?:oline)?|price|population)\b",
                r"\b(?:trend|trajectory|slope)\s+(?:of\s+)?(?:cpi|inflation|wage|wages|gas(?:oline)?|price|population)\b",
                r"\bhow\s+(?:volatil(?:e|ity)|much\s+does\s+\w+\s+swing)\b",
                r"\bhow\s+(?:unusual|rare|extreme)\s+(?:was|is)\s+(?:cpi|inflation|gas|gasoline|wage|prices?)\s+in\s+\d{4}\b",
                r"\b(?:real|inflation[-\s]adjusted)\s+(?:wage|wages|gas|gasoline|price|prices)\b",
                r"\bcorrelat(?:e|ion)\s+(?:between|of)\s+\w+\s+(?:and|with|vs)\s+\w+\b",
            ],
            "parameters": {
                "type": "object",
                "properties": {
                    "series": {
                        "type": "string",
                        "description": (
                            "Series identifier. Examples: 'cpi', "
                            "'cpi:japan', 'wages', 'population', "
                            "'population:us', 'item:gasoline', "
                            "'item:bread'. Region keys mirror "
                            "inflation/population: us, hk_sar, "
                            "cn_mainland, japan, south_korea, world."
                        ),
                    },
                    "op": {
                        "type": "string",
                        "enum": [
                            "peak", "trough", "trend", "volatility",
                            "percentile_rank", "zscore", "deflate",
                            "correlate",
                        ],
                        "description": (
                            "Operation. peak/trough = argmax/argmin "
                            "year. trend = linear OLS slope + R² + "
                            "1-step projection. volatility = std of "
                            "YoY % changes. percentile_rank/zscore "
                            "need `year`. deflate = nominal → real in "
                            "`base_year` dollars (default latest). "
                            "correlate needs `series_b`."
                        ),
                    },
                    "year_from": {
                        "type": "integer",
                        "description": "Optional window start year.",
                    },
                    "year_to": {
                        "type": "integer",
                        "description": "Optional window end year.",
                    },
                    "year": {
                        "type": "integer",
                        "description": (
                            "Reference year for percentile_rank / zscore."
                        ),
                    },
                    "base_year": {
                        "type": "integer",
                        "description": (
                            "For op=deflate: target year for the real "
                            "series. Default = latest year."
                        ),
                    },
                    "series_b": {
                        "type": "string",
                        "description": (
                            "Second series for op=correlate. Same "
                            "format as `series`."
                        ),
                    },
                    "lag": {
                        "type": "integer",
                        "description": (
                            "For op=correlate: shift series_b by this "
                            "many years (positive = B leads A)."
                        ),
                    },
                },
                "required": ["series", "op"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "alice",
            "description": (
                "Estimate the share of US households that are ALICE — "
                "Asset Limited, Income Constrained, Employed (above the "
                "federal poverty line but below a household survival "
                "budget). Returns the budget breakdown (housing / food "
                "/ healthcare / childcare / transport / tech / taxes), "
                "ALICE threshold, % poverty + % ALICE estimates, and a "
                "cross-validation vs the published United for ALICE "
                "national figures. Use when the user asks 'how many "
                "people can't afford to live but don't qualify for "
                "benefits', 'what's the ALICE rate', 'survival budget', "
                "or compares poverty vs working-poor stats."
            ),
            "routing_patterns": [
                r"\bALICE\b",
                r"\b(?:asset[-\s]limited|income[-\s]constrained)\b",
                r"\bworking\s+poor\b",
                r"\bsurvival\s+budget\b",
                r"\bhow\s+many\s+(?:americans|people|households)\s+can'?t\s+afford\b",
                r"\b(?:above|over)\s+poverty\s+(?:line|level)\s+but\s+(?:below|under|can'?t)\b",
                r"\b(?:too\s+rich|too\s+poor)\s+(?:for|to\s+get)\s+(?:benefits|welfare|aid)\b",
            ],
            "parameters": {
                "type": "object",
                "properties": {
                    "year": {
                        "type": "integer",
                        "description": (
                            "Year to estimate (2016, 2018, 2020, "
                            "2022, 2023, 2024). Default = latest."
                        ),
                    },
                    "composition": {
                        "type": "string",
                        "enum": ["1A0K", "2A0K", "1A1K", "2A1K",
                                 "2A2K", "2A3K"],
                        "description": (
                            "Household composition. 1A0K=single adult, "
                            "2A0K=couple no kids, 1A1K=single parent + "
                            "1 child, 2A1K=couple + 1 child, "
                            "2A2K=couple + 2 children (canonical, "
                            "default), 2A3K=couple + 3 children. Each "
                            "composition picks the right housing "
                            "bedrooms, food adult-equivalents, "
                            "healthcare coverage tier, and federal "
                            "filing status (single/MFJ/HoH)."
                        ),
                    },
                    "household_size": {
                        "type": "integer",
                        "description": (
                            "Backward-compat alternative to "
                            "`composition`. 1→single, 2→couple, "
                            "3→couple+1kid, 4→canonical, 5+→larger."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
]
