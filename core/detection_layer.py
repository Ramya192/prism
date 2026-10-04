# core/detection_layer.py
# Which layer of the rules -> ML -> LLM chain actually decided a verdict.
#
# Every domain's detector agent returns one text block ("Risk Level: ...\n
# Reason: ...\nAction: ..."). The UI used to guess the deciding layer back out
# of the reason text, which mislabels an LLM "BLOCK" with a short reason as a
# rule hit. Instead each detector's analyse() stamps a "Decided by:" line via
# tag_layer(), and its parse_response() reads it back via parse_decided_by()
# into parsed["decided_by"]. A parsed dict without the key (older responses,
# a detector that doesn't tag) falls back to the old inference in ui/shared.py.

RULE_ENGINE = "Rule Engine"
ML_DETECTOR = "ML Detector"
LLM_REASONING = "LLM Reasoning"

DECIDING_LAYERS = {RULE_ENGINE, ML_DETECTOR, LLM_REASONING}

_PREFIX = "decided by"


def tag_layer(response: str, layer: str) -> str:
    """Appends the deciding-layer line to a detector response."""
    assert layer in DECIDING_LAYERS, layer
    return f"{response}\nDecided by: {layer}"


def parse_decided_by(line: str) -> str | None:
    """The layer named on a "Decided by: ..." line, or None if `line` is some
    other line or names something that isn't a known layer."""
    if not line.lower().startswith(_PREFIX):
        return None
    _, sep, value = line.partition(":")
    value = value.strip()
    return value if sep and value in DECIDING_LAYERS else None
