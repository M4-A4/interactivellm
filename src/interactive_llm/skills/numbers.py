"""Spoken-number and duration parsing shared by the skills."""
from __future__ import annotations

import re

_ONES = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
         "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
         "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
         "ninety": 90}
_UNIT_SECONDS = {"second": 1, "seconds": 1, "sec": 1, "secs": 1, "minute": 60, "minutes": 60, "min": 60, "mins": 60,
                 "hour": 3600, "hours": 3600, "hr": 3600, "hrs": 3600}
_ARTICLES = {"a", "an"}


def _tokens(text: str) -> list[str]:
    """Lower-case words and numbers; hyphens split ("twenty-five" -> twenty, five)."""
    return re.findall(r"\d+(?:\.\d+)?|[a-z']+", text.lower().replace("-", " "))


def _read_number(tokens: list[str], index: int) -> tuple[float, int] | None:
    """Read a number (digits, "forty five", "twelve") starting at `index`; return (value, next index) or None."""
    token = tokens[index]
    if re.fullmatch(r"\d+(?:\.\d+)?", token):
        return float(token), index + 1
    if token in _TENS:
        value = _TENS[token]
        if index + 1 < len(tokens) and tokens[index + 1] in _ONES and 1 <= _ONES[tokens[index + 1]] <= 9:
            return float(value + _ONES[tokens[index + 1]]), index + 2
        return float(value), index + 1
    if token in _ONES:
        return float(_ONES[token]), index + 1
    return None


def spoken_number(text: str) -> float | None:
    """Return the first number in `text` ("fifty", "40", "twenty five"), or None."""
    tokens = _tokens(text)
    for index in range(len(tokens)):
        found = _read_number(tokens, index)
        if found:
            return found[0]
    return None


def parse_duration(text: str) -> float | None:
    """Return a duration in seconds from phrases like "5 minutes", "half an hour", "an hour and a half",
    "1 hour 30 minutes" or "ninety seconds"; None if no duration is present."""
    tokens = _tokens(text)
    total, found_any, index = 0.0, False, 0
    while index < len(tokens):
        token = tokens[index]
        # "half an hour", "half a minute", "a quarter of an hour"
        if token in ("half", "quarter"):
            skip = index + 1
            while skip < len(tokens) and tokens[skip] in _ARTICLES | {"of"}:
                skip += 1
            if skip < len(tokens) and tokens[skip] in _UNIT_SECONDS:
                total += (0.5 if token == "half" else 0.25) * _UNIT_SECONDS[tokens[skip]]
                found_any, index = True, skip + 1
                continue
        value, next_index = None, index
        number = _read_number(tokens, index)
        if number:
            value, next_index = number
        elif token in _ARTICLES:
            value, next_index = 1.0, index + 1
        if value is not None and tokens[next_index:next_index + 3] == ["and", "a", "half"] \
                and next_index + 3 < len(tokens) and tokens[next_index + 3] in _UNIT_SECONDS:
            total += (value + 0.5) * _UNIT_SECONDS[tokens[next_index + 3]]   # "two and a half minutes"
            found_any, index = True, next_index + 4
            continue
        if value is not None and next_index < len(tokens) and tokens[next_index] in _UNIT_SECONDS:
            unit = _UNIT_SECONDS[tokens[next_index]]
            total += value * unit
            found_any, index = True, next_index + 1
            if tokens[index:index + 3] == ["and", "a", "half"]:   # "an hour and a half"
                total += 0.5 * unit
                index += 3
            continue
        index += 1
    return total if found_any else None
