from __future__ import annotations

import re
from typing import Iterable, List, Optional

_UNITS_M = [  # мужской/средний род (для сотен и при отсутствии тысяч)
    "", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять",
]
_UNITS_F = [  # женский род — для сочетаний с "тысяча"
    "", "одна", "две", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять",
]
_TEENS = [
    "десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать",
    "пятнадцать", "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать",
]
_TENS = [
    "", "", "двадцать", "тридцать", "сорок", "пятьдесят",
    "шестьдесят", "семьдесят", "восемьдесят", "девяносто",
]
_HUNDREDS = [
    "", "сто", "двести", "триста", "четыреста", "пятьсот",
    "шестьсот", "семьсот", "восемьсот", "девятьсот",
]


def _thousand_form(n: int) -> str:
    last_two = n % 100
    last = n % 10
    if 11 <= last_two <= 14:
        return "тысяч"
    if last == 1:
        return "тысяча"
    if 2 <= last <= 4:
        return "тысячи"
    return "тысяч"


def _below_thousand(n: int, feminine: bool = False) -> str:
    assert 0 <= n < 1000
    if n == 0:
        return ""
    parts: List[str] = []
    h = n // 100
    rest = n % 100
    if h:
        parts.append(_HUNDREDS[h])
    if 10 <= rest <= 19:
        parts.append(_TEENS[rest - 10])
    else:
        t = rest // 10
        u = rest % 10
        if t:
            parts.append(_TENS[t])
        if u:
            parts.append((_UNITS_F if feminine else _UNITS_M)[u])
    return " ".join(parts)


def number_to_words(n: int) -> str:
    if n < 0:
        raise ValueError("negative not supported")
    if n == 0:
        return "ноль"

    thousands = n // 1000
    remainder = n % 1000

    parts: List[str] = []
    if thousands:
        if thousands == 1:
            parts.append("одна")
        else:
            parts.append(_below_thousand(thousands, feminine=True))
        parts.append(_thousand_form(thousands))
    if remainder:
        parts.append(_below_thousand(remainder, feminine=False))
    return " ".join(p for p in parts if p)


_WORD2NUM = {}
for i, w in enumerate(_UNITS_M):
    if w:
        _WORD2NUM[w] = i
for i, w in enumerate(_UNITS_F):
    if w:
        _WORD2NUM[w] = i
for i, w in enumerate(_TEENS):
    _WORD2NUM[w] = 10 + i
for i, w in enumerate(_TENS):
    if w:
        _WORD2NUM[w] = i * 10
for i, w in enumerate(_HUNDREDS):
    if w:
        _WORD2NUM[w] = i * 100

_THOUSAND_MARKERS = {"тысяча", "тысячи", "тысяч"}

_ALL_NUM_WORDS = sorted(set(_WORD2NUM) | _THOUSAND_MARKERS, key=len, reverse=True)


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) > len(b):
        a, b = b, a
    prev = list(range(len(a) + 1))
    for j, cb in enumerate(b, 1):
        curr = [j]
        for i, ca in enumerate(a, 1):
            curr.append(min(
                curr[-1] + 1,
                prev[i] + 1,
                prev[i - 1] + (0 if ca == cb else 1),
            ))
        prev = curr
    return prev[-1]


def _fuzzy_fix(token: str) -> Optional[str]:
    if token in _WORD2NUM or token in _THOUSAND_MARKERS:
        return token
    if len(token) < 3:
        return None
    best = None
    best_d = 10**9
    max_d = 1 if len(token) <= 6 else 2
    for cand in _ALL_NUM_WORDS:
        # быстрый предфильтр по длине
        if abs(len(cand) - len(token)) > max_d:
            continue
        d = _levenshtein(token, cand)
        if d < best_d:
            best_d = d
            best = cand
            if d == 0:
                break
    if best_d <= max_d:
        return best
    return None


def words_to_number(text: str, fuzzy: bool = True) -> int:
    if not text:
        return 0
    tokens = re.findall(r"[а-яёa-z]+", text.lower())

    thousands_part = 0  # накопленное до маркера "тысяча"
    current = 0         # текущая сумма в пределах тысяч/единиц
    saw_thousand = False
    total = 0           # финальный результат: thousands_part*1000 + остаток

    for raw in tokens:
        tok = raw
        if tok not in _WORD2NUM and tok not in _THOUSAND_MARKERS:
            if not fuzzy:
                continue
            fixed = _fuzzy_fix(tok)
            if fixed is None:
                continue
            tok = fixed

        if tok in _THOUSAND_MARKERS:
            thousands_part = (current or 1)
            current = 0
            saw_thousand = True
        else:
            current += _WORD2NUM[tok]

    total = thousands_part * 1000 + current
    return total


def vocab_chars() -> List[str]:
    s = set()
    for n in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
              20, 30, 40, 50, 60, 70, 80, 90,
              100, 200, 300, 400, 500, 600, 700, 800, 900,
              1000, 1001, 1234, 99999, 999999):
        s.update(number_to_words(n))
    # гарантированно включим все буквы русских числительных
    for w in list(_WORD2NUM) + list(_THOUSAND_MARKERS):
        s.update(w)
    s.add(" ")
    return sorted(s)


if __name__ == "__main__":
    checks = {
        1000: "одна тысяча",
        1005: "одна тысяча пять",
        139473: "сто тридцать девять тысяч четыреста семьдесят три",
        992597: "девятьсот девяносто две тысячи пятьсот девяносто семь",
        100000: "сто тысяч",
        1001: "одна тысяча один",
        2000: "две тысячи",
        5000: "пять тысяч",
        21000: "двадцать одна тысяча",
        999999: "девятьсот девяносто девять тысяч девятьсот девяносто девять",
    }
    for n, expected in checks.items():
        got = number_to_words(n)
        assert got == expected, f"{n}: got {got!r}, expected {expected!r}"
        assert words_to_number(got) == n, f"round-trip fail for {n}"
    assert words_to_number("сто тридцать девять тысяч четыреста семьдесят три") == 139473
    assert words_to_number("сто тридцат девять тысяч четыреста семдесят три") == 139473  # опечатки
    assert words_to_number("") == 0
    print("text_norm OK, vocab size:", len(vocab_chars()), "chars:", vocab_chars())
