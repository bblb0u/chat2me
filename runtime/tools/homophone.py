from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def normalize_pinyin(value: str) -> str:
    return "".join(value.split()).strip()


def phrase_to_pinyin(value: str) -> str:
    try:
        from pypinyin import Style, pinyin
    except ModuleNotFoundError as exc:
        raise RuntimeError("pypinyin is not installed; cannot convert homophone terms to pinyin") from exc

    syllables = pinyin(
        value,
        style=Style.TONE3,
        heteronym=False,
        neutral_tone_with_five=True,
        errors=lambda chars: list(chars),
    )
    return normalize_pinyin("".join(item[0] for item in syllables if item))


def load_rules(path: Path) -> list[tuple[str, str]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("homophone config must be a YAML object")

    pairs: list[tuple[str, str]] = []
    for index, item in enumerate(data.get("rules", []), start=1):
        if isinstance(item, str):
            target = item.strip()
            variants = []
            explicit_pinyin = []
        elif isinstance(item, dict):
            target = str(item.get("target") or "").strip()
            variants = item.get("variants") or []
            explicit_pinyin = item.get("pinyin") or []
        else:
            raise ValueError(f"rules[{index}] must be a string or YAML object")

        if not target:
            raise ValueError(f"rules[{index}].target must not be empty")

        if isinstance(variants, str):
            variants = [variants]
        if not isinstance(variants, list):
            raise ValueError(f"rules[{index}].variants must be a list")
        if isinstance(explicit_pinyin, str):
            explicit_pinyin = [explicit_pinyin]
        if not isinstance(explicit_pinyin, list):
            raise ValueError(f"rules[{index}].pinyin must be a list")

        for phrase in [target, *[str(variant or "").strip() for variant in variants]]:
            if phrase:
                pairs.append((phrase_to_pinyin(phrase), target))
        for raw_variant in explicit_pinyin:
            variant = normalize_pinyin(str(raw_variant or ""))
            if variant:
                pairs.append((variant, target))

    deduped: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for source, target in pairs:
        key = (source, target)
        if source and key not in seen:
            deduped.append(key)
            seen.add(key)
    return deduped


def generate_fst(pairs: list[tuple[str, str]], output: Path) -> None:
    try:
        import pynini
        from pynini import cdrewrite
        from pynini.lib import utf8
    except ModuleNotFoundError as exc:
        raise RuntimeError("pynini is not installed; cannot generate homophone replace.fst") from exc

    if not pairs:
        raise ValueError("homophone config does not contain any pinyin rules")

    sigma = utf8.VALID_UTF8_CHAR.star
    rule = pynini.union(*(pynini.cross(source, target) for source, target in pairs)).optimize()
    cdrewrite(rule, "", "", sigma).write(str(output))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Generate sherpa-onnx homophone replace.fst from YAML rules.")
    parser.add_argument("config", type=Path, help="Path to homophones.yaml")
    parser.add_argument("output", type=Path, help="Path to write replace.fst")
    args = parser.parse_args(argv)

    try:
        pairs = load_rules(args.config)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        generate_fst(pairs, args.output)
    except Exception as exc:
        print(f"homophone: {exc}", file=sys.stderr)
        return 1

    print(f"generated {args.output} with {len(pairs)} homophone rules")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
