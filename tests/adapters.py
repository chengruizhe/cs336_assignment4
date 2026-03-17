from __future__ import annotations

import os
from typing import Any
import pathlib
import fasttext
from cs336_data import extract_text
from cs336_data import filters


def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    return extract_text.extract_html(html_bytes=html_bytes)


def run_identify_language(text: str) -> tuple[Any, float]:
    return filters.identify_language(text)


def run_mask_emails(text: str) -> tuple[str, int]:
    return filters.mask_emails(text)


def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    return filters.mask_phone_numbers(text)


def run_mask_ips(text: str) -> tuple[str, int]:
    return filters.mask_ip(text)


def run_classify_nsfw(text: str) -> tuple[Any, float]:
    return filters.classify_nsfw(text)


def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    return filters.classify_toxic_speech(text)


def run_classify_quality(text: str) -> tuple[Any, float]:
    return filters.low_quality_filter(text)


def run_gopher_quality_filter(text: str) -> bool:
    return filters.gopher_quality_filter(text)


def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    raise NotImplementedError


def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    raise NotImplementedError
