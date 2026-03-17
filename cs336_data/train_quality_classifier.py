from __future__ import annotations

import argparse
import pathlib
from typing import Iterator, TextIO
import warnings

import fasttext
from fastwarc.stream_io import FileStream, StreamError
from fastwarc.tools import wrap_warc_stream
from fastwarc.warc import ArchiveIterator, WarcRecordType

from cs336_data.extract_text import extract_html
from cs336_data.filters import gopher_quality_filter, identify_language


POSITIVE_LABEL = "wiki"
NEGATIVE_LABEL = "cc"


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def iter_warc_texts(warc_path: pathlib.Path) -> Iterator[str]:
    with FileStream(str(warc_path), "rb") as raw_stream:
        warc_stream = wrap_warc_stream(raw_stream, "rb")
        for record in ArchiveIterator(
            warc_stream,
            parse_http=True,
            record_types=WarcRecordType.response,
            auto_decode="none",
        ):
            try:
                html_bytes = record.reader.read()
            except StreamError as exc:
                warnings.warn(
                    f"Skipping unreadable record in {warc_path}: {exc}", stacklevel=2
                )
                continue
            text = normalize_text(extract_html(html_bytes))
            yield text


def write_examples(
    warc_path: pathlib.Path,
    label: str,
    out_file: TextIO,
    max_examples: int | None,
    gopher_filter: bool = False,
    english_filter: bool = False,
) -> int:
    count = 0
    for text in iter_warc_texts(
        warc_path,
        gopher_filter=gopher_filter,
        english_filter=english_filter,
    ):
        if gopher_filter and not gopher_quality_filter(text):
            continue
        if english_filter and identify_language(text)[0] != "en":
            continue
        if max_examples is not None and count >= max_examples:
            break
        out_file.write(f"__label__{label} {text}\n")
        count += 1
    return count


def write_training_data(
    positive_warc_path: pathlib.Path,
    negative_warc_path: pathlib.Path,
    train_output_path: pathlib.Path,
    max_positive: int | None,
    max_negative: int | None,
) -> None:
    train_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(train_output_path, "w", encoding="utf-8") as out_file:
        positive_count = write_examples(
            positive_warc_path,
            POSITIVE_LABEL,
            out_file,
            max_positive,
            gopher_filter=True,
            english_filter=True,
        )
        negative_count = write_examples(
            negative_warc_path,
            NEGATIVE_LABEL,
            out_file,
            min(max_negative, positive_count),
            gopher_filter=False,
            english_filter=False,
        )

    print(
        f"Wrote {positive_count} {POSITIVE_LABEL} examples and "
        f"{negative_count} {NEGATIVE_LABEL} examples to {train_output_path}"
    )


def train_model(
    train_input_path: pathlib.Path, model_output_path: pathlib.Path
) -> None:
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    model = fasttext.train_supervised(
        input=str(train_input_path),
        lr=0.5,
        epoch=10,
        dim=100,
        wordNgrams=2,
        minn=2,
        maxn=5,
        loss="softmax",
        bucket=200000,
    )
    model.save_model(str(model_output_path))
    print(f"Saved model to {model_output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build quality_classifier_train.txt from two WARC files and train a FastText classifier."
    )
    parser.add_argument("--positive-warc", type=pathlib.Path, required=True)
    parser.add_argument("--negative-warc", type=pathlib.Path, required=True)
    parser.add_argument("--train-output", type=pathlib.Path, required=True)
    parser.add_argument("--model-output", type=pathlib.Path, required=True)
    parser.add_argument("--max-positive", type=int)
    parser.add_argument("--max-negative", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_training_data(
        args.positive_warc,
        args.negative_warc,
        args.train_output,
        args.max_positive,
        args.max_negative,
    )
    train_model(args.train_output, args.model_output)


if __name__ == "__main__":
    main()
