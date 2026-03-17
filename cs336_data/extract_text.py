import pathlib
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
from fastwarc.warc import ArchiveIterator, WarcRecordType


def extract_html(html_bytes: bytes) -> str:
    encoding = detect_encoding(html_bytes)
    return extract_plain_text(
        html_bytes.decode(encoding, errors="replace"),
        main_content=False,
    )


def extract_warc_file(
    warc_path: pathlib.Path,
    output_path: pathlib.Path,
) -> str:
    with (
        open(warc_path, "rb") as warc_file,
        open(output_path, "w", encoding="utf-8") as out_file,
    ):
        first_record = True
        for record in ArchiveIterator(
            warc_file,
            parse_http=True,
            record_types=WarcRecordType.response,
            auto_decode="content",
        ):
            html_bytes = record.reader.read()
            text = extract_html(html_bytes)
            cleaned_text = "\n".join(
                line.strip() for line in text.splitlines() if line.strip()
            )
            if not cleaned_text:
                continue

            if not first_record:
                out_file.write("\n\n")
            out_file.write(cleaned_text)
            first_record = False
    return str(output_path)
