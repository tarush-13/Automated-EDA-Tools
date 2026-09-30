"""Dataset loading: accepts CSV / XLSX / JSON uploads.

Auto-detects file type, encoding and delimiter, and uses chunked/lazy loading
for large files so the app stays responsive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Optional, Tuple

import chardet
import numpy as np
import pandas as pd

from config import CHUNK_SIZE, MAX_ROWS


@dataclass
class LoadResult:
    """Container returned by ``load_dataset``."""

    df: pd.DataFrame
    file_name: str
    file_type: str  # csv | xlsx | json
    encoding: Optional[str] = None
    delimiter: Optional[str] = None
    truncated: bool = False
    original_rows: int = 0
    chunk_count: int = 0
    messages: list = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Encoding / delimiter detection
# --------------------------------------------------------------------------- #
def detect_encoding(raw_bytes: bytes, fallback: str = "utf-8") -> str:
    """Detect file encoding using chardet, with a useful fallback."""
    if not raw_bytes:
        return fallback
    try:
        guess = chardet.detect(raw_bytes[: 100_000])
        enc = guess.get("encoding")
        confidence = guess.get("confidence") or 0
        if enc and confidence > 0.6:
            # Python prefers this name over chardet's alias.
            return enc.lower().replace("iso-8859-1", "latin-1")
    except Exception:
        pass
    return fallback


def detect_delimiter(sample: str, candidates: Tuple[str, ...] = (',', ';', '\t', '|')) -> str:
    """Detect the field delimiter of a delimited text sample."""
    if not sample:
        return ","
    best = ","
    best_score = -1
    first_line = sample.splitlines()[0] if sample.splitlines() else ""
    for sep in candidates:
        score = first_line.count(sep)
        # penalise delimiters that show up inside quoted fields
        if score > best_score:
            best_score = score
            best = sep
    return best


def _peek_csv(raw: bytes) -> Tuple[Optional[str], Optional[str]]:
    """Detect encoding + delimiter, validating that the encoding actually decodes."""
    guess = detect_encoding(raw)
    candidates = [guess, "utf-8", "cp1252", "latin-1", "utf-16"]
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            text = raw.decode(candidate)
            return candidate, detect_delimiter(text)
        except (UnicodeDecodeError, LookupError):
            continue
    # Nothing worked: best effort.
    text = raw.decode("latin-1", errors="replace")
    return guess, detect_delimiter(text)


# --------------------------------------------------------------------------- #
# Sinks
# --------------------------------------------------------------------------- #
def _load_csv(file: BinaryIO, encoding: str, delimiter: str) -> LoadResult:
    sample = file.read(200_000)
    file.seek(0)
    if not sample:
        return LoadResult(df=pd.DataFrame(), file_name=file.name, file_type="csv")

    enc, delim = _peek_csv(sample)
    if not encoding:
        encoding = enc
    if not delimiter:
        delimiter = delim

    # Wrap the binary stream in a text decoder so the parser always sees str lines.
    import io

    text_stream = io.TextIOWrapper(file, encoding=encoding, errors="replace")

    # Stream in chunks; stop once we have enough rows for a full analysis set.
    reader = pd.read_csv(
        text_stream,
        delimiter=delimiter,
        engine="python",
        chunksize=CHUNK_SIZE,
    )
    chunks, total, truncated = [], 0, False
    for _i, chunk in enumerate(reader):
        chunks.append(chunk)
        total += len(chunk)
        if total >= MAX_ROWS:
            truncated = True
            break

    df = pd.concat(chunks, ignore_index=True).reset_index(drop=True)
    if truncated:
        df = df.iloc[:MAX_ROWS].copy()

    # Try to report the true source size when the uploader exposes it.
    size = getattr(file, "size", None)
    original_rows = int(size) if size else total

    return LoadResult(
        df=df,
        file_name=file.name,
        file_type="csv",
        encoding=encoding,
        delimiter=delimiter,
        truncated=truncated,
        original_rows=original_rows,
        chunk_count=len(chunks),
    )


def _load_excel(file: BinaryIO) -> LoadResult:
    file.seek(0)
    xl = pd.ExcelFile(file)
    sheet = xl.sheet_names[0] if xl.sheet_names else None
    df = pd.read_excel(file, sheet_name=sheet, engine="openpyxl") if sheet else pd.DataFrame()
    if len(df) > MAX_ROWS:
        df = df.iloc[:MAX_ROWS].copy()
    return LoadResult(
        df=df,
        file_name=file.name,
        file_type="xlsx",
        messages=[f"Workbook has sheets: {', '.join(xl.sheet_names[:10])}"],
        truncated=len(df) > MAX_ROWS,
        original_rows=len(df),
    )


def _load_json(file: BinaryIO) -> LoadResult:
    # Stream as-is; JSON files are usually modest in size.
    file.seek(0)
    data = file.read()
    enc = detect_encoding(data)
    file.seek(0)
    try:
        df = pd.read_json(file, encoding=enc)
    except ValueError:
        # Fall back to "records" orientation.
        file.seek(0)
        df = pd.read_json(file, encoding=enc, orient="records")
    if len(df) > MAX_ROWS:
        df = df.iloc[:MAX_ROWS].copy()
    return LoadResult(
        df=df,
        file_name=file.name,
        file_type="json",
        encoding=enc,
        truncated=len(df) > MAX_ROWS,
        original_rows=len(df),
    )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def load_dataset(
    uploaded,
    encoding: Optional[str] = None,
    delimiter: Optional[str] = None,
) -> LoadResult:
    """Load an uploaded file (Streamlit UploadedFile / file-like / path).

    ``uploaded`` must expose ``.name`` and read() semantics.
    """
    name = getattr(uploaded, "name", None)
    if isinstance(uploaded, (str, Path)):
        p = Path(uploaded)
        name = p.name
        with open(p, "rb") as fh:
            uploaded = _BytesLikeReadable(fh.read())
    elif hasattr(uploaded, "read"):
        pass
    else:
        raise TypeError("Unsupported uploadable object")

    suffix = Path(name).suffix.lower() if name else ".csv"
    if suffix in (".csv", ".txt", ".tsv"):
        return _load_csv(uploaded, encoding, delimiter)
    if suffix in (".xlsx", ".xls"):
        return _load_excel(uploaded)
    if suffix == ".json":
        return _load_json(uploaded)
    # Unknown extension: assume CSV.
    return _load_csv(uploaded, encoding, delimiter)


class _BytesLikeReadable:
    """Minimal file-like wrapper so memory bytes can be re-read multiple times.

    Implements the ``read``/``readline``/iteration protocol that the C and
    python CSV engines both require.
    """

    def __init__(self, data: bytes, name: str = "upload.csv"):
        self._data = data
        self._pos = 0
        self.name = name
        self.size = len(data)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self) -> bytes:
        line = self.readline()
        if not line:
            raise StopIteration
        return line

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = len(self._data) - self._pos
        out = self._data[self._pos : self._pos + n]
        self._pos += len(out)
        return out

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return True

    def close(self) -> None:
        self.closed = True

    def flush(self) -> None:
        pass

    def tell(self) -> int:
        return self._pos

    def readline(self, size: int = -1) -> bytes:
        """Line-based read required by pandas' python parser engine."""
        if self._pos >= len(self._data):
            return b""
        end = self._data.find(b"\n", self._pos)
        if end == -1:
            data = self._data[self._pos :]
            self._pos = len(self._data)
        else:
            stop = end + 1
            data = self._data[self._pos : stop]
            self._pos = stop
        if 0 < size < len(data):
            data = data[:size]
        return data

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self._pos = offset
        elif whence == 1:
            self._pos += offset
        elif whence == 2:
            self._pos = len(self._data) + offset
        return self._pos


def preview_dataframe(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Small preview helper reused across tabs."""
    return df.head(n) if not df.empty else pd.DataFrame()