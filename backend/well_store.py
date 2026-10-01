"""Adding user-supplied LAS files to the data directory.

Uploads are validated by actually parsing them with lasio before anything is
written, and the stored name is a sanitized well id, never the raw filename.
"""

import re
from pathlib import Path

import lasio

from backend.config import DATA_DIR

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
_WELL_ID_RE = re.compile(r"[^A-Za-z0-9_-]+")


class UploadError(ValueError):
    """Bad upload (wrong type, unparseable, empty)."""


class WellExistsError(UploadError):
    """A well with this id is already loaded."""


def well_id_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    well_id = _WELL_ID_RE.sub("_", stem).strip("_-")[:64]
    return well_id or "UploadedWell"


def save_uploaded_las(filename: str, data: bytes) -> str:
    """Validate and store an uploaded LAS file; returns the new well id."""
    if not filename.lower().endswith(".las"):
        raise UploadError("Only .las files are supported.")
    if not data:
        raise UploadError("The file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadError(f"File is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("latin-1")

    try:
        las = lasio.read(text)
        df = las.df()
    except Exception as e:  # lasio raises a variety of parse errors
        raise UploadError(f"Could not read this file as LAS: {e}") from e
    if df.empty or len(las.curves) < 2:
        raise UploadError("The LAS file has no depth samples or no log curves.")

    well_id = well_id_from_filename(filename)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing = {p.stem.lower() for p in DATA_DIR.glob("*.las")}
    if well_id.lower() in existing:
        raise WellExistsError(f"A well named '{well_id}' is already loaded. Rename the file to add it again.")

    (DATA_DIR / f"{well_id}.las").write_bytes(data)
    return well_id
