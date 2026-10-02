"""pyfm version and HadronsMILC compatibility metadata."""

import dataclasses
import datetime
import functools
import re
import subprocess
from importlib import metadata
from pathlib import Path

# MAJOR.MINOR of the HadronsMILC release whose XML contract pyfm emits.
# pyfm and HadronsMILC share MAJOR.MINOR: keep equal to pyproject.toml's version.
HADRONS_MILC_COMPAT = "0.2"

UNKNOWN = "unknown"

_SOURCE_ROOT = Path(__file__).resolve().parent.parent
_MAJOR_MINOR_PATTERN = re.compile(r"^v?(\d+)\.(\d+)")


def _dist_version() -> str:
    try:
        return metadata.version("pyfm")
    except metadata.PackageNotFoundError:
        return "0+unknown"


__version__ = _dist_version()


def major_minor(version: str) -> str | None:
    """Return "MAJOR.MINOR" from a version like "0.2", "0.2.1" or "v0.2.3"."""
    match = _MAJOR_MINOR_PATTERN.match(version.strip())
    if match is None:
        return None
    return f"{match.group(1)}.{match.group(2)}"


@functools.cache
def git_sha() -> str:
    """`git describe --always --dirty` of the pyfm source checkout, or "unknown"."""
    if not (_SOURCE_ROOT / ".git").exists():
        return UNKNOWN
    try:
        result = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            cwd=_SOURCE_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return UNKNOWN
    return result.stdout.strip() or UNKNOWN


def build_provenance() -> dict[str, str]:
    """Leaves of the <grid><provenance> element read by HadronsMILC."""
    generated = datetime.datetime.now(datetime.timezone.utc)
    return {
        "pyfmVersion": __version__,
        "pyfmSha": git_sha(),
        "hadronsMilcCompat": HADRONS_MILC_COMPAT,
        "generated": generated.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


PROVENANCE_KEYS = ("pyfmVersion", "pyfmSha", "hadronsMilcCompat", "generated")

# HadronsMILC run-log lines (each behind the usual "Hadrons : Message : <t> s : "
# prefix). Values may contain spaces, so every field is matched by its own prefix.
_LOG_PATTERNS = {
    "hadrons_milc_version": re.compile(r"HadronsMILC version=(\S*)"),
    "hadrons_milc_git": re.compile(r"HadronsMILC version=\S* git=(.*)$"),
    "grid_git": re.compile(r"(?:^|\s)Grid git=(.*)$"),
    "hadrons_git": re.compile(r"(?:^|\s)Hadrons git=(.*)$"),
    "grid_milc_version": re.compile(r"(?:^|\s)GridMilc version=(.*)$"),
    "dependency_pins": re.compile(r"(?:^|\s)Dependency pins=(\S+)"),
}
_PROVENANCE_LINE = re.compile(
    r"(?:^|\s)Provenance pyfmVersion=(.*?) pyfmSha=(.*?) "
    r"hadronsMilcCompat=(.*?) generated=(.*)$"
)
_MISMATCH_LINE = re.compile(
    r"Parameter file targets HadronsMILC .* \(MAJOR\.MINOR mismatch\)"
)


@dataclasses.dataclass
class RunLogVersions:
    """Version information reported by a HadronsMILC run log."""

    hadrons_milc_version: str | None = None
    hadrons_milc_git: str | None = None
    grid_git: str | None = None
    hadrons_git: str | None = None
    grid_milc_version: str | None = None
    dependency_pins: str | None = None
    provenance: dict[str, str] | None = None
    mismatch_warning: bool = False

    def expected_compat(self) -> str:
        """hadronsMilcCompat recorded in the log, else this pyfm's constant."""
        if self.provenance and self.provenance.get("hadronsMilcCompat"):
            return self.provenance["hadronsMilcCompat"]
        return HADRONS_MILC_COMPAT

    def compat_status(self) -> str:
        """"match", "mismatch" or "unknown" (binary without a version banner)."""
        if self.mismatch_warning:
            return "mismatch"
        binary = major_minor(self.hadrons_milc_version or "")
        expected = major_minor(self.expected_compat())
        if binary is None or expected is None:
            return "unknown"
        return "match" if binary == expected else "mismatch"


def parse_run_log(text: str) -> RunLogVersions:
    """Extract the HadronsMILC version banner from run-log text; first match wins."""
    info = RunLogVersions()
    for line in text.splitlines():
        line = line.rstrip()
        for field, pattern in _LOG_PATTERNS.items():
            if getattr(info, field) is None and (match := pattern.search(line)):
                setattr(info, field, match.group(1).strip())
        if info.provenance is None and (match := _PROVENANCE_LINE.search(line)):
            info.provenance = dict(
                zip(PROVENANCE_KEYS, (value.strip() for value in match.groups()))
            )
        if _MISMATCH_LINE.search(line):
            info.mismatch_warning = True
    return info
