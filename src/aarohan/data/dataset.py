"""Data structures and representations for dataset metadata and validation results."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class ValidationStatus(StrEnum):
    """Standard validation status outcomes."""

    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


@dataclass
class DatasetMetadata:
    """Metadata describing a Parquet dataset partition."""

    name: str
    path: str
    file_size_bytes: int
    file_size_mb: float
    sha256: str
    num_rows: int
    num_columns: int
    num_row_groups: int
    column_names: list[str]
    schema: dict[str, str]
    min_dt: str
    max_dt: str
    num_days: int
    num_stores: int
    num_products: int
    num_series: int

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to a serializable dictionary."""
        return asdict(self)


@dataclass
class ValidationCheck:
    """Represents the outcome of an individual validation check."""

    name: str
    status: str  # "PASS", "WARNING", "FAIL"
    expected: Any = None
    actual: Any = None
    explanation: str = ""
    counts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert check to a serializable dictionary."""
        return asdict(self)


@dataclass
class ValidationResult:
    """Aggregated validation results for a dataset partition or comparative check."""

    dataset_name: str
    status: str = ValidationStatus.PASS.value
    checks: list[ValidationCheck] = field(default_factory=list)
    metadata: DatasetMetadata | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def add_check(self, check: ValidationCheck) -> None:
        """Add a check and update overall status."""
        self.checks.append(check)
        self.update_status()

    def update_status(self) -> None:
        """Derive overall status from individual check statuses."""
        statuses = [c.status for c in self.checks]
        if ValidationStatus.FAIL.value in statuses:
            self.status = ValidationStatus.FAIL.value
        elif ValidationStatus.WARNING.value in statuses:
            self.status = ValidationStatus.WARNING.value
        else:
            self.status = ValidationStatus.PASS.value

    @property
    def summary_counts(self) -> dict[str, int]:
        """Return counts of PASS, WARNING, and FAIL checks."""
        return {
            "total": len(self.checks),
            "pass": sum(
                1 for c in self.checks if c.status == ValidationStatus.PASS.value
            ),
            "warning": sum(
                1 for c in self.checks if c.status == ValidationStatus.WARNING.value
            ),
            "fail": sum(
                1 for c in self.checks if c.status == ValidationStatus.FAIL.value
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert result to a serializable dictionary."""
        return {
            "dataset_name": self.dataset_name,
            "status": self.status,
            "summary_counts": self.summary_counts,
            "metadata": self.metadata.to_dict() if self.metadata else None,
            "checks": [c.to_dict() for c in self.checks],
            "diagnostics": self.diagnostics,
        }
