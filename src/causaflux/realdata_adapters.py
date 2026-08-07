"""Repository-specific, non-destructive real-data access adapters.

Adapters generate explicit commands or request payloads and lock metadata. They do
not bypass authentication, license acceptance, or controlled-access approval.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from .realdata import SourceSpec


@dataclass(frozen=True)
class AdapterPlan:
    adapter: str
    source_id: str
    accession: str
    destination: str
    execution_mode: str
    metadata_only: bool
    command_or_action: str
    request_payload: dict[str, Any]
    requires_user_authorization: bool
    redistributable_by_causaflux: bool = False

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")
        return target


class RepositoryAdapter:
    name = "manual"

    def plan(self, source: SourceSpec, destination: Path, *, metadata_only: bool) -> AdapterPlan:
        return AdapterPlan(
            adapter=self.name,
            source_id=source.source_id,
            accession=source.accession,
            destination=str(destination),
            execution_mode="manual",
            metadata_only=metadata_only,
            command_or_action=f"Resolve {source.url} into {destination}",
            request_payload=dict(source.query),
            requires_user_authorization="controlled" in source.access.lower(),
        )


class AwsS3Adapter(RepositoryAdapter):
    name = "aws_s3"

    def plan(self, source: SourceSpec, destination: Path, *, metadata_only: bool) -> AdapterPlan:
        bucket = source.query["bucket"]
        action = "ls" if metadata_only else "sync"
        command = f"aws s3 {action} --no-sign-request s3://{bucket}/ {destination}/"
        return AdapterPlan(self.name, source.source_id, source.accession, str(destination),
                           "public-command", metadata_only, command,
                           {"bucket": bucket, "anonymous": True}, False)


class DandiAdapter(RepositoryAdapter):
    name = "dandi"

    def plan(self, source: SourceSpec, destination: Path, *, metadata_only: bool) -> AdapterPlan:
        did = source.query["dandiset"]
        version = source.query.get("version", "latest_published")
        if metadata_only:
            command = f"dandi ls DANDI:{did}@latest"
        else:
            command = f"dandi download DANDI:{did}@latest --output-dir {destination}"
        return AdapterPlan(self.name, source.source_id, source.accession, str(destination),
                           "public-command", metadata_only, command,
                           {"dandiset": did, "version_policy": version}, False)


class SynapseAdapter(RepositoryAdapter):
    name = "synapse"

    def plan(self, source: SourceSpec, destination: Path, *, metadata_only: bool) -> AdapterPlan:
        sid = source.query.get("synapse_id", source.accession)
        command = f"synapse get {'--downloadFile false ' if metadata_only else '-r '}{sid} --downloadLocation {destination}"
        return AdapterPlan(self.name, source.source_id, source.accession, str(destination),
                           "account-or-controlled", metadata_only, command,
                           {"synapse_id": sid, "lock_entity_version": True}, True)


class HtanSynapseAdapter(RepositoryAdapter):
    name = "htan_synapse"

    def plan(self, source: SourceSpec, destination: Path, *, metadata_only: bool) -> AdapterPlan:
        action = (
            f"Query the HTAN portal using {json.dumps(source.query, sort_keys=True)}; export the result; "
            f"resolve every entity to a Synapse ID and immutable version; place approved files under {destination}"
        )
        return AdapterPlan(self.name, source.source_id, source.accession, str(destination),
                           "portal-query-account", metadata_only, action,
                           {"portal_query": source.query, "lock_synapse_versions": True}, True)


class GdcAdapter(RepositoryAdapter):
    name = "gdc"

    def plan(self, source: SourceSpec, destination: Path, *, metadata_only: bool) -> AdapterPlan:
        payload = {"filters": source.query, "format": "JSON", "size": 10000}
        action = (
            f"POST the recorded filters to the GDC files/cases API; write response JSON, GDC release, "
            f"file manifest, MD5 values and case joins under {destination}"
        )
        return AdapterPlan(self.name, source.source_id, source.accession, str(destination),
                           "public-and-controlled", metadata_only, action, payload,
                           "controlled" in source.access.lower())


class PdcAdapter(RepositoryAdapter):
    name = "pdc"

    def plan(self, source: SourceSpec, destination: Path, *, metadata_only: bool) -> AdapterPlan:
        study = source.query.get("pdc_study_id", source.accession)
        action = f"Query the PDC GraphQL API for study {study}; lock study metadata and file checksums under {destination}"
        return AdapterPlan(self.name, source.source_id, source.accession, str(destination),
                           "public-and-controlled", metadata_only, action,
                           {"pdc_study_id": study, "graphql": True},
                           "controlled" in source.access.lower())


class GeoAdapter(RepositoryAdapter):
    name = "geo"

    def plan(self, source: SourceSpec, destination: Path, *, metadata_only: bool) -> AdapterPlan:
        accession = source.query.get("geo_accession", source.accession)
        action = (
            f"Fetch GEO series metadata for {accession} via NCBI HTTPS; "
            + ("list supplementary files" if metadata_only else f"download supplementary files to {destination}")
        )
        return AdapterPlan(self.name, source.source_id, source.accession, str(destination),
                           "public-command", metadata_only, action,
                           {"geo_accession": accession}, False)


class DepMapManualAdapter(RepositoryAdapter):
    name = "depmap_manual"

    def plan(self, source: SourceSpec, destination: Path, *, metadata_only: bool) -> AdapterPlan:
        action = (
            f"Open the official DepMap Downloads page, accept the file-specific terms, download {source.accession}, "
            f"and place it under {destination}. Automated portal scraping is intentionally disabled."
        )
        return AdapterPlan(self.name, source.source_id, source.accession, str(destination),
                           "manual-terms-no-scraping", metadata_only, action,
                           {"release": source.accession, "manual": True}, True)


_ADAPTERS = {
    cls.name: cls()
    for cls in (AwsS3Adapter, DandiAdapter, SynapseAdapter, HtanSynapseAdapter,
                GdcAdapter, PdcAdapter, GeoAdapter, DepMapManualAdapter)
}


def adapter_names() -> tuple[str, ...]:
    return tuple(sorted(_ADAPTERS))


def get_adapter(name: str) -> RepositoryAdapter:
    if name not in _ADAPTERS:
        raise KeyError(f"unknown real-data adapter: {name}")
    return _ADAPTERS[name]


def plan_source(source: SourceSpec, destination: str | Path, *, metadata_only: bool = True) -> AdapterPlan:
    return get_adapter(source.adapter).plan(source, Path(destination), metadata_only=metadata_only)


def write_accession_lock(
    source: SourceSpec,
    destination: str | Path,
    *,
    resolved_version: str,
    files: list[dict[str, Any]],
) -> Path:
    """Write an immutable lock after repository resolution/download.

    Each file record should include the repository identifier and checksum when
    available. This function does not certify data-use compliance; it records it.
    """
    output = Path(destination) / "accession_lock.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark_id": source.benchmark_id,
        "source_id": source.source_id,
        "accession": source.accession,
        "resolved_version": resolved_version,
        "access": source.access,
        "license": source.license,
        "license_url": source.license_url,
        "files": files,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output
