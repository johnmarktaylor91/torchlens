"""Hash-addressed public, private, and local crawler artifact stores."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

from menagerie.crawler.authority import MirrorObject
from menagerie.crawler.identity import hash_bytes, utc_now
from menagerie.crawler.models import JsonObject


class MirrorClass(str, Enum):
    """Physically separate artifact-store classes."""

    PUBLIC = "public"
    PRIVATE = "private"
    LOCAL = "local"


class RetentionClass(str, Enum):
    """Declared durability/privacy retention classes."""

    DURABLE_PUBLIC = "durable-public"
    RESTRICTED_PRIVATE = "restricted-private"
    CAMPAIGN_PRIVATE = "campaign-private"
    LOCAL_EPHEMERAL = "local-ephemeral"


class MirrorError(RuntimeError):
    """Base class for mirror addressing and verification failures."""


class MirrorHashMismatchError(MirrorError):
    """Raised when fetched bytes do not match their addressed hash."""


class MirrorManifestError(MirrorError):
    """Raised when a retention manifest is incomplete or contradictory."""


class MirrorSeparationError(MirrorError):
    """Raised when public/private/local storage boundaries overlap."""


@dataclass(frozen=True)
class ArtifactOrigin:
    """Exact upstream origin retained in every mirror manifest."""

    url: str
    revision: str

    def __post_init__(self) -> None:
        """Reject incomplete origin identities.

        Raises
        ------
        MirrorManifestError
            If the URL or revision is empty.
        """

        if not self.url.strip() or not self.revision.strip():
            raise MirrorManifestError("artifact origin URL and revision must be non-empty")


@dataclass(frozen=True)
class ArtifactManifest:
    """Hash-bound location, origin, and retention declaration."""

    content_sha256: str
    byte_count: int
    media_type: str
    origin: ArtifactOrigin
    retention_class: RetentionClass
    mirror_class: MirrorClass
    object_key: str
    verified_at: str

    def to_dict(self) -> JsonObject:
        """Return the JSON-compatible manifest.

        Returns
        -------
        dict[str, Any]
            Persistable mirror manifest.
        """

        payload = dict(asdict(self))
        payload["retention_class"] = self.retention_class.value
        payload["mirror_class"] = self.mirror_class.value
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArtifactManifest":
        """Parse a typed manifest from persisted JSON data.

        Parameters
        ----------
        payload:
            Manifest mapping.

        Returns
        -------
        ArtifactManifest
            Parsed manifest.

        Raises
        ------
        MirrorManifestError
            If required typed fields are absent or invalid.
        """

        try:
            raw_origin = payload["origin"]
            if not isinstance(raw_origin, Mapping):
                raise TypeError("origin must be an object")
            return cls(
                content_sha256=str(payload["content_sha256"]),
                byte_count=int(payload["byte_count"]),
                media_type=str(payload["media_type"]),
                origin=ArtifactOrigin(
                    url=str(raw_origin["url"]), revision=str(raw_origin["revision"])
                ),
                retention_class=RetentionClass(str(payload["retention_class"])),
                mirror_class=MirrorClass(str(payload["mirror_class"])),
                object_key=str(payload["object_key"]),
                verified_at=str(payload["verified_at"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MirrorManifestError(f"invalid artifact manifest: {exc}") from exc


def _is_sha256(value: str) -> bool:
    """Return whether a digest uses canonical prefixed SHA-256 syntax.

    Parameters
    ----------
    value:
        Candidate digest.

    Returns
    -------
    bool
        Whether the digest is canonical.
    """

    return (
        len(value) == 71
        and value.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in value[7:])
    )


class MirrorStore:
    """Address and reverify three physically separate content stores."""

    def __init__(self, public_root: Path, private_root: Path, local_root: Path) -> None:
        """Bind non-overlapping public, private, and local roots.

        Parameters
        ----------
        public_root, private_root, local_root:
            Store roots. No root may equal or contain another.

        Raises
        ------
        MirrorSeparationError
            If any roots overlap.
        """

        roots = {
            MirrorClass.PUBLIC: public_root.resolve(),
            MirrorClass.PRIVATE: private_root.resolve(),
            MirrorClass.LOCAL: local_root.resolve(),
        }
        for left_name, left in roots.items():
            for right_name, right in roots.items():
                if left_name == right_name:
                    continue
                if left == right or left in right.parents or right in left.parents:
                    raise MirrorSeparationError(
                        f"{left_name.value} and {right_name.value} mirror roots overlap"
                    )
        self._roots = roots

    def root(self, mirror_class: MirrorClass) -> Path:
        """Return one physical mirror root.

        Parameters
        ----------
        mirror_class:
            Store class.

        Returns
        -------
        pathlib.Path
            Resolved physical root.
        """

        return self._roots[mirror_class]

    def address(self, content_sha256: str, mirror_class: MirrorClass) -> Path:
        """Return the class-specific path for a content hash.

        Parameters
        ----------
        content_sha256:
            Canonical prefixed SHA-256 digest.
        mirror_class:
            Public, private, or local store.

        Returns
        -------
        pathlib.Path
            Two-level content-addressed path.

        Raises
        ------
        MirrorManifestError
            If the digest is malformed.
        """

        if not _is_sha256(content_sha256):
            raise MirrorManifestError("content hash must be sha256:<64 lowercase hex>")
        digest = content_sha256.removeprefix("sha256:")
        return self.root(mirror_class) / "sha256" / digest[:2] / digest

    def put(
        self,
        content: bytes,
        *,
        mirror_class: MirrorClass,
        retention_class: RetentionClass,
        origin: ArtifactOrigin,
        media_type: str = "application/octet-stream",
        verified_at: str | None = None,
    ) -> ArtifactManifest:
        """Verify, atomically store, and manifest one artifact by hash.

        Parameters
        ----------
        content:
            Exact artifact bytes.
        mirror_class, retention_class:
            Physical class and compatible retention declaration.
        origin:
            Exact upstream origin.
        media_type:
            Non-empty artifact media type.
        verified_at:
            Optional deterministic verification timestamp.

        Returns
        -------
        ArtifactManifest
            Hash-bound retention manifest.
        """

        self._validate_retention(mirror_class, retention_class)
        if not media_type.strip():
            raise MirrorManifestError("media_type must be non-empty")
        digest = hash_bytes(content)
        destination = self.address(digest, mirror_class)
        if destination.exists():
            existing = destination.read_bytes()
            if hash_bytes(existing) != digest:
                raise MirrorHashMismatchError(f"corrupt addressed mirror object: {destination}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
            try:
                with temporary.open("xb") as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, destination)
                self._fsync_directory(destination.parent)
            finally:
                temporary.unlink(missing_ok=True)
        object_key = destination.relative_to(self.root(mirror_class)).as_posix()
        return ArtifactManifest(
            content_sha256=digest,
            byte_count=len(content),
            media_type=media_type,
            origin=origin,
            retention_class=retention_class,
            mirror_class=mirror_class,
            object_key=object_key,
            verified_at=verified_at or utc_now(),
        )

    def fetch(self, manifest: ArtifactManifest) -> bytes:
        """Fetch by hash and reverify the complete retention manifest.

        Parameters
        ----------
        manifest:
            Hash-bound location and retention declaration.

        Returns
        -------
        bytes
            Exact verified artifact bytes.

        Raises
        ------
        MirrorManifestError
            If the manifest contradicts its canonical address or retention class.
        MirrorHashMismatchError
            If stored bytes no longer match the addressed digest or size.
        """

        self.verify_manifest(manifest)
        destination = self.address(manifest.content_sha256, manifest.mirror_class)
        try:
            content = destination.read_bytes()
        except OSError as exc:
            raise MirrorManifestError(
                f"artifact is not retrievable by hash {manifest.content_sha256}: {exc}"
            ) from exc
        actual = hash_bytes(content)
        if actual != manifest.content_sha256:
            raise MirrorHashMismatchError(
                f"mirror hash mismatch: expected {manifest.content_sha256}, got {actual}"
            )
        if len(content) != manifest.byte_count:
            raise MirrorHashMismatchError(
                f"mirror size mismatch: expected {manifest.byte_count}, got {len(content)}"
            )
        return content

    def fetch_object(
        self,
        obj: MirrorObject,
        *,
        custody_class: Optional[MirrorClass] = None,
    ) -> bytes:
        """Fetch and verify one normalized object projection.

        ``custody_class`` permits the reconstruction consumer to recover a public
        claim from the mandatory retained private copy with the same digest. It
        changes only the physical store selected for reading; all intrinsic size,
        media, address, and digest checks remain strict.

        Parameters
        ----------
        obj:
            Intrinsic object row from the normalized artifact projection.
        custody_class:
            Optional physical mirror class to read instead of ``obj.mirror_class``.

        Returns
        -------
        bytes
            Exact verified object bytes.

        Raises
        ------
        MirrorManifestError
            If the object row or selected physical address is malformed or missing.
        MirrorHashMismatchError
            If the selected bytes differ from the normalized object identity.
        """

        try:
            projected_class = MirrorClass(obj.mirror_class)
        except ValueError as exc:
            raise MirrorManifestError(
                f"normalized object has unsupported mirror class: {obj.mirror_class!r}"
            ) from exc
        selected_class = custody_class or projected_class
        selected_path = self.address(obj.content_sha256, selected_class)
        expected_key = selected_path.relative_to(self.root(selected_class)).as_posix()
        if obj.object_key != expected_key:
            raise MirrorManifestError(
                f"normalized object key {obj.object_key!r} is not canonical {expected_key!r}"
            )
        try:
            if selected_path.is_symlink() or not selected_path.is_file():
                raise MirrorManifestError(
                    f"normalized mirror object is not a regular unaliased file: {selected_path}"
                )
            content = selected_path.read_bytes()
        except OSError as exc:
            raise MirrorManifestError(
                f"normalized mirror object is unavailable: {selected_path}"
            ) from exc
        observed = hash_bytes(content)
        if observed != obj.content_sha256:
            raise MirrorHashMismatchError(
                f"normalized mirror object hash mismatch: expected {obj.content_sha256}, "
                f"got {observed}"
            )
        if len(content) != obj.byte_count:
            raise MirrorHashMismatchError(
                f"normalized mirror object size mismatch: expected {obj.byte_count}, "
                f"got {len(content)}"
            )
        if not obj.media_type.strip():
            raise MirrorManifestError("normalized mirror object media type must be non-empty")
        return content

    def iter_objects(
        self, mirror_class: Optional[MirrorClass] = None
    ) -> Iterator[tuple[MirrorClass, str]]:
        """Enumerate every physical object in deterministic key order.

        Enumeration deliberately returns physical ``(class, object_key)`` pairs,
        not manifest claims.  Callers must join these facts to independently
        derived :class:`~menagerie.crawler.authority.MirrorObject` inventory
        rows before trusting media type, size, or provenance.

        Parameters
        ----------
        mirror_class:
            Optional single store class.  Omission enumerates public, private,
            and local stores in enum order.

        Yields
        ------
        tuple[MirrorClass, str]
            Physical mirror class and root-relative POSIX object key.
        """

        classes = (mirror_class,) if mirror_class is not None else tuple(MirrorClass)
        for selected in classes:
            root = self.root(selected)
            if not root.exists():
                continue
            for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
                if path.is_file():
                    yield selected, path.relative_to(root).as_posix()

    def verify_manifest(self, manifest: ArtifactManifest) -> None:
        """Validate origin, retention, and canonical hash address.

        Parameters
        ----------
        manifest:
            Manifest to inspect.

        Raises
        ------
        MirrorManifestError
            If any manifest field is incomplete or contradictory.
        """

        self._validate_retention(manifest.mirror_class, manifest.retention_class)
        if (
            manifest.byte_count < 0
            or not manifest.media_type.strip()
            or not manifest.verified_at.strip()
        ):
            raise MirrorManifestError("manifest size, media type, and verified_at must be valid")
        destination = self.address(manifest.content_sha256, manifest.mirror_class)
        expected_key = destination.relative_to(self.root(manifest.mirror_class)).as_posix()
        if manifest.object_key != expected_key:
            raise MirrorManifestError(
                f"manifest object_key {manifest.object_key!r} is not canonical {expected_key!r}"
            )

    @staticmethod
    def _validate_retention(mirror_class: MirrorClass, retention_class: RetentionClass) -> None:
        """Enforce the physical-store/retention boundary.

        Parameters
        ----------
        mirror_class, retention_class:
            Proposed physical and retention classes.

        Raises
        ------
        MirrorManifestError
            If the pair could expose or misstate retained bytes.
        """

        allowed = {
            MirrorClass.PUBLIC: {RetentionClass.DURABLE_PUBLIC},
            MirrorClass.PRIVATE: {
                RetentionClass.RESTRICTED_PRIVATE,
                RetentionClass.CAMPAIGN_PRIVATE,
            },
            MirrorClass.LOCAL: {RetentionClass.LOCAL_EPHEMERAL},
        }
        if retention_class not in allowed[mirror_class]:
            raise MirrorManifestError(
                f"{retention_class.value} is invalid for {mirror_class.value} mirror"
            )

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        """Synchronize a directory after an atomic object write.

        Parameters
        ----------
        path:
            Directory containing the new object.
        """

        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
