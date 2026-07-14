# Crawler mirror manifests

The public release store and private restricted mirror are physically separate content-addressed
stores. Every object is fetched by `sha256:<64 lowercase hex>` and verified against its manifest's
digest, byte count, canonical object key, media type, exact upstream URL and revision, retention
class, mirror class, and verification time.

The durable public store contains only license-evidence-backed redistributable source archives and
package artifacts. Its committed public manifest is sufficient to fetch and reverify those bytes by
hash. Public retention is `durable-public`; JMT owns credentials and backup policy, and credentials
never enter manifests, the repository, or worker environments.

GPL/AGPL, no-license, and unresolved-license full bytes stay in the private mirror under
`restricted-private` or `campaign-private` retention. The committed private manifest may disclose
URL, revision, digest, size, media type, license finding, retention, and deterministic fetch recipe,
but not the restricted bytes or access credentials. Local ephemeral artifacts use a third,
non-overlapping local store.

Consumers request an object from the manifest's declared store by content hash and reject missing,
misaddressed, size-mismatched, or hash-mismatched bytes. The pre-public-merge license sweep rereads
the entire staged artifact set, fails closed on restricted or unknown disposition, verifies public
objects by hash, and emits the hash-bound report before merge.
