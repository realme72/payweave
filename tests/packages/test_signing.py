import shutil
import stat
from datetime import UTC, datetime
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from payweave.packages import __main__ as cli
from payweave.packages import signing
from payweave.packages.runtime import PackageLoadError, PackageProvider, load_package
from payweave.packages.signing import ApprovalError, load_signed_package
from payweave.verify import report as report_mod

RAZORPAY = Path(__file__).parents[2] / "packages" / "razorpay" / "v1"
NOW = datetime(2026, 10, 8, 12, 0, tzinfo=UTC)
PASSPHRASE = "correct horse battery"


@pytest.fixture
def package_dir(tmp_path: Path) -> Path:
    target = tmp_path / "razorpay" / "v1"
    target.mkdir(parents=True)
    for name in ("manifest.toml", "mappers.py"):
        shutil.copy(RAZORPAY / name, target / name)
    return target


@pytest.fixture
def key() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.generate()


@pytest.fixture
def trust(key: Ed25519PrivateKey) -> dict[str, Ed25519PublicKey]:
    return {"alice": key.public_key()}


def promotable_report(directory: Path) -> None:
    """Stand-in for `python -m payweave.verify` (tests/verify covers the real thing)."""
    sha = load_package(directory).sha256
    report_mod.write_report(
        directory, {"package": {"sha256": sha}, "promotable": True, "gates": {}}
    )


def approve(directory: Path, key: Ed25519PrivateKey, signer: str = "alice") -> signing.Approval:
    approval = signing.sign_package(load_package(directory), key, signer, now=NOW)
    (directory / signing.APPROVAL_FILE).write_text(approval.to_toml())
    return approval


# --- the gate -------------------------------------------------------------------------------


def test_approved_package_loads_and_serves(
    package_dir: Path, key: Ed25519PrivateKey, trust: dict[str, Ed25519PublicKey]
) -> None:
    approval = approve(package_dir, key)
    package = load_signed_package(package_dir, trust)
    assert package.approval == approval
    assert signing.read_approval(package_dir) == approval  # TOML round-trips
    PackageProvider(package)  # no allow_unsigned needed


def test_unsigned_package_is_refused(package_dir: Path, trust: dict[str, Ed25519PublicKey]) -> None:
    with pytest.raises(ApprovalError, match="never approved"):
        load_signed_package(package_dir, trust)
    with pytest.raises(PackageLoadError, match="not approved"):
        PackageProvider(load_package(package_dir))


def test_edit_after_approval_is_refused_before_mappers_run(
    package_dir: Path, key: Ed25519PrivateKey, trust: dict[str, Ed25519PublicKey]
) -> None:
    approve(package_dir, key)
    mappers = package_dir / "mappers.py"
    # Would be rejected by the sandbox; the hash check must fire first.
    mappers.write_text("import os\n" + mappers.read_text())
    with pytest.raises(ApprovalError, match="changed since it was approved"):
        load_signed_package(package_dir, trust)


def test_manifest_edit_is_refused(
    package_dir: Path, key: Ed25519PrivateKey, trust: dict[str, Ed25519PublicKey]
) -> None:
    approve(package_dir, key)
    manifest = package_dir / "manifest.toml"
    manifest.write_text(manifest.read_text() + "\n")  # even whitespace
    with pytest.raises(ApprovalError, match="changed"):
        load_signed_package(package_dir, trust)


def test_untrusted_or_revoked_signer_is_refused(package_dir: Path, key: Ed25519PrivateKey) -> None:
    approve(package_dir, key)
    with pytest.raises(ApprovalError, match="'alice' is not trusted"):
        load_signed_package(package_dir, {})
    other = Ed25519PrivateKey.generate().public_key()
    with pytest.raises(ApprovalError, match="bad signature"):
        load_signed_package(package_dir, {"alice": other})


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ('signer = "alice"', 'signer = "bob"'),  # claim someone else approved
        ("2026-10-08T12:00:00Z", "2026-10-09T12:00:00Z"),  # back/forward-date
    ],
)
def test_tampered_approval_fields_break_the_signature(
    package_dir: Path, key: Ed25519PrivateKey, old: str, new: str
) -> None:
    approve(package_dir, key)
    path = package_dir / signing.APPROVAL_FILE
    assert old in path.read_text()
    path.write_text(path.read_text().replace(old, new))
    trust = {"alice": key.public_key(), "bob": key.public_key()}
    with pytest.raises(ApprovalError, match="bad signature"):
        load_signed_package(package_dir, trust)


def test_approval_cant_be_moved_to_another_package(
    package_dir: Path, key: Ed25519PrivateKey, trust: dict[str, Ed25519PublicKey]
) -> None:
    approve(package_dir, key)
    other = package_dir.parent / "v2"
    other.mkdir()
    manifest = (package_dir / "manifest.toml").read_text().replace('version = "1"', 'version = "2"')
    (other / "manifest.toml").write_text(manifest)
    shutil.copy(package_dir / "mappers.py", other / "mappers.py")
    shutil.copy(package_dir / signing.APPROVAL_FILE, other / signing.APPROVAL_FILE)
    with pytest.raises(ApprovalError, match="changed"):
        load_signed_package(other, trust)


def test_malformed_approval_is_refused(
    package_dir: Path, trust: dict[str, Ed25519PublicKey]
) -> None:
    (package_dir / signing.APPROVAL_FILE).write_text('signer = "alice"\n')
    with pytest.raises(ApprovalError, match="malformed"):
        load_signed_package(package_dir, trust)


# --- keys and trust -------------------------------------------------------------------------


def test_private_key_is_encrypted_private_and_never_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PAYWEAVE_KEY_DIR", str(tmp_path / "keys"))
    path, public = signing.write_private_key("alice", PASSPHRASE)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert b"ENCRYPTED PRIVATE KEY" in path.read_bytes()
    assert signing.public_key_b64(signing.read_private_key("alice", PASSPHRASE)) == public
    with pytest.raises(ApprovalError, match="wrong passphrase"):
        signing.read_private_key("alice", PASSPHRASE + "x")
    with pytest.raises(FileExistsError):
        signing.write_private_key("alice", PASSPHRASE)
    with pytest.raises(ValueError, match="at least 12"):
        signing.write_private_key("bob", "short")
    with pytest.raises(ApprovalError, match="run keygen"):
        signing.read_private_key("carol", PASSPHRASE)


def test_trust_file(tmp_path: Path, key: Ed25519PrivateKey) -> None:
    path = tmp_path / "trusted_signers.toml"
    assert signing.load_trust(path) == {}  # missing file trusts nobody
    signing.add_trusted_signer("alice", signing.public_key_b64(key), path)
    loaded = signing.load_trust(path)
    assert signing.public_key_b64(loaded["alice"]) == signing.public_key_b64(key)
    with pytest.raises(ValueError, match="already"):
        signing.add_trusted_signer("alice", signing.public_key_b64(key), path)
    path.write_text('[signers.bob]\npublic_key = "not base64!"\n')
    with pytest.raises(ApprovalError, match="bad public key"):
        signing.load_trust(path)


# --- CLI ------------------------------------------------------------------------------------


class Person:
    """Stands in for the human at the terminal."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *, interactive: bool = True) -> None:
        self.typed: list[str] = []
        self.secrets: list[str] = []
        monkeypatch.setattr(cli, "_is_interactive", lambda: interactive)
        monkeypatch.setattr(cli, "_input", lambda _prompt: self.typed.pop(0))
        monkeypatch.setattr(cli, "_getpass", lambda _prompt: self.secrets.pop(0))


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("PAYWEAVE_KEY_DIR", str(tmp_path / "keys"))
    return tmp_path / "trusted_signers.toml"


def test_keygen_review_sign_verify(
    package_dir: Path,
    env: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    person = Person(monkeypatch)
    trust = ["--trust-file", str(env)]
    person.secrets = [PASSPHRASE, PASSPHRASE]
    assert cli.main([*trust, "keygen", "alice"]) == 0

    assert cli.main([*trust, "review", str(package_dir)]) == 0
    out = capsys.readouterr().out
    assert "razorpay v1" in out
    assert "NOT APPROVED" in out
    assert "api.razorpay.com" in out

    assert cli.main([*trust, "verify", str(package_dir)]) == 1

    sha = load_package(package_dir).sha256
    sign = [*trust, "sign", str(package_dir), "--signer", "alice"]
    assert cli.main(sign) == 1  # no report.json yet
    assert "no report.json" in capsys.readouterr().err
    promotable_report(package_dir)
    person.typed, person.secrets = [sha[:8]], [PASSPHRASE]
    assert cli.main(sign) == 0
    assert cli.main([*trust, "verify", str(package_dir)]) == 0
    assert "approved by alice" in capsys.readouterr().out


def test_sign_refuses_without_a_person(
    package_dir: Path, env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    Person(monkeypatch, interactive=False)
    assert cli.main(["--trust-file", str(env), "sign", str(package_dir), "--signer", "a"]) == 2
    assert not (package_dir / signing.APPROVAL_FILE).exists()


def test_sign_refuses_wrong_confirmation_or_untrusted_key(
    package_dir: Path, env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    person = Person(monkeypatch)
    trust = ["--trust-file", str(env)]
    sign = [*trust, "sign", str(package_dir), "--signer", "alice"]
    sha = load_package(package_dir).sha256

    promotable_report(package_dir)
    signing.write_private_key("alice", PASSPHRASE)  # key exists but isn't trusted
    person.typed, person.secrets = ["deadbeef"], []
    assert cli.main(sign) == 1
    person.typed, person.secrets = [sha[:8]], [PASSPHRASE]
    assert cli.main(sign) == 1
    person.typed, person.secrets = [sha[:8]], ["wrong passphrase!"]
    assert cli.main(sign) == 1
    assert not (package_dir / signing.APPROVAL_FILE).exists()


def test_committed_trust_file_parses() -> None:
    # Fine to be empty until someone runs keygen; must never be malformed.
    signing.load_trust(Path(__file__).parents[2] / signing.DEFAULT_TRUST_FILE)
