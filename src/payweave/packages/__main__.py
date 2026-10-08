"""Gateway Package approval CLI (gate G8).

    uv run python -m payweave.packages keygen <your-name>      # once: key + trusted_signers.toml
    uv run python -m payweave.packages review packages/razorpay/v1
    uv run python -m payweave.packages sign packages/razorpay/v1 --signer <your-name>
    uv run python -m payweave.packages verify                  # every package; exit 1 on failure

``sign`` only works in an interactive terminal: you type part of the hash you
reviewed and your key's passphrase. Nothing can approve a package for you. It
also needs a current, promotable report.json (``python -m payweave.verify <dir>``).
"""

from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

from payweave.packages import signing
from payweave.packages.runtime import GatewayPackage, PackageLoadError, load_package
from payweave.verify.report import read_report, report_problem

PACKAGES = Path("packages")
CONFIRM_CHARS = 8

# Indirection so tests can stand in for the person at the keyboard.
_input = input
_getpass = getpass.getpass


def _is_interactive() -> bool:
    return sys.stdin.isatty()


def describe(package: GatewayPackage) -> list[str]:
    """What a reviewer should look at before approving."""
    m = package.manifest
    ops = m.operations
    lines = [
        f"{m.gateway} v{m.version}",
        f"  sha256        {package.sha256}",
        f"  base_url      {m.api.base_url}   (allowed hosts: {', '.join(m.api.allowed_hosts)})",
        f"  auth          {m.auth.scheme}"
        + (f" {m.auth.username}:{m.auth.password}" if m.auth.username else f" {m.auth.password}"),
        "  credentials   "
        + ", ".join(
            name
            + ("" if not (c.secret or c.public) else f" ({'secret' if c.secret else 'public'})")
            for name, c in m.credentials.items()
        ),
    ]
    if m.live_guard:
        lines.append(f"  live guard    {m.live_guard.credential} ~ {m.live_guard.live_pattern}")
    requests = [
        ("create_payment", ops.create_payment),
        ("capture", ops.capture),
        ("refund", ops.refund),
        ("get_refund", ops.get_refund),
    ]
    if ops.get_payment:
        requests.insert(1, ("get_payment", ops.get_payment.primary))
        requests[2:2] = [(f"  + {n}", r) for n, r in ops.get_payment.related.items()]
    lines.append("  operations")
    lines += [f"    {name:<16}{r.method:<5} {r.path}" for name, r in requests if r is not None]
    lines.append(f"  flags         {', '.join(m.flags) or '-'}")
    if m.webhooks:
        w = m.webhooks
        lines.append(f"  webhooks      {w.scheme} in {w.signature_header}, id {w.event_id_header}")
    if cr := m.checkout_return:
        lines.append(f"  checkout      {cr.scheme} over {cr.message_template!r}")
    lines.append(f"  error rules   {len(m.errors.rules)}")
    lines.append(f"  mappers       {', '.join(package.mappers)}")
    return lines


def approval_status(directory: Path, trust_file: Path) -> str:
    try:
        signing.load_signed_package(directory, signing.load_trust(trust_file))
    except PackageLoadError as exc:
        return f"NOT APPROVED: {exc}"
    approval = signing.read_approval(directory)
    return f"approved by {approval.signer} at {approval.signed_at.isoformat()}"


def cmd_keygen(args: argparse.Namespace) -> int:
    print(f"Creating an Ed25519 signing key for {args.name!r} in {signing.key_dir()}")
    passphrase = _getpass(f"Passphrase (min {signing.MIN_PASSPHRASE} chars): ")
    if passphrase != _getpass("Repeat passphrase: "):
        print("Passphrases differ; nothing written.", file=sys.stderr)
        return 1
    path, public = signing.write_private_key(args.name, passphrase)
    signing.add_trusted_signer(args.name, public, args.trust_file)
    print(f"Private key: {path} (never commit it; losing the passphrase means a new key)")
    print(f"Public key added to {args.trust_file}; commit that file.")
    return 0


def report_lines(package: GatewayPackage, directory: Path) -> list[str]:
    report = read_report(directory)
    problem = report_problem(package, report)
    lines = [f"  gate report   {problem or 'promotable'}"]
    if report is not None and (report.get("package") or {}).get("sha256") == package.sha256:
        for name, g in report["gates"].items():
            lines.append(f"    {name} {g['status']:<8}{g['summary']}")
    return lines


def cmd_review(args: argparse.Namespace) -> int:
    package = load_package(args.directory)  # runs every load-time gate
    print("\n".join(describe(package)))
    print("\n" + "\n".join(report_lines(package, args.directory)))
    print(f"\n  status        {approval_status(args.directory, args.trust_file)}")
    print(f"\nRead the files themselves too: {args.directory}/manifest.toml, mappers.py")
    return 0


def cmd_sign(args: argparse.Namespace) -> int:
    if not _is_interactive():
        print("sign needs an interactive terminal: a person has to approve.", file=sys.stderr)
        return 2
    package = load_package(args.directory)
    if problem := report_problem(package, read_report(args.directory)):
        print(f"Not signing: {problem}.", file=sys.stderr)
        return 1
    print("\n".join(describe(package)))
    print("\n" + "\n".join(report_lines(package, args.directory)))
    print(f"\nYou are approving exactly these bytes for real traffic: {package.sha256}")
    typed = _input(f"Type the first {CONFIRM_CHARS} characters of the sha256 to approve: ")
    if typed.strip() != package.sha256[:CONFIRM_CHARS]:
        print("Didn't match; not signed.", file=sys.stderr)
        return 1
    key = signing.read_private_key(args.signer, _getpass("Key passphrase: "))
    trusted = signing.load_trust(args.trust_file).get(args.signer)
    if trusted is None or signing.public_key_b64(trusted) != signing.public_key_b64(key):
        print(f"{args.signer!r}'s key is not in {args.trust_file}; not signed.", file=sys.stderr)
        return 1
    approval = signing.sign_package(package, key, args.signer)
    (args.directory / signing.APPROVAL_FILE).write_text(approval.to_toml())
    signing.load_signed_package(args.directory, signing.load_trust(args.trust_file))
    print(f"Approved. Wrote {args.directory / signing.APPROVAL_FILE}; commit it with the package.")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Approval valid *and* report.json current and promotable, for each package."""
    directories = args.directories or sorted(p.parent for p in PACKAGES.glob("*/*/manifest.toml"))
    trust = signing.load_trust(args.trust_file)
    failed = 0
    for directory in directories:
        try:
            package = signing.load_signed_package(directory, trust)
        except PackageLoadError as exc:
            problem: str | None = str(exc)
        else:
            problem = report_problem(package, read_report(directory))
        if problem:
            failed += 1
            print(f"FAIL {directory}: {problem}")
        else:
            print(f"ok   {directory}: approved by {signing.read_approval(directory).signer}")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m payweave.packages")
    parser.add_argument("--trust-file", type=Path, default=signing.DEFAULT_TRUST_FILE)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("keygen", help="create your signing key and trust its public half")
    p.add_argument("name")
    p.set_defaults(run=cmd_keygen)
    p = sub.add_parser("review", help="summarise a package and its approval status")
    p.add_argument("directory", type=Path)
    p.set_defaults(run=cmd_review)
    p = sub.add_parser("sign", help="approve a package (interactive)")
    p.add_argument("directory", type=Path)
    p.add_argument("--signer", required=True)
    p.set_defaults(run=cmd_sign)
    p = sub.add_parser("verify", help="check approvals (all packages by default)")
    p.add_argument("directories", nargs="*", type=Path)
    p.set_defaults(run=cmd_verify)
    args = parser.parse_args(argv)
    try:
        code: int = args.run(args)
    except (PackageLoadError, ValueError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return code


if __name__ == "__main__":
    sys.exit(main())
