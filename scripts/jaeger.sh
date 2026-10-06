#!/usr/bin/env bash
# Run Jaeger v2 locally (no Docker): download once, verify its checksum, start it.
#   UI:   http://localhost:16686
#   OTLP: http://localhost:4318 (HTTP), localhost:4317 (gRPC)
# Traces live in memory and are gone when Jaeger stops.
set -euo pipefail

VERSION="${JAEGER_VERSION:-2.22.0}"
case "$(uname -s)-$(uname -m)" in
  Darwin-arm64) PLATFORM=darwin-arm64 ;;
  Darwin-x86_64) PLATFORM=darwin-amd64 ;;
  Linux-x86_64) PLATFORM=linux-amd64 ;;
  Linux-aarch64) PLATFORM=linux-arm64 ;;
  *) echo "unsupported platform: $(uname -s)-$(uname -m)" >&2; exit 1 ;;
esac
NAME="jaeger-${VERSION}-${PLATFORM}"
DIR="$(cd "$(dirname "$0")/.." && pwd)/.tools"
BIN="$DIR/$NAME/jaeger"

if [[ ! -x "$BIN" ]]; then
  mkdir -p "$DIR"
  URL="https://github.com/jaegertracing/jaeger/releases/download/v${VERSION}"
  curl -fsSL -o "$DIR/$NAME.tar.gz" "$URL/$NAME.tar.gz"
  curl -fsSL -o "$DIR/$NAME.sha256sum.txt" "$URL/$NAME.sha256sum.txt"
  tar -xzf "$DIR/$NAME.tar.gz" -C "$DIR"
  # The release checksum file covers the binaries inside the archive.
  (cd "$DIR" && grep "$NAME/jaeger\$" "$NAME.sha256sum.txt" | sed 's/ \*/  /' | shasum -a 256 -c -)
fi

echo "Jaeger UI: http://localhost:16686  (OTLP/HTTP on :4318)"
exec "$BIN"
