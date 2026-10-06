#!/usr/bin/env bash
# Re-download gateway doc snapshots listed in docs/sources/<gateway>/urls.txt and
# report pages whose content changed since SNAPSHOT.tsv was recorded.
# Raw copies are not committed (third-party content); only urls + hashes are.
set -euo pipefail
cd "$(dirname "$0")/../docs/sources"
for dir in */; do
  gw="${dir%/}"; [[ -f "$gw/urls.txt" ]] || continue
  changed=0
  while read -r url; do
    [[ -z "$url" ]] && continue
    file=$(echo "$url" | sed 's#https://[^/]*/docs/build/llm-docs/##; s#/#__#g')
    curl -sfL "$url" -o "$gw/$file"
    new=$(shasum -a 256 "$gw/$file" | cut -c1-16)
    old=$(awk -F'\t' -v f="$file" '$1==f {print $3}' "$gw/SNAPSHOT.tsv")
    if [[ -n "$old" && "$old" != "$new" ]]; then echo "CHANGED  $gw/$file"; changed=1; fi
  done < "$gw/urls.txt"
  [[ $changed -eq 0 ]] && echo "$gw: all pages match SNAPSHOT.tsv"
done
