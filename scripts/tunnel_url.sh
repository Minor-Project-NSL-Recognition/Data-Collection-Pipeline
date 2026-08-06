#!/usr/bin/env bash
# Print the public URL of the running Cloudflare tunnel, and check it works.
#
# The quick tunnel's hostname is random and only ever appears in cloudflared's
# log, so it has to be scraped back out.
#
#     ./scripts/tunnel_url.sh
#
# Works from Git Bash on Windows as well as Linux/macOS.

set -uo pipefail

CONTAINER="${1:-nsl-tunnel-quick}"

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  echo "Container '$CONTAINER' is not running." >&2
  echo >&2
  echo "  docker compose --profile quick up -d" >&2
  echo >&2
  echo "For a named tunnel the hostname is the one you configured in the" >&2
  echo "Cloudflare dashboard — there is nothing to scrape." >&2
  exit 1
fi

echo "Waiting for the tunnel to register..."
URL=""
for _ in $(seq 1 60); do
  URL=$(docker logs "$CONTAINER" 2>&1 \
        | grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' \
        | tail -1)
  [ -n "$URL" ] && break
  sleep 1
done

if [ -z "$URL" ]; then
  echo "No URL found in the logs after 60s. Recent output:" >&2
  docker logs --tail 30 "$CONTAINER" >&2
  exit 1
fi

echo
echo "  $URL"
echo
echo "Enter that in the app (gear icon). The client turns https:// into wss://"
echo "for the stream automatically."
echo

printf 'Checking /health ... '
if command -v curl >/dev/null 2>&1; then
  BODY=$(curl -fsS --max-time 20 "$URL/health" 2>/dev/null)
  if [ -n "$BODY" ]; then
    echo "ok"
    echo "$BODY"
    case "$BODY" in
      *'"auth_required":true'*)
        echo
        echo "Auth is on — the app needs the API key too." ;;
      *'"auth_required":false'*)
        echo
        echo "WARNING: this URL is public and unauthenticated."
        echo "Set NSL_API_KEY in .env and re-run 'docker compose up -d'." ;;
    esac
  else
    echo "FAILED"
    echo "The tunnel is up but the server did not answer. Try: docker compose logs server" >&2
    exit 1
  fi
else
  echo "skipped (no curl)"
fi
