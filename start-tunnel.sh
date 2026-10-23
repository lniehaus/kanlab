#!/usr/bin/env bash
# Start the Cloudflare Tunnel for kanlab.app as a standalone container.
# Reads TUNNEL_TOKEN from .env and attaches to the compose network; all
# traffic routed to the tunnel is forwarded to the site at http://kanlab:80.
set -euo pipefail

cd "$(dirname "$0")"

NAME=kanlab-cloudflared
NETWORK=kanlab_default

# Shells started before joining the docker group lack it; re-run under it
if ! docker info >/dev/null 2>&1 && id -nG "$USER" | grep -qw docker && ! id -nG | grep -qw docker; then
    exec sg docker -c "$(printf '%q ' "$0" "$@")"
fi

if ! grep -qE '^TUNNEL_TOKEN=.+' .env 2>/dev/null; then
    echo "error: TUNNEL_TOKEN is not set in .env (see .env.example)" >&2
    exit 1
fi

# The site container must be running so the network and the "kanlab" name exist
docker compose up -d --build kanlab

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d \
    --name "$NAME" \
    --network "$NETWORK" \
    --env-file .env \
    --restart unless-stopped \
    --read-only \
    --security-opt no-new-privileges:true \
    cloudflare/cloudflared:latest tunnel --no-autoupdate --url http://kanlab:80 run

echo "Tunnel started. Follow the logs with: docker logs -f $NAME"
