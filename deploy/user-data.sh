#!/bin/bash
# deploy/user-data.sh
# EC2 "user data" bootstrap script — paste this into the "User data" field
# when launching the instance (Advanced details section) and AWS runs it
# automatically as root on first boot. No manual SSH-and-install-Docker
# step needed.
#
# Targets Amazon Linux 2023 (the default AMI suggested in deploy/README.md).

set -euo pipefail
exec > >(tee /var/log/user-data.log) 2>&1   # so `cat /var/log/user-data.log` shows what happened if anything fails

dnf update -y
dnf install -y docker git

systemctl enable docker
systemctl start docker
usermod -aG docker ec2-user

# Docker Compose v2 plugin (the `docker compose` subcommand, not the old
# standalone `docker-compose` binary) — not bundled with the docker package
# on Amazon Linux 2023, has to be fetched separately.
mkdir -p /usr/local/lib/docker/cli-plugins
curl -SL https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64 \
  -o /usr/local/lib/docker/cli-plugins/docker-compose
chmod +x /usr/local/lib/docker/cli-plugins/docker-compose

# Buildx plugin -- `docker compose up --build` needs it (Compose v5
# requires buildx >= 0.17). AL2023's base `docker` package DOES bundle a
# buildx plugin already, but as of this writing it's v0.12.1 -- older than
# Compose v5 requires -- so `docker compose up --build` fails outright with
# "compose build requires buildx 0.17.0 or later". There's no
# docker-buildx-plugin RPM in AL2023's repos to upgrade via dnf, so fetch a
# current release binary instead and place it in a cli-plugins dir that
# takes priority over wherever the bundled one lives (Docker's plugin
# search order checks /usr/local/lib/docker/cli-plugins before the system
# dirs the bundled version is likely in).
BUILDX_VERSION=$(curl -s https://api.github.com/repos/docker/buildx/releases/latest | grep '"tag_name"' | cut -d'"' -f4)
curl -SL "https://github.com/docker/buildx/releases/download/${BUILDX_VERSION}/buildx-${BUILDX_VERSION}.linux-amd64" \
  -o /usr/local/lib/docker/cli-plugins/docker-buildx
chmod +x /usr/local/lib/docker/cli-plugins/docker-buildx

# 2GB swap file. t3.micro's 1GB RAM is genuinely tight for `docker compose
# build` (resolving/installing langchain + chromadb + ragas + datasets in
# one image) and can OOM partway through without this. Swap turns "OOM
# killed" into "a bit slower," which is the right trade for a free-tier box.
fallocate -l 2G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab

echo "user-data.sh done" >> /var/log/user-data.log
