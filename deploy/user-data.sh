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
