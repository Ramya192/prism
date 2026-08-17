# Deploying Prism to AWS (EC2 free tier, raw IP — phase 1)

Phase 1 goal: one public `http://<ip>:8501` URL serving the real app, on
AWS Free Tier only (no ALB, no ECS, no NAT gateway — those all cost money).
TLS + a proper domain is phase 2, layered on top once this is confirmed
working — see the note at the bottom.

**You run these steps** (AWS account access + this repo's GitHub URL is
all you need — nothing here requires AWS CLI credentials in the Claude
Code sandbox).

---

## 0. Prerequisites

- An AWS account (note the 2025 Free Tier change: $200 credit for 6
  months on new accounts, vs. the older "12 months of specific free
  resources" on older accounts — check what your account has).
- An SSH key pair you can create during instance launch (AWS will offer
  to generate one and download a `.pem` file — save it, you can't
  re-download it later).
- Your `OPENAI_API_KEY` (and optionally `COHERE_API_KEY`) ready to paste
  into a `.env` file on the instance.

---

## 1. Create a Security Group

EC2 Console → **Security Groups** → **Create security group**.

- Name: `prism-sg`
- Inbound rules:
  | Type | Port | Source |
  |---|---|---|
  | SSH | 22 | My IP (not 0.0.0.0/0 — no reason to expose SSH to the whole internet) |
  | Custom TCP | 8501 | Anywhere (0.0.0.0/0) — this is the public demo URL |
- Outbound: leave the default (all traffic allowed) — the instance needs
  to reach OpenAI's API, PyPI, GitHub, etc.

## 2. Launch the instance

EC2 Console → **Launch instance**.

- Name: `prism`
- AMI: **Amazon Linux 2023** (free tier eligible)
- Instance type: **t3.micro** (or t2.micro if t3 isn't offered in your
  region's free tier — check the "Free tier eligible" badge next to
  whichever you pick)
- Key pair: create/select one, download the `.pem`
- Network settings → Security group: select the existing `prism-sg`
  created above
- Storage: default 8GB gp3 is fine (free tier covers up to 30GB)
- **Advanced details → User data**: paste the entire contents of
  [`deploy/user-data.sh`](user-data.sh) here. This auto-installs Docker,
  the Compose plugin, and a swap file on first boot — no manual SSH setup
  step needed for that part.

Launch it. Wait ~1-2 minutes for it to boot and for user-data to finish
(you can check later via `cat /var/log/user-data.log` once you SSH in).

## 3. Allocate + associate an Elastic IP

Without this, the instance's public IP changes every time you stop/start
it. EC2 Console → **Elastic IPs** → **Allocate Elastic IP address** →
**Allocate**. Then select it → **Actions → Associate Elastic IP address**
→ pick the `prism` instance. (Elastic IPs are free *while attached to a
running instance* — if you ever stop the instance without releasing the
IP, AWS starts charging for the idle allocation, so release it if you
tear the instance down.)

Note the Elastic IP address — that's your app's URL: `http://<that-ip>:8501`.

## 4. SSH in and verify Docker

```bash
ssh -i /path/to/your-key.pem ec2-user@<elastic-ip>
cat /var/log/user-data.log   # confirm it finished without errors
docker --version
docker compose version
```

If `docker compose version` fails, user-data likely hasn't finished yet —
wait another minute and retry, or re-run the relevant commands from
`user-data.sh` manually.

## 5. Get the code and configure `.env`

```bash
git clone https://github.com/Ramya192/prism.git
cd prism
cp deploy/.env.production.example .env
nano .env   # fill in OPENAI_API_KEY (and COHERE_API_KEY if you have one)
```

`.env` here is **not** the same file as your local dev `.env` — it sets
`ENV=production`, which routes `bfsi_documents` to OpenAI instead of a
local Ollama that doesn't exist on this box. See the comments in
`deploy/.env.production.example` for why.

## 6. Build and run

```bash
docker compose up --build -d
```

First build will take a few minutes (installing langchain/chromadb/ragas
into the image) — the swap file from user-data.sh is there specifically
so this doesn't OOM-kill on a 1GB-RAM instance. Watch it if you want:

```bash
docker compose logs -f
```

## 7. Verify

Open `http://<elastic-ip>:8501` in a browser. You should see the Prism
UI. Walk both domains through once (a fraud transaction, and ingest+query
one of the committed demo statement PDFs) to confirm it's not just
serving a blank page.

---

## Troubleshooting

- **Container keeps restarting / OOM**: `docker stats` to check memory
  while it's running. If `bfsi_documents`'s combined dependency footprint
  (langchain + chromadb + ragas + datasets) is still too tight even with
  the swap file, the honest fallback is a `t3.small` (2GB RAM, ~$15/mo —
  no longer free) rather than fighting a 1GB box indefinitely. Try swap
  first; only go there if it's still failing.
- **Can't reach the app from a browser**: double check the `prism-sg`
  security group actually has the 8501 inbound rule, and that you're
  using the Elastic IP (not the private IP shown inside the instance).
- **`docker compose up` says `.env` vars are empty**: confirm you're
  running the command from the `prism/` directory (the one containing
  `docker-compose.yml`) — `env_file: .env` in that file is a relative
  path.

---

## Phase 2 (later): nginx reverse proxy + free TLS

Once phase 1 is confirmed working, the next layer is: install
Caddy or nginx+Certbot on the instance, point a free subdomain (e.g. via
DuckDNS) at the Elastic IP, and get a free Let's Encrypt certificate for
it — so the public URL becomes `https://prism-ramya.duckdns.org` instead
of `http://<ip>:8501`, with the reverse proxy listening on 80/443 and
forwarding to the container's 8501 internally. Open the Security Group's
80/443 ports for that when we get there; not needed yet.
