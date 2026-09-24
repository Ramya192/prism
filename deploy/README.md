# Deploying Prism to AWS — full walkthrough, starting from account sign-up

Phase 1 goal: one public `http://<ip>:8501` URL serving the real app, on
AWS Free Tier only (no ALB, no ECS, no NAT gateway — those all cost
money). TLS + a proper domain is phase 2, layered on top once this is
confirmed working — see the note at the very end.

**You run every step below** — nothing here needs AWS credentials stored
locally. If a step doesn't match what's described, capture the exact
error text before debugging further.

---

## Part 1 — Create your AWS account

1. Go to **https://aws.amazon.com** → click **Create an AWS Account** (top right).
2. Enter your email address and choose an **AWS account name** (e.g.
   "Ramya Personal" — this is just a label, not public).
3. Check your email for a **verification code** from AWS, enter it.
4. Set a **root user password** (this is the account owner login — strong,
   unique, save it in a password manager. You'll create a separate,
   more restricted login in Part 2).
5. **Contact information** — choose **Personal** account type, fill in
   name/phone/address.
6. **Billing information** — AWS requires a valid credit/debit card even
   though you're only using free-tier resources. This is for identity
   verification; free-tier usage should show **$0.00** on your bill as
   long as you stay within the limits this guide sticks to. (The budget
   alert in Part 2 is your safety net if something ever doesn't.)
7. **Identity verification** — enter your phone number, choose SMS or
   voice call, enter the code AWS sends you.
8. **Choose a support plan** — select **Basic support - Free**. (Do not
   pick a paid support tier — Basic is free and enough for this project.)
9. Click through to finish. You'll land on the **AWS Management Console**.
   Sign in at **https://console.aws.amazon.com** going forward, or from
   now on I'll just say "the Console."

---

## Part 2 — Account safety basics (do this before touching EC2)

Two things that cost nothing and directly protect the "this should never
charge me" requirement:

### 2a. Set a billing alert

Console → search **"Billing"** in the top search bar → **Billing and Cost
Management** → left sidebar **Budgets** → **Create budget**.

- Choose **Zero spend budget** (a built-in template — alerts you the
  moment *any* charge appears) — simplest option, pick this.
- Or, for more headroom, a **Cost budget**: set it to **$5/month**, add
  your email as an alert recipient at, say, 80% and 100% thresholds.
- Save it. You'll now get an email the moment AWS is about to charge you
  anything, instead of finding out at the end of the month.

### 2b. Enable MFA on your root login (recommended, 2 minutes)

Console → click your account name (top right) → **Security credentials**
→ **Multi-factor authentication (MFA)** → **Assign MFA device**. Use an
authenticator app (Google Authenticator, Authy, etc.) — scan the QR code,
enter two consecutive codes to confirm.

### 2c. (Optional but good practice) Create an IAM user instead of using root day-to-day

For a small personal project, using the root login for the steps below
is *acceptable* — but if you want to follow the real-world pattern
(never use root for daily work), Console → search **IAM** → **Users** →
**Create user** → give it **AdministratorAccess** policy for now (you
can tighten this later) → sign in as that user instead of root from here
on. Skip this if you'd rather keep things simple for a one-off deploy —
nothing below requires it.

---

## Part 3 — Create a Security Group

A Security Group is AWS's firewall for your instance — it decides what
traffic is allowed in.

Console → search **EC2** → left sidebar **Security Groups** → **Create
security group**.

- Name: `prism-sg`
- Description: `Prism app — SSH from me, 8501 public`
- VPC: leave the default VPC selected
- Inbound rules → **Add rule** (twice):

  | Type | Port | Source |
  |---|---|---|
  | SSH | 22 | **My IP** (dropdown — auto-fills your current IP; do not use 0.0.0.0/0 here) |
  | Custom TCP | 8501 | **Anywhere-IPv4** (0.0.0.0/0) — this is the public demo URL |

- Outbound rules: leave the default (**All traffic, 0.0.0.0/0**) — the
  instance needs to reach OpenAI's API, PyPI, GitHub, etc.
- Click **Create security group**.

---

## Part 4 — Launch the EC2 instance

EC2 Console → **Instances** (left sidebar) → **Launch instances**.

1. **Name**: `prism`
2. **Application and OS Images (AMI)**: select **Amazon Linux 2023** —
   it's the default suggestion and has a **Free tier eligible** badge.
3. **Instance type**: select **t3.micro** — confirm it shows the **Free
   tier eligible** badge. (If your region doesn't offer t3.micro under
   free tier, use **t2.micro** instead — same idea.)
4. **Key pair (login)**: click **Create new key pair**.
   - Name: `prism-key`
   - Type: RSA, Format: `.pem` (or `.ppk` if you'll use PuTTY on Windows —
     `.pem` is fine if you use PowerShell's built-in `ssh`, which is what
     the rest of this guide assumes)
   - Click **Create key pair** — it downloads automatically. **Save this
     file somewhere safe** (e.g. `C:\Users\<you>\.ssh\prism-key.pem`) —
     AWS will not let you download it again.
5. **Network settings** → click **Edit** → **Select existing security
   group** → choose `prism-sg` (the one created in Part 3).
6. **Configure storage**: leave the default **8 GiB gp3** — well within
   the free tier's 30GB allowance.
7. **Advanced details** → scroll down to **User data** → paste the
   *entire* contents of [`deploy/user-data.sh`](user-data.sh) into the
   text box. This auto-installs Docker, the Compose plugin, and a swap
   file the moment the instance boots — you won't need to do that part
   by hand over SSH.
8. Review the **Summary** panel on the right — confirm it says **Free
   tier eligible**. Click **Launch instance**.
9. Click **View all instances**, wait for **Instance state** to show
   **Running** and **Status check** to show **2/2 checks passed**
   (~1-2 minutes).

---

## Part 5 — Allocate and attach an Elastic IP

Without this, the instance's public IP changes every time you stop/start
it.

EC2 Console → left sidebar **Elastic IPs** → **Allocate Elastic IP
address** → leave defaults → **Allocate**.

Select the new address (checkbox) → **Actions** → **Associate Elastic IP
address** → **Instance**: select `prism` → **Associate**.

**Write down this IP address** — this is your app's URL:
`http://<that-ip>:8501`.

⚠️ Elastic IPs are free **only while attached to a running instance**. If
you ever stop the instance without releasing the IP, AWS starts charging
a small hourly fee for the idle allocation — release it (**Actions →
Release Elastic IP address**) if you tear the instance down for good.

---

## Part 6 — Connect and deploy

### 6a. SSH in

From PowerShell on your machine (adjust the key path and IP):

```powershell
ssh -i "C:\Users\<you>\.ssh\prism-key.pem" ec2-user@<elastic-ip>
```

First connection: type `yes` when asked about the host's fingerprint.

If you get a "Permissions are too open" / "UNPROTECTED PRIVATE KEY FILE"
error, PowerShell needs the key file locked down:

```powershell
icacls "C:\Users\<you>\.ssh\prism-key.pem" /inheritance:r
icacls "C:\Users\<you>\.ssh\prism-key.pem" /grant:r "$($env:USERNAME):(R)"
```

### 6b. Verify the bootstrap script finished

```bash
cat /var/log/user-data.log
docker --version
docker compose version
```

If `docker compose version` errors, user-data likely hasn't finished yet
— wait another minute and retry.

### 6c. Get the code

```bash
git clone https://github.com/Ramya192/prism.git
cd prism
```

### 6d. Configure `.env`

```bash
cp deploy/.env.production.example .env
nano .env
```

Fill in `OPENAI_API_KEY` (required) and `COHERE_API_KEY` (optional — the
reranker degrades gracefully without it). Save and exit nano: `Ctrl+O`,
`Enter`, `Ctrl+X`.

This `.env` is **not** the same as your local dev `.env` — it sets
`ENV=production`, which routes banking document chat to OpenAI instead of a
local Ollama that doesn't exist on this box. See the comments in
`deploy/.env.production.example` for why.

### 6e. Build and run

```bash
docker compose up --build -d
```

First build takes a few minutes (installing langchain/chromadb/ragas into
the image) — the swap file from `user-data.sh` exists specifically so
this doesn't OOM-kill on a 1GB-RAM instance. Watch progress if you want:

```bash
docker compose logs -f
```
(`Ctrl+C` exits the log follow — the container keeps running in the
background either way, `-d` already detached it.)

### 6f. Ingest the reference corpora (once per fresh vector store)

The instance's `vector_store/` starts empty, so chat has no domain policy
grounding (Regulation E, CMS Ch. 26, IRS Pub 15-T, FinCEN CVC guidance)
until each domain's reference corpus is ingested. Run all four — each is
idempotent, and the store persists across restarts via the compose volume:

```bash
docker compose exec prism python -m domains.banking.documents.data.ingest_reference_corpus
docker compose exec prism python -m domains.insurance.data.ingest_reference_corpus
docker compose exec prism python -m domains.payroll_hr.data.ingest_reference_corpus
docker compose exec prism python -m domains.financial_services.data.ingest_reference_corpus
```

Redeploying code later is just `git pull && docker compose up --build -d`
— the ingest only needs repeating if `vector_store/` is wiped.

The ML scorers need no data on the instance: their trained models ship in
the image (`models/*.joblib`, committed to the repo). If startup logs show
`Ignoring model artifact ... built with ...`, the image's
scikit-learn/LightGBM/XGBoost differ from the ones the artifacts were built
with — rebuild them locally (`python -m core.model_store --rebuild`), commit,
and redeploy.

---

## Part 7 — Verify

Open `http://<elastic-ip>:8501` in a browser. You should see the Prism
UI. Walk all 4 domains through once — pick a sample file on the landing
page, confirm the detected domain, ingest it, read the fraud verdict, and
ask a chat question that should cite the domain's reference corpus — to
confirm it's not just serving a blank page.

---

## Troubleshooting

- **`compose build requires buildx 0.17.0 or later`**: AL2023's base
  `docker` package bundles an older buildx (v0.12.1 as of this writing).
  `deploy/user-data.sh` now fetches a current buildx binary automatically
  on first boot — if you hit this anyway (e.g. an instance launched before
  this fix), run:
  ```bash
  BUILDX_VERSION=$(curl -s https://api.github.com/repos/docker/buildx/releases/latest | grep '"tag_name"' | cut -d'"' -f4)
  sudo curl -SL "https://github.com/docker/buildx/releases/download/${BUILDX_VERSION}/buildx-${BUILDX_VERSION}.linux-amd64" \
    -o /usr/local/lib/docker/cli-plugins/docker-buildx
  sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-buildx
  docker buildx version   # confirm it now shows >= 0.17.0
  ```
- **Container keeps restarting / looks OOM-killed**: `docker stats` to
  watch memory live. If it's still failing with the swap file in place,
  the honest fallback is a `t3.small` (2GB RAM, ~$15/mo — no longer
  free) rather than fighting a 1GB box indefinitely. Try swap first;
  only go there if it's still failing.
- **Can't reach the app from a browser**: double-check `prism-sg` really
  has the 8501 inbound rule, and that you're using the **Elastic IP**
  (not the private IP shown inside the instance via `hostname -I`).
- **`docker compose up` says env vars are empty**: confirm you're running
  the command from the `prism/` directory (the one containing
  `docker-compose.yml`) — `env_file: .env` in that file is a relative path.
- **SSH connection times out**: check the security group's SSH rule
  source — if your IP changed since Part 3 (e.g. different wifi), edit
  the rule to "My IP" again to refresh it.
- **Old uploads linger in the vector store**: documents uploaded before
  per-visitor naming have no owner tag, so the duplicate check (correctly)
  ignores them and no visitor's chat can reach them. Clear them once with
  `docker compose exec prism python -m core.rag.purge_legacy_uploads`
  (lists only), then re-run with `--apply` to delete. Reference corpora
  are never touched.

---

## Phase 2 (later): nginx reverse proxy + free TLS

Once phase 1 is confirmed working, the next layer is: install Caddy or
nginx+Certbot on the instance, point a free subdomain (e.g. via DuckDNS)
at the Elastic IP, and get a free Let's Encrypt certificate for it — so
the public URL becomes `https://prism-ramya.duckdns.org` instead of
`http://<ip>:8501`, with the reverse proxy listening on 80/443 and
forwarding to the container's 8501 internally. Open the security group's
80/443 ports for that when we get there — not needed yet.
