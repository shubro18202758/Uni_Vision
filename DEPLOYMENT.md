# Uni_Vision — Production Deployment Guide

This document covers deploying Uni_Vision in a production environment. For local
development see [README.md](README.md).

---

## Prerequisites

| Component | Minimum Version | Notes |
|---|---|---|
| Docker Engine | 24.0+ | with Compose V2 |
| NVIDIA Driver | 535+ | for GPU passthrough |
| NVIDIA Container Toolkit | 1.14+ | `nvidia-ctk` configured |
| Disk | 40 GB free | models + data volumes |
| RAM | 16 GB | PostgreSQL, Redis, application |
| GPU | NVIDIA with ≥ 8 GB VRAM | RTX 4070 / T4 / A10 |

---

## 1. Environment Variables

Create a `.env` file at the project root. All variables use the `UV_` prefix.

### Core Services

| Variable | Default | Description |
|---|---|---|
| `UV_POSTGRES_DSN` | `postgresql://uni_vision:changeme@localhost:5432/uni_vision` | PostgreSQL connection string |
| `UV_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama LLM server URL |
| `UV_S3_ENDPOINT` | `http://localhost:9000` | MinIO / S3 endpoint |
| `UV_S3_ACCESS_KEY` | `minioadmin` | S3 access key |
| `UV_S3_SECRET_KEY` | `minioadmin` | S3 secret key |
| `UV_S3_BUCKET` | `uni-vision-images` | Image archive bucket |
| `UV_REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |

### API & Security

| Variable | Default | Description |
|---|---|---|
| `UV_API_HOST` | `0.0.0.0` | Bind address |
| `UV_API_PORT` | `8000` | Listen port |
| `UV_API_API_KEYS` | *(empty — auth disabled)* | Comma-separated API keys |
| `UV_API_RATE_LIMIT_RPM` | `120` | Requests/minute per IP (0 = off) |
| `UV_API_CORS_ORIGINS` | *(empty — allow all)* | Comma-separated allowed origins |

### Logging

| Variable | Default | Description |
|---|---|---|
| `UV_LOG_LEVEL` | `INFO` | Log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `UV_LOG_FORMAT` | `json` | Log format (`json` or `console`) |

### Data Retention

| Variable | Default | Description |
|---|---|---|
| `UV_RETENTION_ENABLED` | `false` | Enable automated data cleanup |
| `UV_RETENTION_MAX_AGE_DAYS` | `90` | Delete detections older than N days |
| `UV_RETENTION_AUDIT_MAX_AGE_DAYS` | `180` | Delete audit logs older than N days |
| `UV_RETENTION_CHECK_INTERVAL_HOURS` | `24` | Hours between cleanup runs |
| `UV_RETENTION_BATCH_SIZE` | `1000` | Rows deleted per batch (limits lock time) |

### Pipeline Tuning

| Variable | Default | Description |
|---|---|---|
| `UV_INFERENCE_QUEUE_MAXSIZE` | `10` | Max frames queued for inference |
| `UV_INFERENCE_QUEUE_HIGH_WATER` | `8` | Throttle new frames above this |
| `UV_INFERENCE_QUEUE_LOW_WATER` | `3` | Resume accepting frames below this |
| `UV_VRAM_CEILING_MB` | `8192` | Max VRAM budget (MB) |

---

## 2. Docker Compose Deployment

```bash
# 1. Clone and enter the project
git clone <repo-url> && cd Uni_Vision

# 2. Create your production .env (see section 1)
cp .env.example .env
# Edit .env — at minimum set strong API keys and DB password

# 3. Build and start all services
docker compose up -d --build

# 4. Run database migrations
docker compose exec app alembic upgrade head

# 5. Verify health
curl -s http://localhost:8000/health | jq .
```

The stack includes 7 services: `app`, `postgres`, `ollama`, `minio`,
`redis`, `prometheus`, `grafana`. All have health checks and automatic
restart (`unless-stopped`).

### GPU Passthrough

The `app` and `ollama` services reserve 1 NVIDIA GPU each (they can
share the same physical GPU). Ensure the NVIDIA Container Toolkit is
installed:

```bash
nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Volume Persistence

Docker volumes store all stateful data:

| Volume | Service | Content |
|---|---|---|
| `pgdata` | PostgreSQL | Database files |
| `ollama_data` | Ollama | Downloaded model weights |
| `minio_data` | MinIO | Archived plate images |
| `redis_data` | Redis | Pub/sub state |
| `prometheus_data` | Prometheus | Metrics history |
| `grafana_data` | Grafana | Dashboard configs |

---

## 3. Reverse Proxy / HTTPS

In production, place a reverse proxy (NGINX, Caddy, Traefik) in front of
the `app` service on port 8000.

**Caddy example** (`Caddyfile`):

```
anpr.example.com {
    reverse_proxy localhost:8000
}
```

Caddy auto-provisions TLS via Let's Encrypt. For WebSocket support, ensure
the proxy forwards the `Upgrade` and `Connection` headers.

---

## 4. Database

### Migrations

Uni_Vision uses Alembic for schema migrations:

```bash
# Apply all pending migrations
make db-upgrade

# Generate a new migration after model changes
make db-revision MSG="add_new_column"

# Rollback one migration
make db-downgrade
```

### Backups

Schedule `pg_dump` via cron or use the PostgreSQL container directly:

```bash
docker compose exec postgres pg_dump -U uni_vision uni_vision | gzip > backup_$(date +%Y%m%d).sql.gz
```

For point-in-time recovery, enable WAL archiving on the PostgreSQL
instance and ship WAL files to durable storage.

---

## 5. Data Retention

When `UV_RETENTION_ENABLED=true`, a background task periodically purges
old records:

- **Detection events** older than `max_age_days` (default 90)
- **Audit logs** older than `audit_max_age_days` (default 180)

Records are deleted in batches (default 1,000 rows) to avoid long
table-level locks. The cleanup interval defaults to once every 24 hours.

Monitor purge activity via the `retention_purged` log line.

---

## 6. Monitoring & Alerting

### Prometheus (port 9090)

Scrapes the `/metrics` endpoint every 15 seconds. Pre-configured alert
rules in `config/prometheus_alerts.yml` cover:

- High request latency (p95 > 2 s)
- Elevated error rate (> 5% 5xx)
- Pipeline queue depth saturation
- VRAM budget exceeded
- Database connection pool exhaustion

### Grafana (port 3000)

Default credentials: `admin` / `admin` (change on first login).

Pre-provisioned dashboard: **Uni_Vision Overview** — 14 panels covering
API latency, throughput, pipeline queue depth, VRAM usage, OCR accuracy,
and detection rate.

### Structured Logging

Logs are emitted in JSON format by default (`UV_LOG_FORMAT=json`). Each
line includes timestamp, level, logger name, and a structured event key.
Forward logs to your aggregation platform (ELK, Loki, CloudWatch) via a
Docker log driver or sidecar.

---

## 7. Scaling Considerations

| Dimension | Approach |
|---|---|
| **Horizontal API** | Run multiple `app` containers behind a load balancer; use `UV_API_WORKERS` > 1 |
| **Pipeline throughput** | Tune `UV_INFERENCE_QUEUE_MAXSIZE` and water-mark thresholds |
| **Database** | Increase `UV_POSTGRES_POOL_MAX`; add read replicas for query-heavy workloads |
| **Object storage** | Configure MinIO erasure-coding or switch to AWS S3 |
| **Multi-GPU** | Assign Ollama and the pipeline to separate `cuda_device_index` values |

---

## 8. Security Checklist

- [ ] Set strong, unique `UV_API_API_KEYS` (SHA-256 compared at runtime)
- [ ] Restrict `UV_API_CORS_ORIGINS` to your frontend domains
- [ ] Change default PostgreSQL and MinIO passwords in `.env`
- [ ] Change Grafana admin password on first login
- [ ] Terminate TLS at the reverse proxy — never expose port 8000 raw
- [ ] Use Docker network isolation — only expose ports that must be public
- [ ] Enable `UV_API_RATE_LIMIT_RPM` to mitigate abuse
- [ ] Rotate API keys periodically
- [ ] Review Prometheus alert rules and configure a notification channel

---

## 9. Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `429 Too Many Requests` | Rate limit exceeded | Lower request rate or increase `UV_API_RATE_LIMIT_RPM` |
| Pipeline stalls | VRAM exhaustion | Lower `UV_VRAM_CEILING_MB` or use INT8 TensorRT engines |
| `connection refused` on DB | PostgreSQL not ready | Check `docker compose logs postgres`; increase `start_period` |
| Ollama timeout | Model not loaded | Run `ollama run qwen3.5:9b-q4_K_M` manually or wait for `ollama-init` |
| WebSocket disconnects | Reverse proxy buffering | Enable `proxy_buffering off` in NGINX |
| Images not archived | MinIO bucket missing | Create via `mc mb minio/uni-vision-images` |
