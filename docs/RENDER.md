# Hosting Chronicle on Render

The app is a normal FastAPI + NiceGUI service, so Render runs it as a **Web
Service**. With no LLM and no perception, the free plan is enough.

- `/ask` — the hosted one-box page (`Ask your memory`)
- `/` — the full mission-control dashboard
- `/docs` — FastAPI's interactive API docs
- `/api/health` — health check / cold-start probe

## 1. Deploy

Commit the app and push the branch that contains `render.yaml`, then:

1. Open <https://dashboard.render.com>.
2. **New → Blueprint**.
3. Connect GitHub and pick `g1gowda123/chronicle`.
4. Select the branch you want to host.
5. Review the service (`chronicle-vfm`) and click **Deploy Blueprint**.

The Blueprint sets:

| Setting | Value |
| --- | --- |
| Runtime | `python` (uses `.python-version` → 3.12) |
| Build | `pip install -e ".[ui]"` |
| Start | `python run_dashboard.py --port $PORT` |
| Health check | `/api/health` |
| Plan | `free` |

Do not set `CHRONICLE_PORT`: `run_dashboard.py --port $PORT` is what makes the
app bind Render's port on `0.0.0.0`.

### Manual service instead of a Blueprint

**New → Web Service**, then use `pip install -e ".[ui]"` as the build command
and `python run_dashboard.py --port $PORT` as the start command. Add the
environment variables from `render.yaml`.

## 2. Give `/ask` a compiled vault

`data/vfm/` is gitignored, so a clean deploy has **no memory** and `/ask` will
say `no vault yet`. The text box only needs a vault (not the video, DB or
browser assets).

The default clip is `events:demo_events.jsonl`, which resolves to the job
directory `data/vfm/events-demo/`. The quickest demo setup:

```bash
# Option A: compile the demo clip locally by opening /, selecting the demo
# clip and pressing "Compile memory". That writes data/vfm/events-demo/.
# Then force-add it:
git add -f data/vfm/events-demo data/demo_events.jsonl

# Option B: reuse an existing compiled vault as the demo memory.
mkdir -p data/vfm/events-demo
cp -r data/vfm/events-synthetic/. data/vfm/events-demo/
git add -f data/vfm/events-demo
```

Commit and push. `/api/vfm/status` should now return `"status": "done"` and the
badge in `/ask` turns green.

### Other clips

`/ask?clip=events:<relative-path>` selects a specific timeline, but the events
file must also exist on the server (for example `data/real/cook.events.jsonl`).
A vault by itself is not enough for `memory.resolve()`.

### Persisting uploads (paid)

Free services have an ephemeral filesystem: vaults, imports and uploads vanish
on every deploy/restart/spin-down. To persist them, upgrade to a paid instance,
attach a disk at `/var/data`, and set `CHRONICLE_DATA_DIR=/var/data` (both lines
are commented in `render.yaml`). Upload vaults through the Render Shell.

## 3. Look into it

On the service page in the Render dashboard:

| Tab | What it shows |
| --- | --- |
| **Logs** | Live app output, request errors, tracebacks, startup banner |
| **Events** | Deploys, restarts, spin-downs, build failures |
| **Metrics** | CPU, memory, bandwidth |
| **Environment** | Env vars and secret files |
| **Disk** | Persistent disk (paid plans only) |
| **Shell** | SSH-style shell (paid plans only; unavailable on Free) |

Free services spin down after 15 minutes without traffic and take ~30–60 s to
wake. Run `curl` once before a demo to warm it up.

### From your laptop

```bash
BASE=https://chronicle-vfm.onrender.com

curl -sS "$BASE/api/health" | jq
curl -sS "$BASE/api/vfm/status" | jq
curl -sS -X POST "$BASE/api/vfm/ask" \
  -H 'content-type: application/json' \
  -d '{"question":"Where is the mug?"}' | jq

open "$BASE/ask"
```

### Troubleshooting

| Symptom | Check |
| --- | --- |
| Deploy fails at build | Python version and that `pip install -e ".[ui]"` works locally |
| Health check fails / 502 | Start command is exactly `python run_dashboard.py --port $PORT` |
| `/ask` says `backend offline` | It must be served by the app, not a static site |
| `/ask` says `no vault yet` | Ship `data/vfm/events-demo/vault.db` (see above) |
| Answers vanish after a while | Free filesystem is ephemeral; re-upload or use a paid disk |
