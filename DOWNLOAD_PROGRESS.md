# Download Progress — Current State & Options

## What was just removed

`download_progress` was an `INTEGER` column on the `models` table that stored a 0–100 percentage. It was populated by the download handler in real-time and exposed through the REST API. The frontend read it to power a `<v-progress-linear>` bar. This entire chain has been removed.

---

## What is actually still there

### Backend — the progress pipeline

**Where progress numbers come from (`model_operation_handler.py`)**

The download handler iterates over a file manifest. For each file it registers a per-file callback (`_make_progress_handler`). The provider calls that callback with `{ "percent": 0‒100 }` as each chunk arrives.

The handler converts per-file percentage into an *overall* percentage:

```
base         = int(file_idx / total_files * 100)   # % already fully done
contribution = int(file_pct / total_files)          # this file's slice of total
overall_pct  = base + contribution                  # 0–100 across all files
```

Example — 3 files, file 2 is at 60%:
- base = 1/3 × 100 = 33
- contribution = 60/3 = 20
- overall = 53%

The handler still:
1. **Logs** every 10% milestone via `logger.info("[download] PROGRESS …")` — visible in Docker container stdout.
2. **Writes job progress** into the `jobs` table (`progress_percent`, `current_step`, `total_steps`, `message`) via `JobRepository.update_progress()` — throttled to 1% increments.
3. **Publishes a `job.progress` SSE event** to every connected client via the event bus, with payload:
   ```json
   {
     "model_id": "org/model",
     "progress_percent": 53,
     "file": "model.safetensors",
     "file_index": 2,
     "total_files": 3
   }
   ```

On completion it publishes `model.downloaded`. On error, `model.download_failed`.

**SSE stream (`/sse/observability?subscribe=job,model`)**

All events flow through `sse_hub.broadcast_typed()`. The frontend connects once per download session and receives all `job.*` and `model.*` events as plain `data:` frames.

### Frontend — what's connected

**Store (`stores/models.ts`)**

- Opens the SSE connection when a download starts (`connectSSE()`).
- Handles `job.progress`: receives `model_id` from the payload and flips the in-registry model to `status: 'downloading'` (previously also updated `download_progress`; that part was removed).
- Handles `model.downloaded`: shows success notification and refetches the registry.
- Handles `model.download_failed`: shows error notification and refetches.

**View (`views/ModelsView.vue`)**

While `isModelDownloading(model_id)` is true, shows:
- An **indeterminate** `<v-progress-linear>` (no percentage, just animation).
- A disabled "Downloading..." button.

The `job.progress` events still arrive in the frontend — `progress_percent` and `file` are in the payload — but nothing reads them to display a number right now.

---

## Why the progress bar showed nothing before

The old polling-based flow updated `download_progress` on the model object server-side and the FE polled. When polling was replaced with SSE events, the `job.progress` handler in the store correctly updated `download_progress` in the local registry (not DB). But the `v-progress-linear` was bound to `model.download_progress`, which defaults to 0 on the model object fetched from the server — so it always showed 0% instead of the live value.

---

## Options to display progress in the frontend

### Option A — Use the SSE payload directly (recommended, no BE changes)

The `job.progress` event already carries `progress_percent`, `file`, `file_index`, and `total_files`. The frontend can maintain a local progress map keyed by `model_id`.

**Store changes only:**
```ts
// in store
const downloadProgress = ref<Map<string, { pct: number; file: string }>>(new Map())

// in _handleSSEEvent, job.progress branch:
const pct = payload.progress_percent as number
const file = payload.file as string
downloadProgress.value = new Map(downloadProgress.value).set(modelId, { pct, file })
```

**View:**
```html
<v-progress-linear
  v-if="modelsStore.isModelDownloading(model.model_id)"
  :model-value="modelsStore.downloadProgress.get(model.model_id)?.pct ?? 0"
  height="8" color="blue" rounded
>
  <strong class="text-caption">
    {{ modelsStore.downloadProgress.get(model.model_id)?.pct ?? 0 }}%
    — {{ modelsStore.downloadProgress.get(model.model_id)?.file ?? '' }}
  </strong>
</v-progress-linear>
```

**Pros:** No backend changes. Progress resets to 0 if page is refreshed (the SSE resumes and catches up quickly).
**Cons:** Progress is lost on page refresh until the next `job.progress` event arrives.

---

### Option B — Persist progress in a `job_progress` reactive store + REST fallback

Add a `GET /jobs/{job_id}/progress` endpoint that returns the current `progress_percent` from the `jobs` table (already written there). When the FE connects SSE mid-download it can fetch the current progress immediately instead of waiting for the next event.

**Backend:** One new endpoint, no schema changes.
**Frontend:** Call `GET /jobs/{job_id}` on reconnect to seed the initial percentage.

**Pros:** Survives page refresh cleanly.
**Cons:** Requires knowing the `job_id` in the frontend (the download API already returns it — store it alongside `isDownloading`).

---

### Option C — Structured container log tailing

The handler already logs:
```
[download] PROGRESS job=<uuid> | 2/3 'model.safetensors': 40%  (overall ~46%)
[download] FILE     job=<uuid> | 2/3 'model.safetensors' (3700.0 MB)
[download] DONE     job=<uuid> | 2/3 'model.safetensors' complete
```

**To see progress in container logs without any code changes:**
```bash
docker compose logs -f backend | grep "\[download\]"
```

**To add a richer live view**, expose a `GET /jobs/{job_id}/logs` SSE endpoint that streams filtered log lines from a `job_events` or an in-memory ring buffer. The frontend subscribes when a download starts and renders the raw log text in a `<pre>` block or a terminal-style component.

**Pros:** Zero client noise in the UI; developers get full detail in the terminal.
**Cons:** Requires a log-streaming endpoint and a FE terminal component if you want it in the UI.

---

### Recommendation

Start with **Option A** — it requires only frontend changes, uses data already in the SSE stream, and gives you percentage + filename in the progress bar. Add **Option B**'s REST fallback afterwards if page-refresh survivability matters. **Option C** is always available for free right now via `docker compose logs`.
