# Deploying CDFuse

CDFuse is a Streamlit application: a Python process that must be *running* to serve
requests. That single fact determines which hosts will work.

> [!IMPORTANT]
> **GitHub Pages cannot host CDFuse.** Pages is a static file host — it serves HTML,
> CSS, JavaScript and images, and never executes Python. Keep the *source* on GitHub,
> run the *app* on a Python host, and link to it from your website.

| Option | Cost | Effort | Best for |
| --- | --- | --- | --- |
| [Streamlit Community Cloud](#option-1--streamlit-community-cloud-recommended) | Free | Lowest | Public academic tools — **recommended** |
| [Hugging Face Spaces](#option-2--hugging-face-spaces) | Free tier | Low | Extra visibility in the ML/data community |
| [Docker](#option-3--docker-self-hosted) | Varies | Medium | Institutional servers, full control |
| [Render / Railway / Fly.io](#option-4--generic-paas) | Free tier upward | Medium | Custom domains, always-on |

---

## Option 1 · Streamlit Community Cloud (recommended)

Free, purpose-built for exactly this, and it redeploys on every push.

### Prerequisites

- A **public** GitHub repository containing this project.
- A GitHub account (the same one).

### Steps

1. **Make sure the code is on GitHub.** The canonical repository is already
   published at <https://github.com/hrishikesh618/CDFuse>. If you are deploying a
   fork or a copy, push it first:

   ```bash
   cd CDFuse
   git init
   git add .
   git commit -m "Initial commit: CDFuse v1.0.0"
   git branch -M main
   git remote add origin https://github.com/<your-account>/CDFuse.git
   git push -u origin main
   ```

2. Go to **<https://share.streamlit.io>** and sign in with GitHub.

3. Click **Create app** → choose **Deploy a public app from GitHub**.

4. Fill in:

   | Field | Value |
   | --- | --- |
   | Repository | `hrishikesh618/CDFuse` |
   | Branch | `main` |
   | Main file path | `app.py` |
   | App URL | e.g. `cdfuse` → `https://cdfuse.streamlit.app` |

5. Click **Deploy**. The first build takes 3–8 minutes while the geospatial stack
   installs.

### What the repo already provides

- `requirements.txt` — Python dependencies.
- `packages.txt` — the apt packages (`libgdal-dev`, `libgeos-dev`, `libproj-dev`)
  that Community Cloud installs before pip.
- `.streamlit/config.toml` — headless mode, a 500 MB upload cap, the theme, and
  `showErrorDetails = false` so tracebacks are not exposed publicly.

### Resource limits

Community Cloud gives roughly **1 GB of RAM** per app. Since CDFuse loads datasets
into memory, aim for files in the tens of megabytes.

### Keeping the app awake

Community Cloud suspends an app once it has gone a while without visitors. The next
person to open it sees a **"Zzzz — this app has gone to sleep due to inactivity"**
screen with a button to wake it, which takes roughly a minute while the container
restarts.

This is deliberate platform behaviour. There is **no setting in the app or
`.streamlit/config.toml` that disables it.** Three ways to handle it:

1. **Set expectations (recommended).** Note on the page that links to the app that a
   first visit after a quiet spell may need a moment. Visitors are never blocked —
   anyone can press the wake button. In practice a tool with steady traffic rarely
   sleeps.
2. **Ping it externally.** An uptime service (UptimeRobot, cron-job.org) requesting
   the URL every few hours keeps it from idling out. It works, but it consumes free
   shared infrastructure to serve nobody, and Streamlit discourages it. Weigh that
   before doing it on a publicly attributed app.
3. **Host it somewhere that stays up.** Hugging Face Spaces is free with a more
   generous idle window; Fly.io, Render and Railway offer always-on instances for a
   few dollars a month; an institutional server costs nothing but admin time. All
   three can use the `Dockerfile` in this repository.

Self-hosted deployments (Docker, VPS, institutional server) never sleep.

### Linking it from your website

```html
<a href="https://cdfuse.streamlit.app" target="_blank" rel="noopener">
  Launch CDFuse
</a>
```

Embedding in an `<iframe>` also works, though a full-tab link gives a better
experience on mobile:

```html
<iframe src="https://cdfuse.streamlit.app/?embed=true"
        style="width:100%;height:80vh;border:0;" loading="lazy"
        title="CDFuse"></iframe>
```

---

## Option 2 · Hugging Face Spaces

1. Create a Space at <https://huggingface.co/new-space>, choosing the **Streamlit** SDK.
2. Push this repository to the Space's git remote.
3. Add a YAML header to the top of `README.md`:

   ```yaml
   ---
   title: CDFuse
   emoji: 🛰️
   colorFrom: blue
   colorTo: green
   sdk: streamlit
   sdk_version: 1.38.0
   app_file: app.py
   pinned: false
   license: mit
   ---
   ```

Spaces reads `packages.txt` and `requirements.txt` the same way Community Cloud does.

---

## Option 3 · Docker (self-hosted)

Use this for an institutional server, or anywhere you need full control.

```bash
docker build -t cdfuse .
docker run --rm -p 8501:8501 cdfuse
```

Then open <http://localhost:8501>.

To give the container more memory for larger datasets:

```bash
docker run --rm -p 8501:8501 --memory=4g cdfuse
```

### Behind a reverse proxy

Streamlit uses WebSockets, so the proxy must forward upgrade headers. nginx:

```nginx
location / {
    proxy_pass         http://127.0.0.1:8501;
    proxy_http_version 1.1;
    proxy_set_header   Upgrade $http_upgrade;
    proxy_set_header   Connection "upgrade";
    proxy_set_header   Host $host;
    proxy_set_header   X-Real-IP $remote_addr;
    proxy_set_header   X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header   X-Forwarded-Proto $scheme;
    proxy_read_timeout 86400;
}
```

Serving under a sub-path (`https://example.org/cdfuse/`) needs:

```bash
streamlit run app.py --server.baseUrlPath=cdfuse
```

---

## Option 4 · Generic PaaS

Render, Railway and Fly.io all work. Use:

- **Build command** — `pip install -r requirements.txt`
- **Start command** —
  `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true`

If the platform cannot install system packages, deploy the `Dockerfile` instead —
it already includes GDAL, GEOS and PROJ.

---

## Post-deployment checklist

- [ ] The app loads and the header renders.
- [ ] **Load demonstration data** works, and a comparison completes.
- [ ] Maps render, and the PNG / NetCDF / GeoTIFF / CSV downloads all succeed.
- [ ] Uploading a real NetCDF pair of your own works.
- [ ] Boundary upload (a zipped shapefile) draws and clips correctly.
- [ ] The layout is usable on a phone.
- [ ] `showErrorDetails = false` is set, so tracebacks stay private.

---

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `ImportError` for `geopandas`, `rasterio` or `pyproj` | Missing system libraries. Confirm `packages.txt` is at the repo root, or deploy the Dockerfile. |
| `AttributeError: 'DataArray' object has no attribute 'rio'` | `rioxarray` was not imported. `cdfuse/io.py` and `cdfuse/export.py` import it deliberately — do not "tidy away" those imports. |
| App restarts, or "Oh no" error, on large files | Out of memory. Crop or coarsen the data, or move to a host with more RAM. |
| Maps have no coastlines | Cartopy is not installed. Optional — `pip install cartopy` to enable them. |
| Boundary rejected for a missing CRS | The shapefile has no `.prj`. Include every component, or upload a GeoJSON. |
| Blank maps after enabling clipping | The boundary does not overlap the data. CDFuse reports the two extents so they can be compared. |
| "Zzzz — this app has gone to sleep" | Normal free-tier behaviour after a spell with no visitors. Anyone can click **Yes, get this app back up**; it takes about a minute. It cannot be disabled from the app or `config.toml` — see [Keeping the app awake](#keeping-the-app-awake) if that matters for your audience. |
| First visit is slow | The app was waking from sleep, or the container had just restarted. Normal on free tiers. |

---

## Updating a deployed app

```bash
git add .
git commit -m "Describe the change"
git push
```

Community Cloud and Spaces redeploy automatically. For Docker, rebuild and restart
the container.
