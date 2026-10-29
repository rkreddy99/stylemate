# Hosting StyleMate on GitHub + Render

The `stylemate/` folder is the whole app and becomes the GitHub repository.
Render builds it from GitHub and runs it at a public `https://….onrender.com`
address. The Gemini key is entered in Render only: it is never in the repo,
never sent to the browser, and visitors cannot read it.

## 1. Put the code on GitHub (once)

Create an **empty** repository on github.com (no README), e.g. `stylemate`.
Private is fine; Render can read private repos. Then in Terminal:

```bash
cd ~/Downloads/agentcicd2c/stylemate
git init
git add .
git status          # check: no backend/.env, no .venv, no var/ in the list
git commit -m "StyleMate"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/stylemate.git
git push -u origin main
```

`.gitignore` keeps out `backend/.env` (your key), `.venv`, `backend/var/`
(the local database and caches) and log files.

## 2. Create the service on Render (once)

1. render.com → sign in with GitHub → **New → Blueprint**.
2. Pick the `stylemate` repository. Render reads `render.yaml`.
3. It asks for **GEMINI_API_KEY**: paste your key → **Apply**.
4. The first build takes a few minutes. The address is on the service page,
   e.g. `https://stylemate-xxxx.onrender.com`. Anyone can open it.

Check it is working:

| Open | You should see |
|---|---|
| `/api/ping` | `{"ok":true}` |
| `/api/config` | `"public":true`, `"geminiKeyPresent":true` |
| `/` | the storefront; sign in with `test` / `test` |

## 3. Updating the site

Commit and push. Every push to `main` redeploys:

```bash
cd ~/Downloads/agentcicd2c/stylemate
git add .
git commit -m "v4: ..."
git push
```

The build runs `backend/sync_static.py`, which serves the **highest-numbered**
`frontend/StyleMate_frontend_*_gemini_vN.html`, exactly as
`start_stylemate.command` does locally.

## What is different when hosted

`render.yaml` sets `SM_PUBLIC=1`, which switches on `backend/app/guard.py`:

- **AI limits.** 15 AI calls per visitor per minute and 600 for the whole site
  per day (UTC). Past that, visitors get a polite message and the AI features
  pause; browsing and keyword search still work. Change `SM_AI_PER_MINUTE` /
  `SM_AI_PER_DAY` under the service's **Environment** tab.
- **Model settings are locked.** The in-page "Agent model" panel cannot change
  the key, provider or model for everyone. Change them in **Environment**
  (`SM_GEMINI_TEXT_MODEL`, `SM_GEMINI_VISION_MODEL`, `GEMINI_API_KEY`).
- **The relay only uses your configured models.** A visitor cannot request a
  pricier one on your key.
- **Audit log.** One line per model call appears in the service's **Logs** tab
  (`[audit] shop-a-look call status=ok model=… prompt_tokens=…`). Nothing is
  written to disk.
- **No cross-site API calls.** Other websites cannot call your API from their
  visitors' browsers.

Things to know:

- **Free plan sleeps** after ~15 minutes without visitors. The next visit takes
  about a minute to wake it. The paid Starter plan stays awake.
- **The disk is wiped on every deploy and restart.** That is fine here: the
  catalogue (`backend/app/data/myntra_catalogue.json`) and its search index
  (`backend/app/data/emb-*.npz`) ship with the code, and the database is
  rebuilt from them at startup.
- **Sign-in is a demo.** `test` / `test` is in the page itself; it is not a
  real account system.
- **Set a spending cap on the key** in Google AI Studio / Google Cloud as a
  second line of defence behind the limits above.
- **Product photos** load from Myntra's image servers, as they do locally.

## Files involved

| File | Purpose |
|---|---|
| `render.yaml` | the Render service: build, start, health check, environment |
| `.python-version` | Python 3.12 (Render's own default is newer) |
| `.gitignore` | keeps the key, local environments and logs out of GitHub |
| `backend/sync_static.py` | build step: puts the newest page and 3D assets in `app/static` |
| `backend/app/guard.py` | public-mode limits and the settings lock |
| `backend/app/data/emb-*.npz` | precomputed search index, so cold starts need no re-embedding |
