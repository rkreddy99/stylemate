# StyleMate backend — AI search, Shop a Look, and the 3D fitting room

Python application layer for the StyleMate storefront. FastAPI does the
thinking, measurement maths and geometry; the browser does the rendering.

---

## 1. Quick start (Windows)

1. Install **Python 3.11 or 3.12** from python.org. Tick *Add Python to PATH*.
2. Double-click **`run.bat`**.
   - First run creates `.venv\` and installs dependencies. This happens **once**;
     later runs reuse the same environment and start in a couple of seconds.
3. The browser opens at **http://localhost:8000**. Sign in with `test` / `test`.

macOS / Linux: `./run.sh`

Manual start, if you prefer:

```bat
.venv\Scripts\python -m uvicorn app.main:app --port 8000
```

API docs at http://localhost:8000/docs

---

## 2. Choosing the AI model

Open **Agent model** in the feature bar. Three options:

| Setting | What it does |
|---|---|
| **Auto** (default) | Gemini when an API key is saved, local Ollama otherwise |
| **Google Gemini** | Always Gemini |
| **Local Ollama** | Always the local model |

### Gemini
Paste your API key into the panel and press save. That is the whole setup —
the backend immediately asks Google what that key can run and **selects the
newest usable model for you**, for both search and photo recognition. On a
current key that is `gemini-3.8-flash`.

The key goes to **your own backend**, is held in memory, and is never written
to disk, never logged, and never returned to the page.

The dropdowns are populated **live** from your key and sorted newest-first, so
new Google releases appear without a code change. Models that cannot do chat —
TTS, live-audio, image-generation, transcription — are filtered out. If the key
cannot be reached, this known-good list is offered instead:

`gemini-4-argon`, `gemini-3.8-flash`, `gemini-3.6-flash`,
`gemini-3.5-flash-lite`, `gemini-3.1-pro-preview`, `gemini-3.1-flash-lite`,
`gemini-3-flash-preview`, `gemini-2.5-pro`, `gemini-2.5-flash`,
`gemini-2.5-flash-lite`, plus `gemini-embedding-2-preview` and
`gemini-embedding-001`.

Gemini models are multimodal, so the same model does both the search
understanding and the Shop-a-Look recognition.

To set it outside the UI, use `GEMINI_API_KEY` (see `.env.example`).

### Ollama
```
ollama serve
ollama pull qwen2.5:7b        # search understanding
ollama pull qwen2.5vl:7b      # Shop a Look recognition
```
The backend talks to Ollama's **native** `/api/chat`, not the OpenAI-compatible
`/v1` route — that route ignores `num_ctx` and silently truncates at 4096
tokens, which would quietly cut the catalogue out of the prompt.

**Nothing here is required.** Every feature degrades to a deterministic path
and says so on screen when no model is available.

---

## 3. What runs locally and what needs something extra

### Works offline, no model, no key, no downloads
- Full catalogue search with budget, gender, occasion, season, colour and slot
  reasoning, including *"goes with silver jewellery"* and *"to wear with a skirt"*
- The whole fit engine — every number in a fit report
- Avatar generation from measurements, and the 3D viewer
- Garment measurement shells
- Shop a Look colour-and-placement detection and catalogue matching
- Photo analysis via the OpenCV silhouette estimator

### Needs a model (Gemini key, or Ollama running)
- **Sentence understanding in search.** Without it, a rule parser handles
  budgets, months, genders, occasions, metals and pair phrases — which covers
  most real queries. The model adds the awkward ones.
- **Garment recognition in Shop a Look.** Without it, the fallback reads the
  dominant colour of each part of the figure. It knows *where* an item sits,
  not *what kind of garment* it is.

### Not present, and not faked
- **Real 3D garment meshes.** The dataset has photographs and text, nothing
  else. A flat photo cannot be turned into an accurate garment pattern, so the
  app builds a *measurement shell* instead and labels it as one everywhere.
  To use real assets: drop `.glb` / `.gltf` files into `var/garment_assets/`
  named `<sku>.glb`, or map them in `var/garment_assets/garment_assets.json`.
  The API then returns `assetKind: "supplier_mesh"` and the viewer loads the
  real geometry instead.
- **Cloth simulation.** No solver, no collision, no gravity. The integration
  point is `app/services/garment.py` → `GarmentAssetResolver`.
- **SMPL-X or any statistical body model.** Those weights are licence-gated and
  must be obtained from their authors. The avatar is a parametric
  anthropometric mesh. `app/services/avatar.py` → `ESTIMATORS` is the adapter
  list; a reconstruction engine plugs in as one class with
  `.estimate(image) -> BodyEvidence`.
- **MediaPipe pose.** Implemented as the preferred photo estimator and selected
  automatically when importable. It could **not** be exercised in the build
  environment (the Python 3.13 wheel there ships only the `tasks` namespace
  with no bundled model), so treat that path as untested until it runs on your
  machine. `pip install -r requirements-optional.txt` on Python 3.11/3.12.
- **Supplier size charts.** The charts shipped are a generic Indian ready-to-wear
  reference, labelled `"reference"` in every response. Real charts go in
  `app/data/supplier_charts.json`:
  ```json
  {"charts":[{"supplier":"Roadster","unit":"in","skus":["hf-1234"],
    "brandArticleKey":"Roadster|Tshirts",
    "sizes":{"M":{"chest":40,"shoulder":17.5,"length":28,"sleeve":8.5}}}]}
  ```
  Anything matched there is reported as `"supplier"` at higher confidence.

---

## 4. How each feature works

### The status light
The left rail shows a live light, refreshed every 20 seconds:

| Light | Meaning |
|---|---|
| **Green** | Key accepted, model answering. Smart search and photo recognition are on. |
| **Red** | A key is saved but Google rejected it, or the backend is not running. The reason is printed. |
| **Amber** | No model connected. Everything still works from rules; the panel says what you are missing. |

It also lists, line by line, whether smart search, photo recognition and the
fit engine are each live — so you can see at a glance what is actually running.

### Ask StyleMate anything
Two stages, deliberately separated.

**Understanding** (model, `app/services/search.py`): the sentence becomes a
`QuerySpec` — intent, audience, occasion, month, budget, colours, slots, and
the *pair anchor* (the thing you already own).

**Retrieval** (deterministic): BM25 over product text, plus colour affinity,
occasion→usage, month→season and slot rules from `app/catalogue/compat.py`.

A model is never asked whether a product is under ₹2,000. It is asked what the
sentence means; the rules do the rest. That is why the budget is always exact.

The agent also **writes an answer**, not just a filtered grid: two to four
sentences naming at most two of the products it actually returned, with the
reasoning. It can only mention products from the result set and cannot quote a
price that is not in the data. With no model connected it still writes one from
the data and labels it "no model connected".

Relational queries are the interesting case. *"Something that goes good on
silver jewellery"* contains no searchable keyword:
- `silver jewellery` is recognised as something you **own**, so jewellery is
  excluded from the results entirely;
- the anchor's own words are stripped from the search terms, so you do not get
  silver-coloured clothes;
- `silver` maps to the colours that read well against it — navy, charcoal,
  white, grey, black, blue.

*"Something to wear with a skirt"* never returns another bottom. So does
*"what will suit me better of a blue jeans for casual date"* — the shape
"what suits me with X" is recognised as a pair query, X is trimmed of its
trailing qualifiers, and the answer is shirts, not more jeans.

When the request does not name who it is for, the search shops for **you**,
using the gender saved in My Size.

### Colour Studio
The camera reads an **undertone** (warm / cool / neutral) and a **depth**
(light / medium / deep) off your face. Those two axes place you in the
twelve-season system stylists use — warm + deep is Deep Autumn, cool + light is
Light Summer, and so on.

The studio then does the part that actually matters: it shops the palette for
you. Tops, trousers, dresses, layers, footwear, jewellery and bags are each
returned **in your colours**, spread across the palette rather than eight items
in the same shade. It also names your neutrals, your metals, and what to go
easy on, and offers the two neighbouring seasons if the reading feels off.

It also re-orders **what you are already looking at**. The intended flow is:
search for something in Ask StyleMate → get results → open the Colour Studio →
read your tone → press *Show only these colours*, and those same results come
back ordered by what suits you, your palette first. Nothing is thrown away;
items outside the palette drop to the back.

Seasonal analysis is normally done by draping fabric in daylight. Two numbers
from a webcam frame put you in the right family, not necessarily the exact
sub-season — the panel says so, and you can switch season by hand.

Palettes follow standard seasonal analysis, cross-checked against the
ajeworld.com reference "What are my colours".

### Shop a Look
A vision model lists every item with slot, colour, pattern and material.
Each detected item then becomes a slot + colour constraint and the catalogue is
ranked against it by the same deterministic retrieval. Matches are *similar
StyleMate products*, not the same product, and no brand is ever identified from
a photograph.

Each detected item also carries whether the garment is **menswear, womenswear
or unisex**, judged from the cut rather than from the person, and that decides
which part of the catalogue is searched. Your own saved gender is the default
when the model does not say. A man's photo no longer comes back with women's
dresses.

Without a vision model the fallback **anchors on the person's face** — found by
skin colour and connected components, since this OpenCV build ships no face
cascade — then reads each anatomical band from there. If it cannot find a
person it returns nothing rather than guessing. An earlier version sampled
fixed bands of the whole frame and reported a night sky as "turquoise
headwear"; it now also rejects any band that is mostly background, and will not
mistake a white shirt for skin.

### Fitting room
1. **Add to Fitting Room** on any product card.
2. The avatar **builds itself** — on opening the room, after a photo upload,
   when you switch Male/Female, and whenever you add or remove a piece. There
   is no "now press Rebuild" step.
3. Garments are **worn**, not floated: a shirt gets sleeves down the arms at
   the chart's sleeve length and a shoulder line at the chart's shoulder
   width; trousers get two legs down to the chart's inseam; shoes go on the
   feet; a cap goes on the crown. Each dimension traces back to a number in
   the size chart. Jewellery, bags and watches are deliberately **not** drawn —
   there is no 3D model of them and inventing one would misrepresent the
   product.
4. **Fit analysis** per garment: ease per dimension, a verdict, a recommended
   size, what could not be checked, and why confidence is what it is.
5. **Add to Cart** once you have reviewed it.

What the photo is used for: proportion checking (shoulder-to-hip width) and
skin tone. It is **not** used to infer body shape — a single photograph carries
no depth. Measurements you have not entered use a population default and are
listed under `defaulted` in the response and in the on-screen status.

The fit engine is pure arithmetic. Ease = garment measurement − body
measurement at the same landmark, judged against bands that differ by garment
family and landmark, with fabric stretch allowed for when the material is
stated. No LLM touches a number.

---

## 5. Privacy

- Uploads limited to JPEG/PNG/WebP, 12 MB.
- Body photos are held **in process memory only**, keyed by a random id. They
  are never written to disk and never served from any URL.
- **Forget my photo** removes it immediately (`POST /api/avatar/forget`).
- The Gemini key lives on the backend. No endpoint returns it; responses only
  ever state whether a key is present.
- Nothing is sent anywhere except the provider you selected.

---

## 6. API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | catalogue size, provider state, per-feature readiness |
| GET/POST | `/api/settings/llm` | read / change provider, key, models |
| GET | `/api/settings/gemini/models` | live model list for your key |
| POST | `/api/catalogue/sync` | browser pushes its catalogue |
| GET | `/api/catalogue/stats` | counts and facets |
| POST | `/api/search` | AI search |
| POST | `/api/look` | Shop a Look (multipart photo) |
| POST | `/api/palette` | season + products in your colours |
| GET | `/api/palette/seasons` | all twelve seasons |
| POST | `/api/fit` | one fit report |
| POST | `/api/fit/batch` | many fit reports |
| POST | `/api/avatar/photo` | analyse a body photo, returns a `photoId` |
| POST | `/api/avatar/forget` | delete that photo |
| POST | `/api/avatar` | body mesh |
| POST | `/api/fittingroom` | mesh + garment shells + fit reports in one call |

---

## 7. Layout

```
backend/
  run.bat / run.sh          launchers
  requirements.txt          what the app needs
  requirements-optional.txt mediapipe, trimesh
  .env.example
  app/
    main.py                 FastAPI routes
    config.py               settings, all env-overridable
    models.py               request schemas
    llm/
      router.py             provider selection
      ollama.py             native /api/chat client
      gemini.py             Gemini client + model lists
    catalogue/
      store.py              SQLite + BM25 index
      compat.py             styling rules: metals, slots, occasions, seasons
    services/
      search.py             understanding + retrieval
      palette.py            twelve-season analysis + palette shopping
      look.py               vision + zone fallback + matching
      fit.py                the fit engine
      avatar.py             parametric body mesh, photo estimators
      garment.py            measurement shells, real-asset resolver
    data/
      size_charts.py        garment charts, fabric stretch
    static/                 index.html, stylemate-room.js, three.js
  tests/test_all.py
  var/                      database, uploads, garment assets (created on run)
```

---

## 8. Test report — what was actually validated

`python -m tests.test_all` → **116 checks, all passing.**
Browser suites (Playwright, Chromium with WebGL) → **48 checks, all passing.**

Validated by running it:

- **Size charts** — family resolution including compound names
  ("Kurta & Trouser Set" → kurta), monotonic sizing, men's and women's cuts
  genuinely differing, fabric blends resolving to the stretchiest fibre,
  unknown fabric reported as unknown.
- **Fit engine** — ease computed per dimension; verdicts loosening as size
  increases; an unwearable size always triggering a different recommendation;
  missing measurements named rather than filled in; confidence dropping when
  data is missing; inch↔cm conversion; shoes judged on foot length; non-sized
  items returning no family.
- **Avatar** — valid closed mesh, no NaNs, no degenerate triangles, no
  out-of-range indices, feet on the floor plane, mesh height matching the
  measurement. **Gendered silhouette measured, not assumed**: male shoulders
  wider, female hips wider, female waist narrower, female chest carrying
  forward volume.
- **Garments** — a shirt reaching out over the arms, trousers enclosing the
  legs at knee and ankle, shoes at floor level rather than on the torso, a cap
  on the crown, and watches producing nothing at all.
- **Gemini** — newest-flash selection from a live list, flagship selection on
  request, TTS and image-only models filtered out, newest-first ordering.
- **Gemini request shape** — a regression test asserts the request body carries
  no field Google rejects. An earlier build put the model id in the payload as
  `_model`, and Google answered **400 Invalid JSON payload / Unknown name
  "_model"** to every single call, so the vision and search models never ran at
  all. The test now fails if any unknown top-level field reappears, and checks
  the key travels as a header rather than in the URL.
- **Status light** — amber with no model, red on a rejected key, red when the
  backend is unreachable, each with the reason shown.
- **Smart search** — "what will suit me better of a blue jeans for casual date"
  understood as a pair query, returning tops and no jeans, with a written
  answer that references what the shopper already owns.
- **Shop a Look gender** — a menswear reading returns only men's and unisex
  products; a male shopper with no model reading gets no womenswear.
- **Photo analysis** — silhouette estimator on a synthetic figure; implausible
  ratios rejected with a stated reason; a saved shoulder measurement never
  overwritten by photo evidence; corrupt uploads handled.
- **Garment shells** — valid indices, correct vertical span, trousers reaching
  lower than a t-shirt, footwear correctly producing nothing.
- **Search** — budget/audience filters exact; "wedding dress" returning only
  dresses; the silver-jewellery query returning no jewellery **and** no
  silver-coloured clothes; the skirt query returning no bottoms; nonsense
  queries not crashing.
- **Shop a Look** — zone detection on a synthetic figure finding top, bottom
  and footwear, naming navy correctly, not reporting the background as a
  garment, matches staying in the detected slot.
- **API** — every endpoint over HTTP; 415 on a wrong file type; a bad Gemini
  key degrading to rule parsing with a 200 rather than a 500; the key never
  appearing in any response.
- **Colour Studio** — all twelve seasons defined with colours, neutrals,
  metals and an avoid list; the mapping from undertone and depth; every
  recommended product actually sitting in the palette; colours varied within a
  row; a season never both recommending and avoiding the same colour; warm and
  cool producing different product sets; budget and audience filters.
- **Browser** — the avatar building itself on open, on gender switch and on
  adding a piece, each verified by **reading pixels back off the WebGL canvas**
  rather than trusting the data (an earlier version passed every assertion
  while rendering a blank stage); garments appearing on the model; the Colour
  Studio returning Deep Autumn for warm+deep with 56 products across 7
  categories and switching to Light Summer for cool+light; Shop a Look
  detecting three garment zones and proposing 18 products; search honouring
  budget, gender and garment type. No JavaScript errors.

**Not tested, and not claimed:**
- Gemini against a real API key — outbound access to
  `generativelanguage.googleapis.com` was blocked in the build environment.
  The request/response shapes follow the documented API and every failure path
  (no key, bad key, 429, 404, proxy refusal, timeout) is handled and was
  exercised with a fake key.
- Ollama against a real model — not installed in the build environment. The
  failure paths were exercised; the success path was not.
- MediaPipe pose — see §3.
- Real supplier GLB assets — none exist to test with; the resolver and the
  `supplier_mesh` branch are implemented but unexercised.
- Cloth simulation — not implemented, by design.
