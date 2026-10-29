# StyleMate UI redesign and catalogue v2 — design

Status: DRAFT for sign-off · Label: KEEP · Author: Claude (Opus 5.5) with Rama · 2026-10-29
Reference mockup (THROWAWAY): `http://127.0.0.1:8811/`, source in the session scratchpad.

## 1. Intent

StyleMate is an AI personal-styling storefront. What it does that a normal shop does not:
ask in plain words and get an answer, shop in your colours, match an outfit photo,
check fit against your measurements, try pieces on a 3D avatar, and know what you
already own. The current UI buries all of that under text, duplicate controls and data
plumbing. The goal is a storefront where a shopper knows where to start, sees products
first, and meets each personal signal (size, palette, wardrobe) once, where it matters.

## 2. Decisions log

| # | Decision | Who | When |
|---|---|---|---|
| D1 | Rebuild the frontend in place. FastAPI backend stays. No build step. Not adopting Spree, Saleor or svelte-commerce. | Rama | 2026-10-29 |
| D2 | Visual direction: clean monochrome editorial, one font family. | Rama | 2026-10-29 |
| D3 | Throwaway mockup first, then this doc, then the build. Mockup approved ("looks actually good"). | Rama | 2026-10-29 |
| D4 | Replace the 1,000-row 2023 catalogue with a subset of the 44k Kaggle "Fashion Product Images" dataset in `~/myntai`. Supersedes the earlier "filter to 819" decision. | Rama | 2026-10-29 |
| D5 | About 2,500 products. | Rama | 2026-10-29 |
| D6 | Main photo self-hosted from the zip (the zip holds only the main photo). Angle shots (front/back/left/right) linked to Myntra's CDN; any angle that fails to load is hidden. Supersedes "all images self-hosted". | Rama | 2026-10-29 |
| D7 | Show simulated ratings and the generated reviews, with no per-review "synthetic" tag. Note: the ratings are also borrowed (each copied from a different real 2023 product of similar type and price band, `donor_style_id`), not this product's own. | Rama | 2026-10-29 |
| D8 | One footer line: "Demo store. Ratings and reviews are illustrative." Required before any public deploy; included locally because it is free. | Claude's call; Rama asked whether ratings were real (answered: no). Revisit at deploy. | 2026-10-29 |
| D9 | Typeface Archivo (variable width), monochrome tokens, the user's palette as the only accent colour. | Claude's call, shown in the mockup | 2026-10-29 |
| D10 | Catalogue v2 is local only for now. Render keeps the current version until a separate deploy decision. | Rama | 2026-10-29 |
| D11 | The versioned single-file convention becomes versioned folders: `frontend/StyleMate_frontend_v16/` (index.html, css/, js/). `sync_static.py` and the launcher copy and check the newest folder. Version stamp in `<meta name="sm-version">`, also returned by `/api/config`. | Claude's call | 2026-10-29 |
| D12 | Profile: measurements are the source of truth. Sizes for tops, bottoms and shoes are derived from them by the fit engine (`fit.best_size`) and stored alongside. The header chip shows the top size; the product panel preselects the size for that product's family, or the nearest available, with a note. | Claude's call | 2026-10-29 |
| D13 | Only products with ratings and generated reviews enter the catalogue (26,414 of 44,446 qualify). | Claude's call | 2026-10-29 |
| D14 | When no model answers, the answer area shows the data-built answer without the "connect a model" line. Acceptance run 8 is done with Gemini on, plus one run with it off. | Claude's call | 2026-10-29 |

## 3. Goals, non-goals, anti-goals

**Goals**
- G1: One search field, handled the same way for every query. No AI toggle. The rule-parser fallback is silent.
- G2: A clear start: one heading, one search, three entry points (Match a photo, Find my colours, Try on in 3D).
- G3: Product cards show photo, brand, name, price, MRP, % off and rating. At most one badge per card.
- G4: Size, palette and wardrobe each appear once per surface: size in the header chip and the product panel, palette dots in the header, "You own this" as the card badge.
- G5: Every differentiator keeps working: Ask, Colour Studio, Shop a Look, fit report, 3D room, wardrobe check.
- G6: Correct, complete data: explicit gender, colour, season, real per-product sizes, fit/fabric/pattern, multi-angle gallery, correct discount.
- G7: Accessible: native dialogs (focus moves in, Esc closes, focus returns), WCAG AA contrast, minimum 12px text, 44px tap targets on mobile.

**Non-goals**
- Real checkout, stock or orders.
- Real garment measurements or size charts. The data does not have them, so the fit engine keeps its reference charts.
- New 3D fidelity. The fitting room is restyled, not re-engineered.
- A React or Svelte migration.

**Anti-goals** (would count as failure even if the page "works")
- Any copy that is false: "demo price" on real prices, "no internet" when Gemini is in use, invented product facts.
- Showing a men's shopper women's products by default.
- Re-introducing a second search box, a second category bar, or developer status in the shopper's view.

## 4. Data model — catalogue v2 (lock before UI)

**Identity:** `id = "k44-" + <Myntra style id>`. Unique. Stable across rebuilds.
**Ownership:** server truth. Built offline by `build_catalogue_v2.py` from `~/myntai` into `app/data/catalogue_v2.json` (pinned). Read-only at runtime. The page never writes the catalogue; the `/api/catalogue/sync` calls are removed from the page.
**Mutability:** changed only by re-running the build script; the output is versioned by file hash.

```
Product {
  id            "k44-15970"
  styleId       15970                                  // source key
  name          productDisplayName, cleaned
  brand         brandName
  audience      men | women | unisex                   // from Myntra `gender`; Boys/Girls excluded
  cat           western | ethnic | footwear | accessories | sports   // mapped from master/sub/articleType
  slot          top | bottom | dress | outerwear | footwear | headwear | watch | bag | jewellery | eyewear | belt
  articleType   e.g. "Shirts"
  colour        baseColour (required)
  colour2       colour1 if not "NA"
  season        Summer | Winter | Fall | Spring (required)
  usage         Casual | Formal | Ethnic | Sports | Party | ...
  price         discountedPrice (INR)
  mrp           price (INR)
  discountPct   round(100 - price/mrp*100), 0 if none  // computed, never read from source
  sizes         ["39","40","42","44"]                  // real list from styleOptions
  attrs         { fit, fabric, pattern, sleeve, neck, occasion }  // subset present in source
  images        { main, angles: [url, ...] }           // main: local /img/<id>.jpg, 600x800 JPEG q75 (~60 KB);
                                                       // angles: Myntra CDN URLs, hidden on load error (D6)
  details       plain text from description (composition, fit, wash care), HTML stripped
  rating        number | null      // from ratings_44k.jsonl (simulated, see D7/D8)
  ratingCount   int
  reviews       [{stars, text}]    // from generated/reviews_v1 where present
  provenance    { source:"kaggle-fashion-product-images", ratings:"simulated", reviews:"generated" }  // kept in data, not shown per item
}
```

**Selection, about 2,500 rows:**
- Adult only: gender Men, Women or Unisex, and `ageGroup` from the style JSON is an adult group (drops about 228 kids' items labelled Men/Women).
- masterCategory in Apparel, Footwear, Accessories. Personal Care and Free Items excluded.
- Required fields: colour, season, usage and sizes; apparel also needs a fabric attribute. A local main image, plus at least one angle URL.
- Has a rating (`rating_count` > 0) and generated reviews (D13).
- Category quotas: western 34%, accessories 26%, footwear 17%, ethnic 13%, sports 10% (the current mix).
- Within each category, round-robin across articleType for variety. The sample is deterministic with a fixed seed.

**Client state (browser) — per viewer, not shared:**

| Key | Holds | Store | Survives reload |
|---|---|---|---|
| session | signed-in demo user | sessionStorage (today, broken on reload, see B4) | must: yes |
| profile | gender, size, measurements, palette | localStorage | yes |
| bag, saved | product ids | localStorage (new; in memory today) | yes |
| wardrobe | owned ids + dates | seeded demo data | yes |
| fit queue | ids in the fitting room | localStorage (exists) | yes |

## 5. Screens (as in the mockup)

- **Header:** wordmark, categories (Western, Ethnic, Footwear, Accessories, Sports), the search field (results mode only), a profile chip (initial, size, palette dots), saved, bag. On mobile: wordmark, profile, saved, bag; categories in a scrollable row; search under the header in results mode.
- **Home:** H1 "What are you dressing for?", the search field, 5 example queries, 3 entry tiles, "Picked for you" grid (shopper's gender plus unisex; palette-first once a palette is set).
- **Results:** AI answer at the top, with up to 2 product chips taken from the answer. Filters on the left: For (Men/Women/Everyone, preset from profile), Price bands, Colour swatches from the results, **Size (new)**, **Fit (new)**, **Fabric (new)**, Rated 4+, Hide things I own. Toolbar: count, palette status, sort. Grid with "Show more".
- **Product panel (side sheet):** gallery (main plus angles), brand and name, price, MRP, % off, rating; size picker from the product's own sizes with the shopper's size preselected (or nearest, with a note); Add to bag, Save; Colour (in or outside your palette); Wear it with (3 items, same audience, other slots); Fit and Fabric line; Details (composition, wash care); Reviews (stars plus up to 5 texts, "Show all").
- **Dialogs:** Match a photo, Find my colours, Fitting room (existing 3D, restyled; step labels fixed; empty-state CTAs disabled), Bag, Profile and settings (model status and API key move here).
- **Footer:** one line of provenance and disclosure (D8).

## 6. Lifecycle tables (actor × state × action)

Actor: shopper (signed-in demo user). Untested cells get listed at verification time.

**Page mode**

| State \ Action | Search submit | Category click | Logo click | Reload |
|---|---|---|---|---|
| Home | → Results (the input moves to the header, keeping its text) | → Results filtered to category, query cleared | stays Home | stays Home, still signed in |
| Results (query) | new query; filters For and Price reset, colour cleared | → category results, query cleared, answer removed | → Home, query cleared | restores query and results |
| Results (category) | → query results | switches category | → Home | restores category |

**Dialogs:** any dialog — open: focus moves to the first control; Esc or the × button: closes and focus returns to the opener; backdrop click: closes. Only one dialog at a time (opening another closes the first).

**Palette:** none → set (from Colour dialog) → shown as header dots, reorders results and home, dot badge on matching cards → "Turn off" clears it (profile keeps the last reading for "Start again").

**Bag and Saved:** add → count increments, card and panel buttons change to "In bag"/"Saved" → survives reload. Empty bag: checkout button disabled with "Your bag is empty".

## 7. Behaviour fixes carried into the build

- B1: Discount computed from price and MRP (the source discount field was rupees on 62 rows).
- B2: Remove "Demo price / no price in dataset" copy (prices are real).
- B3: Colour Studio privacy line names the actual provider: Gemini when connected, otherwise "read on this device".
- B4: Reload must not sign out. Root cause not traced yet; trace before fixing.
- B5: No stale state: the results subtitle, active tab and search text always reflect the current mode.
- B6: Search audience comes from the explicit `audience` field. The `profile.gender` default stays.
- B7: "Try on" only on slots the 3D room draws (top, bottom, dress, outerwear, footwear, headwear).
- B8: Currency is always ₹, en-IN grouping, including in the AI answer text.
- B9: Remove the page → server catalogue sync calls.

## 8. Frontend structure (no build)

Source of truth: `frontend/StyleMate_frontend_v16/` (D11), copied to `app/static/` by `sync_static.py`.
```
app/static/
  index.html          shell only, carries <meta name="sm-version">
  css/tokens.css      the only place colours, type scale and spacing are defined
  css/app.css         components, no hex literals outside tokens
  js/main.js          ES module entry
  js/api.js  state.js  router.js (mode)  card.js  results.js  product.js
  js/colours.js  look.js  bag.js  profile.js
  stylemate-room.js   existing 3D room, restyled
```
Rules: no inline `style=` attributes in templates; one `card()` used everywhere; native `<dialog>` for every overlay.

## 9. Backend changes

- Load `catalogue_v2.json` (pinned). Serve images from `/img/<id>/<angle>.jpg`.
- Search, palette, look and fit read the new fields (audience, season, sizes, attrs.fabric for stretch).
- The semantic embeddings cache rebuilds on the new hash (needs Gemini; BM25 is the fallback). Gemini was fixed on 2026-10-29: the old server process was pointing at a deleted `.venv` CA file; a restart cleared it.
- `/api/config` returns the frontend version (D11).

## 10. Acceptance criteria

1. Home at 1440×900: H1, search and tiles above the fold; first product row starts within 1.1 screens. At 390×844: header ≤ 60px, first product ≤ 560px from top, no horizontal scroll.
2. Exactly one text input for search exists in the DOM at any time.
3. A men's profile with no audience in the query returns 0 items tagged `women` in the first 24 results (test with 10 queries).
4. No card shows "Not specified", "Not provided", "Demo price" or a discount above 95%.
5. Every dialog: focus inside on open, Esc closes, focus returns (checked per dialog in Chrome).
6. Lighthouse accessibility ≥ 98, zero contrast failures, no text under 12px.
7. Reload keeps sign-in, bag, saved, palette and the current results mode.
8. All six differentiators exercised end to end in the real app, with screenshots: Ask, Colour Studio → shop in colours, Shop a Look upload → matches, fit report on a product, 3D room with 2 pieces, wardrobe badge and "Hide things I own".
9. Catalogue build is deterministic: the same inputs produce the same file hash; row count 2,500 ± 50; every row passes the required-field check.
10. Local image folder ≤ 200 MB; every product's main image loads from the backend. With Myntra's CDN blocked (offline test), the product panel still shows the main image and no broken thumbnails.
11. The launcher serves v16 after a restart and the page's version stamp matches the folder name.

## 11. What this makes impossible or invisible (negative space)

- No keyword-only search mode for users.
- Model and key status are visible only in Profile → Settings, not in the header.
- Fit, wardrobe and palette notes no longer appear on cards (badge or dot only); the detail lives in the product panel.
- The 2023 products and their real ratings and 2023 prices are gone; prices are 2011–2017 Myntra prices.
- Per-review provenance is not shown; only the footer line discloses it (if D8 is approved).
- Kids', personal care and free items are not in the store.
- Unrated products (18,032 of the 44k) are not in the store.
- The gallery's angle shots depend on Myntra's CDN; if it stops serving them, products fall back to one photo.
- v2 is not on the public site until a separate deploy decision.
- No server-side rendering or SEO; still a single-page app.

## 12. Left to the builder

Exact copy beyond what's shown in the mockup, icon set, image resize quality, the articleType → slot/cat mapping table (must be written down in the build script), and how the reviews are paginated.
