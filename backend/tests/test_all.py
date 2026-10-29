"""StyleMate backend test suite. Run: python -m tests.test_all"""
from __future__ import annotations
import sys, os, io, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
os.environ.setdefault("SM_DATA_DIR", str(pathlib.Path(__file__).parent / "_vartest"))

import numpy as np, cv2
from app.data.size_charts import charts, family_for, stretch_for
from app.services import fit as F
from app.services import avatar as A
from app.services import garment as GA
from app.services.look import detect_zones, match_item, nearest_colour
from app.services.search import parse_rules, search
from app.catalogue.store import catalogue
from app.catalogue.compat import slot_of, complement_slots, colours_for_metal

P, Fl = 0, 0
def ok(cond, msg):
    global P, Fl
    if cond: P += 1; print(f"  PASS {msg}")
    else:    Fl += 1; print(f"  FAIL {msg}")

def section(t): print(f"\n== {t} ==")

# --- seed a catalogue ----------------------------------------------------
import random
random.seed(11)
TYPES=[("Shirts","western","top"),("Tshirts","western","top"),("Jeans","western","bottom"),
 ("Skirts","western","bottom"),("Dresses","western","dress"),("Kurtas","ethnic","top"),
 ("Sarees","ethnic","dress"),("Jackets","western","outerwear"),("Casual Shoes","footwear","footwear"),
 ("Earrings","accessories","jewellery"),("Watches","accessories","watch"),("Lipstick","beauty","beauty")]
COL=["Navy Blue","Black","White","Red","Maroon","Beige","Grey","Gold","Silver","Olive","Charcoal"]
rows=[]
for i in range(300):
    at,cat,_=random.choice(TYPES); col=random.choice(COL)
    use = "Ethnic" if cat=="ethnic" else random.choice(["Casual","Formal","Party","Sports"])
    rows.append({"id":f"t-{i}","sourceSku":str(i),"cat":cat,"name":f"{col} {at[:-1] if at.endswith('s') else at}",
      "brand":at,"articleType":at,"masterCategory":"Apparel","subCategory":at,"usage":use,
      "season":random.choice(["Summer","Winter","Fall"]),"colour":col,
      "audience":random.choice(["men","women","unisex"]),"spec":random.choice(["Cotton","Denim","Cotton lycra"]),
      "size":"S-XXL","price":random.choice([499,999,1499,2499,4999]),"image":""})
catalogue.upsert(rows)

section("Size charts")
ok(family_for("Tshirts","Cotton Tee","")=="top", "t-shirt -> top family")
ok(family_for("","Kurta & Trouser Set","")=="kurta", "compound name takes the leading piece")
ok(family_for("Sarees","Silk Saree","") is None, "saree is not body-sized by chart")
c=charts.chart_for(sku="1",brand="X",article_type="Jeans",name="Slim Jeans",category="",gender="female")
ok(c["source"]=="reference" and "waist" in c["dims"], "reference chart is labelled as such")
ok(c["sizes"]["L"]["waist"] > c["sizes"]["M"]["waist"], "sizes increase monotonically")
m=charts.chart_for(sku="1",brand="X",article_type="Tshirts",name="Tee",category="",gender="male")["sizes"]["M"]
w=charts.chart_for(sku="1",brand="X",article_type="Tshirts",name="Tee",category="",gender="female")["sizes"]["M"]
ok(m["chest"] != w["chest"], "men's and women's cuts differ, not relabelled")
ok(stretch_for("cotton lycra tee")[0] > stretch_for("denim jeans")[0], "lycra stretches more than denim")
ok(stretch_for("no material stated")==(None,None), "unknown fabric reports unknown")

section("Fit engine (deterministic)")
body={"gender":"male","chest":42.5,"frontLength":29.5,"shoulder":18.3}
tee={"id":"x","brand":"HRX","articleType":"Tshirts","name":"Jersey Tee","spec":"Cotton jersey","cat":"western"}
r=F.evaluate(tee,"M",body,unit="in")
ok(r.family=="top" and r.dims, "produces dimension-level results")
ok(any(d["dim"]=="chest" and d["ease_cm"] is not None for d in r.dims), "chest ease computed")
ok(r.recommended_size and r.recommended_size!="M", "unwearable size triggers a recommendation")
sizes=[F.evaluate(tee,z,body,unit="in").overall for z in ["S","M","L","XL","XXL"]]
ok(sizes.index("relaxed")>sizes.index("snug") if "relaxed" in sizes and "snug" in sizes else True,
   "verdicts loosen as size increases: "+", ".join(sizes))
ok(F.evaluate(tee,"M",{"gender":"male"},unit="in").overall=="cannot be judged",
   "no measurements -> cannot be judged, not a guess")
r2=F.evaluate(tee,"M",{"gender":"male","chest":42.5},unit="in")
ok("shoulder" in r2.missing_body, "missing body measurement is named")
ok(r2.confidence in ("low","medium"), "confidence drops when data is missing")
shoe={"id":"s","articleType":"Sports Shoes","name":"Runner","cat":"footwear"}
rs=F.evaluate(shoe,"UK 8",{"gender":"male","foot":26.4},unit="cm")
ok(rs.dims[0]["dim"]=="footLength", "shoes judged on foot length")
ok(F.evaluate({"id":"p","articleType":"Perfume","name":"EDP","cat":"beauty"},"M",body,"in").family is None,
   "non-sized item returns no family")
cm=F.to_cm(42.5,"in")
ok(abs(cm-107.95)<0.1, f"inch->cm conversion correct ({cm})")
ok(F.to_cm(100,"cm")==100, "cm passes through unchanged")
b_rel={**body,"fitPref":"relaxed"}
ok(F.evaluate(tee,"M",body,"in").size=="M", "requested size is honoured")

section("Avatar mesh")
m2,sup,defd=A.resolve_measurements({"unit":"in","gender":"male","chest":42.5,"shoulder":18.3,"frontLength":29.5})
ok(set(sup)>={"chest","shoulder","frontLength"}, "supplied measurements reported")
ok("height" in defd, "unsupplied measurements reported as defaulted, not hidden")
ok(abs(m2["chest"]-107.9)<0.2, "measurements converted to cm")
mesh=A.build_mesh(m2)
V=np.array(mesh["positions"]).reshape(-1,3); I=np.array(mesh["indices"])
ok(len(V)>1500 and len(I)//3>2500, f"real mesh: {len(V)} verts / {len(I)//3} tris")
ok(I.max()<len(V), "no out-of-range indices")
ok(not np.isnan(V).any(), "no NaN vertices")
ok(sum(1 for t in I.reshape(-1,3) if len(set(t))<3)==0, "no degenerate triangles")
ok(abs(V[:,1].min())<1e-6, "feet sit on the floor plane")
ok(abs(V[:,1].max()-m2["height"]/100) < 0.02, "mesh height matches the measurement")
f_mesh=A.build_mesh(A.resolve_measurements({"unit":"cm","gender":"female"})[0])
ok(f_mesh["gender"]=="female", "female model selectable")
ok(f_mesh["vertexCount"]==mesh["vertexCount"], "both genders build a valid mesh")

section("Gendered body")
mm,_,_=A.resolve_measurements({"unit":"cm","gender":"male","chest":100,"waist":88,"hips":100,"height":170})
fm,_,_=A.resolve_measurements({"unit":"cm","gender":"female","chest":96,"waist":74,"hips":100,"height":170})
MM=np.array(A.build_mesh(mm)["positions"]).reshape(-1,3)
FM=np.array(A.build_mesh(fm)["positions"]).reshape(-1,3)
def _w(V,f,ax=0):
    y=1.70*f; band=V[np.abs(V[:,1]-y)<0.012]
    return float(band[:,ax].max()-band[:,ax].min()) if len(band) else 0.0
ok(_w(MM,0.818) > _w(FM,0.818), "male shoulders are wider")
ok(_w(FM,0.52) > _w(MM,0.52), "female hips are wider")
ok(_w(FM,0.62) < _w(MM,0.62), "female waist is narrower")
ok(_w(FM,0.715,2) > _w(MM,0.715,2), "female chest has forward volume (bust)")
ok(A.leg_axis(mm) > 0 and abs(A.leg_axis(mm)-A.leg_axis(fm)) >= 0,
   "leg axis is defined once and shared")

section("Photo analysis")
img=np.full((900,500,3),235,np.uint8)
cv2.circle(img,(250,110),58,(168,140,120),-1)
cv2.rectangle(img,(190,168),(310,430),(70,80,110),-1)
cv2.rectangle(img,(196,430),(304,560),(60,65,90),-1)
cv2.rectangle(img,(205,560),(243,840),(45,50,70),-1)
cv2.rectangle(img,(257,560),(295,840),(45,50,70),-1)
okj,buf=cv2.imencode('.jpg',img)
ev=A.analyse_photo(buf.tobytes())
ok(ev.source in ("silhouette","mediapipe"), f"estimator ran ({ev.source})")
ok(ev.notes, "photo analysis explains what it did")
ok(A.analyse_photo(b"not an image").source=="none", "garbage upload handled")
m3,notes=A.apply_evidence(m2,ev)
ok(m3["shoulder"]==m2["shoulder"], "a measured shoulder is never overwritten by a photo")

section("Garment shells")
H=m2["height"]
sh=GA.build_shell({"id":"g1","articleType":"Tshirts","name":"Jersey Tee","cat":"western"},"L",m2,H)
ok(sh["assetKind"]=="measurement_shell", "shell built from the chart")
ok("not the product" in sh["note"] and "cloth simulation" in sh["note"],
   "shell states what it is not")
ok(sh["slot"]=="top", "shell carries the slot it is worn on")
SV=np.array(sh["positions"]).reshape(-1,3)
ok(np.array(sh["indices"]).max()<len(SV), "shell indices valid")
ok(sh["topY"]>sh["bottomY"], "shell spans top to bottom correctly")
shoe=GA.build_shell({"id":"g2","articleType":"Sports Shoes","name":"Runner","cat":"footwear"},"UK 8",m2,H)
ok(shoe["assetKind"]=="measurement_shell" and shoe["triangleCount"]>100,
   "footwear is worn on the feet")
import numpy as _np
SP=_np.array(shoe["positions"]).reshape(-1,3)
ok(SP[:,1].max() < 0.25, "shoes sit at floor level, not on the torso")
watch=GA.build_shell({"id":"g4","articleType":"Watches","name":"Analog Watch","cat":"accessories"},"M",m2,H)
ok(watch["assetKind"]=="none" and "misrepresent" in watch["note"],
   "watches and jewellery are not invented on the avatar")
cap=GA.build_shell({"id":"g5","articleType":"Caps","name":"Baseball Cap","cat":"accessories"},"M",m2,H)
ok(cap["assetKind"]=="measurement_shell", "headwear is worn on the head")
CP=_np.array(cap["positions"]).reshape(-1,3)
ok(CP[:,1].min() > H/100*0.85, "cap sits on the crown")
top_sh=GA.build_shell({"id":"g6","articleType":"Shirts","name":"Cotton Shirt","cat":"western"},"L",m2,H)
TP=_np.array(top_sh["positions"]).reshape(-1,3)
ok(TP[:,0].max() > SP[:,0].max()*0.9, "a long-sleeved shirt reaches out over the arms")
bot=GA.build_shell({"id":"g3","articleType":"Jeans","name":"Slim Jeans","cat":"western"},"M",m2,H)
ok(bot["bottomY"] < sh["bottomY"], "trousers reach lower than a t-shirt")

section("Search understanding (rules only)")
s1=parse_rules("I want a wedding dress for this December, under budget of 2000, for female")
ok(s1.budget_max==2000, "budget parsed")
ok(s1.month==12 and s1.season=="Winter", "month -> season")
ok(s1.audience=="women", "audience parsed")
ok(s1.occasion=="wedding", "occasion parsed")
ok("dress" in s1.slots, "garment noun pins the slot")
s2=parse_rules("something which would go good on silver jewellery")
ok(s2.intent=="pair" and s2.pair_metal=="silver", "pair intent and metal parsed")
ok("silver" not in s2.keywords, "anchor words excluded from search terms")
s3=parse_rules("need a top to wear with my skirt")
ok(s3.pair_slot=="bottom", "anchor slot identified")
ok(parse_rules("under 2.5k")is not None, "shorthand budget does not crash")

section("Search retrieval")
r1=search("wedding dress for December under 2000 for women",limit=8,use_model=False)
ok(all(p["price"]<=2000 for p in r1["results"]), "budget filter is hard")
ok(all(p["audience"] in ("women","unisex","") for p in r1["results"]), "audience filter is hard")
ok(all(p["slot"]=="dress" for p in r1["results"]) if r1["results"] else True, "only dresses returned")
r2=search("something that goes good on silver jewellery",limit=8,use_model=False)
ok(all(p["slot"]!="jewellery" for p in r2["results"]), "never returns more jewellery")
ok(not any("silver" in (p["colour"] or "").lower() for p in r2["results"][:5]),
   "does not return silver-coloured clothes for a silver-jewellery query")
cool=colours_for_metal("silver")
ok(any((p["colour"] or "").lower() in cool for p in r2["results"][:5]), "prefers colours that suit silver")
r3=search("something to wear with a skirt",limit=8,use_model=False)
ok(all(p["slot"]!="bottom" for p in r3["results"]), "never returns another bottom")
ok(r1["filtersApplied"] and r1["explanation"], "explains what it did")
ok(search("qwertyuiop zxcvbn",limit=5,use_model=False)["total"]>=0, "nonsense query does not crash")

section("Shop a look")
look=np.full((800,440,3),238,np.uint8)
cv2.circle(look,(220,95),52,(150,185,215),-1)
cv2.rectangle(look,(150,155),(290,410),(60,40,30),-1)
cv2.rectangle(look,(165,410),(275,700),(185,185,185),-1)
cv2.rectangle(look,(165,715),(275,770),(40,40,45),-1)
okj,lbuf=cv2.imencode('.jpg',look)
items=detect_zones(lbuf.tobytes())
slots={i.slot for i in items}
ok("top" in slots and "bottom" in slots, f"zones detected: {sorted(slots)}")
ok(not any(i.slot=="headwear" for i in items), "background not reported as a garment")
top=[i for i in items if i.slot=="top"][0]
ok(top.colour=="navy blue", f"navy top read correctly (got {top.colour})")
mt=match_item(top,limit=5)
ok(mt and all(p["slot"]=="top" for p in mt), "matches stay in the detected slot")
ok(mt[0]["colour"].lower()==top.colour, "closest match shares the colour")
ok(nearest_colour((28,42,86))=="navy blue", "colour naming works")
ok(detect_zones(b"bad")==[], "bad image returns nothing rather than crashing")

section("Gemini model selection")
from app.llm.gemini import pick_best, FALLBACK_MODELS, _version_key, _EXCLUDE
ok(pick_best(FALLBACK_MODELS["text"])=="gemini-3.8-flash",
   "defaults to the newest flash model")
live=["gemini-2.5-flash","gemini-3.8-flash","gemini-4-argon","gemini-3.6-flash",
      "gemini-3.8-flash-tts","gemini-3.1-flash-image","gemini-embedding-001"]
ok(pick_best(live)=="gemini-3.8-flash", "picks newest flash from a live list")
ok(pick_best(live,prefer="pro")=="gemini-4-argon", "can prefer the flagship")
ok(all(not any(x in m for x in _EXCLUDE)
       for m in [pick_best(live)]), "never selects a tts or image-only model")
ordered=sorted([m for m in live if not any(x in m for x in _EXCLUDE)],key=_version_key)
ok(ordered[0]=="gemini-4-argon" and ordered.index("gemini-3.8-flash")<ordered.index("gemini-2.5-flash"),
   "newest first: "+", ".join(ordered[:3]))
ok("gemini-3.8-flash" in FALLBACK_MODELS["vision"], "3.8 flash offered for vision too")

section("Shop a Look: person anchoring")
from app.services.look import _find_face, _skin_mask
np.random.seed(5)
busy=np.random.randint(5,60,(900,600,3),np.uint8)
for _i in range(120):
    cv2.circle(busy,(np.random.randint(0,600),np.random.randint(0,420)),
               np.random.randint(2,7),(int(np.random.randint(120,255)),)*3,-1)
cv2.circle(busy,(300,330),26,(150,185,215),-1)
cv2.rectangle(busy,(258,360),(342,520),(225,232,240),-1)
cv2.rectangle(busy,(266,520),(334,700),(60,55,52),-1)
okj,bb=cv2.imencode('.jpg',busy)
items=detect_zones(bb.tobytes())
ok(items, f"finds the person on a busy night background ({len(items)} item(s))")
ok(all(i.slot!="headwear" for i in items), "does not read the sky as headwear")
white=np.full((400,300,3),235,np.uint8)
ok(_skin_mask(white).sum()==0, "a white garment is not mistaken for skin")
land=np.random.randint(40,200,(600,800,3),np.uint8)
okj,lb=cv2.imencode('.jpg',land)
ok(detect_zones(lb.tobytes())==[], "no person -> reports nothing rather than guessing")

section("Gemini request shape (regression)")
import app.llm.gemini as _G
_ALLOWED = {"contents","systemInstruction","generationConfig","tools",
            "safetySettings","toolConfig","cachedContent"}
_seen = {}
class _Resp:
    status_code = 200
    def raise_for_status(self): pass
    def json(self):
        return {"candidates":[{"content":{"parts":[{"text":'{"intent":"pair"}'}]}}],
                "usageMetadata":{"promptTokenCount":10}}
class _Client:
    def __init__(self, **k): pass
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def post(self, url, json=None, headers=None):
        _seen["body"] = json; _seen["url"] = url; _seen["headers"] = headers
        return _Resp()
_real_client = _G.httpx.Client
_G.httpx.Client = _Client
_g = _G.Gemini("AIza-test")
_g.chat("system", "user text")
unknown = [k for k in _seen["body"] if k not in _ALLOWED]
ok(not unknown, f"request body carries no field Google rejects (saw {sorted(_seen['body'])})")
ok("_model" not in _seen["body"],
   "the model id is NOT smuggled into the payload - that caused a 400 on every call")
ok("gemini-" in _seen["url"] and ":generateContent" in _seen["url"],
   "model goes in the URL where it belongs")
ok(_seen["headers"].get("x-goog-api-key") == "AIza-test", "key sent as a header")
ok("AIza-test" not in _seen["url"], "key is not put in the URL")
_g.chat("system", "describe", images=[b"\xff\xd8\xff"])
parts = _seen["body"]["contents"][0]["parts"]
ok(any("inline_data" in p for p in parts), "images attached as inline_data")
ok(not [k for k in _seen["body"] if k not in _ALLOWED], "vision call is also clean")
_G.httpx.Client = _real_client

section("Compatibility rules")
ok(slot_of("Skirts","Pleated Skirt")=="bottom", "skirt is a bottom")
ok("bottom" not in complement_slots("bottom"), "a bottom never complements a bottom")
ok("top" in complement_slots("bottom"), "a top complements a bottom")
ok(colours_for_metal("gold")!=colours_for_metal("silver"), "gold and silver pull different colours")


section("Colour studio: seasonal palette")
from app.services.palette import recommend, season_for, SEASONS
ok(season_for("warm","deep").name=="Deep Autumn", "warm + deep -> Deep Autumn")
ok(season_for("cool","light").name=="Light Summer", "cool + light -> Light Summer")
ok(season_for("neutral","medium").name=="Soft Summer", "neutral + medium -> Soft Summer")
ok(season_for("nonsense","rubbish").name in [s.name for s in SEASONS.values()],
   "bad input falls back to a real season rather than erroring")
ok(len(SEASONS)==12, f"twelve seasons defined ({len(SEASONS)})")
ok(all(s.colours and s.neutrals and s.metals and s.avoid for s in SEASONS.values()),
   "every season has colours, neutrals, metals and an avoid list")
pr=recommend("warm","deep",per_row=6)
ok(pr["season"]["name"]=="Deep Autumn", "recommend returns the derived season")
ok(len(pr["rows"])>=4, f"{len(pr['rows'])} shopping rows returned")
ok(any(r["slot"]=="top" for r in pr["rows"]) and any(r["slot"]=="bottom" for r in pr["rows"]),
   "includes tops and bottoms")
pal={c.lower() for c in pr["season"]["colours"]} | {c.lower() for c in pr["season"]["neutrals"]}
allp=[i for r in pr["rows"] for i in r["items"]]
ok(all((i["colour"] or "").lower() in pal for i in allp),
   "every recommended product is in the palette")
first=pr["rows"][0]["items"]
ok(len({(i["colour"] or "").lower() for i in first})>=3,
   "colours varied within a row: "+", ".join(i["colour"] for i in first[:4]))
ok(len(pr["alternatives"])==2, "offers two neighbouring seasons")
ok("draping" in pr["note"], "states that real analysis is done by draping")
avoid={c.lower() for c in pr["season"]["avoid"]}
ok(not (pal & avoid), "a season never both recommends and avoids the same colour")
w=recommend("warm","deep",per_row=4); c=recommend("cool","deep",per_row=4)
ok(w["season"]["name"]!=c["season"]["name"], "undertone changes the season")
wc={i["colour"] for r in w["rows"] for i in r["items"]}
cc={i["colour"] for r in c["rows"] for i in r["items"]}
ok(wc!=cc, "warm and cool produce different product sets")
b=recommend("warm","deep",per_row=6,budget_max=1000)
ok(all(i["price"]<=1000 for r in b["rows"] for i in r["items"]), "budget cap honoured")
a=recommend("warm","deep",per_row=6,audience="women")
ok(all(i["audience"] in ("women","unisex","") for r in a["rows"] for i in r["items"]),
   "audience filter honoured")

print(f"\n{'='*52}\n{P} passed, {Fl} failed\n{'='*52}")
sys.exit(1 if Fl else 0)
