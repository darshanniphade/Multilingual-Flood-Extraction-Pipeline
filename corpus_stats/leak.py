import json, re
from collections import Counter
from pathlib import Path
ROOT=Path(r"C:\darsh\pipeline")
YEARS=["2021","2022","2023"]
CANON=set("""country state province district city village location river stream lake dam water_body
event_date start_date end_date cause trigger rainfall_mm water_depth water_level water_height
water_unit flood_duration duration_unit affected_people evacuated displaced deaths injured missing
affected_people_type evacuated_type displaced_type deaths_type injured_type missing_type
houses_damaged houses_destroyed roads_damaged bridges_damaged schools_damaged hospitals_damaged
crop_damage crop_damage_unit livestock_loss livestock_loss_type economic_loss
economic_loss_currency response_agencies summary""".split())
LEAK={"affected_people":(15000,450),"evacuated":(600,),"deaths":(14,),"injured":(82,),"missing":(5,),"livestock_loss":(350,)}
PH=re.compile(r"^(not\s+\w+|none|n/?a|unknown|unspecified|null)",re.I)
def num(v):
    try: return float(str(v).replace(",",""))
    except: return None
def real(v):
    if v in (None,"",[],{}) or isinstance(v,bool): return False
    if isinstance(v,(int,float)) and v==0: return False
    if isinstance(v,str) and (not v.strip() or PH.match(v.strip())): return False
    return True
def evs(ex):
    e=ex.get("events") or ([ex["event_details"]] if isinstance(ex.get("event_details"),dict) and ex["event_details"] else [])
    return [x for x in e if isinstance(x,dict)]
print(f"{'year':6}{'events':>9}{'>=1 leak':>11}{'>=2 leak':>11}{'>=3 leak':>11}")
best_clean={}
for y in YEARS:
    seen=set(); n=0; c=Counter(); per=Counter(); best=[]
    for p in sorted((ROOT/"data"/"extracted"/y).glob(f"{y}_??.jsonl")):
        for line in open(p,encoding="utf-8"):
            line=line.strip()
            if not line: continue
            try: r=json.loads(line)
            except: continue
            k=(r.get("month"),r.get("article_id"))
            if k in seen: continue
            seen.add(k)
            ex=r.get("extraction") or {}
            if not ex.get("contains_flood_event"): continue
            ver=bool(ex.get("is_verifiable_flood"))
            mon=(r.get("month") or "").replace("_","-")
            inm=any(str(d).startswith(mon) for d in (ex.get("flood_dates") or []))
            for ev in evs(ex):
                n+=1
                hits=0
                for f,vals in LEAK.items():
                    x=num(ev.get(f))
                    if x is not None and x in vals:
                        hits+=1; per[f]+=1
                c[min(hits,3)]+=1
                if ver and inm and hits==0:
                    clean={kk:vv for kk,vv in ev.items() if real(vv)}
                    nc=sum(1 for kk in clean if kk in CANON)
                    best.append((nc,len(clean),r.get("article_id"),r.get("month"),r.get("translated_title",""),ex,clean))
        best.sort(key=lambda t:(-t[0],-t[1])); del best[4:]
    g1=n-c[0]
    print(f"{y:6}{n:>9,}{g1:>11,}{c[2]+c[3]:>11,}{c[3]:>11,}   ({100*g1/n:.1f}% / {100*(c[2]+c[3])/n:.1f}% / {100*c[3]/n:.1f}%)")
    print("      per-field:", ", ".join(f"{f}={v:,}" for f,v in per.most_common()))
    best_clean[y]=best
json.dump({y:[{"canon":b[0],"total":b[1],"id":b[2],"month":b[3],"title":b[4],"gate":{"dates":b[5].get("flood_dates"),"locs":b[5].get("flooded_locations")},"event":b[6]} for b in v] for y,v in best_clean.items()},
          open(r"C:\Users\DNK\AppData\Local\Temp\claude\c--darsh-pipeline\05957676-323a-4620-a828-22c520f766fe\scratchpad\clean_best.json","w",encoding="utf-8"),indent=1,ensure_ascii=False)
print("wrote clean_best.json")
