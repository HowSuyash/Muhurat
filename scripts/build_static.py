"""Build the deployable, self-contained dashboard.

The served app reads its data from `data/runs/`, which is generated output and
therefore git-ignored. A hosted deployment has no such directory, so a serverless
FastAPI build would 404 on every page.

This produces a single file that carries everything it needs:

    public/index.html   <- the whole dashboard, data embedded, no backend

The advisor still works because `scripts/export_replay.py` precomputes advice for
all 110 published reasons through the same code path as `/api/advise`. Nothing is
lost by dropping the server; it is simply a build step rather than a request.

    python -m scripts.build_static
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "app" / "static" / "dashboard.html"
DATA = ROOT / "data" / "runs" / "replay.json"
OUT = ROOT / "public" / "index.html"

#: The served page fetches its data; the static build reads it from the document.
FETCH_CALL = 'fetch("/api/replay")'

#: Same for the advisor: swap the API round-trip for a lookup in the embedded
#: table, keeping the prior-attempts back-off maths identical to app/advisor.py.
ADVISE_FETCH = '''    fetch("/api/advise?reason="+encodeURIComponent(r)+"&prior_attempts="+ADV_PRIOR)
      .then(function(x){if(!x.ok)throw new Error("HTTP "+x.status);return x.json()})
      .then(function(a){ADV=a;document.getElementById("aout").innerHTML=adviceCard(a);mountAdvisor()})
      .catch(function(e){document.getElementById("aout").innerHTML=
        '<div class="panel"><p class="sub">Could not get advice ('+esc(e.message)+').</p></div>'});'''

ADVISE_LOCAL = '''    const base=D.advice&&D.advice[r];
    if(!base){document.getElementById("aout").innerHTML=
      '<div class="panel"><p class="sub">No advice for <b>'+esc(r)+'</b>.</p></div>';return}
    const a=JSON.parse(JSON.stringify(base));
    if(ADV_PRIOR&&a.wait_hours>0){
      a.wait_hours=+(a.wait_hours*Math.pow(1.6,Math.min(ADV_PRIOR,3))).toFixed(2);
      const hh=a.wait_hours;
      a.retry_in_words=hh<1?("in about "+Math.round(hh*60)+" minutes"):
        hh<48?("in about "+(+hh.toFixed(1))+" hours"):("in about "+Math.round(hh/24)+" days");
      a.retry_at=new Date(Date.now()+hh*36e5).toISOString();
    }
    ADV=a;document.getElementById("aout").innerHTML=adviceCard(a);mountAdvisor();'''


def main() -> None:
    for f in (SRC, DATA):
        if not f.exists():
            raise SystemExit(
                f"missing {f}\nRun: python -m scripts.run_baseline && "
                "python -m scripts.evaluate && python -m scripts.export_replay"
            )

    html = SRC.read_text(encoding="utf-8")
    data = DATA.read_text(encoding="utf-8")
    payload = json.loads(data)  # fail loudly here rather than in the browser

    html = html.replace(
        '<div class="wrap" id="root"><div class="err mono">Loading…</div></div>',
        '<div class="wrap" id="root"><div class="err mono">Loading…</div></div>\n'
        '<script id="rep" type="application/json">' + data + "</script>",
    )

    if ADVISE_FETCH not in html:
        raise SystemExit("advisor fetch block not found -- dashboard.html changed shape")
    html = html.replace(ADVISE_FETCH, ADVISE_LOCAL)

    i = html.index(FETCH_CALL)
    j = html.index("</script>", i)
    html = html[:i] + 'boot(JSON.parse(document.getElementById("rep").textContent));\n' + html[j:]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")

    kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT}  ({kb:.0f} KB)")
    print(f"  {len(payload['payments'])} payments · {len(payload['arms'])} arms · "
          f"{len(payload['advice'])} advice entries embedded")
    print("  no backend required — open it directly or deploy the public/ directory")


if __name__ == "__main__":
    main()
