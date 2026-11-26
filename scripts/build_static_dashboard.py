"""Bundle the dashboard and its data into one offline HTML file
(for demos without a running API, or for sharing a snapshot)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "dist" / "vidyutrakshak_console.html"
html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
data = json.loads((ROOT / "static" / "dashboard_data.json").read_text())
blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
html = html.replace("/*__DATA__*/null", blob, 1)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(html, encoding="utf-8")
print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
