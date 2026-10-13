# Bind a proved native installation to the local console

The current console accepts `--activation` and `--requirements` only at trusted
terminal startup. These paths are never accepted from browser fields. This
closes a portability gap between the native setup tier's relocated activation
files and the earlier console's original-checkout defaults.

First follow `native-setup/README.md` in a fresh source extraction, supplying the
separate unchanged model/protocol/freeze and vendor prerequisites. Installation
must include the independent JVM proof. The source archive contains no model,
geoid, vendor runtime or raw satellite imagery. MIT covers original code only;
the upstream assets retain their separate terms.

From that same installed source root, use Python 3.11–3.13 and the pinned overlay:

```sh
uv sync --frozen
uv run --with scikit-learn==1.7.2 python scripts/serve_fresh_console_20261012.py \
  --activation data/native-setup/activation-1/activation.json \
  --requirements /path/to/the/supplied/Track-B.txt --port 8825
```

Use the actual new activation path after moving an installation and proving it
again. `--requirements` is the separately supplied organizer text; the console
does not guess an absent sibling research folder. `--offline` may be added only
when all Python dependencies already exist in the local uv cache. It controls
dependency resolution and does not prohibit a subsequently authorized raw run
from contacting declared data providers.

Startup checks strict bounded JSON, canonical in-root activation paths, exact
installed asset records, the pinned geoid/checkpoint/protocol and original
freeze, source bindings, the corrected DEM overlay and its saved analytic proof.
It forwards the derived model freeze, overlay, proof, geoid and organizer text
to the actual runner parser. The portable SNAP layout remains fixed to the
checked macOS arm64 preparation; no alternate runtime is silently selected.
All checked sources are bound into each new plan and checked before execution.

These checks read files and hash prerequisites. They start no JVM, load no model,
run no raw processing and make no network request. The new relocation test uses
explicitly synthetic installed records and pins in a temporary source fixture;
it is not a fresh real-satellite replay or an independent JVM proof. The earlier
completed raw case retains its exact R6 source snapshot and unchanged receipts.
The R7 source ZIP and review kit are historical, byte-preserved artifacts.

The optical adapter is v3: its default body budget is 60 MB, its configurable
maximum is 100 MB, and its deadline maximum is 600 s. The completed recorded
case declared a 90 MB cap and received 60,069,247 bytes. Its 11.10% assessed
optical coverage, failed historical-map attempts, frozen model recall and
other scientific limitations remain unchanged. This startup fix does not
establish arbitrary judge-selected-area/date success or mapping accuracy.
