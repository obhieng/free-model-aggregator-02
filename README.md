# Free model aggregator 02

Node standard-library entrypoint: `node main.ts`. Reads the trial brief, registers one dual-wire server, and polls until done. `node main.ts --list` prints the same quality-ordered catalog; outside a trial it prints an empty catalog.

Routing is rating descending and ID ascending, constrained by each request's own requirements (response_format, tools, image content), with the brief's manifest as fallback. JSON requests prefer super-vl; other requests prefer inkling. At most three attempts per prompt and 60 per trial. HTTP 429 excludes a model for the whole trial; 429/5xx permit failover with a truthful failure header. Auth failures stop immediately. Missing provider tokens permit one replacement completion and are never appended locally. Streaming is buffered before client output for validation and final usage translation.

## Verification status

IN-PROCESS STUB only: 18 practice requests passed, plus 429/503 recovery, cooldown, token-omission replacement, no retry on 401, deterministic replay, both model-list shapes, and catalog byte equality. No live provider quality or served score was measured. Socket HTTP tests were blocked by sandbox EPERM. Nobody was blocked by setpriv setresuid. The real grade-local command refused to launch without a provider key or configured gateway; no 401 trial was actually observed. Socket-level stub evidence is kept with the build logs, not in this repo.

No key, wallet operation, payment, or submission was performed. Scripts for the local socket stub and in-process stub are at /tmp/fma02-key2-test.mjs and /tmp/fma02-key2-inprocess.mjs in the build environment.
