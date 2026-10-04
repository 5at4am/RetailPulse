"""RetailPulse source package.

Every module here is runnable on its own: it takes already-clean DataFrames in and
returns DataFrames plus a metrics dict out. Nothing reads a global cache, and nothing
imports another module's internals. That is what makes `tests/` cheap and the pipeline
order in the design spec explicit rather than emergent.
"""