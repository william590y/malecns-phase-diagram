"""Verification driver (2026-11-09): regenerate the four redraw_fss part-B figures after the readability patches in
pd3_fss_analyze.figures() WITHOUT rewriting results/pd3/redraw_fss/fss_analysis.json (same seeded bootstrap, NBOOT 2000)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pd3_fss_analyze as m
R = m.main(); S = m.summary(R); m.figures(R, S)
print("figures regenerated", flush=True)
