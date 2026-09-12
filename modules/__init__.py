# This file makes "modules" a Python package.
# Each module (detection, drift, vessel_analysis, risk_analysis, alerts)
# lives in its own subfolder and exposes its own public interface via
# its own __init__.py, so modules never need to reach into each other's
# internals.
