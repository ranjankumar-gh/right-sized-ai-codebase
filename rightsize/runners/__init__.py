"""Runner implementations: one module per real execution path.

Each runner wraps a runtime behind the bench.Runner protocol so the harness
can treat them interchangeably. Imports of heavy dependencies are deferred
into the methods that need them, so importing this package never requires a
GPU stack to be installed.
"""
