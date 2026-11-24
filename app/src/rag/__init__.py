"""Retrieval over the clinic's policy corpus: chunking, embeddings and the Qdrant policy index.

Holds clinic documents only, never patient data (patient memory lives in the runtime store, keyed by
patient_ref). Must not import the tool, graph or workflow layers (enforced by tests/test_architecture.py).
"""
