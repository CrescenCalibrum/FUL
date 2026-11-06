from __future__ import annotations


def validate_history_request(history, forget_client_ids):
    if not history:
        raise ValueError("Empty training history")
    requested = set(map(str, forget_client_ids))
    for item in history:
        if "clients" in item:
            continue
        if "forget_client_ids" not in item:
            raise ValueError("Compact history lacks forget-set provenance; rerun training")
        if set(map(str, item["forget_client_ids"])) != requested:
            raise ValueError("Compact history is bound to a different forget set")
