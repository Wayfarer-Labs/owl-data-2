# analyzers/registry.py
_ANALYZERS = []

def register(analyzer):
    _ANALYZERS.append(analyzer)
    return analyzer

def all_analyzers():
    return list(_ANALYZERS)

