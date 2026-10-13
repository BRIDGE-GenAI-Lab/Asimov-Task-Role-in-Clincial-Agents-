"""Helpers that check whether the provider and model reported by an API call match the pinned endpoint: split_endpoint separates provider and permaslug, normalise strips a name to lower-case letters and digits, and provider_ok tests a returned provider against the pinned list. It reads and writes nothing."""
import re


def split_endpoint(endpoint_name):
    provider, sep, permaslug = (endpoint_name or "").partition(" | ")
    return (provider, permaslug) if sep else (None, None)


def normalise(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


def provider_ok(returned, pinned, endpoint_provider=None):
    got = normalise(returned)
    if endpoint_provider and got == normalise(endpoint_provider):
        return True
    return any(got.startswith(normalise(tag.split("/", 1)[0])) for tag in pinned)
