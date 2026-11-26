"""Normalize only truncated HTTP-body reads for the existing retry policy.

Installed solely in a new recovery process. No source runtime file, successful
request body, HTTP status policy, JSON parser, artifact policy or system setting
is changed. A partial body is never parsed or accepted as a model response.
"""
import hashlib
import http.client
import urllib.request


class ResponseReadInterrupted(OSError):
    """A transport interruption, not a valid/invalid completed model response."""


class ResponseAdapter:
    def __init__(self, response, body_sha256, observer):
        self.response = response
        self.body_sha256 = body_sha256
        self.observer = observer

    def __enter__(self):
        self.response.__enter__()
        return self

    def __exit__(self, *args):
        return self.response.__exit__(*args)

    def __getattr__(self, name):
        return getattr(self.response, name)

    def read(self, *args, **kwargs):
        try:
            return self.response.read(*args, **kwargs)
        except http.client.IncompleteRead as exc:
            partial = exc.partial
            if not isinstance(partial, bytes): raise
            if self.observer is not None:
                self.observer({'kind': 'http_response_body_incomplete', 'body_sha256': self.body_sha256,
                    'partial_bytes': len(partial), 'partial_bytes_sha256': hashlib.sha256(partial).hexdigest(),
                    'expected_remaining_bytes': exc.expected, 'response_not_accepted': True,
                    'remote_acceptance_and_billing_unknown': True})
            raise ResponseReadInterrupted('HTTP response body interrupted; partial response not accepted') from exc


def install(observer=None):
    original = urllib.request.urlopen
    if getattr(original, '_opctx_incomplete_read_adapter', False): raise RuntimeError('Adapter already installed')
    def wrapped(request, *args, **kwargs):
        response = original(request, *args, **kwargs)
        data = getattr(request, 'data', None)
        body_hash = hashlib.sha256(data).hexdigest() if isinstance(data, bytes) else None
        return ResponseAdapter(response, body_hash, observer)
    wrapped._opctx_incomplete_read_adapter = True
    urllib.request.urlopen = wrapped
    return original


def restore(original):
    urllib.request.urlopen = original
