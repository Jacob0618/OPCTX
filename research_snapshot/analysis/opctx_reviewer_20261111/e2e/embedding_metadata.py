"""Read-only metadata probe for exact historical embedding model."""
import json
import os
from pathlib import Path
from creamem3.experiments.memeye_resume_20260921 import network as n

r = n.client().get(n.g.GEMINI_API_BASE + '/models/gemini-embedding-001',
    headers={'x-goog-api-key': os.environ['GEMINI_API_KEY']}, timeout=30)
data = r.json()
result = {'http_status': r.status_code, 'name': data.get('name'),
          'supportedGenerationMethods': data.get('supportedGenerationMethods'),
          'inputTokenLimit': data.get('inputTokenLimit'), 'error_status': data.get('error', {}).get('status')}
print(json.dumps(result), flush=True)
with (Path(__file__).parent / 'embedding_metadata.json').open('x', encoding='utf-8') as f:
    json.dump(result, f, indent=2)
