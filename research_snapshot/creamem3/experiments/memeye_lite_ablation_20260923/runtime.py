"""Explicit single-backbone requests, immutable receipts, no old cache writes."""
import base64
from dataclasses import asdict
import mimetypes
from pathlib import Path
from ..memeye_backbones_20260921 import runtime as prior

b = prior.b
MODELS = {'g25lite': 'gemini-2.5-flash-lite', 'g35lite': 'gemini-3.5-flash-lite'}
JUDGES = {'g35flash': 'gemini-3.5-flash', 'g35lite': 'gemini-3.5-flash-lite'}
CONSTRUCTION = {'caption', 'writer', 'view', 'online_revision', 'online_revision_repair'}


def config(model, stage, cap, json_mode=False):
    if model not in set(MODELS.values()) | set(JUDGES.values()):
        raise ValueError('Unregistered model')
    construct = stage in CONSTRUCTION
    thinking = ({'thinkingBudget': 512 if construct else 0} if model == MODELS['g25lite']
                else {'thinkingLevel': 'low' if construct else 'minimal'})
    cfg = {'maxOutputTokens': int(cap), 'thinkingConfig': thinking}
    if model == MODELS['g25lite'] and not construct:
        cfg['temperature'] = 0.0
    if json_mode:
        cfg['responseMimeType'] = 'application/json'
    return cfg


def request(model, stage, prompt, system, cap, json_mode, image_path=None):
    cfg = config(model, stage, cap, json_mode)
    parts = []
    image_ref = None
    if image_path is not None:
        if stage != 'caption':
            raise ValueError('Images only in caption')
        p = Path(image_path)
        blob = p.read_bytes()
        image_ref = {'path': str(p), 'sha256': b.sha(blob), 'mime_type': mimetypes.guess_type(str(p))[0]}
        part = {'inlineData': {'mimeType': image_ref['mime_type'], 'data': base64.b64encode(blob).decode('ascii')}}
        if model.startswith('gemini-3.'):
            part['mediaResolution'] = {'level': 'MEDIA_RESOLUTION_HIGH'}
        else:
            cfg['mediaResolution'] = 'MEDIA_RESOLUTION_HIGH'
        parts.append(part)
    parts.append({'text': prompt})
    wire = {'contents': [{'role': 'user', 'parts': parts}], 'generationConfig': cfg}
    if system:
        wire['systemInstruction'] = {'parts': [{'text': system}]}
    return wire, image_ref


class Runtime:
    def __init__(self, root, model, allow):
        self.root, self.model, self.allow = Path(root), model, allow

    def generate(self, stage, prompt, *, system=None, json_mode=False, max_output_tokens=500,
                 max_provider_attempts=3, cache_metadata=None, image_path=None, caption_context_count=None):
        if stage == 'judge' and self.model not in JUDGES.values():
            raise ValueError('Wrong Judge model')
        if stage != 'judge' and self.model not in MODELS.values():
            raise ValueError('Wrong generation backbone')
        wire, ref = request(self.model, stage, prompt, system, max_output_tokens, json_mode, image_path)
        if image_path is not None and not 0 <= caption_context_count <= 2:
            raise ValueError('Caption context must be previous two or fewer')
        manifest = dict(model=self.model, stage=stage, prompt=prompt, system=system, image=ref,
                        generation_config=wire['generationConfig'], wire_sha256=b.sha(wire),
                        metadata=cache_metadata or {}, caption_context_count=caption_context_count)
        def send():
            if not self.allow:
                raise RuntimeError('Model calls disabled')
            return prior.n.VerifiedProvider().generate_content(model=self.model, request=wire)
        def validate(raw):
            cs = raw.get('candidates') or []
            if len(cs) != 1 or cs[0].get('finishReason') != 'STOP':
                raise RuntimeError('Non-STOP generation: '+str([c.get('finishReason') for c in cs]))
            if not prior.n.g._normalize_text(raw).strip():
                raise RuntimeError('Empty generation')
            served = raw.get('modelVersion', '')
            if served and not served.startswith(self.model):
                raise RuntimeError('Served model mismatch: '+served)
        result, hit = prior.durable(self.root, manifest, send, validate, min(3, max_provider_attempts))
        raw = result['response']
        return prior.Reply(prior.n.g._normalize_text(raw).strip(), result['request_sha256'],
                           result['response_sha256'], self.model, raw.get('modelVersion', ''),
                           asdict(prior.n.g._normalize_usage(raw)), hit)

    def judge(self, question, gold, prediction, metadata):
        prompt = prior.n.g.build_mgtm_judge_prompt(question, gold, prediction)
        out = self.generate('judge', prompt, json_mode=True, max_output_tokens=400, cache_metadata=metadata)
        return out, prior.n.g.parse_mgtm_judge_response(out.text)
