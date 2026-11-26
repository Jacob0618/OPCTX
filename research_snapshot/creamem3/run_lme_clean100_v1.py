"""Bounded clean-input LME control: frozen B1 or expanded B3, never Top3 Writer.

The scientific change is only literal-empty-session no-op handling. Original
physical B3 boundaries, Writer/QA/Judge prompts and Online v4 remain frozen.
Without --allow-live this command performs an offline preflight only.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from creamem3 import lme_clean_selection_v1 as selection
from creamem3 import lme_clean_writer_v1 as clean_writer
from creamem3 import run_lme_opctx100_v2 as core
from creamem3 import run_lme_opctx400_v2 as harness

ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = ROOT / 'creamem3' / 'audit' / 'lme_clean100_20260905'
BASE_CORE_SHA = 'b331c56345ca28e38da0fcdea08897c4d0216e6e94605388753103ac9c8c718a'
BASE_HARNESS_SHA = '99a98f46f2db610fbeb6e599c2e35371807b53f0d257a016c3e0fd15bba4a8c9'
_BASE_CONFIG = core._run_config
_BASE_PROCESS = core._process_question
_VARIANT = 'b1'
adapter: Any = None


def schema(variant: str) -> str:
    if variant not in ('b1', 'b3_scaled'):
        raise ValueError('only b1 and b3_scaled controls are authorized')
    return f'creamem3.lme_clean100.{variant}.v1'


def roots(variant: str, smoke: bool = False) -> tuple[Path, Path]:
    schema(variant)
    tag = f'lme_clean100_{variant}_d768_20260905_r1'
    if smoke:
        tag += '_smoke'
    return ROOT / 'creamem3' / f'results_{tag}', Path('D:/creamem3_cache') / tag


def _load_inputs(contract: Any):
    data, manifest, selected = selection.load_inputs()
    if contract.file_sha256['lme_dataset'] != core._sha_file(core.legacy.LME_DATASET):
        raise RuntimeError('dataset changed after prompt contract validation')
    return data, manifest, selected


def _run_config(contract: Any) -> dict[str, Any]:
    config = _BASE_CONFIG(contract)
    config['architecture'].update({
        'physical_sessions_per_writer_frontier': adapter.FRONTIER_SIZE,
        'final_frontier_policy': 'preserve_original_boundaries_remove_empty_members_skip_empty_groups',
        'causality_granularity': 'nonempty_physical_session' if _VARIANT == 'b1' else 'original_b3_frontier',
        'writer_reads_history': False,
        'writer_top3': False,
        'writer_fact_count_policy': 'original_soft_10_70' if _VARIANT == 'b1' else 'soft_10n_70n_effective_nonempty_members',
        'writer_output_tokens': 8192 if _VARIANT == 'b1' else 16384,
        'model_summary_preserved': True,
        'input_empty_policy': 'literal_turns_empty_list_noop',
        'physical_execution': 'independent_writer_then_causal_online_v4_replay',
        'treatment': f'clean_empty_only_{_VARIANT}',
    })
    config['implementation_source_sha256'].update({
        'clean_runner': core._sha_file(Path(__file__)),
        'clean_writer': core._sha_file(Path(clean_writer.__file__)),
        'clean_selection': core._sha_file(Path(selection.__file__)),
        'writer_base': core._sha_file(ROOT / 'creamem3' / 'lme_opctx_v2.py'),
        'b1_base': core._sha_file(ROOT / 'creamem3' / 'lme_opctx_b1_v1.py'),
        'scaled_b3_base': core._sha_file(ROOT / 'creamem3' / 'lme_opctx_v2_b3_scaled_facts.py'),
        'harness': core._sha_file(Path(harness.__file__)),
    })
    config['campaign'] = {
        'seed': 20260905, 'questions': 100,
        'manifest': str(selection.DEFAULT_MANIFEST),
        'excluded': 'initial100_and_repeated_same78_100',
        'interpretation': 'new_stratified_subsample_of_previously_evaluated_500_not_unseen_test_set',
        'optimization_during_run': False,
    }
    config['cache_contract'].update({'old_writer_or_graph_reuse': False, 'top3_reuse': False})
    return config


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        if core._load_json(path) != payload:
            raise RuntimeError(f'existing immutable artifact differs: {path}')
    else:
        core._atomic_json(path, payload)


def _process_question(**kwargs: Any) -> dict[str, Any]:
    item = kwargs['item']
    ordered, _ = core.legacy._ordered_session_specs(item)
    groups, audit = adapter.audit_and_group(ordered)
    if not groups:
        raise RuntimeError('all source sessions empty; do not invent memory')
    binding = {
        'position': int(kwargs['position']), 'question_id': str(item['question_id']),
        'record_sha256': str(kwargs['manifest_binding']['record_sha256']),
        'manifest_sha256': core.MANIFEST_SHA256,
        'config_sha256': str(kwargs['config_sha256']),
    }
    receipt = core._sealed_payload({'schema_version': f'{schema(_VARIANT)}.input_audit',
        'binding': binding, 'input_audit': audit, 'frontier_plan_sha256': core._canonical(groups)})
    _write_immutable_json(Path(kwargs['cache_root']) / 'questions' /
        f"{int(kwargs['position']):03d}-{item['question_id']}" / 'input_audit.json', receipt)
    result = _BASE_PROCESS(**kwargs)
    # Keep the frozen result untouched. The separate sealed input audit is the
    # authority for source vs processed counts; legacy physical_sessions means
    # original source count and may include no-ops.
    print(json.dumps({'event': 'question_finished', 'variant': _VARIANT,
        'position': kwargs['position'], 'question_id': item['question_id'],
        'status': result.get('status'), 'correct': result.get('correct'),
        'total_wall': result.get('timings_seconds', {}).get('total_wall')}, ensure_ascii=False), flush=True)
    return result


def configure(variant: str, *, freeze: bool = True) -> Any:
    global _VARIANT, adapter
    schema(variant)
    if core._sha_file(Path(core.__file__)) != BASE_CORE_SHA:
        raise RuntimeError('frozen core implementation drift')
    if core._sha_file(Path(harness.__file__)) != BASE_HARNESS_SHA:
        raise RuntimeError('frozen campaign harness drift')
    _, manifest, _ = selection.load_inputs()
    if freeze:
        selection.persist_manifest(manifest)
    if not selection.DEFAULT_MANIFEST.is_file():
        raise RuntimeError('freeze the new manifest before execution')
    _VARIANT = variant
    adapter = clean_writer.make_adapter(variant)
    core.adapter = adapter
    # Same isolated runtime binding used by the historical expanded-B3 runner.
    # PhysicalChat and release import validate the derived Writer contract too.
    core.legacy.assert_prompt_contract = adapter.assert_prompt_contract
    core.legacy.assert_live_prompt_contract = adapter.assert_live_prompt_contract
    core.RUN_SCHEMA = schema(variant)
    core.MANIFEST = selection.DEFAULT_MANIFEST
    core.MANIFEST_SHA256 = core._sha_file(core.MANIFEST)
    core._run_config = _run_config
    core._process_question = _process_question
    harness.adapter = adapter
    harness.RUN_SCHEMA = core.RUN_SCHEMA
    harness.MANIFEST = core.MANIFEST
    harness.MANIFEST_SHA256 = core.MANIFEST_SHA256
    harness._load_inputs = _load_inputs
    harness._run_config = _run_config
    harness._patch_core = lambda: None  # Identity already bound and hash-checked above.
    return adapter


def run_preflight(output: Path, *, limit: int = 100,
                  question_workers: int = 6, writer_workers: int = 1) -> dict[str, Any]:
    if not 1 <= limit <= 100 or not 1 <= question_workers <= 8 or writer_workers != 1:
        raise ValueError('limit 1..100, question workers 1..8, writer workers fixed at 1')
    contract = adapter.assert_prompt_contract()
    _, manifest, selected = _load_inputs(contract)
    rows = []
    for pos, item in enumerate(selected[:limit]):
        ordered, _ = core.legacy._ordered_session_specs(item)
        groups, audit = adapter.audit_and_group(ordered)
        if not groups:
            raise RuntimeError(f"empty entire question: {item['question_id']}")
        if [g['frontier_id'] for g in groups] != list(range(1, len(groups) + 1)):
            raise RuntimeError('non-contiguous cleaned causal frontier IDs')
        for group in groups:
            prompt = adapter.render_write_prompt(contract, adapter.render_lme_frontier(group), unit_count=len(group['members']))
            if not prompt or any(not member['turns'] for member in group['members']):
                raise RuntimeError('empty member reached Writer preflight')
        rows.append({'position': pos, 'question_id': item['question_id'],
            'source_sessions': len(ordered), 'empty_sessions': sum(not s['turns'] for s in ordered),
            'writer_frontiers': len(groups), 'online_calls': len(groups) - 1,
            'input_audit_sha256': core._canonical(audit), 'frontier_plan_sha256': core._canonical(groups)})
    receipt = {'schema_version': f'{schema(_VARIANT)}.preflight', 'status': 'PASS',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'network_calls': 0, 'model_calls': 0,
        'limit': limit, 'question_workers': question_workers, 'writer_workers': 1,
        'manifest_sha256': core.MANIFEST_SHA256, 'run_config': _run_config(contract),
        'selection_items_sha256': core._canonical(manifest['items'][:limit]),
        'questions': rows, 'expected_calls': {
            'writer': sum(r['writer_frontiers'] for r in rows),
            'online': sum(r['online_calls'] for r in rows), 'planner': limit, 'qa': limit,
            'judge': limit, 'qwen': limit, 'empty_sessions_skipped': sum(r['empty_sessions'] for r in rows)}}
    core._atomic_json(output, core._sealed_payload(receipt))
    return receipt


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', choices=('b1', 'b3_scaled'), required=True)
    p.add_argument('--allow-live', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--limit', type=int, default=100)
    p.add_argument('--question-workers', type=int, default=6)
    p.add_argument('--writer-workers', type=int, default=1)
    p.add_argument('--output', type=Path)
    p.add_argument('--cache-root', type=Path)
    p.add_argument('--preflight-output', type=Path)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    configure(args.variant)
    default_output, default_cache = roots(args.variant)
    args.output = args.output or default_output
    args.cache_root = args.cache_root or default_cache
    args.preflight_output = args.preflight_output or AUDIT_ROOT / f'{args.variant}_preflight.json'
    harness.run_preflight = run_preflight
    # Reuse the frozen harness with already parsed bounded arguments.
    class FixedParser:
        def parse_args(self, _argv=None):
            return args
    harness._parser = FixedParser
    return harness.main([])


if __name__ == '__main__':
    raise SystemExit(main())
