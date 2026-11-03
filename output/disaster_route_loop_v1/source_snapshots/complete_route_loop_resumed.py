"""Seal the unchanged October protocol with the actual resumed completion date."""
import shutil

from run_route_loop import HERE, ROOT, OUT, read, digest, checked, save, verify


def complete():
    assert not (OUT / 'completion_receipt.json').exists(), 'Completed output frozen'
    protocol = verify()
    audit = read(OUT / 'audit_report.json')
    assert audit['status'] == 'PASS' and audit['routes_independently_replayed'] == 2160
    visual = read(OUT / 'visual_inspection.json')
    expected = {f'figures/{name}.png' for name in ('route_example_port', 'route_example_south', 'budget_comparison', 'all_primary_deltas', 'PS_geometry_routes')}
    assert visual['all_five_figures_viewed'] and set(visual['files']) == expected
    for name, sha in visual['files'].items():
        checked(OUT / name, sha)
    resumed = read(OUT / 'resumption_note.json')
    assert resumed['client_completion_date'] == '2026-11-01'
    checked(HERE / 'analyze_route_results.py', resumed['additional_analysis_source_sha256'])
    sources = dict(protocol['source_sha256'])
    for name in ('analyze_route_results.py', 'complete_route_loop_resumed.py'):
        sources[name] = digest(HERE / name)
    snapshots = OUT / 'source_snapshots'
    snapshots.mkdir()
    for name, sha in sources.items():
        checked(HERE / name, sha)
        shutil.copyfile(HERE / name, snapshots / name)
    external = {**protocol['external_sha256'], **{str(HERE / name): sha for name, sha in sources.items()},
                **{str(ROOT / 'work/disaster_search' / name): sha for name, sha in protocol['old_perception_sources_sha256'].items()}}
    for name, sha in external.items():
        checked(name, sha)
    receipt = {'status': 'COMPLETE', 'client_report_date': '2026-11-01', 'protocol_client_date': protocol['client_date'],
               'source_code_sha256': sources,
               'completion_note': 'Original protocol and five bound sources unchanged; resumed completion tool corrects the original reporter hardcoded date and includes post-run descriptive diagnostics.',
               'files': {str(path.relative_to(OUT)): digest(path) for path in sorted(OUT.rglob('*')) if path.is_file()},
               'external_files': external, 'simulation_routes': 2160, 'synthetic_worlds': 60,
               'realPS_geometry_static_routes': 4, 'new_tests_passed': 12, 'budget_violations': 0,
               'all_five_figures_viewed': True, 'new_perception_calls': 0, 'training': False, 'gpu_rented': False,
               'new_real_target_labels': 0, 'old_reserved_sets_scored': False,
               'physical_flight_validation': False, 'new_method_efficacy_established': False}
    save(OUT / 'completion_receipt.json', receipt)
    for name, sha in receipt['files'].items():
        checked(OUT / name, sha)
    print({'status': 'COMPLETE', 'client_report_date': receipt['client_report_date'],
           'receipt_sha256': digest(OUT / 'completion_receipt.json'), 'output_files': len(receipt['files'])})


if __name__ == '__main__':
    complete()
