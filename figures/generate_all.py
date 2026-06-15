#!/usr/bin/env python3
"""
Generate all figures for the Third Matter paper.

Runs each figure script in sequence and reports success/failure.
Usage: python generate_all.py
"""

import sys
import time
import importlib
import traceback


FIGURE_MODULES = [
    'fig1_architecture',
    'fig2_pentlandite_neb',
    'fig3_mackinawite_neb',
    'fig4_barrier_comparison',
    'fig5_protocell_model',
    'fig6_energy_balance',
]


def main():
    print('=' * 60)
    print('  Third Matter Paper: Generating All Figures')
    print('=' * 60)
    print()

    # Add script directory to path
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    results = {}
    t_total_start = time.time()

    for i, module_name in enumerate(FIGURE_MODULES, 1):
        print(f'[{i}/{len(FIGURE_MODULES)}] Generating {module_name}...')
        t_start = time.time()

        try:
            # Import (or reload) the module
            if module_name in sys.modules:
                mod = importlib.reload(sys.modules[module_name])
            else:
                mod = importlib.import_module(module_name)

            # Run main()
            mod.main()
            elapsed = time.time() - t_start
            results[module_name] = ('OK', elapsed)
            print(f'    Done in {elapsed:.1f}s')

        except Exception as e:
            elapsed = time.time() - t_start
            results[module_name] = ('FAIL', elapsed)
            print(f'    FAILED in {elapsed:.1f}s: {e}')
            traceback.print_exc()

        print()

    t_total = time.time() - t_total_start

    # Summary
    print('=' * 60)
    print('  Summary')
    print('=' * 60)
    n_ok = sum(1 for s, _ in results.values() if s == 'OK')
    n_fail = sum(1 for s, _ in results.values() if s == 'FAIL')

    for name, (status, elapsed) in results.items():
        symbol = '+' if status == 'OK' else 'X'
        print(f'  [{symbol}] {name:30s}  {elapsed:6.1f}s')

    print(f'\n  Total: {n_ok} OK, {n_fail} FAILED ({t_total:.1f}s)')

    if n_fail > 0:
        sys.exit(1)
    else:
        print('\n  All figures generated successfully!')


if __name__ == '__main__':
    main()
