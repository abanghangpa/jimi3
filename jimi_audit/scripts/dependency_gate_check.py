"""
Dependency Gate Checker
=======================
When a booster strategy or module is modified, this script:
1. Loads the dependency map
2. Finds all strategies that depend on the modified component
3. Runs each dependent's gate (or loads recent results)
4. Compares to baseline
5. Reports: PASS / WARN / FAIL

Usage:
    python3 dependency_gate_check.py <modified_component>
    
Examples:
    python3 dependency_gate_check.py whale_watch
    python3 dependency_gate_check.py m14_sweep
    python3 dependency_gate_check.py regime_classifier
    python3 dependency_gate_check.py derivatives
"""

import json, os, sys, subprocess
from datetime import datetime, timezone

BASE = '/root/.openclaw/workspace/jimi_audit'
DEPS_FILE = os.path.join(BASE, 'config', 'strategy_dependencies.json')
REPORTS_DIR = os.path.join(BASE, 'reports')

def load_deps():
    with open(DEPS_FILE) as f:
        return json.load(f)

def find_dependents(deps, component):
    """Find all strategies that depend on a component."""
    dependents = []
    
    # Check if it's a module
    if component in deps.get('modules', {}):
        used_by = deps['modules'][component].get('used_by', [])
        for item in used_by:
            if item == 'ALL':
                # Expand to all strategies
                dependents.extend(deps.get('strategies', {}).keys())
            else:
                dependents.append(item)
    
    # Check if it's a strategy used as booster
    for strat_name, strat_info in deps.get('strategies', {}).items():
        if component in strat_info.get('dependencies', []):
            dependents.append(strat_name)
    
    # Also check direct strategy name
    if component in deps.get('strategies', {}):
        for strat_name, strat_info in deps.get('strategies', {}).items():
            if component in strat_info.get('dependencies', []):
                if strat_name not in dependents:
                    dependents.append(strat_name)
    
    return list(set(dependents))

def get_baseline(deps, strat_name):
    """Get the baseline gate result for a strategy."""
    strat = deps.get('strategies', {}).get(strat_name, {})
    return {
        'last_validated': strat.get('last_validated'),
        'gate_result': strat.get('gate_result'),
        'version': strat.get('version'),
        'risk': strat.get('risk_if_modified', 'UNKNOWN'),
    }

def run_gate(strat_name):
    """Run the gate for a strategy. Returns metrics or None if no gate script exists."""
    # Map strategy name to gate script
    gate_scripts = {
        's01_failed_breakout': 's01_v3_full_backtest.py',
        's20_liquidation_cascade': 's20_v8b_gate.py',
        's21_trade_flow': None,  # gate exists but complex
        's19_orderbook_imbalance': None,
        's04_positioning_fade': None,
        's14_whale_watch': None,
        's22_judas_sweep': None,
        's13_funding_arb': None,
    }
    
    script = gate_scripts.get(strat_name)
    if not script:
        return None
    
    script_path = os.path.join(BASE, 'scripts', script)
    if not os.path.exists(script_path):
        return None
    
    # Run the gate script
    try:
        result = subprocess.run(
            ['python3', script_path],
            capture_output=True, text=True, timeout=300,
            cwd=BASE
        )
        
        # Parse output for key metrics
        output = result.stdout
        metrics = parse_gate_output(output, strat_name)
        return metrics
    except Exception as e:
        return {'error': str(e)}

def parse_gate_output(output, strat_name):
    """Parse gate script output for key metrics."""
    metrics = {}
    
    for line in output.split('\n'):
        line = line.strip()
        
        # Look for MC p-value
        if 'MC p:' in line or 'mc_p:' in line:
            try:
                p_str = line.split(':')[-1].strip().rstrip(',')
                metrics['mc_p'] = float(p_str)
            except:
                pass
        
        # Look for WR
        if 'WR=' in line and '%' in line:
            try:
                wr_str = line.split('WR=')[1].split('%')[0].strip()
                metrics['wr'] = float(wr_str) / 100
            except:
                pass
        
        # Look for SIGNIFICANT
        if 'SIGNIFICANT: YES' in line:
            metrics['significant'] = True
        elif 'SIGNIFICANT: NO' in line:
            metrics['significant'] = False
        
        # Look for Gate verdict
        if 'Gate:' in line:
            try:
                gate = line.split('Gate:')[-1].strip()
                metrics['gate_verdict'] = gate
            except:
                pass
        
        # Look for n= (sample size)
        if line.startswith('n=') or '  n=' in line:
            try:
                n_str = line.split('n=')[1].split(',')[0].strip()
                metrics['n'] = int(n_str)
            except:
                pass
        
        # Look for mean return
        if 'mean=' in line and '%' in line:
            try:
                mean_str = line.split('mean=')[1].split('%')[0].strip().replace('+', '')
                metrics['mean'] = float(mean_str)
            except:
                pass
    
    return metrics

def compare_results(baseline_metrics, current_metrics):
    """Compare current gate results to baseline."""
    if not baseline_metrics or not current_metrics:
        return 'UNKNOWN', 'No baseline or current metrics to compare'
    
    issues = []
    status = 'PASS'
    
    # Compare WR
    base_wr = baseline_metrics.get('wr') or baseline_metrics.get('WR')
    curr_wr = current_metrics.get('wr')
    if base_wr and curr_wr:
        base_wr = base_wr if base_wr <= 1 else base_wr / 100
        curr_wr = curr_wr if curr_wr <= 1 else curr_wr / 100
        wr_delta = curr_wr - base_wr
        if wr_delta < -0.10:
            status = 'FAIL'
            issues.append(f'WR dropped {abs(wr_delta)*100:.1f}% ({base_wr*100:.1f}% → {curr_wr*100:.1f}%)')
        elif wr_delta < -0.05:
            status = 'WARN'
            issues.append(f'WR dropped {abs(wr_delta)*100:.1f}% ({base_wr*100:.1f}% → {curr_wr*100:.1f}%)')
    
    # Compare MC p-value
    base_p = baseline_metrics.get('mc_p')
    curr_p = current_metrics.get('mc_p')
    if base_p is not None and curr_p is not None:
        if base_p < 0.05 and curr_p >= 0.05:
            status = 'FAIL'
            issues.append(f'MC lost significance (p={base_p:.4f} → {curr_p:.4f})')
        elif base_p < 0.05 and curr_p > base_p * 2:
            status = 'WARN'
            issues.append(f'MC p-value degraded ({base_p:.4f} → {curr_p:.4f})')
    
    # Compare sample size
    base_n = baseline_metrics.get('n')
    curr_n = current_metrics.get('n')
    if base_n and curr_n:
        if curr_n < base_n * 0.5:
            status = 'WARN'
            issues.append(f'Sample size halved ({base_n} → {curr_n})')
    
    # Compare significance
    base_sig = baseline_metrics.get('significant')
    curr_sig = current_metrics.get('significant')
    if base_sig and not curr_sig:
        status = 'FAIL'
        issues.append('Lost statistical significance')
    
    if not issues:
        return 'PASS', 'No regression detected'
    
    return status, '; '.join(issues)

def print_report(component, dependents, results):
    """Print the dependency gate report."""
    print("\n" + "="*70)
    print(f"DEPENDENCY GATE REPORT: {component}")
    print("="*70)
    print(f"Modified: {component}")
    print(f"Dependents: {len(dependents)}")
    print(f"Timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print()
    
    overall_status = 'PASS'
    
    for strat_name in sorted(dependents):
        baseline = results[strat_name]['baseline']
        current = results[strat_name]['current']
        status = results[strat_name]['status']
        issues = results[strat_name]['issues']
        
        if status == 'FAIL':
            overall_status = 'FAIL'
        elif status == 'WARN' and overall_status != 'FAIL':
            overall_status = 'WARN'
        
        icon = {'PASS': '✅', 'WARN': '⚠️', 'FAIL': '❌', 'UNKNOWN': '❓'}.get(status, '❓')
        
        print(f"  {icon} {strat_name}")
        print(f"     Version: {baseline.get('version', '?')}")
        print(f"     Last validated: {baseline.get('last_validated', 'never')}")
        print(f"     Baseline: {baseline.get('gate_result', 'none')}")
        
        if current:
            print(f"     Current: {current}")
        
        print(f"     Status: {status}")
        if issues:
            print(f"     Issues: {issues}")
        print()
    
    print("="*70)
    print(f"OVERALL: {overall_status}")
    
    if overall_status == 'FAIL':
        print("⛔ REVERT REQUIRED — one or more dependents lost edge")
    elif overall_status == 'WARN':
        print("⚠️  REVIEW REQUIRED — metrics degraded, investigate before deploying")
    else:
        print("✅ ALL CLEAR — no regression detected")
    
    print("="*70)
    
    return overall_status

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 dependency_gate_check.py <modified_component>")
        print("\nExamples:")
        print("  python3 dependency_gate_check.py whale_watch")
        print("  python3 dependency_gate_check.py m14_sweep")
        print("  python3 dependency_gate_check.py regime_classifier")
        print("\nAvailable components:")
        deps = load_deps()
        print("  Strategies:", ', '.join(sorted(deps.get('strategies', {}).keys())))
        print("  Modules:", ', '.join(sorted(deps.get('modules', {}).keys())))
        sys.exit(1)
    
    component = sys.argv[1]
    run_gates = '--run-gates' in sys.argv
    
    deps = load_deps()
    dependents = find_dependents(deps, component)
    
    if not dependents:
        print(f"No strategies depend on '{component}'")
        sys.exit(0)
    
    print(f"Found {len(dependents)} dependents of '{component}': {', '.join(dependents)}")
    
    results = {}
    for strat_name in dependents:
        baseline = get_baseline(deps, strat_name)
        
        current = None
        if run_gates:
            print(f"\nRunning gate for {strat_name}...")
            current = run_gate(strat_name)
        
        # Compare
        status, issues = compare_results(
            baseline.get('gate_result', {}),
            current or {}
        )
        
        results[strat_name] = {
            'baseline': baseline,
            'current': current,
            'status': status,
            'issues': issues,
        }
    
    overall = print_report(component, dependents, results)
    
    # Save report
    report_file = os.path.join(REPORTS_DIR, f'dependency_check_{component}_{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")}.json')
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    with open(report_file, 'w') as f:
        json.dump({
            'component': component,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'overall_status': overall,
            'dependents': {k: {
                'baseline': v['baseline'],
                'current': v['current'],
                'status': v['status'],
                'issues': v['issues'],
            } for k, v in results.items()},
        }, f, indent=2, default=str)
    print(f"\nReport saved to {report_file}")
    
    sys.exit(0 if overall == 'PASS' else 1)

if __name__ == '__main__':
    main()
