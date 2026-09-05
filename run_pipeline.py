import argparse
import subprocess
import sys
import time
from pathlib import Path
 
PROJECT_ROOT = Path(__file__).resolve().parent
 
 
def _section(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
 
 
def run_tests():
    _section("Running test suite (pytest)")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q"],
        cwd=PROJECT_ROOT,
    )
    if result.returncode != 0:
        print("\nTest suite failed. Aborting.")
        sys.exit(result.returncode)
 
 
def run_generation():
    _section("Step 1/2: Generating synthetic dataset")
    from src.generate.run_generation import main as generate_main
    generate_main()
    print("Synthetic ledger, network log, and wallet metadata written to data/clean/")
 
 
def run_graph_build():
    _section("Step 2/2: Ingest -> validate -> correlate -> build graph")
    from src.ingestion.loader import load_ledger, load_wallets, load_network_log, validate_ingested_data
    from src.graph.builder import build_graph, save_graph, graph_stats, Wallet
 
    t0 = time.time()
    print("Loading + validating ledger.csv ...")
    ledger_df = load_ledger()
    print(f"  ledger: {len(ledger_df)} rows ({time.time() - t0:.1f}s)")
 
    t1 = time.time()
    print("Loading + validating wallet_metadata.csv ...")
    wallets_df = load_wallets()
    print(f"  wallets: {len(wallets_df)} rows ({time.time() - t1:.1f}s)")
 
    t2 = time.time()
    print("Loading + validating network_log.csv ...")
    network_log_df = load_network_log()
    print(f"  network log: {len(network_log_df)} rows ({time.time() - t2:.1f}s)")
 
    t3 = time.time()
    print("Cross-validating ledger <-> network log ...")
    validate_ingested_data(ledger_df, network_log_df)
    print(f"  cross-validation OK ({time.time() - t3:.1f}s)")
 
    all_wallets = [Wallet(row["address"], row["script_type"]) for _, row in wallets_df.iterrows()]
 
    t4 = time.time()
    print("Building + validating graph ...")
    graph = build_graph(ledger_df, network_log_df, all_wallets)
    print(f"  graph built ({time.time() - t4:.1f}s)")
 
    t5 = time.time()
    output_path = save_graph(graph)
    print(f"  graph saved ({time.time() - t5:.1f}s)")
 
    stats = graph_stats(graph)
    print(f"\nGraph artifact saved to: {output_path}")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print(f"\nTotal graph-build time: {time.time() - t0:.1f}s")
    return graph, output_path
 
 
def main():
    parser = argparse.ArgumentParser(description="Run the full data-foundation pipeline.")
    parser.add_argument("--skip-generate", action="store_true",
                         help="Skip synthetic data generation; use existing data/clean/*.csv")
    parser.add_argument("--test", action="store_true",
                         help="Run the pytest suite before running the pipeline")
    parser.add_argument("--test-only", action="store_true",
                         help="Only run the pytest suite; skip data generation and graph build")
    args = parser.parse_args()
 
    start = time.time()
 
    if args.test or args.test_only:
        run_tests()
        if args.test_only:
            _section(f"Done in {time.time() - start:.1f}s (tests only)")
            return
 
    if not args.skip_generate:
        run_generation()
    else:
        _section("Skipping generation step (--skip-generate)")
 
    run_graph_build()
 
    _section(f"Pipeline complete in {time.time() - start:.1f}s")
 
 
if __name__ == "__main__":
    main()