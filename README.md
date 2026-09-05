# AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic

Smart India Hackathon 2026 — Problem Statement 26146.

Offline pipeline for ingesting Bitcoin transaction and network metadata, validating and cleaning the data, correlating ledger and network observations through TXIDs, and building a transaction/entity graph.

## Setup

### Linux
sudo apt-get install -y libmagic1
pip install -r requirements.txt
 

### Windows
pip install -r requirements.txt

### macOS
brew install libmagic
pip install -r requirements.txt
 

## Run

# Run from the project root.
python run_pipeline.py --test
 
Runs the test suite and then the pipeline.

# To reuse the existing generated data:
python run_pipeline.py --skip-generate
 

# To run tests only:
python run_pipeline.py --test-only
 
# Tests can also be run directly:
pytest tests/ -v

## Output

# Generated datasets:

data/clean/ledger.csv
data/clean/network_log.csv
data/clean/wallet_metadata.csv
 

# Graph artifact:

data/graph/transaction_network.graphml

Ground-truth data used for evaluation:

data/ground_truth/wallet_ground_truth.csv
 
The ground-truth file is not used as model input.

## Pipeline
CSV / JSON / XML
       |
       v
   Ingestion
       |
       v
Validation + Cleaning
       |
       v
Canonical Data
       |
       v
Ledger <-> Network Correlation
       |
       v
   Graph Builder
       |
       v
transaction_network.graphml


## Project Structure
bitcoin-transaction/
├── src/
│   ├── ingestion/
│   ├── graph/
│   └── generate/
├── data/
│   ├── clean/
│   ├── dirty_data/
│   ├── geoip/
│   ├── graph/
│   └── ground_truth/
├── tests/
├── dashboard/
├── docs/
├── notebooks/
├── requirements.txt
└── run_pipeline.py

## Current Scope

Implemented:

* CSV / JSON / XML ingestion
* Data validation and cleaning
* Bitcoin address, TXID, IP, port and amount validation
* Wallet metadata ingestion
* Ledger/network correlation
* Transaction and entity graph construction
* GraphML output
* Automated test suite

The following components are planned separately:

* `src/models/` — anomaly detection and entity clustering
* `src/explainability/` — explainable ranked alerts
* `dashboard/` — dashboard and link-analysis views
* `docs/` — technical documentation

`src/generate/` contains the synthetic dataset generator used for development and testing.
