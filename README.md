# PPML for Higher Education — hands-on lab (supplementary material)

Companion scripts for Section 8 of the chapter. Everything runs on a laptop in under a minute.

## Setup (Ubuntu 24.04 / Python 3.10+; Windows: use Anaconda prompt or WSL)
    pip install numpy pandas scikit-learn matplotlib dp-accounting

## Run in order
    python3 00_make_dataset.py                                      # Step 0: synthetic data (2,445 rows)
    PSEUDO_KEY=$(openssl rand -hex 16) python3 uc1_pseudonymise_kanon.py   # UC1: pseudonymise + k-anonymity
    rm -f privacy_ledger.json && python3 uc2_dp_release.py          # UC2: DP statistics + budget ledger
    python3 uc3_federated.py                                        # UC3: FedAvg + secure aggregation
    python3 uc4_dpsgd.py 2>/dev/null                                # UC4: DP-SGD with (eps, delta)
    python3 uc5_mpc.py                                              # UC5: 3-party secret sharing

`expected_output/` holds the exact console output and screenshots obtained by the author,
so you can diff your run against them. No real student data is used anywhere.
