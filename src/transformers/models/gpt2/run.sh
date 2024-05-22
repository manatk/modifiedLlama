#!/bin/bash
# Run across different alpha values
rm benchmark_scores.csv
touch benchmark_scores.csv
python3 eval.py r 0 "BookSum"
python3 eval.py r 0.5 "BookSum"
python3 eval.py r 0.6 "BookSum"
python3 eval.py r 0.7 "BookSum"
python3 eval.py r 0.8 "BookSum"
python3 eval.py r 0.9 "BookSum"
python3 eval.py r 0.95 "BookSum"
python3 eval.py r 0.97 "BookSum"
python3 eval.py r 0.99 "BookSum"
python3 eval.py r 1 "BookSum"
python3 eval.py r 0 "BookSum"
python3 eval.py r 0.5 "LegalBench"
python3 eval.py r 0.6 "LegalBench"
python3 eval.py r 0.7 "LegalBench"
python3 eval.py r 0.8 "LegalBench"
python3 eval.py r 0.9 "LegalBench"
python3 eval.py r 0.95 "LegalBench"
python3 eval.py r 0.97 "LegalBench"
python3 eval.py r 0.99 "LegalBench"
python3 eval.py r 1 "LegalBench"