
#### demo_1
Dewei’s Member’s Week demo, April 2025. Contrastive learning.

#### demo_2
Member’s Week demo, October 2025. Transformer-based model.
All data collected on the morning of Oct 22, 2025, 3×10 min (cumin + cloves + oregano).

1. Put training data into `data/train`, realtime test data into `data/online`.
2. If you don’t know the best parameters, run `train.py`. It will test all combinations in `config.py` and save results to `log.txt`.
   Run `to_csv.py` to convert `log.txt` → `log.csv`, then check which parameters perform best.
3. If you already know the best parameters, run `quick_train.py` to train directly.

#### deploy

Realtime data collection and prediction for the live demo.

* `run.py`: collect data
* `monitor.py`: visualize realtime collection
* `demo.py`: set correct model directory in `checkpoints` and collected data, then run demo.py to show visualization
