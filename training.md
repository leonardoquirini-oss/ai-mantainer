# Scoring notturno (ogni notte alle 3)
0 3 * * * cd /home/berni/maintainer && python scripts/score_fleet.py

# Drift check mensile (1° del mese alle 9)
0 9 1 * * cd /home/berni/maintainer && python scripts/check_drift.py --alert

# Retraining trimestrale (1° gen/apr/lug/ott alle 2)
0 2 1 1,4,7,10 * cd /home/berni/maintainer && python scripts/train_models.py --refresh-data