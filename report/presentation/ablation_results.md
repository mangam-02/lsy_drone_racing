# Ablation-Suche — Level 2 (Seeds 0–200)

_201 Seeds getestet · all_on erfolgreich in 174/201 · jede Zeile = ein fester Seed, gleicher Track für alle 4 Varianten._

**Legende:** ✓ = Ziel erreicht (Zeit) · ✗ = Fehlschlag (`coll`=Kollision, `grnd`=Boden, `oob`=außerhalb, `time`=Timeout; `@gN`=an Gate N)


## ⭐ Ideale Seeds (all_on ✓, alle drei Ablationen ✗)

| Seed | all_on | curvature_speed_limit aus | gate_track_boost aus | use_caution aus |
|---|---|---|---|---|
| **96** | ✓ 17.1s | ✗ coll@g3 | ✗ coll@g2 | ✗ coll@g2 |
| **146** | ✓ 9.0s | ✗ coll@g2 | ✗ coll@g2 | ✗ coll@g2 |
| **150** | ✓ 7.6s | ✗ coll@g3 | ✗ time@g2 | ✗ coll@g3 |
| **169** | ✓ 8.1s | ✗ coll@g3 | ✗ coll@g3 | ✗ coll@g3 |

**Empfehlung: Seed 169** — all_on 8.1 s, und alle drei Ablationen scheitern einheitlich mit Kollision an Gate 3 (klare, konsistente Story). Alternative **146** (alle Kollision an Gate 2). Seed 96 (min-cover-Default des Skripts) hat all_on 17.1 s — vermutlich mit Retry, als „so soll es aussehen"-Video weniger schön.


## 🎬 Aufnahme-Befehle — Seed 169 (empfohlen)

```bash
venv_drone/bin/python scripts/record_video.py --config level2.toml --seed 169 --out report/presentation/lvl2_seed169_all_on.mp4
venv_drone/bin/python scripts/record_video.py --config level2.toml --seed 169 --ablate no_curvature --out report/presentation/lvl2_seed169_no_curvature.mp4
venv_drone/bin/python scripts/record_video.py --config level2.toml --seed 169 --ablate no_gate_boost --out report/presentation/lvl2_seed169_no_gate_boost.mp4
venv_drone/bin/python scripts/record_video.py --config level2.toml --seed 169 --ablate no_caution --out report/presentation/lvl2_seed169_no_caution.mp4
```

## 🎬 Aufnahme-Befehle — Seed 146 (Alternative)

```bash
venv_drone/bin/python scripts/record_video.py --config level2.toml --seed 146 --out report/presentation/lvl2_seed146_all_on.mp4
venv_drone/bin/python scripts/record_video.py --config level2.toml --seed 146 --ablate no_curvature --out report/presentation/lvl2_seed146_no_curvature.mp4
venv_drone/bin/python scripts/record_video.py --config level2.toml --seed 146 --ablate no_gate_boost --out report/presentation/lvl2_seed146_no_gate_boost.mp4
venv_drone/bin/python scripts/record_video.py --config level2.toml --seed 146 --ablate no_caution --out report/presentation/lvl2_seed146_no_caution.mp4
```

## 🎬 Aufnahme-Befehle — Seed 96 (Skript-Default)

```bash
venv_drone/bin/python scripts/record_video.py --config level2.toml --seed 96 --out report/presentation/lvl2_seed96_all_on.mp4
venv_drone/bin/python scripts/record_video.py --config level2.toml --seed 96 --ablate no_curvature --out report/presentation/lvl2_seed96_no_curvature.mp4
venv_drone/bin/python scripts/record_video.py --config level2.toml --seed 96 --ablate no_gate_boost --out report/presentation/lvl2_seed96_no_gate_boost.mp4
venv_drone/bin/python scripts/record_video.py --config level2.toml --seed 96 --ablate no_caution --out report/presentation/lvl2_seed96_no_caution.mp4
```

## Per-Ablation-Coverage (all_on ✓, aber dieses Feature-aus ✗)

Falls du pro Feature einen eigenen Seed zeigen willst statt einem idealen:

- **curvature_speed_limit aus** (no_curvature): 23 Seeds — [3, 4, 15, 21, 26, 33, 40, 41, 43, 53, 62, 73, 74, 96, 140, 145, 146, 150, 163, 169, 170, 171, 199]
- **gate_track_boost aus** (no_gate_boost): 21 Seeds — [17, 20, 23, 30, 35, 49, 52, 78, 87, 96, 99, 105, 106, 146, 150, 153, 155, 164, 169, 170, 187]
- **use_caution aus** (no_caution): 29 Seeds — [3, 8, 9, 35, 37, 40, 45, 57, 58, 77, 84, 87, 89, 96, 98, 106, 107, 117, 141, 146, 150, 151, 163, 169, 177, 182, 183, 194, 197]


## Vollständige Tabelle (nach Seed sortiert)

| Seed | all_on | no_curvature | no_gate_boost | no_caution |
|---|---|---|---|---|
| 0 | ✓ 6.8s | ✓ 6.5s | ✓ 6.6s | ✓ 5.5s |
| 1 | ✓ 7.4s | ✓ 7.1s | ✓ 7.3s | ✓ 5.9s |
| 2 | ✓ 7.2s | ✓ 6.9s | ✓ 7.0s | ✓ 5.7s |
| 3 | ✓ 6.8s | ✗ coll@g3 | ✓ 7.1s | ✗ coll@g3 |
| 4 | ✓ 6.6s | ✗ coll@g3 | ✓ 6.9s | ✓ 5.7s |
| 5 | ✓ 8.2s | ✓ 7.7s | ✓ 8.1s | ✓ 6.2s |
| 6 | ✓ 8.4s | ✓ 8.2s | ✓ 6.5s | ✓ 6.1s |
| 7 | ✓ 8.3s | ✓ 7.4s | ✓ 8.7s | ✓ 6.2s |
| 8 | ✓ 7.3s | ✓ 7.3s | ✓ 7.6s | ✗ coll@g3 |
| 9 | ✓ 6.4s | ✓ 6.0s | ✓ 7.0s | ✗ coll@g3 |
| 10 | ✓ 7.1s | ✓ 6.7s | ✓ 7.4s | ✓ 5.7s |
| 11 | ✓ 6.8s | ✓ 6.4s | ✓ 6.7s | ✓ 6.4s |
| 12 | ✓ 7.5s | ✓ 7.2s | ✓ 7.0s | ✓ 6.4s |
| 13 | ✓ 8.2s | ✓ 8.2s | ✓ 8.0s | ✓ 6.5s |
| 14 | ✓ 6.2s | ✓ 6.2s | ✓ 6.3s | ✓ 5.4s |
| 15 | ✓ 8.8s | ✗ oob@g3 | ✓ 9.2s | ✓ 6.9s |
| 16 | ✓ 6.8s | ✓ 6.6s | ✓ 6.8s | ✓ 5.7s |
| 17 | ✓ 7.7s | ✓ 6.9s | ✗ time@g2 | ✓ 5.9s |
| 18 | ✗ time@g2 | ✓ 7.5s | ✗ coll@g3 | ✓ 8.1s |
| 19 | ✗ grnd@g3 | ✓ 8.8s | ✓ 8.3s | ✗ grnd@g3 |
| 20 | ✓ 6.5s | ✓ 6.3s | ✗ coll@g3 | ✓ 5.4s |
| 21 | ✓ 7.0s | ✗ coll@g3 | ✓ 6.9s | ✓ 6.1s |
| 22 | ✓ 7.0s | ✓ 7.2s | ✓ 7.5s | ✓ 6.0s |
| 23 | ✓ 6.8s | ✓ 6.3s | ✗ coll@g2 | ✓ 5.9s |
| 24 | ✓ 6.6s | ✓ 6.7s | ✓ 6.9s | ✓ 5.8s |
| 25 | ✗ time@g2 | ✓ 6.6s | ✓ 6.7s | ✗ coll@g2 |
| 26 | ✓ 6.6s | ✗ grnd@g1 | ✓ 7.6s | ✓ 6.0s |
| 27 | ✓ 7.4s | ✓ 7.0s | ✓ 7.3s | ✓ 5.9s |
| 28 | ✓ 7.5s | ✓ 7.3s | ✓ 7.2s | ✓ 6.5s |
| 29 | ✓ 9.8s | ✓ 8.7s | ✓ 9.4s | ✓ 8.7s |
| 30 | ✓ 6.9s | ✓ 7.5s | ✗ oob@g3 | ✓ 6.3s |
| 31 | ✗ grnd@g3 | ✗ coll@g3 | ✓ 6.9s | ✓ 5.9s |
| 32 | ✗ coll@g0 | ✗ time@g0 | ✗ coll@g1 | ✗ time@g0 |
| 33 | ✓ 7.6s | ✗ coll@g3 | ✓ 6.5s | ✓ 6.2s |
| 34 | ✓ 7.6s | ✓ 7.1s | ✓ 7.3s | ✓ 6.3s |
| 35 | ✓ 6.1s | ✓ 6.5s | ✗ oob@g2 | ✗ coll@g0 |
| 36 | ✗ coll@g3 | ✓ 6.9s | ✗ coll@g3 | ✓ 6.2s |
| 37 | ✓ 8.6s | ✓ 21.6s | ✓ 8.9s | ✗ coll@g3 |
| 38 | ✓ 7.5s | ✓ 7.1s | ✓ 7.4s | ✓ 6.1s |
| 39 | ✗ time@g3 | ✗ coll@g3 | ✓ 18.5s | ✗ coll@g3 |
| 40 | ✓ 6.6s | ✗ coll@g0 | ✓ 6.4s | ✗ coll@g2 |
| 41 | ✓ 6.4s | ✗ coll@g3 | ✓ 6.4s | ✓ 6.6s |
| 42 | ✓ 6.5s | ✓ 6.6s | ✓ 6.8s | ✓ 5.9s |
| 43 | ✓ 7.3s | ✗ coll@g1 | ✓ 6.9s | ✓ 5.9s |
| 44 | ✓ 6.8s | ✓ 6.5s | ✓ 6.7s | ✓ 6.0s |
| 45 | ✓ 7.6s | ✓ 7.5s | ✓ 8.2s | ✗ coll@g0 |
| 46 | ✓ 7.5s | ✓ 7.2s | ✓ 7.4s | ✓ 6.2s |
| 47 | ✓ 7.3s | ✓ 7.1s | ✓ 7.3s | ✓ 6.2s |
| 48 | ✓ 7.4s | ✓ 6.9s | ✓ 6.6s | ✓ 5.9s |
| 49 | ✓ 7.0s | ✓ 7.1s | ✗ grnd@g3 | ✓ 5.9s |
| 50 | ✓ 7.8s | ✓ 7.5s | ✓ 7.5s | ✓ 8.9s |
| 51 | ✓ 7.3s | ✓ 7.1s | ✓ 7.1s | ✓ 6.2s |
| 52 | ✓ 9.6s | ✓ 7.6s | ✗ coll@g3 | ✓ 6.1s |
| 53 | ✓ 6.7s | ✗ time@g2 | ✓ 6.8s | ✓ 5.5s |
| 54 | ✓ 7.5s | ✓ 7.0s | ✓ 7.3s | ✓ 6.0s |
| 55 | ✗ time@g2 | ✗ time@g2 | ✗ oob@g2 | ✗ time@g2 |
| 56 | ✓ 6.5s | ✓ 6.2s | ✓ 6.7s | ✓ 5.4s |
| 57 | ✓ 7.7s | ✓ 7.5s | ✓ 7.8s | ✗ coll@g3 |
| 58 | ✓ 7.0s | ✓ 6.9s | ✓ 7.2s | ✗ coll@g1 |
| 59 | ✓ 7.2s | ✓ 7.4s | ✓ 7.4s | ✓ 6.8s |
| 60 | ✗ time@g2 | ✓ 6.5s | ✓ 7.1s | ✓ 6.0s |
| 61 | ✓ 6.5s | ✓ 5.9s | ✓ 6.3s | ✓ 5.7s |
| 62 | ✓ 9.3s | ✗ coll@g3 | ✓ 7.7s | ✓ 7.0s |
| 63 | ✓ 6.9s | ✓ 6.3s | ✓ 6.9s | ✓ 5.7s |
| 64 | ✗ time@g2 | ✗ coll@g3 | ✓ 7.4s | ✓ 6.5s |
| 65 | ✓ 6.9s | ✓ 6.6s | ✓ 6.8s | ✓ 6.4s |
| 66 | ✓ 6.6s | ✓ 6.6s | ✓ 6.5s | ✓ 5.7s |
| 67 | ✓ 8.0s | ✓ 8.4s | ✓ 8.6s | ✓ 6.6s |
| 68 | ✓ 7.2s | ✓ 7.0s | ✓ 7.1s | ✓ 5.9s |
| 69 | ✓ 7.9s | ✓ 7.4s | ✓ 6.8s | ✓ 6.2s |
| 70 | ✗ coll@g1 | ✓ 5.8s | ✓ 6.4s | ✗ coll@g0 |
| 71 | ✓ 6.7s | ✓ 6.3s | ✓ 7.0s | ✓ 5.8s |
| 72 | ✓ 7.2s | ✓ 7.0s | ✓ 7.2s | ✓ 6.1s |
| 73 | ✓ 6.2s | ✗ coll@g1 | ✓ 6.7s | ✓ 5.9s |
| 74 | ✓ 6.4s | ✗ coll@g3 | ✓ 6.7s | ✓ 6.0s |
| 75 | ✗ oob@g2 | ✗ grnd@g3 | ✓ 6.7s | ✗ oob@g2 |
| 76 | ✓ 8.9s | ✓ 7.1s | ✓ 7.3s | ✓ 6.3s |
| 77 | ✓ 8.7s | ✓ 7.3s | ✓ 7.6s | ✗ coll@g0 |
| 78 | ✓ 8.4s | ✓ 7.5s | ✗ oob@g2 | ✓ 8.1s |
| 79 | ✓ 7.1s | ✓ 7.0s | ✓ 7.2s | ✓ 5.8s |
| 80 | ✗ time@g2 | ✗ time@g2 | ✓ 7.1s | ✗ coll@g1 |
| 81 | ✗ coll@g3 | ✓ 5.6s | ✓ 6.5s | ✓ 5.7s |
| 82 | ✓ 7.1s | ✓ 6.8s | ✓ 6.8s | ✓ 5.8s |
| 83 | ✓ 6.3s | ✓ 6.1s | ✓ 6.9s | ✓ 6.3s |
| 84 | ✓ 8.7s | ✓ 7.2s | ✓ 7.0s | ✗ coll@g2 |
| 85 | ✓ 7.4s | ✓ 7.5s | ✓ 7.3s | ✓ 5.9s |
| 86 | ✓ 8.4s | ✓ 7.6s | ✓ 8.3s | ✓ 7.1s |
| 87 | ✓ 6.9s | ✓ 6.9s | ✗ coll@g1 | ✗ coll@g1 |
| 88 | ✓ 7.6s | ✓ 7.0s | ✓ 7.2s | ✓ 6.0s |
| 89 | ✓ 8.9s | ✓ 8.4s | ✓ 9.4s | ✗ coll@g1 |
| 90 | ✗ grnd@g3 | ✓ 6.7s | ✓ 7.5s | ✓ 5.8s |
| 91 | ✓ 6.4s | ✓ 6.3s | ✓ 6.4s | ✓ 5.5s |
| 92 | ✓ 6.6s | ✓ 6.1s | ✓ 6.3s | ✓ 5.3s |
| 93 | ✓ 7.3s | ✓ 7.0s | ✓ 7.6s | ✓ 6.5s |
| 94 | ✓ 6.2s | ✓ 6.1s | ✓ 6.1s | ✓ 5.4s |
| 95 | ✓ 6.8s | ✓ 6.3s | ✓ 6.9s | ✓ 5.7s |
| 96 ⭐ | ✓ 17.1s | ✗ coll@g3 | ✗ coll@g2 | ✗ coll@g2 |
| 97 | ✗ coll@g3 | ✗ coll@g3 | ✓ 7.1s | ✓ 6.0s |
| 98 | ✓ 8.0s | ✓ 6.4s | ✓ 6.8s | ✗ coll@g1 |
| 99 | ✓ 7.4s | ✓ 7.4s | ✗ time@g2 | ✓ 6.2s |
| 100 | ✓ 6.7s | ✓ 6.6s | ✓ 6.6s | ✓ 5.4s |
| 101 | ✓ 7.2s | ✓ 7.2s | ✓ 7.4s | ✓ 5.9s |
| 102 | ✓ 7.3s | ✓ 6.6s | ✓ 6.9s | ✓ 6.2s |
| 103 | ✓ 7.0s | ✓ 7.2s | ✓ 7.1s | ✓ 6.0s |
| 104 | ✓ 8.1s | ✓ 7.5s | ✓ 7.1s | ✓ 6.5s |
| 105 | ✓ 6.7s | ✓ 6.6s | ✗ coll@g3 | ✓ 5.9s |
| 106 | ✓ 7.4s | ✓ 7.8s | ✗ coll@g3 | ✗ coll@g3 |
| 107 | ✓ 7.6s | ✓ 7.0s | ✓ 7.1s | ✗ coll@g0 |
| 108 | ✓ 7.1s | ✓ 6.9s | ✓ 7.0s | ✓ 6.1s |
| 109 | ✓ 6.5s | ✓ 5.8s | ✓ 6.0s | ✓ 5.8s |
| 110 | ✓ 6.3s | ✓ 6.1s | ✓ 6.2s | ✓ 5.6s |
| 111 | ✓ 6.9s | ✓ 6.4s | ✓ 6.8s | ✓ 6.6s |
| 112 | ✓ 6.9s | ✓ 7.1s | ✓ 6.9s | ✓ 5.8s |
| 113 | ✓ 7.3s | ✓ 6.6s | ✓ 7.4s | ✓ 6.4s |
| 114 | ✓ 7.3s | ✓ 6.9s | ✓ 7.1s | ✓ 6.4s |
| 115 | ✓ 7.6s | ✓ 7.2s | ✓ 7.4s | ✓ 6.0s |
| 116 | ✓ 6.6s | ✓ 6.8s | ✓ 6.5s | ✓ 6.4s |
| 117 | ✓ 6.9s | ✓ 6.9s | ✓ 6.9s | ✗ coll@g3 |
| 118 | ✓ 7.8s | ✓ 7.8s | ✓ 7.1s | ✓ 6.1s |
| 119 | ✓ 7.4s | ✓ 7.1s | ✓ 7.3s | ✓ 6.1s |
| 120 | ✓ 8.2s | ✓ 6.8s | ✓ 6.7s | ✓ 6.0s |
| 121 | ✗ coll@g3 | ✓ 7.1s | ✓ 7.0s | ✓ 5.5s |
| 122 | ✓ 7.1s | ✓ 6.8s | ✓ 7.0s | ✓ 8.4s |
| 123 | ✓ 7.1s | ✓ 6.7s | ✓ 7.2s | ✓ 5.9s |
| 124 | ✗ time@g2 | ✓ 7.0s | ✓ 6.9s | ✓ 5.9s |
| 125 | ✓ 6.2s | ✓ 6.2s | ✓ 6.3s | ✓ 5.5s |
| 126 | ✗ coll@g3 | ✓ 7.3s | ✓ 7.3s | ✓ 6.5s |
| 127 | ✓ 7.2s | ✓ 7.1s | ✓ 7.6s | ✓ 6.0s |
| 128 | ✓ 6.8s | ✓ 6.7s | ✓ 6.8s | ✓ 5.8s |
| 129 | ✓ 6.5s | ✓ 6.5s | ✓ 6.8s | ✓ 5.7s |
| 130 | ✗ coll@g3 | ✓ 6.8s | ✓ 6.8s | ✓ 6.5s |
| 131 | ✓ 5.9s | ✓ 6.1s | ✓ 6.2s | ✓ 5.4s |
| 132 | ✓ 6.9s | ✓ 6.6s | ✓ 6.6s | ✓ 6.1s |
| 133 | ✓ 6.6s | ✓ 6.6s | ✓ 7.3s | ✓ 5.8s |
| 134 | ✓ 7.1s | ✓ 6.9s | ✓ 6.8s | ✓ 5.9s |
| 135 | ✗ coll@g0 | ✓ 7.1s | ✓ 7.6s | ✗ coll@g1 |
| 136 | ✓ 7.2s | ✓ 7.0s | ✓ 7.1s | ✓ 6.7s |
| 137 | ✓ 6.1s | ✓ 6.0s | ✓ 6.3s | ✓ 5.5s |
| 138 | ✓ 6.9s | ✓ 6.7s | ✓ 8.7s | ✓ 5.7s |
| 139 | ✗ coll@g3 | ✓ 7.1s | ✓ 7.4s | ✗ coll@g2 |
| 140 | ✓ 6.6s | ✗ oob@g3 | ✓ 6.6s | ✓ 5.6s |
| 141 | ✓ 8.3s | ✓ 7.7s | ✓ 7.6s | ✗ coll@g1 |
| 142 | ✓ 9.4s | ✓ 8.5s | ✓ 7.9s | ✓ 6.5s |
| 143 | ✓ 7.5s | ✓ 7.1s | ✓ 6.6s | ✓ 6.6s |
| 144 | ✓ 6.7s | ✓ 6.5s | ✓ 7.4s | ✓ 6.1s |
| 145 | ✓ 6.9s | ✗ coll@g2 | ✓ 6.6s | ✓ 6.4s |
| 146 ⭐ | ✓ 9.0s | ✗ coll@g2 | ✗ coll@g2 | ✗ coll@g2 |
| 147 | ✓ 7.5s | ✓ 7.5s | ✓ 7.9s | ✓ 6.3s |
| 148 | ✗ time@g3 | ✓ 7.1s | ✓ 7.5s | ✓ 6.7s |
| 149 | ✓ 7.1s | ✓ 6.2s | ✓ 7.6s | ✓ 6.1s |
| 150 ⭐ | ✓ 7.6s | ✗ coll@g3 | ✗ time@g2 | ✗ coll@g3 |
| 151 | ✓ 7.2s | ✓ 6.6s | ✓ 7.1s | ✗ coll@g0 |
| 152 | ✓ 7.0s | ✓ 6.7s | ✓ 6.7s | ✓ 5.7s |
| 153 | ✓ 6.0s | ✓ 6.2s | ✗ time@g2 | ✓ 5.4s |
| 154 | ✓ 6.9s | ✓ 6.7s | ✓ 6.9s | ✓ 5.6s |
| 155 | ✓ 5.8s | ✓ 5.8s | ✗ coll@g2 | ✓ 5.4s |
| 156 | ✗ coll@g3 | ✓ 6.9s | ✓ 7.2s | ✓ 6.5s |
| 157 | ✓ 6.5s | ✓ 6.2s | ✓ 6.3s | ✓ 6.5s |
| 158 | ✓ 6.6s | ✓ 6.7s | ✓ 6.3s | ✓ 5.6s |
| 159 | ✗ time@g2 | ✓ 6.3s | ✓ 6.1s | ✗ time@g2 |
| 160 | ✓ 7.3s | ✓ 7.0s | ✓ 6.8s | ✓ 10.2s |
| 161 | ✓ 7.2s | ✓ 7.0s | ✓ 7.1s | ✓ 6.2s |
| 162 | ✓ 6.2s | ✓ 6.1s | ✓ 6.3s | ✓ 5.5s |
| 163 | ✓ 7.5s | ✗ coll@g2 | ✓ 7.2s | ✗ coll@g3 |
| 164 | ✓ 8.2s | ✓ 7.3s | ✗ coll@g3 | ✓ 7.2s |
| 165 | ✓ 6.7s | ✓ 6.5s | ✓ 6.8s | ✓ 5.6s |
| 166 | ✓ 7.2s | ✓ 7.0s | ✓ 7.2s | ✓ 5.9s |
| 167 | ✗ time@g2 | ✓ 7.6s | ✓ 7.6s | ✓ 6.7s |
| 168 | ✓ 8.1s | ✓ 7.3s | ✓ 7.6s | ✓ 6.9s |
| 169 ⭐ | ✓ 8.1s | ✗ coll@g3 | ✗ coll@g3 | ✗ coll@g3 |
| 170 | ✓ 7.7s | ✗ coll@g3 | ✗ coll@g3 | ✓ 6.4s |
| 171 | ✓ 6.9s | ✗ coll@g3 | ✓ 6.8s | ✓ 5.6s |
| 172 | ✓ 7.8s | ✓ 7.3s | ✓ 8.5s | ✓ 7.0s |
| 173 | ✓ 6.6s | ✓ 6.1s | ✓ 6.5s | ✓ 5.6s |
| 174 | ✓ 6.5s | ✓ 6.5s | ✓ 6.7s | ✓ 5.5s |
| 175 | ✓ 7.3s | ✓ 7.2s | ✓ 8.2s | ✓ 5.9s |
| 176 | ✓ 6.8s | ✓ 7.0s | ✓ 7.0s | ✓ 6.5s |
| 177 | ✓ 7.8s | ✓ 7.9s | ✓ 7.9s | ✗ coll@g0 |
| 178 | ✓ 6.3s | ✓ 6.0s | ✓ 6.4s | ✓ 5.6s |
| 179 | ✓ 6.4s | ✓ 7.1s | ✓ 7.2s | ✓ 5.8s |
| 180 | ✓ 6.6s | ✓ 6.4s | ✓ 6.9s | ✓ 6.8s |
| 181 | ✓ 6.9s | ✓ 6.8s | ✓ 6.9s | ✓ 6.0s |
| 182 | ✓ 7.6s | ✓ 6.9s | ✓ 7.0s | ✗ coll@g1 |
| 183 | ✓ 6.7s | ✓ 6.8s | ✓ 6.9s | ✗ coll@g3 |
| 184 | ✗ coll@g1 | ✓ 7.1s | ✓ 7.2s | ✓ 5.6s |
| 185 | ✓ 7.2s | ✓ 6.9s | ✓ 7.3s | ✓ 6.1s |
| 186 | ✓ 7.7s | ✓ 7.2s | ✓ 7.3s | ✓ 6.4s |
| 187 | ✓ 7.1s | ✓ 7.0s | ✗ coll@g3 | ✓ 6.2s |
| 188 | ✓ 6.3s | ✓ 6.1s | ✓ 6.5s | ✓ 5.4s |
| 189 | ✓ 6.4s | ✓ 6.2s | ✓ 6.3s | ✓ 6.4s |
| 190 | ✓ 6.1s | ✓ 5.8s | ✓ 7.1s | ✓ 5.1s |
| 191 | ✓ 7.3s | ✓ 7.5s | ✓ 7.8s | ✓ 6.5s |
| 192 | ✓ 7.8s | ✓ 7.5s | ✓ 7.3s | ✓ 6.7s |
| 193 | ✓ 7.3s | ✓ 6.8s | ✓ 6.9s | ✓ 5.9s |
| 194 | ✓ 7.6s | ✓ 7.2s | ✓ 7.4s | ✗ coll@g1 |
| 195 | ✓ 6.8s | ✓ 6.1s | ✓ 6.9s | ✓ 5.5s |
| 196 | ✓ 7.3s | ✓ 7.6s | ✓ 7.7s | ✓ 7.5s |
| 197 | ✓ 7.2s | ✓ 6.9s | ✓ 7.1s | ✗ coll@g3 |
| 198 | ✓ 7.0s | ✓ 6.9s | ✓ 7.7s | ✓ 6.1s |
| 199 | ✓ 7.6s | ✗ oob@g2 | ✓ 9.1s | ✓ 6.2s |
| 200 | ✓ 8.1s | ✓ 8.6s | ✓ 7.3s | ✓ 7.6s |
