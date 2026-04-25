# macOS Setup Guide - LSY Drone Racing

## ✅ Was wurde bereits installiert?

Auf deinem Mac wurde folgendes eingerichtet:

### ✓ Automatische Installation durchgeführt:
- Python 3.12 Virtual Environment (`venv_drone/`)
- Alle Simulation-Abhängigkeiten:
  - **Gymnasium** - RL Environment
  - **JAX** - Numerische Berechnungen (CPU-Version für macOS)
  - **CrazyFlow** - Drone Simulator
  - **Drone-Models & Controllers** - Drohnen-Modelle
  - **Warp-Lang** - Performance-optimierte Simulationen
  - **ACADOS** Python Interface - Optimal Control

## 🚀 Simulation starten

### 1. Aktiviere die Virtual Environment:

```bash
cd /Users/timomatuszewski/Desktop/SS26/Autonomous_Drone_Racing_Project_Course/lsy_drone_racing
source venv_drone/bin/activate
```

### 2. Starte die Simulation:

```bash
# Einfacher Test (ohne Visualisierung)
python scripts/sim.py --config level0.toml --render=False

# Mit Rendering (Visualisierung der Drohne)
python scripts/sim.py --config level0.toml --render=True

# Mit benutzerdefinierten Einstellungen
python scripts/sim.py --config level0.toml --render=True --n_runs=1
```

## 📊 Verfügbare Schwierigkeitsstufen

| Config | Schwierigkeit | Beschreibung |
|--------|---------------|------------|
| `level0.toml` | ⭐ Einfach | Perfektes Wissen über die Umgebung |
| `level1.toml` | ⭐⭐ Mittel | Adaptive Kontrolle |
| `level2.toml` | ⭐⭐⭐ Schwer | Re-planning erforderlich |
| `level3.toml` | ⭐⭐⭐⭐ Experte | Online Planning |

## 📝 Beispiele

### Test ausführen (30 Sekunden Simulation):
```bash
python scripts/sim.py --config level0.toml --n_runs=1 --render=False
```

**Erwartete Ausgabe:**
```
INFO:__main__:Flight time (s): 13.38
Finished: True
Gates passed: 4
```

### Mehrere Läufe durchführen:
```bash
python scripts/sim.py --config level1.toml --n_runs=5 --render=False
```

### Mit deinem eigenen Controller:
```bash
python scripts/sim.py --config level0.toml --controller=attitude_controller.py
```

## 🎮 Verfügbare Controller

Die folgenden Controller sind implementiert:

| Controller | Beschreibung |
|-----------|------------|
| `attitude_controller.py` | PID-basierter Controller mit Haltungsregelung |
| `state_controller.py` | State-basierter Controller |
| `our_controller.py` | MPC-basierter Controller (benötigt ACADOS) |

## 🔧 Fehlerbehebung

### Problem: JAX Warnung "overflow encountered in cast"
**Lösung:** Das ist eine Warnung, kein Fehler. Die Simulation läuft trotzdem korrekt.

### Problem: "No module named 'acados_template'"
**Lösung:** ACADOS wurde möglicherweise nicht korrekt installiert. Führe aus:
```bash
pip install -e acados/interfaces/acados_template
```

### Problem: Rendering öffnet kein Fenster
**Lösung:** Auf macOS funktioniert OpenGL-Rendering in Terminal-Sessions manchmal nicht. Verwende `--render=False` oder führe das Skript aus der Anwendung aus.

### Problem: "cannot execute binary file" für t_renderer
**Lösung:** Das ist ein macOS-Kompatibilitätsproblem mit ACADOS. Verwende einen anderen Controller wie `attitude_controller.py`.

## 📚 Weitere Ressourcen

- [Dokumentation](../docs/getting_started/general.rst)
- [Projektrepository](https://github.com/utiasDSL/lsy_drone_racing)
- [CrazyFlow Simulator](https://github.com/utiasDSL/crazyflow)

## ⚡ Performance-Tipps

1. **GPU-Beschleunigung nicht verfügbar auf macOS** - Die Installation nutzt nur CPU (JAX CPU)
2. **Mehrere Cores nutzen** - JAX nutzt automatisch alle verfügbaren CPU-Cores
3. **Rendering deaktivieren** für schnellere Simulationen

## 🛠️ Wiederherstellen der Installation

Falls etwas schiefgeht, kannst du die Installation neu starten:

```bash
# Virtual Environment löschen
rm -rf venv_drone/

# Neu installieren
./setup_mac.sh
```

## ✅ Installation verifizieren

```bash
# Teste, ob alles funktioniert
python scripts/sim.py --config level0.toml --render=False --n_runs=1
```

**Erfolg:** Du solltest eine Ausgabe wie diese sehen:
```
Flight time (s): 13.38
Finished: True
Gates passed: 4
```

---

**Letzte Aktualisierung:** 25. April 2026  
**macOS Version:** Getestet auf M-Serie Mac (ARM64) mit Python 3.12
