# Kollege mit Linux/Windows (WSL) - Kompatibilitätsguide

## ✅ Gute Nachrichten - WSL ist perfekt!

Dein Kollege mit **Linux auf Windows (WSL)** hat **KEINE Probleme**:

### 🎯 WSL Setup für deinen Kollegen:

**Schritt 1-2: Gleich wie auf deinem Mac**
```bash
# Repository klonen
git clone https://github.com/<USERNAME>/lsy_drone_racing.git
cd lsy_drone_racing

# Python 3.12 nutzen (oder 3.11)
python3.12 -m venv venv_drone

# Environment aktivieren
source venv_drone/bin/activate

# Installation (IDENTISCH mit macOS)
pip install --upgrade pip setuptools wheel
pip install fire numpy toml 'gymnasium[array-api]>=1.2.0' \
    'ml-collections>=1.0' 'packaging>=24.0' 'jax[cpu]>=0.7' \
    drone-models drone-controllers crazyflow warp-lang

# ACADOS (optional, aber wichtig für Lehrstuhl)
pip install -e acados/interfaces/acados_template
pip install -e .
```

## 📊 Unterschiede Mac vs. Linux/WSL

| Aspekt | macOS | Linux/WSL | Problem? |
|--------|-------|-----------|---------|
| **Python Version** | 3.12 | 3.11, 3.12, 3.13 | ✅ NEIN |
| **JAX Backend** | CPU | CPU oder GPU! | ✅ NEIN (sogar Vorteil) |
| **Gymnasium** | JAX + CPU | JAX + CPU/GPU | ✅ NEIN |
| **CrazyFlow** | Funktioniert | Funktioniert besser | ✅ NEIN |
| **Rendering** | Software | GPU-beschleunigt möglich | ✅ NEIN (Vorteil) |
| **pyproject.toml** | Original | Original | ✅ NEIN - selbe Datei! |
| **Git Kompatibilität** | ✅ | ✅ | ✅ NEIN |

## 🚀 Warum Linux/WSL sogar BESSER ist:

1. **GPU Support** - Er kann `jax[cuda12]` installieren (schneller!)
2. **Lehrstuhl-Kompatibilität** - Server nutzen auch Linux
3. **Keine macOS Spezialprobleme** - Keine .DS_Store, keine ACADOS t_renderer Fehler
4. **Deploy auf echte Drohne** - Funktioniert nahtlos
5. **Online-Tests** - Server-Umgebung ist identisch

## 🌳 Braucht er einen eigenen Branch?

### **NEIN, solange:**

✅ Beide nutzen **denselben Code** (dein `our_controller.py`)  
✅ Beide arbeiten an **unterschiedlichen Features/Teilen**  
✅ Die **pyproject.toml bleibt original**  

### **JA, er braucht einen Branch, wenn:**

❌ Er andere Controller testet (z.B. `his_controller.py`)  
❌ Er die Dependencies ändert  
❌ Er experimentiert (GPU-Versionen, etc.)  

## 📋 Empfohlenes Team-Setup:

```
Repository: lsy_drone_racing
├── main                          (Original vom Lehrstuhl)
├── development                   (Gemeinsam arbeitend)
│   └── DEIN Controller: our_controller.py
├── feature/timomatuszewski       (Deine Features)
│   └── Nur DEIN Code
└── feature/kollege               (Sein Features)
    └── Nur SEIN Code
```

### Branch Workflow:

```bash
# 1. Development Branch nutzen (gemeinsam)
git checkout development

# 2. Dein Feature-Branch für Experimente
git checkout -b feature/timomatuszewski
# ... dein Code ...
git push origin feature/timomatuszewski

# 3. Sein Feature-Branch für Experimente
git checkout -b feature/kollege
# ... sein Code ...
git push origin feature/kollege

# 4. Finaler Code mergen in development
git checkout development
git merge feature/timomatuszewski
git push origin development
```

## ✅ Checklist für deinen Kollegen:

- [ ] WSL ist installiert (Windows Terminal mit WSL 2)
- [ ] Python 3.12 oder 3.11 verfügbar
- [ ] `git clone` Repository
- [ ] `venv_drone/` erstellt und aktiviert
- [ ] Abhängigkeiten installiert (pip install ...)
- [ ] Test laufen lassen: `python scripts/sim.py --config level0.toml`
- [ ] Sein Code in `lsy_drone_racing/control/` erstellen

## 🎯 Wichtig: .gitignore nutzen!

Dank deines aktualisierten `.gitignore`:
- Er kann auch `venv_drone/` haben (wird ignoriert)
- Er kann auch `.DS_Store` haben auf WSL (wird ignoriert)
- Sein Repository bleibt sauber ✅

## ⚠️ Häufige WSL Probleme (NICHT relevant für euch):

Diese Probleme sollte er **NICHT** haben:
- ❌ "Cannot find gcc" → Linux hat alles
- ❌ "Module not found" → Pip funktioniert besser auf Linux
- ❌ "Permission denied" → WSL 2 hat volle Permissions

## 🚀 Tipp für schnellere Tests

Wenn dein Kollege GPU hat (RTX/Tesla):

```bash
# GPU-Version installieren
pip install 'jax[cuda12]>=0.7'

# Tests werden ~10-50x schneller! 🚀
python scripts/sim.py --config level2.toml  # Läuft viel schneller!
```

---

**Fazit:** Ihr könnt beide dasselbe Repository nutzen, **solange** der Hauptcode (`our_controller.py`, `pyproject.toml`) gleich bleibt. WSL hat sogar Vorteile! 🎉
