# Git Workflow - Wichtig für dich und deinen Team

## 📋 Was wurde aufgeräumt:

✅ **`.gitignore` erweitert** - folgende Dateien werden jetzt ignoriert:
- `venv_drone/` - deine lokale virtuelle Umgebung
- `venv/`, `env/` - andere venv-Namen
- `.DS_Store` - macOS Systemdateien
- `__pycache__/`, `.pytest_cache/` - Python Build-Dateien
- IDE-Dateien (`.vscode/`, `.idea/`)

✅ **.DS_Store aus Git entfernt** - diese macOS-Dateien werden nicht mehr getracked

✅ **pixi.lock gelöscht** - wird bei Bedarf automatisch regeneriert

## 🎯 Was du pushen solltest:

### ✅ Pushen (für Lehrstuhl + Team):
```
lsy_drone_racing/control/our_controller.py  ← DEIN CODE (wichtig!)
config/level*.toml                           ← Deine Konfigurationen
tests/                                        ← Deine Tests
pyproject.toml                                ← Dependencies (ORIGINAL)
.gitignore                                    ← Aufgeräumt ✅
MACOS_SETUP.md                                ← Optional (für Dokumentation)
setup_mac.sh                                  ← Optional (hilfreiche Tools)
quick_test.sh                                 ← Optional (Test-Skript)
```

### ❌ NICHT pushen (lokal nur):
```
venv_drone/                                   ← .gitignore ✅
.DS_Store                                     ← .gitignore ✅
__pycache__/                                  ← .gitignore ✅
.vscode/                                      ← .gitignore ✅
```

## 🧹 Git Cleanup durchführen (optional aber empfohlen):

```bash
# 1. Aktuelle Änderungen committen
git add .gitignore config/level0.toml
git commit -m "cleanup: Update .gitignore and remove macOS files"

# 2. Oder wenn du nur .gitignore willst:
git add .gitignore
git commit -m "cleanup: Improve .gitignore for Python venvs and macOS"

# 3. Optional: Neue Setup-Dateien hinzufügen
git add MACOS_SETUP.md setup_mac.sh quick_test.sh
git commit -m "docs: Add macOS setup instructions and helper scripts"
```

## 📊 Git Status NACH Aufräumen:

```
Unversionierte Dateien:
  MACOS_SETUP.md        ← Du kannst das pushen oder ignorieren
  quick_test.sh         ← Du kannst das pushen oder ignorieren
  setup_mac.sh          ← Du kannst das pushen oder ignorieren
  venv_drone/           ← Wird durch .gitignore ignoriert ✅

Geänderte Dateien:
  config/level0.toml    ← Pushen (deine Änderungen)
  pyproject.toml        ← (Wir haben das NICHT geändert, original!)
```

## 🔄 Branches für Team-Arbeit

Die Lehrstuhl wird wahrscheinlich nur die **Originalversion** (keine venv_drone/ Dateien) erwarten!
