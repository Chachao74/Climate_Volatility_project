#!/usr/bin/env python3
"""
CLIMATE TRADING BOT - LANCEMENT COMPLET
========================================
Lance le bot de stratégie avec mise à jour météo live et génération de rapport complet.

Usage:
    python3 bot/run_strategy.py
"""

import sys
import subprocess
from pathlib import Path

def main():
    print("🚀 Lancement du Climate Trading Bot...")
    print("="*80)
    
    # 1. Vérifier données live
    live_data = Path('bot/live_data.csv')
    if not live_data.exists():
        print("⚠️  Pas de données live trouvées. Récupération...")
        result = subprocess.run(['python3', 'bot/live_weather_fetcher.py'], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Erreur lors de la récupération: {result.stderr}")
            return 1
    
    # 2. Lancer stratégie
    print("\n🤖 Génération des signaux de trading...")
    result = subprocess.run(['python3', 'bot/strategy_bot.py'], 
                          capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    if result.returncode == 0:
        print("\n✅ Stratégie générée avec succès!")
        print("📄 Rapport: bot/STRATEGY_REPORT.md")
        print("📊 Signaux: bot/strategy_signals.csv")
    else:
        print("\n❌ Erreur lors de la génération")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
