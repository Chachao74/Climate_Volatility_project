# Climate Trading Bot - Résultats de Stratégie

**Date**: 5 Décembre 2025, 15:32

## 📊 Vue d'Ensemble

Ce dossier contient les résultats de la stratégie de trading automatisée basée sur 7 secteurs performants.

## 📁 Fichiers

### 1. `STRATEGY_REPORT.md`
Rapport complet markdown avec:
- Signaux de trading pour chaque secteur
- Analyse détaillée par secteur
- Résumé des positions recommandées
- Rationale pour chaque signal

### 2. `strategy_signals.csv`
Données exportables contenant:
- Secteur, Ticker, Modèle utilisé
- Volatilité prédite (RV)
- Signal (LONG/SHORT/HEDGE/NEUTRAL)
- Niveau de confiance
- Directional Accuracy historique
- Rationale

### 3. `README.md`
Documentation complète du bot incluant:
- Description des 7 secteurs
- Instructions d'utilisation
- Structure du code
- Configuration

## 🎯 Résumé des Signaux (5 Dec 2025)

**🟢 LONG (1)**: Agriculture (DBA) - RV prédit: 0.97%, Confiance: HIGH (90% DA)

**🔴 SHORT/HEDGE (2)**: 
- Consumer Staples (XLP) - RV prédit: 2.38%, Confiance: HIGH (72% DA)
- SP500 (SPY) - RV prédit: 4.00%, Confiance: HIGH (71% DA)

**⚪ NEUTRAL (4)**: Financials, Utilities, Materials, Technology

## 🌍 Données Utilisées

- **Source**: Open-Meteo Historical Weather API
- **Période**: 6 septembre - 5 décembre 2025 (91 jours)
- **Couverture**: 188 pays
- **Variables**: Température (t2m), Précipitations (tp)

## 🤖 Modèles

| Secteur | Modèle | DA Historique |
|---|---|---|
| Financials | LSTM | 72% |
| Consumer Staples | LSTM | 72% |
| SP500 | LSTM | 71% |
| Utilities | RF | 75% |
| Materials | RF | 68% |
| Technology | RF | 70% |
| Agriculture | LSTM Global | 90% |

## 📈 Performance Historique

Voir dossiers:
- `results/LSTM SECTORS/` - Métriques LSTM détaillées
- `results/RANDOM FOREST/` - Métriques RF détaillées
- `Agriculture/` - Analyse complète Agriculture

## 🔄 Mise à Jour

Pour générer de nouveaux signaux:
```bash
python3 bot/run_strategy.py
```

---
*Généré automatiquement par Climate Trading Bot*
