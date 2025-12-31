# 🌍 Climate Trading Bot - Documentation

Bot de trading automatisé utilisant données climatiques et modèles ML pour prédire la volatilité de 7 secteurs de marché.

## 📊 Secteurs Couverts

| Secteur | Ticker | Modèle | Directional Accuracy | Sharpe |
|---|---|---|---|---|
| **Financials** | XLF | LSTM | 72% | 1.75 |
| **Consumer Staples** | XLP | LSTM | 72% | 1.58 |
| **Technology** | XLK | LSTM | 72% | 1.53 |
| **SP500** | SPY | LSTM | 71% | 1.63 |
| **Utilities** | XLU | RF | 75% | 1.90 |
| **Materials** | XLB | RF | 68% | 1.70 |
| **Agriculture** | DBA | LSTM Global | 90% | - |

## 🚀 Utilisation

### Lancement Rapide
```bash
python3 bot/run_strategy.py
```

### Étape par Étape

1. **Récupérer données météo**:
```bash
python3 bot/live_weather_fetcher.py
```

2. **Générer signaux**:
```bash
python3 bot/strategy_bot.py
```

3. **Voir rapport**:
```bash
cat bot/STRATEGY_REPORT.md
```

## 📁 Structure

```
bot/
├── run_strategy.py          # 🚀 Lancement principal
├── strategy_bot.py           # 🤖 Moteur de stratégie
├── live_weather_fetcher.py   # 🌍 Récupération météo (Open-Meteo API)
├── models/                   # 📦 Modèles ML (LSTM + RF)
│   ├── lstm_*.h5
│   ├── rf_*.pkl
│   └── scaler_*.pkl
├── STRATEGY_REPORT.md        # 📄 Rapport généré
└── strategy_signals.csv      # 📊 Signaux CSV
```

## 🌐 Données Météo

**Source**: Open-Meteo Historical Weather API (gratuit)
- **Fréquence**: Quotidienne
- **Couverture**: 188 pays
- **Variables**: Température (t2m), Précipitations (tp), Désastres (EM-DAT)
- **Historique**: 90 derniers jours

## 📈 Signaux de Trading

### Types de Signaux
- **LONG**: Volatilité basse prédite → Environnement stable, opportunité d'achat
- **SHORT/HEDGE**: Volatilité élevée → Alerte risque, protection recommandée  
- **NEUTRAL**: Volatilité modérée → Maintenir position

### Niveaux de Confiance
- **HIGH**: DA > 70%
- **MEDIUM**: DA 60-70%

## 🔧 Configuration

Modifiez `bot/strategy_bot.py` pour:
- Ajuster seuils de volatilité
- Changer ISOs météo
- Ajouter nouveaux secteurs

## 📊 Outputs

### strategy_signals.csv
```csv
sector,ticker,model,predicted_rv,signal,confidence,da
Financials,XLF,LSTM,0.0180,LONG,HIGH,0.72
...
```

### STRATEGY_REPORT.md
Rapport markdown avec:
- Tableau des signaux
- Rationale par secteur
- Timestamp de génération

## ⚙️ Requirements

```
python >= 3.9
pandas
numpy
tensorflow
scikit-learn
requests
yfinance
```

## 📝 Notes

- **Données Live**: Mise à jour auto ou manuelle via `live_weather_fetcher.py`
- **Fréquence**: Recommandé 1x/semaine (vendredi soir après bourse US)
- **Backtest**: Non inclus (utiliser résultats historiques dans `results/`)

## 🎯 Performance Historique

Voir dossiers individuels:
- `results/LSTM SECTORS/` - Métriques LSTM
- `results/RANDOM FOREST/` - Métriques RF
- `Agriculture/` - Analyse Agriculture complète

---

**Développé avec ERA5 Climate Data + EM-DAT Disasters + ML**
