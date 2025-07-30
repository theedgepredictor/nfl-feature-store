# nfl-feature-store

****

[![Feature Store Data trigger](https://github.com/theedgepredictor/nfl-feature-store/actions/workflows/feature_store_data_trigger.yaml/badge.svg)](https://github.com/theedgepredictor/nfl-feature-store/actions/workflows/feature_store_data_trigger.yaml)

## ETL Process for generating NFL Feature Stores for downstream ML models

****


## Event Based Feature Stores

**Regular Season Game**
- Pre Game Elo Rating
- Vegas Lines
- Rolling Avg EPA (last 10)
- Rolling Avg Points (last 10)
- Avg Vegas Cover (last 10)

**Post Season Game**
- Coming Soon

****

## Player Based Feature Stores

This repo builds feature stores for NFL players and teams, suitable for ML and analytics. Key modules:

### Main Components

- **src/components/player_state.py**
  - `PlayerDataComponent`: Extracts, merges, and builds weekly states for all NFL players. Handles roster, starter, injury, depth chart, and game participation data. For PoC, focuses on QBs.

- **src/components/player_rating.py**
  - `PlayerRatingComponent`: Calculates weekly and overall ratings for players (PoC: QBs) using normalized stats, rolling averages, and regression to mean. Integrates Madden ratings and performance metrics.

- **src/components/game.py**
  - `GameComponent`: Processes game-level features, including rolling averages, Vegas lines, and Elo ratings for each matchup.

- **src/components/team.py**
  - `TeamComponent`: Aggregates team-level features and stats for modeling and analysis.

- **src/extracts/elo.py**
  - Functions to fetch and project team and QB Elo ratings, including logic for future weeks and regression to mean.

- **src/transforms/player.py**
  - Utility functions for player feature engineering, rating imputation, and preseason adjustments.

### Data Sources
- Play by Play: [NFLVerse](https://github.com/nflverse/nflverse-data)
- Schedule: [NFLGameData](http://www.habitatring.com/schedule.php)
- ELO: [TheEdgePredictor](https://github.com/theedgepredictor/elo-rating)

### Wishful Thinking Data Sources

**Collect Player data and dump as parquet**
- Add career stats as attributes
- Add Player Depth Chart Positions as attributes
- Add NGS info as attributes
- Add PFR advanced stats as attributes
- Add Madden Ratings as attributes

---


## Folder Structure

```
nfl-feature-store/
│
├── data/                      # Raw and processed data storage
│   ├── feature_store/         # Feature store outputs
│   └── pump/                  # Pumped/intermediate data
│
├── src/                       # Main source code for feature engineering
│   ├── components/            # High-level feature store builders (player, team, game)
│   │   ├── player_state.py    # Builds weekly player states (PoC: QBs)
│   │   ├── player_rating.py   # Calculates player ratings (PoC: QBs)
│   │   ├── game.py            # Game-level feature engineering
│   │   ├── team.py            # Team-level feature engineering
│   │   └── ...                # Other components
│   ├── extracts/              # Data extraction utilities (Elo, games, stats, etc.)
│   ├── transforms/            # Feature engineering utilities (player, game, stats)
│   ├── formatters/            # Data formatting and cleaning helpers
│   ├── feature_stores/        # Feature store creation scripts
│   ├── pumps/                 # Data pumping utilities
│   ├── utils.py               # General utility functions
│   ├── consts.py              # Constants and config
│   └── data_types.py          # Data type definitions
│
├── feature_store_runner.py    # Main runner script for ETL pipeline
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── .github/                   # GitHub Actions and CI/CD configs
```

---



