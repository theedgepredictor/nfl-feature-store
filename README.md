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

**Season**
- Coming Soon

**Game**
- Coming Soon

****

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


