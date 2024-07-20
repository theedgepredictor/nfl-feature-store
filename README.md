# nfl-feature-store

### Data Pipeline

**Collect Player data and dump as parquet**
- Add career stats as attributes
- Add Player Depth Chart Positions as attributes
- Add NGS info as attributes
- Add PFR advanced stats as attributes
- Add Madden Ratings as attributes

### Backend

**Hosted ML Model**

### Frontend

**Streamlit App**

Will be broken into its own microservices but currently were just going to build out a PoC. 

1. Make a baseline model that uses a function to estimate points
2. Make a way to load a position, a team and look at all the features for a specific player
3. Make a 'Create a Model' button to allow a user to select features from the feature store and run those through a simple RandomForest Regressor
4. Add a way to save these attributes which will download the .txt of the attributes to the users computer. Now they have a way to reload their model
5. Once model is trained we will need to show feature attributions