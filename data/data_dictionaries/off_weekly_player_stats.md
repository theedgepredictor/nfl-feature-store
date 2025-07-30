### Off Weekly Player Stats

- player feature store
- Weekly offensive performance metrics
- Contains game-level statistics
- Used for rating calculations
- Stats include: passing_yards, passing_tds, interceptions, completion_pct
- Different aggregations (season_avg, season_total, form) see below

| Prefix          | Meaning                                                                                                                                  | Example                      |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------- |
| `season_avg_`   | Season‑to‑date average of the underlying metric. On **Week 1**, this is the *previous* season’s full‑season average (useful for priors). | `season_avg_passing_yards`   |
| `season_total_` | Season‑to‑date cumulative sum. On **Week 1**, this is the prior season’s full‑season total.                                              | `season_total_rushing_tds`   |
| `form_`         | Rolling average over the **last 5 games** (or player‑career average if fewer than five).                                                 | `form_completion_percentage` |


| column                                    | dtype   | description                                                         |
| ----------------------------------------- | ------- | ------------------------------------------------------------------- |
| `position_group`                          | object  | Mid‑level positional bucket—e.g., Skill (QB/RB/WR/TE), Line (OL/DL), Secondary, etc. |
| `player_id`                               | object   | Stable identifier for the player across seasons and teams.          |
| `season`                                  | int64   | Season year (e.g., 2024).                                           |
| `week`                                    | int64   | Week number within the season (1 – 18, incl. playoffs if present).  |
| `completions`                             | int64   | Number of completed passes.                                         |
| `attempts`                                | int64   | Number of pass attempts.                                            |
| `passing_yards`                           | int64   | Total passing yards.                                                |
| `passing_tds`                             | int64   | Passing touchdowns.                                                 |
| `interceptions`                           | int64   | Interceptions thrown.                                               |
| `sacks`                                   | int64   | Times sacked.                                                       |
| `sack_yards`                              | int64   | Yards lost on sacks.                                                |
| `sack_fumbles`                            | int64   | Fumbles occurring on sacks.                                         |
| `sack_fumbles_lost`                       | int64   | Sack fumbles lost to defense.                                       |
| `passing_air_yards`                       | int64   | Air yards on pass attempts.                                         |
| `passing_yards_after_catch`               | int64   | Yards after catch on completed passes.                              |
| `passing_first_downs`                     | int64   | Pass completions resulting in first downs.                          |
| `passing_epa`                             | float64 | Expected Points Added from passing plays.                           |
| `passing_2pt_conversions`                 | int64   | Successful 2‑pt conversions via pass.                               |
| `pacr`                                    | float64 | Pass‑catch‑adjusted ratio (air‑yard conversion efficiency).         |
| `dakota`                                  | float64 | Composite QB efficiency metric (Dakota).                            |
| `avg_time_to_throw`                       | float64 | Avg. time from snap to pass (seconds).                              |
| `avg_completed_air_yards`                 | float64 | Avg. air yards on completed passes.                                 |
| `avg_intended_air_yards_passing`          | float64 | Avg. intended air yards per attempt.                                |
| `avg_air_yards_differential`              | float64 | Intended minus completed air yards.                                 |
| `aggressiveness`                          | float64 | Share of passes into tight windows.                                 |
| `max_completed_air_distance`              | int64   | Longest completed air distance (yards).                             |
| `avg_air_yards_to_sticks`                 | float64 | Avg. air yards relative to the 1st‑down marker.                     |
| `passer_rating`                           | float64 | Traditional NFL passer rating.                                      |
| `VALUE_ELO`                               | float64 | ELO‑style QB value metric.                                          |
| `completion_percentage`                   | float64 | Pass completion percentage.                                         |
| `expected_completion_percentage`          | float64 | Model‑expected completion percentage.                               |
| `completion_percentage_above_expectation` | float64 | Completion % minus expected %.                                      |
| `avg_air_distance`                        | float64 | Avg. ball‑flight distance.                                          |
| `max_air_distance`                        | float64 | Longest air distance on any pass.                                   |
| `net_passing_yards`                       | int64   | Passing yards net of sack yardage.                                  |
| `yards_per_pass_attempt`                  | float64 | Yards gained per pass attempt.                                      |
| `sack_rate`                               | float64 | Sacks divided by dropbacks.                                         |
| `air_yards_per_pass_attempt`              | float64 | Air yards per attempt.                                              |
| `carries`                                 | int64   | Rushing attempts.                                                   |
| `rushing_yards`                           | int64   | Rushing yards gained.                                               |
| `rushing_tds`                             | int64   | Rushing touchdowns.                                                 |
| `rushing_fumbles`                         | int64   | Fumbles on rushing plays.                                           |
| `rushing_fumbles_lost`                    | int64   | Rushing fumbles lost.                                               |
| `rushing_first_downs`                     | int64   | Rushes resulting in first downs.                                    |
| `rushing_epa`                             | float64 | Expected Points Added from rushing.                                 |
| `rushing_2pt_conversions`                 | int64   | Successful rushing 2‑pt conversions.                                |
| `efficiency`                              | float64 | EPA per play (rush or pass) for the player.                         |
| `percent_attempts_gte_eight_defenders`    | float64 | Rush % with ≥ 8 defenders in the box.                               |
| `avg_time_to_los`                         | float64 | Avg. time to line of scrimmage on rushes.                           |
| `avg_rush_yards`                          | float64 | Avg. yards per rush.                                                |
| `expected_rush_yards`                     | float64 | Model‑expected rush yards.                                          |
| `rush_yards_over_expected`                | float64 | Actual minus expected rush yards.                                   |
| `rush_yards_over_expected_per_att`        | float64 | Rush YOE per attempt.                                               |
| `rush_pct_over_expected`                  | float64 | Percent of rush yards over expectation.                             |
| `yards_per_rush_attempt`                  | float64 | Yards per rushing attempt.                                          |
| `receptions`                              | int64   | Pass receptions.                                                    |
| `targets`                                 | int64   | Pass targets.                                                       |
| `receiving_yards`                         | int64   | Receiving yards gained.                                             |
| `receiving_tds`                           | int64   | Receiving touchdowns.                                               |
| `receiving_fumbles`                       | int64   | Fumbles after reception.                                            |
| `receiving_fumbles_lost`                  | int64   | Receiving fumbles lost.                                             |
| `receiving_air_yards`                     | int64   | Air yards on targets.                                               |
| `receiving_yards_after_catch`             | int64   | Yards after catch on receptions.                                    |
| `receiving_first_downs`                   | int64   | Receptions resulting in first downs.                                |
| `receiving_epa`                           | float64 | Expected Points Added from receiving.                               |
| `receiving_2pt_conversions`               | int64   | Successful receiving 2‑pt conversions.                              |
| `racr`                                    | float64 | Receiver air‑conversion ratio (yards ÷ air yards).                  |
| `target_share`                            | float64 | Share of team targets.                                              |
| `air_yards_share`                         | float64 | Share of team air yards.                                            |
| `wopr`                                    | float64 | Weighted Opportunity Rating (targets + air yards).                  |
| `avg_cushion`                             | float64 | Avg. pre‑snap cushion vs. defender (yards).                         |
| `avg_separation`                          | float64 | Avg. separation at pass arrival (yards).                            |
| `avg_intended_air_yards_receiving`        | float64 | Avg. intended air yards on targets.                                 |
| `percent_share_of_intended_air_yards`     | float64 | Share of team intended air yards.                                   |
| `catch_percentage`                        | float64 | Receptions divided by targets.                                      |
| `avg_yac`                                 | float64 | Avg. yards after catch per reception.                               |
| `avg_expected_yac`                        | float64 | Model‑expected YAC per reception.                                   |
| `avg_yac_above_expectation`               | float64 | YAC minus expected YAC.                                             |
| `special_teams_tds`                       | int64   | Touchdowns scored on special‑teams plays.                           |
| `fantasy_points_ppr`                      | float64 | Fantasy points (PPR scoring).                                       |
| `total_plays`                             | int64   | Total offensive plays involving the player.                         |
| `total_yards`                             | int64   | Total yards gained (pass + rush + receive).                         |
| `total_fumbles`                           | int64   | Total fumbles committed.                                            |
| `total_fumbles_lost`                      | int64   | Total fumbles lost.                                                 |
| `total_turnovers`                         | int64   | Total turnovers (INT + fumbles lost).                               |
| `total_touchdowns`                        | int64   | Total touchdowns scored.                                            |
| `total_first_downs`                       | int64   | Total first downs generated.                                        |
| `touchdown_per_play`                      | float64 | Touchdowns divided by total plays.                                  |
| `yards_per_play`                          | float64 | Yards gained per play.                                              |
| `fantasy_point_per_play`                  | float64 | Fantasy points per play.                                            |
