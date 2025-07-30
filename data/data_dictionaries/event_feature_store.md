
## Event Feature Store

EWMA = Exponentially Weighted Moving Average (α ≈ 0.2 unless otherwise noted).

home_avg_rushing_yards_offense – Average rushing yards gained per game by the home team’s offense.

away_avg_turnover_defense – Average turnovers forced per game by the away team’s defense.

home_avg_points_offense_rank – Where the home offense ranks league‑wide in points per game (1 = most).

| Affix pattern                                       | Example column                         | Interpretation                                                          |
| --------------------------------------------------- | -------------------------------------- | ----------------------------------------------------------------------- |
| **Team side prefix**<br>`home_` / `away_`           | `home_avg_points_offense`              | Metric is evaluated for the **home** (or **away**) team.                |
| **Phase‑of‑play suffix**<br>`_offense` / `_defense` | `home_avg_points_defense`              | Same metric split by the team’s **offense** or **defense** performance. |
| **League‑rank suffix**<br>`_rank`                   | `away_avg_yards_per_play_offense_rank` | League ranking of the metric (1 = best, 32 = worst).                    |
| **Standalone (no affix)**                           | `spread_line`                          | Applies to the entire game or betting market, not team‑specific.        |



Data Table

| column                                                        | dtype   | description                                                        |
| ------------------------------------------------------------- | ------- | ------------------------------------------------------------------ |
| `home_team` / `away_team`                                     | object  | Team abbreviation (e.g., “KC”, “SF”).                              |
| `season`                                                      | int64   | Season year.                                                       |
| `week`                                                        | int64   | Week number (1‑18, or 0 for playoffs).                             |
| `rest`                                                        | int64   | Days since the team’s previous game.                               |
| `spread_line`                                                 | float64 | Vegas point‑spread closing line. (Positive ⇒ home underdog.)       |
| `total_line`                                                  | float64 | Vegas over/under points line.                                      |
| `home_moneyline` / `away_moneyline`                           | int64   | Money‑line odds (American format).                                 |
| `actual_home_score` / `actual_away_score`                     | int64   | Final team score.                                                  |
| `actual_point_total`                                          | int64   | Game total points (`home_score + away_score`).                     |
| `actual_away_spread`                                          | int64   | Actual margin vs. spread (positive ⇒ away covered).                |
| `actual_away_team_win`                                        | bool    | `True` if away team won outright.                                  |
| `actual_away_team_covered_spread`                             | bool    | `True` if away covered the spread.                                 |
| `actual_under_covered`                                        | bool    | `True` if game stayed *under* `total_line`.                        |
| `rolling_spread_cover`                                        | float64 | EWMA\* of spread covers over recent games (team‑side).             |
| `rolling_under_cover`                                         | float64 | EWMA\* of under covers over recent games (team‑side).              |
| `elo_pre`                                                     | float64 | Pre‑game ELO rating.                                               |
| `elo_prob`                                                    | float64 | Win probability derived from `elo_pre`.                            |
| `ewma_score`                                                  | float64 | Exponentially‑weighted moving‑average points scored / allowed.     |
| `avg_fantasy_points{_*}`                                      | float64 | Season‑to‑date average fantasy points (standard / half‑PPR / PPR). |
| `avg_total_plays`                                             | float64 | Avg. offensive plays per game.                                     |
| `avg_total_yards`                                             | float64 | Avg. scrimmage yards per game.                                     |
| `avg_total_fumbles`                                           | float64 | Avg. fumbles committed per game.                                   |
| `avg_total_turnovers`                                         | float64 | Avg. turnovers (INT + lost fumbles) per game.                      |
| `avg_total_touchdowns`                                        | float64 | Avg. touchdowns scored per game.                                   |
| `avg_total_first_downs`                                       | float64 | Avg. first downs gained per game.                                  |
| `avg_touchdown_per_play`                                      | float64 | TD ÷ offensive plays.                                              |
| `avg_yards_per_play`                                          | float64 | Yards ÷ offensive plays.                                           |
| `avg_fantasy_point_per_play`                                  | float64 | Fantasy points ÷ offensive plays.                                  |
| `avg_completions` / `avg_attempts`                            | float64 | Passing completions / attempts per game.                           |
| `avg_passing_yards`                                           | float64 | Passing yards per game.                                            |
| `avg_passing_tds` / `avg_interceptions`                       | float64 | Passing TDs / INTs per game.                                       |
| `avg_sacks` / `avg_sack_yards`                                | float64 | Sacks taken and yards lost per game.                               |
| `avg_sack_fumbles_lost`                                       | float64 | Sack‑fumble turnovers per game.                                    |
| `avg_passing_air_yards`                                       | float64 | Air yards thrown per game.                                         |
| `avg_passing_yards_after_catch`                               | float64 | YAC gained per game.                                               |
| `avg_passing_first_downs`                                     | float64 | Pass first‑downs per game.                                         |
| `avg_passing_epa`                                             | float64 | EPA generated by passing per game.                                 |
| `avg_pacr`                                                    | float64 | Pass‑air‑conversion ratio (yards ÷ air yards).                     |
| `avg_dakota`                                                  | float64 | Dakota composite QB metric (team‑offense or allowed).              |
| `avg_time_to_throw`                                           | float64 | Mean time from snap to release (seconds).                          |
| `avg_completed_air_yards`                                     | float64 | Air yards on completed passes.                                     |
| `avg_intended_air_yards_passing`                              | float64 | Intended air yards on attempts.                                    |
| `avg_air_yards_differential`                                  | float64 | Intended – completed air yards.                                    |
| `avg_aggressiveness`                                          | float64 | % of attempts into tight windows.                                  |
| `avg_max_completed_air_distance`                              | float64 | Longest completed air distance.                                    |
| `avg_air_yards_to_sticks`                                     | float64 | Avg. throw depth relative to the first‑down marker.                |
| `avg_passer_rating`                                           | float64 | Classic NFL passer rating.                                         |
| `avg_completion_percentage`                                   | float64 | Pass completion percentage.                                        |
| `avg_expected_completion_percentage`                          | float64 | Model‑expected completion %.                                       |
| `avg_completion_percentage_above_expectation`                 | float64 | CPAE = actual – expected %                                         |
| `avg_air_distance` / `avg_max_air_distance`                   | float64 | Avg./max ball‑flight distance.                                     |
| `avg_air_yards_per_pass_attempt`                              | float64 | Air yards ÷ attempts.                                              |
| `avg_pass_to_rush_ratio`                                      | float64 | Dropbacks ÷ (rush + dropbacks).                                    |
| `avg_pass_to_rush_first_down_ratio`                           | float64 | Pass 1st‑downs ÷ rush 1st‑downs.                                   |
| `avg_yards_per_pass_attempt`                                  | float64 | Yards ÷ pass attempts.                                             |
| `avg_sack_rate`                                               | float64 | Sacks ÷ dropbacks.                                                 |
| `avg_carries`                                                 | float64 | Rushing attempts per game.                                         |
| `avg_rushing_yards`                                           | float64 | Rushing yards per game.                                            |
| `avg_rushing_tds`                                             | float64 | Rush TDs per game.                                                 |
| `avg_rushing_fumbles_lost`                                    | float64 | Lost rush fumbles per game.                                        |
| `avg_rushing_first_downs`                                     | float64 | Rush first‑downs per game.                                         |
| `avg_rushing_epa`                                             | float64 | EPA from rushing per game.                                         |
| `avg_efficiency`                                              | float64 | EPA per play (run + pass blend).                                   |
| `avg_percent_attempts_gte_eight_defenders`                    | float64 | Rush % vs. ≥8‑man box.                                             |
| `avg_time_to_los`                                             | float64 | Avg. time to line of scrimmage on runs.                            |
| `avg_expected_rush_yards`                                     | float64 | Model‑expected rush yards per game.                                |
| `avg_rush_yards_over_expected`                                | float64 | Total YOE per game.                                                |
| `avg_rush_yards_over_expected_per_att`                        | float64 | YOE ÷ rush attempts.                                               |
| `avg_rush_pct_over_expected`                                  | float64 | % of rush yards over expectation.                                  |
| `avg_yards_per_rush_attempt`                                  | float64 | Yards ÷ rush attempts.                                             |
| `avg_cushion` / `avg_separation`                              | float64 | Coverage cushion / separation (Next Gen Stats).                    |
| `avg_intended_air_yards_receiving`                            | float64 | Intended air yards on targets.                                     |
| `avg_yac_above_expectation`                                   | float64 | YAC – expected YAC.                                                |
| `ewma_rushing` / `ewma_passing`                               | float64 | EWMA rushing / passing EPA.                                        |
| `avg_offensive_penalty_yards` / `avg_defensive_penalty_yards` | float64 | Penalty yards committed per game.                                  |
| `avg_first_down`                                              | float64 | 1st‑downs gained per game.                                         |
| `avg_third_down_converted` / `avg_third_down_failed`          | float64 | 3rd‑down successes / failures.                                     |
| `avg_fourth_down_converted` / `avg_fourth_down_failed`        | float64 | 4th‑down successes / failures.                                     |
| `avg_first_down_penalty`                                      | float64 | 1st‑downs earned by defensive penalty.                             |
| `avg_shotgun` / `avg_no_huddle`                               | float64 | % of plays in shotgun / no‑huddle.                                 |
| `avg_qb_dropback` / `avg_qb_scramble`                         | float64 | Dropback / scramble rate.                                          |
| `avg_goal_to_go`                                              | float64 | % of plays with goal‑to‑go.                                        |
| `avg_is_redzone`                                              | float64 | % of plays in the red zone.                                        |
| `avg_third_down_percentage` / `avg_fourth_down_percentage`    | float64 | Conversion % on 3rd / 4th downs.                                   |
| `avg_points`                                                  | float64 | Points scored / allowed per game.                                  |
| `avg_point_differential`                                      | float64 | Points for – points against.                                       |
| `avg_epa` / `avg_wpa`                                         | float64 | Expected / Win Probability Added per game.                         |
| `avg_time_of_possession`                                      | float64 | Minutes of possession per game.                                    |
| `avg_field_goal_made` / `avg_field_goal_attempt`              | float64 | FGs made / attempted per game.                                     |
| `avg_field_goal_distance`                                     | float64 | Mean FG attempt distance (yards).                                  |
| `avg_extra_point_made` / `avg_extra_point_attempt`            | float64 | XPs made / attempted per game.                                     |
| `avg_turnover`                                                | float64 | Turnovers committed / forced per game.                             |
| `avg_field_goal_percentage`                                   | float64 | FG make %.                                                         |
| `avg_extra_point_percentage`                                  | float64 | XP make %.                                                         |
| `avg_q{1‑5}_point_diff` / `avg_q{1‑5}_points`                 | float64 | Per‑quarter point diff / points.                                   |
| `offensive_rank` / `defensive_rank` / `net_rank`              | int64   | League rank (1 = best).                                            |


Post Processing Formatters

| formatter          | what it does                                                                                                                                                | key arguments                                                                          | result / example                                                                                                                           |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `df_rename_pivot`  | **Splits** one “generic” set of columns into two **prefixed** sets (e.g., `team_id` → `away_team_id`, `home_team_id`).                                      | `all_cols`, `pivot_cols` (columns to keep unchanged), `t1_prefix`, `t2_prefix`         | Returns one wide DF with both prefixed column groups.<br>`df_rename_pivot(teams_df, ["game_id","team_id"], ["game_id"], "away_", "home_")` |
| `df_rename_fold`   | **Folds** two prefixed column groups back into a single generic set (inverse of *pivot*).                                                                   | `t1_prefix`, `t2_prefix`                                                               | Stacked long DF with a single set of column names.<br>`df_rename_fold(wide_df, "away_", "home_")`                                          |
| `df_rename_dif`    | Creates **difference** features between matching prefixed columns and drops the originals.                                                                  | supply either:<br>• `t1_prefix`, `t2_prefix` **or**<br>• explicit `t1_cols`, `t2_cols` | Adds `dif_<metric>` columns, e.g. `dif_team_turnovers` = `away_team_turnovers – home_team_turnovers`.                                      |
| `df_rename_exavg`  | Creates **expected‑average** features (mean of the two sides) and drops originals.                                                                          | same arg pattern as `df_rename_dif`                                                    | Adds `exavg_<metric>` columns, e.g. `exavg_team_turnovers` = average of home/away turnovers.                                               |
| `df_rename_shift`  | **Long‑formats** a game‑level DF: duplicates every row, swaps home/away metric names to a neutral schema, and adds `is_home` flag (`1` = home, `0` = away). | optional `drop_cols` to remove beforehand                                              | Ideal for model matrices that treat each team‑game as its own observation.                                                                 |
| `suffix_to_prefix` | Bulk‑renames columns from `<metric>_home` / `<metric>_away` style to `home_<metric>` / `away_<metric>` style.                                               | `suffix`, `prefix`                                                                     | Harmonises naming before other transforms.                                                                                                 |
