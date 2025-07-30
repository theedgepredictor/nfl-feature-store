## Player State

Built through the PlayerStateDataComponent

- Weekly player state information
- Contains merged weekly player data
- Tracks player STATUS (PLAYED, INJURED, etc.)
- Used to determine player availability and game participation

Not official. Staging table used as part of the PlayerRatingComponent

| column               | dtype  | description                                                                          |
| -------------------- | ------ | ------------------------------------------------------------------------------------ |
| `game_id`            | int64  | Unique identifier for the game (one row per game‑player combination).                |
| `player_id`          | object  | Stable identifier for the player across seasons and teams.                           |
| `season`             | int64  | Season year (e.g., 2024).                                                            |
| `week`               | int64  | Week number within the season (1‑18, including playoffs if present).                 |
| `team`               | object | Team abbreviation (franchise the player suited up for that week).                    |
| `high_pos_group`     | object | Top‑level position grouping—e.g., Offense, Defense, Special Teams.                   |
| `position_group`     | object | Mid‑level positional bucket—e.g., Skill (QB/RB/WR/TE), Line (OL/DL), Secondary, etc. |
| `position`           | object | Specific on‑field position abbreviation (QB, RB, WR, LB, CB…).                       |
| `starter`            | bool   | Boolean flag indicating if the player was in the starting lineup.                    |
| `status`             | object | Roster status at kickoff (Active, Inactive, IR, PUP, etc.).                          |
| `report_status`      | object | Injury‑report designation entering the game (Questionable, Doubtful, Out…).          |
| `playerverse_status` | object | Aggregated fantasy‑platform status (e.g., Healthy, OUT, IR‑R, COVID‑19).             |
