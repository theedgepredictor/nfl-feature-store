
## Static Players

- Contains unchanging player metadata
- Used for joining against rating system
- Provides player names and additional visualization data
- Fields: player_id, name, birth_date, college, draft info, etc.
Static player attributes: Identity, biographical details, draft information, and NFL Combine metrics that do not vary week‑to‑week.

Primary use: Joins with season/weekly performance tables to enrich analyses with baseline traits and historical context.

| column               | dtype           | description                                                            |
| -------------------- | --------------- | ---------------------------------------------------------------------- |
| `player_id`          | object           | Stable internal identifier for the player (unique across all seasons). |
| `name`               | object          | Full display name in “First Last” format.                              |
| `first_name`         | object          | Player’s given name only.                                              |
| `last_name`          | object          | Player’s family/surname only.                                          |
| `pfr_id`             | object          | Pro‑Football‑Reference slug (e.g., `BradyTo00`).                       |
| `espn_id`            | int64          | ESPN player identifier string or numeric code.                         |
| `birth_date`         | datetime64\[ns] | Date of birth (YYYY‑MM‑DD).                                            |
| `height`             | int64           | Height in **inches** (e.g., 74 = 6′2″).                                |
| `weight`             | int64           | Playing weight in **pounds**.                                          |
| `headshot`           | object          | URL or path to the player’s headshot image.                            |
| `college_name`       | object          | Name of the college attended (e.g., “Alabama”).                        |
| `college_conference` | object          | NCAA conference of the college (SEC, Big Ten, etc.).                   |
| `rookie_season`      | int64           | First NFL season on an active roster.                                  |
| `draft_year`         | int64           | NFL Draft year.                                                        |
| `draft_round`        | int64           | Draft round selected (1 – 7; 0 if undrafted).                          |
| `draft_pick`         | int64           | Overall pick number (1 – 262; 0 if undrafted).                         |
| `draft_team`         | object          | Team abbreviation that drafted the player (e.g., “NE”).                |
| `forty`              | float64         | 40‑yard‑dash time (seconds).                                           |
| `bench`              | int64           | Bench‑press reps at 225 lb.                                            |
| `vertical`           | float64         | Vertical‑jump height (inches).                                         |
| `broad_jump`         | int64           | Broad‑jump distance (inches).                                          |
| `cone`               | float64         | 3‑cone‑drill time (seconds).                                           |
| `shuttle`            | float64         | 20‑yard shuttle time (seconds).                                        |
