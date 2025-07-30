
## Preseason Players

- Initial player ratings before season starts
- Used for week 1 rating calculations
- Includes Madden ratings and other baseline metrics

Scope: Pre‑season baseline ratings used to seed Week 1 player values and ELO/rating calculations.
Sources: Madden attribute set (0‑99 integers) plus last_season_av from Sports‑Reference.
Usage: Combines scouting ratings with prior‑year value to generate initial weekly projections.

| column               | dtype | description                                                    |
| -------------------- | ----- | -------------------------------------------------------------- |
| `season`             | int64 | Season year the ratings apply to (e.g., 2025).                 |
| `player_id`          | object | Stable internal player identifier.                             |
| `madden_id`          | object | EA Sports Madden identifier (consistent across game releases). |
| `years_exp`          | int64 | Completed NFL seasons heading into this year.                  |
| `is_rookie`          | bool  | `True` if entering first NFL season.                           |
| `last_season_av`     | int64 | Previous‑season *Approximate Value* from Sports‑Reference.     |
| `overallrating`      | int64 | Composite Madden rating (0‑99).                                |
| `agility`            | int64 | Lateral quickness and change‑of‑direction ability.             |
| `acceleration`       | int64 | Rate at which top speed is reached.                            |
| `speed`              | int64 | Top‑end straight‑line speed.                                   |
| `stamina`            | int64 | Endurance before fatigue effects.                              |
| `strength`           | int64 | Physical power in contact situations.                          |
| `toughness`          | int64 | Ability to play through pain and minor injuries.               |
| `injury`             | int64 | Likelihood of avoiding or quickly recovering from injury.      |
| `awareness`          | int64 | Play recognition, decision‑making, and football IQ.            |
| `jumping`            | int64 | Vertical leap capability.                                      |
| `trucking`           | int64 | Power to run through would‑be tacklers.                        |
| `throwpower`         | int64 | Maximum ball velocity (arm strength).                          |
| `throwaccuracyshort` | int64 | Pass accuracy on throws < 20 yards.                            |
| `throwaccuracymid`   | int64 | Pass accuracy on throws 20‑40 yards.                           |
| `throwaccuracydeep`  | int64 | Pass accuracy on throws > 40 yards.                            |
| `playaction`         | int64 | Effectiveness selling play‑action fakes.                       |
| `throwonrun`         | int64 | Pass accuracy while moving outside the pocket.                 |
| `carrying`           | int64 | Ball security rating against fumbles.                          |
| `ballcarriervision`  | int64 | Ability to identify rushing lanes and cutbacks.                |
| `stiffarm`           | int64 | Effectiveness of stiff‑arm move.                               |
| `spinmove`           | int64 | Effectiveness of spin move.                                    |
| `jukemove`           | int64 | Effectiveness of juke move.                                    |
| `catching`           | int64 | Reliability catching routine passes.                           |
| `shortrouterunning`  | int64 | Precision on routes < 10 yards.                                |
| `midrouterunning`    | int64 | Precision on routes 10‑20 yards.                               |
| `deeprouterunning`   | int64 | Precision on routes > 20 yards.                                |
| `spectacularcatch`   | int64 | Ability to complete highlight‑reel catches.                    |
| `catchintraffic`     | int64 | Securing catches in tight coverage.                            |
| `release`            | int64 | Beating press coverage at the line.                            |
| `runblocking`        | int64 | Effectiveness blocking on rushing plays.                       |
| `passblocking`       | int64 | Effectiveness blocking on pass plays.                          |
| `impactblocking`     | int64 | Power to deliver blocks that displace defenders.               |
| `mancoverage`        | int64 | Skill in man‑to‑man pass coverage.                             |
| `zonecoverage`       | int64 | Skill in zone pass coverage.                                   |
| `tackle`             | int64 | Ability to bring down ball‑carriers.                           |
| `hitpower`           | int64 | Force delivered on tackles (causing fumbles).                  |
| `press`              | int64 | Ability to jam receivers off the snap.                         |
| `pursuit`            | int64 | Closing speed and pursuit angles to the ball.                  |
| `kickaccuracy`       | int64 | Place‑kicker field‑goal/PAT accuracy.                          |
| `kickpower`          | int64 | Leg strength for kick distance and hang‑time.                  |
| `return`             | int64 | Effectiveness as a kick/punt returner.                         |
