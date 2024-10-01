def event_targets(schedule):
    s = schedule[
        [
            'season',
            'week',
            'home_team',
            'away_team',
            'home_score',
            'away_score',
            'spread_line',
            'total_line',
        ]
    ].drop_duplicates(subset=['season', 'week', 'home_team', 'away_team']).reset_index(drop=True) \
        .assign(
        actual_away_team_win=lambda x: (x.home_score < x.away_score),
        actual_away_spread=lambda x: (x.home_score - x.away_score),
        actual_point_total=lambda x: (x.home_score + x.away_score),
    )
    s['actual_away_team_covered_spread'] = (s['away_score'] + s['spread_line'] >= s['home_score'])

    # Calculate if the game covered the under
    s['actual_under_covered'] = (s['home_score'] + s['away_score'] <= s['total_line'])
    return s.rename(columns={
        'home_score':'actual_home_score',
        'away_score':'actual_away_score',
    }).drop(columns=['spread_line','total_line'])