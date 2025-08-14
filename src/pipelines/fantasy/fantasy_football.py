from nfl_data_loader.workflows.components.players.fantasy import PlayerFantasyComponent


def make_fantasy_feature_store(load_seasons):
    #game_player_component = GamePlayerComponent(load_seasons, season_type='REG')
    #game_player_df = game_player_component.run_pipeline()
    #del game_player_component

    pfc = PlayerFantasyComponent(load_seasons)

    #df = off_player_df.merge(game_player_df, on=['player_id','position_group', 'season', 'week'], how='left')
    #df = df.dropna(subset=['player_id'])

    ### Handle Future
    #inference_df = game_player_df[game_player_df.player_id.isnull()].copy()
    #latest_off_player_df = off_player_df[off_player_df].sort_values(['player_id', 'season', 'week']).drop_duplicates(['player_id'], keep='last')

    return pfc.df

if __name__ == '__main__':
    df = make_fantasy_feature_store([2022,2023,2024])