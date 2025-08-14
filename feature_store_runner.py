from nfl_data_loader.utils.utils import get_seasons_to_update, put_dataframe

from src.pipelines.events.event_regular_season_game import make_event_regular_season_feature_store
from src.pipelines.fantasy.fantasy_football import make_fantasy_feature_store
from src.pipelines.players.player_regular_season_game import make_off_player_regular_season_feature_store

event_meta = {
    "name":'event/regular_season_game',
    "start_season": 2002,
    "obj": make_event_regular_season_feature_store
    }
player_off = {
    "name":'player/off/regular_season_game',
    "start_season": 2002,
    "obj": make_off_player_regular_season_feature_store
    }
fantasy = {
    "name":'player/fantasy',
    "start_season": 2019,
    "obj": make_fantasy_feature_store
    }
FEATURE_STORE_METAS = [
    event_meta,
    #player_off,
    fantasy
]



def main():

    root_path = './data/feature_store'
    for fs_meta_obj in FEATURE_STORE_METAS:
        feature_store_name = fs_meta_obj['name']
        start_season = fs_meta_obj['start_season']
        ## Determine pump mode
        update_seasons = get_seasons_to_update(root_path, feature_store_name)
        if min(update_seasons) < start_season:
            update_seasons = [i for i in update_seasons if i >= start_season]

        #update_seasons = [2004, 2005, 2006,2007,2008,2009,2010,2011,2012,2013,2014]

        mode = 'refresh' if start_season in update_seasons else 'upsert'

        # Use the last 2 seasons for aggregate stats for upsert mode
        load_seasons = update_seasons if mode == 'refresh' else list(range(min(update_seasons) - 2, max(update_seasons)+1))

        if 'player' in fs_meta_obj['name']:
            # Career stats so we have to load entire history
            load_seasons = list(range(start_season, max(update_seasons)+1))


        print(f"Running Feature Store: {feature_store_name} from {min(update_seasons)}-{max(update_seasons)} (loads: {min(load_seasons)}-{max(load_seasons)})")

        fs_df = fs_meta_obj['obj'](load_seasons)
        print(f"Adds: {round(fs_df.memory_usage(deep=True).sum() / (1024 ** 2), 2)} MB to the Feature Store")
        for season in update_seasons:
            put_dataframe(fs_df[fs_df.season==season].copy(), f"{root_path}/{feature_store_name}/{season}.parquet")

if __name__ == '__main__':
    main()