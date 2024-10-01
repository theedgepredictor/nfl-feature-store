import datetime




def fill_qbr(df):
    # Step 1: Calculate the average qbr for each season-week group
    qbr_avg = df.groupby(['season', 'week'])['qbr'].transform('mean')

    # Step 2: Fill missing qbr values based on the team_win condition
    df['qbr'] = df.apply(lambda row: 0.4 * qbr_avg[row.name] if pd.isna(row['qbr']) and row['team_win'] == 0
                         else 0.6 * qbr_avg[row.name] if pd.isna(row['qbr']) and row['team_win'] == 1
                         else row['qbr'], axis=1)
    return df



def load_players():
    df = pd.read_parquet('https://github.com/nflverse/nflverse-data/releases/download/players/players.parquet')
    return df
