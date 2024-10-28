import pandas as pd

years = ['2017-2018', '2018-2019', '2019-2020', '2020-2021', '2021-2022']
dataframesCollection = {}
for i, year in enumerate(years) :
    yearURL = 'https://fbref.com/en/comps/9/' + year + '/' + year +'-Premier-League-Stats'
    dataframesCollection[year] = pd.read_html(yearURL)

from functools import reduce

gamesPerSeason = 38
data = {}
for i, year in enumerate(years) :
    standard = dataframesCollection[year][2].drop(columns=['Unnamed: 1_level_0','Unnamed: 2_level_0','Unnamed: 3_level_0','Playing Time','Expected', 'Per 90 Minutes'], axis=1, level=0)
    standard.columns = standard.columns.droplevel()
    standard = standard.drop(columns=['G+A','G-PK','PK','PKatt'])
    standard[['Gls','Ast','CrdY','CrdR','PrgC','PrgP']] = standard[['Gls','Ast','CrdY','CrdR','PrgC','PrgP']].div(gamesPerSeason)

    goalkeeping = dataframesCollection[year][4]
    goalkeeping.columns = goalkeeping.columns.droplevel()
    goalkeeping = goalkeeping[['Squad','Saves']]
    goalkeeping[['Saves']] = goalkeeping[['Saves']].div(gamesPerSeason)

    shooting = dataframesCollection[year][8].drop(columns=['Unnamed: 1_level_0','Unnamed: 2_level_0','Expected'], axis=1, level=0)
    shooting.columns = shooting.columns.droplevel()
    shooting = shooting[['Squad','Sh','SoT']]
    shooting[['Sh','SoT']] = shooting[['Sh','SoT']].div(gamesPerSeason)

    passtypes = dataframesCollection[year][12].drop(columns=['Unnamed: 1_level_0','Unnamed: 2_level_0','Unnamed: 3_level_0','Corner Kicks','Outcomes'], axis=1, level=0)
    passtypes.columns = passtypes.columns.droplevel()
    passtypes = passtypes[['Squad','FK','TB','Sw','Crs','CK']]
    passtypes[['FK','TB','Sw','Crs','CK']] = passtypes[['FK','TB','Sw','Crs','CK']].div(gamesPerSeason)

    creativity = dataframesCollection[year][14].drop(columns=['Unnamed: 1_level_0','Unnamed: 2_level_0','SCA Types','GCA Types'], axis=1, level=0)
    creativity.columns = creativity.columns.droplevel()
    creativity = creativity[['Squad','SCA','GCA']]
    creativity[['SCA','GCA']] = creativity[['SCA','GCA']].div(gamesPerSeason)

    defensive = dataframesCollection[year][16].drop(columns=['Unnamed: 1_level_0','Unnamed: 2_level_0','Challenges','Unnamed: 16_level_0'],axis=1,level=0)
    defensive.columns = defensive.columns.droplevel()
    defensive = defensive[['Squad','TklW','Blocks','Int','Clr','Err']]
    defensive[['TklW','Blocks','Int','Clr','Err']] = defensive[['TklW','Blocks','Int','Clr','Err']].div(gamesPerSeason)

    possession = dataframesCollection[year][18].drop(
        columns=['Unnamed: 1_level_0','Unnamed: 3_level_0','Touches','Take-Ons','Carries','Receiving'],axis=1,level=0)
    possession.columns = possession.columns.droplevel()

    misc = dataframesCollection[year][22].drop(columns=['Unnamed: 1_level_0','Unnamed: 2_level_0','Aerial Duels'],axis=1,level=0)
    misc.columns = misc.columns.droplevel()
    misc = misc[['Squad','Fls','Fld','Off','PKwon','PKcon','Recov']]
    misc[['Fls','Fld','Off','PKwon','PKcon','Recov']] = misc[['Fls','Fld','Off','PKwon','PKcon','Recov']].div(gamesPerSeason)

    data[year] = {'standard' : standard, 'goalkeeping' : goalkeeping, 'shooting' : shooting, 'passtypes' : passtypes,
    'creativity' : creativity, 'defensive' : defensive, 'possession' : possession, 'misc' : misc}


for i, year in enumerate(years) :
    dataframes = [data[year]['standard'],data[year]['goalkeeping'],data[year]['shooting'],
              data[year]['passtypes'],data[year]['defensive'],data[year]['possession'],data[year]['misc']]
    df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['Squad'],how='outer'), dataframes)   
    df_merged.to_csv("data/" + year + '_teamstats.csv') 

# Get match data from CSV file
shortYears = ['2017-18', '2018-19', '2019-20', '2020-21', '2021-22']

resultsMax = pd.read_csv("data/results.csv", encoding="ANSI")
resultsMax = resultsMax.drop(columns=['DateTime', 'Referee'])
results = resultsMax.drop(columns=['FTHG','FTAG','HTHG','HTAG','HTR','HS','AS','HST','AST','HC','AC','HF','AF','HY','AY','HR','AR'])
results['FTR'] = results['FTR'].replace({'H':0, 'D':1, 'A':2})
results[['HomeTeam', 'AwayTeam']] = results[['HomeTeam', 'AwayTeam']].replace({'Cardiff':'Cardiff City', 'Leeds':'Leeds United', 'Leicester':'Leicester City', 'Man City':'Manchester City', 
                                                                               'Man United':'Manchester Utd','Newcastle':'Newcastle Utd', 'Norwich':'Norwich City', 'Sheffield United':'Sheffield Utd', 
                                                                               'Stoke':'Stoke City','Swansea':'Swansea City'})
shortResults = results[results['Season'].isin(shortYears)]

# Combine the two datasets
teamStats = {}
for i, year in enumerate(years) :
    teamStats[shortYears[i]] = pd.read_csv("data/" + year + '_teamstats.csv')
    teamStats[shortYears[i]] = teamStats[shortYears[i]].drop(columns=['Unnamed: 0'])
    teamStats[shortYears[i]]['Season'] = shortYears[i] 

homeTeamStats = {}
awayTeamStats = {}
for i, year in enumerate(shortYears) :
    homeTeamStats[year] = teamStats[year].add_prefix('home_')
    homeTeamStats[year] = teamStats[year].rename(columns={c: 'home_'+ c for c in teamStats[year].columns if c not in ['Squad', 'Season']})
    homeTeamStats[year] = homeTeamStats[year].rename(columns={'Squad':'HomeTeam'})
    awayTeamStats[year] = teamStats[year].rename(columns={c: 'away_'+ c for c in teamStats[year].columns if c not in ['Squad', 'Season']})
    awayTeamStats[year] = awayTeamStats[year].rename(columns={'Squad':'AwayTeam'})

homeTeamConcat = pd.concat(homeTeamStats, ignore_index=True)
awayTeamConcat = pd.concat(awayTeamStats, ignore_index=True)

fulldata = shortResults
fulldata = fulldata.merge(homeTeamConcat, how='left', left_on=['HomeTeam','Season'], right_on = ['HomeTeam','Season'])
fulldata = fulldata.merge(awayTeamConcat, how='left', left_on=['AwayTeam','Season'], right_on = ['AwayTeam','Season'])
fulldata.to_csv("data/fulldata.csv")