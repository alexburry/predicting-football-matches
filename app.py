import sys
import joblib
import pandas as pd
from functools import partial
from functools import reduce
from sklearn.preprocessing import StandardScaler
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QGridLayout,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QComboBox,
    QWidget,
    QLabel,
    QScrollArea,
    QGroupBox,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 700
DISPLAY_HEIGHT = 35
BUTTON_HEIGHT = 30
BUTTON_WIDTH = 120

class AppWindow(QMainWindow):
    ###Main window GUI/View###
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Predicting Match Outcomes")
        self.setFixedSize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.generalLayout = QVBoxLayout()
        centralWidget = QWidget(self)
        centralWidget.setLayout(self.generalLayout)
        self.setCentralWidget(centralWidget)
        self._createMainTitle()
        self._createTeamSelection()
        self._createOutputButton()
        self._createOutput()
        self._createHistoryButton()

    def _createMainTitle(self):
        self.mainTitle = QLabel("<h1>Predicting Match Outcomes</h1>")
        self.mainTitle.setFixedHeight(DISPLAY_HEIGHT)
        self.mainTitle.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.generalLayout.addWidget(self.mainTitle)

    def _createTeamSelection(self):
        teamsLayout = QHBoxLayout()
        currentyeardata = pd.read_csv('project\data\currentyeardata.csv')
        teams = currentyeardata.loc[:, 'Squad']

        self.teamSelection1 = QComboBox(self)
        for col, val in enumerate(teams):
            self.teamSelection1.addItem(val)
        teamsLayout.addWidget(self.teamSelection1)

        self.teamSelection2 = QComboBox(self)
        for col, val in enumerate(teams):
            self.teamSelection2.addItem(val)
        teamsLayout.addWidget(self.teamSelection2)

        self.generalLayout.addLayout(teamsLayout)

    def _createOutputButton(self):
        self.outputButton = QPushButton("Predict Outcome")
        self.outputButton.setFixedSize(BUTTON_WIDTH, BUTTON_HEIGHT)
        self.generalLayout.addWidget(self.outputButton)

    def _createOutput(self):
        outputLayout = QVBoxLayout()

        self.outputDisplay = QLineEdit()
        self.outputDisplay.setFixedHeight(DISPLAY_HEIGHT)
        self.outputDisplay.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.outputDisplay.setReadOnly(True)
        outputLayout.addWidget(self.outputDisplay)

        fig = Figure(figsize=(4,4))
        self.axes = fig.add_subplot()
        self.canvas = FigureCanvas(fig)
        outputLayout.addWidget(self.canvas)

        self.generalLayout.addLayout(outputLayout)

    def _createHistoryButton(self):
        self.historyButton = QPushButton("History")
        self.historyButton.setFixedSize(BUTTON_WIDTH, BUTTON_HEIGHT)
        self.generalLayout.addWidget(self.historyButton)

    def _createSeasonsButton(self):
        self.seasonButton = QPushButton("Season")
        self.seasonButton.setFixedSize(BUTTON_WIDTH, BUTTON_HEIGHT)
        self.generalLayout.addWidget(self.seasonButton)

    def setOutputDisplay(self, result):
        if (result == '[0]') :
            text = "Home Win"
        elif (result == '[1]'):
            text = "Draw"
        elif (result == '[2]'):
            text = "Away Win"
        else :
            text = result;
            
        self.outputDisplay.setText(text)
        self.outputDisplay.setFocus()

    def updateOutputGraph(self, probas):
        labels = ["Home Win", "Draw", "Away Win"]
        self.axes.cla()
        self.axes.pie(probas, labels=labels, autopct='%1.1f%%')
        self.canvas.draw()

class HistoryWindow(QWidget):
    def __init__(self, history):
        super().__init__()
        self.setWindowTitle("Prediction History")
        self.setFixedSize(WINDOW_WIDTH-350, WINDOW_HEIGHT)
        self.generalLayout = QVBoxLayout(self)
        self._createHistoryLabel(history)
        scroll = QScrollArea()
        scroll.setWidget(self.groupBox)
        scroll.setWidgetResizable(False)
        self.generalLayout.addWidget(scroll)

    def _createMainTitle(self):
        self.mainTitle = QLabel("<h1>Prediction History</h1>")
        self.mainTitle.setFixedHeight(DISPLAY_HEIGHT)
        self.mainTitle.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.generalLayout.addWidget(self.mainTitle)

    def _createHistoryLabel(self, history):
        self.groupBox = QGroupBox("Prediction History")
        if not history:
            self.historyLabel = QLabel("No prediction history.")
            tempLayout = QHBoxLayout()
            tempLayout.addWidget(self.historyLabel)
            self.groupBox.setLayout(tempLayout)
        else:
            self.historyStack = {}
            self.graphStack = {}
            historyLayout = QGridLayout()
            for i in range(len(history)):
                # Label
                key = history[i]['HomeTeam'] + history[i]['AwayTeam']
                if (history[i]['Pred'][0] == 0):
                    result = "Home Win"
                elif (history[i]['Pred'][0] == 1):
                    result = "Draw"
                elif (history[i]['Pred'][0] == 2):
                    result = "Away Win"
                self.historyStack[key] = QLabel(f"Home Team: {history[i]['HomeTeam']}, Away Team: {history[i]['AwayTeam']}, Result: {result}")
                historyLayout.addWidget(self.historyStack[key], i, 0)

                # Graph
                fig = Figure(figsize=(2,1))
                self.axes = fig.add_subplot()
                self.graphStack[key] = FigureCanvas(fig)
                probas = history[i]["Proba"][0]
                labels = ["Home Win", "Draw", "Away Win"]
                self.axes.pie(probas, labels=labels, autopct='%1.1f%%')
                self.graphStack[key].draw()
                historyLayout.addWidget(self.graphStack[key], i, 1)
                  
            self.groupBox.setLayout(historyLayout)

class PredModel:
    ###Model###
    def __init__(self):
        rawdata = self.retrieveData()
        matchesData = self.storeData(rawdata)
        self.currentYearData = self.scaleData(matchesData)
        print("Data Loaded")

        self.predictingModel = joblib.load('project\models\\randomtree.sav')
        # self.predictingModel = joblib.load('project\models\\logisticreg.sav')
        print("Model loaded")

        self.history = [] #stack of history

    def retrieveData(self) :
        # Get from data source
        yearURL = 'https://fbref.com/en/comps/9/2022-2023/2022-2023-Premier-League-Stats'
        year = pd.read_html(yearURL)

        # Calculate the average number of games played in the season so far
        temp = pd.DataFrame(year[0])
        averageGamesPlayed = temp.loc[:, 'MP'].mean()

        # Clean standard stats
        standard = year[2].drop(columns=['Unnamed: 1_level_0','Unnamed: 2_level_0','Unnamed: 3_level_0','Playing Time','Expected', 'Per 90 Minutes'], axis=1, level=0)
        standard.columns = standard.columns.droplevel()
        standard = standard.drop(columns=['G+A','G-PK','PK','PKatt'])
        standard[['Gls','Ast','CrdY','CrdR','PrgC','PrgP']] = standard[['Gls','Ast','CrdY','CrdR','PrgC','PrgP']].div(averageGamesPlayed)

        # Clean goalkeeping stats
        goalkeeping = year[4]
        goalkeeping.columns = goalkeeping.columns.droplevel()
        goalkeeping = goalkeeping[['Squad','Saves']]
        goalkeeping[['Saves']] = goalkeeping[['Saves']].div(averageGamesPlayed)

        # Clean shooting stats  
        shooting = year[8].drop(columns=['Unnamed: 1_level_0','Unnamed: 2_level_0','Expected'], axis=1, level=0)
        shooting.columns = shooting.columns.droplevel()
        shooting = shooting[['Squad','Sh','SoT']]
        shooting[['Sh','SoT']] = shooting[['Sh','SoT']].div(averageGamesPlayed)

        # Clean pass types stats
        passtypes = year[12].drop(columns=['Unnamed: 1_level_0','Unnamed: 2_level_0','Unnamed: 3_level_0','Corner Kicks','Outcomes'], axis=1, level=0)
        passtypes.columns = passtypes.columns.droplevel()
        passtypes = passtypes[['Squad','FK','TB','Sw','Crs','CK']]
        passtypes[['FK','TB','Sw','Crs','CK']] = passtypes[['FK','TB','Sw','Crs','CK']].div(averageGamesPlayed)

        # Clean creativity stats
        creativity = year[14].drop(columns=['Unnamed: 1_level_0','Unnamed: 2_level_0','SCA Types','GCA Types'], axis=1, level=0)
        creativity.columns = creativity.columns.droplevel()
        creativity = creativity[['Squad','SCA','GCA']]
        creativity[['SCA','GCA']] = creativity[['SCA','GCA']].div(averageGamesPlayed)

        # Clean defensive stats
        defensive = year[16].drop(columns=['Unnamed: 1_level_0','Unnamed: 2_level_0','Challenges','Unnamed: 16_level_0'],axis=1,level=0)
        defensive.columns = defensive.columns.droplevel()
        defensive = defensive[['Squad','TklW','Blocks','Int','Clr','Err']]
        defensive[['TklW','Blocks','Int','Clr','Err']] = defensive[['TklW','Blocks','Int','Clr','Err']].div(averageGamesPlayed)

        # Clean possesion stats
        possession = year[18].drop(
            columns=['Unnamed: 1_level_0','Unnamed: 3_level_0','Touches','Take-Ons','Carries','Receiving'],axis=1,level=0)
        possession.columns = possession.columns.droplevel()

        # Clean misc stats
        misc = year[22].drop(columns=['Unnamed: 1_level_0','Unnamed: 2_level_0','Aerial Duels'],axis=1,level=0)
        misc.columns = misc.columns.droplevel()
        misc = misc[['Squad','Fls','Fld','Off','PKwon','PKcon','Recov']]
        misc[['Fls','Fld','Off','PKwon','PKcon','Recov']] = misc[['Fls','Fld','Off','PKwon','PKcon','Recov']].div(averageGamesPlayed)

        currentYear = {'standard' : standard, 'goalkeeping' : goalkeeping, 'shooting' : shooting, 'passtypes' : passtypes,
            'creativity' : creativity, 'defensive' : defensive, 'possession' : possession, 'misc' : misc}

        dataframe = [currentYear['standard'],currentYear['goalkeeping'],currentYear['shooting'],
                    currentYear['passtypes'],currentYear['defensive'],currentYear['possession'],currentYear['misc']]

        df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['Squad'],how='outer'), dataframe) 
        return df_merged

    def storeData(self, rawdata):
        matchesData = pd.DataFrame()
        teams = rawdata.loc[:, 'Squad']
        for i in range(len(teams)):
            temp = list(teams)
            temp.remove(teams[i])
            for j in range(len(temp)):
                data = self.cleanAndProcessData(teams[i], temp[j], rawdata)
                matchesData = pd.concat([matchesData, data], ignore_index=True, axis=0)

        return matchesData

    def scaleData(self, matchesData):
        matchups = matchesData[['HomeTeam','AwayTeam']].copy()
        matchstats = matchesData.drop(columns=['HomeTeam', 'AwayTeam'])

        scaler = StandardScaler()
        scaler.fit(matchstats)
        scaledMatchstats = pd.DataFrame(scaler.transform(matchstats))

        scaledData = pd.merge(matchups, scaledMatchstats, left_index=True, right_index=True)

        return scaledData
    
    def cleanAndProcessData(self, homeTeam, awayTeam, rawdata):
        # Retrieve corresponding team stats from data and rename variables `home_'
        homeTeamStats = rawdata.loc[rawdata['Squad'] == homeTeam]
        homeTeamStats = homeTeamStats.drop(columns=['CrdY','CrdR'])
        homeTeamStats = homeTeamStats.rename(columns={c: 'home_'+ c for c in homeTeamStats.columns if c not in ['Squad']})
        homeTeamStats = homeTeamStats.rename(columns={'Squad':'HomeTeam'})

        # Retrieve corresponding team stats from data and rename variables `away_'
        awayTeamStats = rawdata.loc[rawdata['Squad'] == awayTeam]
        awayTeamStats = awayTeamStats.drop(columns=['CrdY','CrdR'])
        awayTeamStats = awayTeamStats.rename(columns={c: 'away_'+ c for c in awayTeamStats.columns if c not in ['Squad']})
        awayTeamStats = awayTeamStats.rename(columns={'Squad':'AwayTeam'})

        # temporary merge key
        homeTeamStats['key'] = 1
        awayTeamStats['key'] = 1

        # merge records together for one match-up record
        teamStats = pd.merge(homeTeamStats, awayTeamStats, on=['key'])
        temp = teamStats['AwayTeam']
        teamStats = teamStats.drop(columns=['AwayTeam'])
        teamStats.insert(loc=1, column='AwayTeam', value=temp)
        teamStats = teamStats.drop(columns=['key'])

        return teamStats

    def predict(self, homeTeam, awayTeam):
        if (homeTeam == awayTeam):
            return ["ERROR", "NULL"]
        else:
            teamstats = self.currentYearData.loc[(self.currentYearData['HomeTeam'] == homeTeam) & 
                                     (self.currentYearData['AwayTeam'] == awayTeam)]
            pureData = teamstats.drop(columns=['HomeTeam', 'AwayTeam'])
            # Predict outcome and probas
            pred = self.predictingModel.predict(pureData)
            proba = self.predictingModel.predict_proba(pureData)

            # Add predictions to new dictionary
            predResult = {'HomeTeam' : homeTeam,
                          'AwayTeam' : awayTeam,
                          'Pred' : pred,
                          'Proba' : proba}
            
            self.history.append(predResult)

            return [pred, proba]

class Controller:
    ###Controller Class###
    def __init__(self, model, view):
        self._predModel = model
        self._view = view
        self.historyWindow = None
        self._connectSignalsAndSlots()
    
    def _predictResult(self):
        result = self._predModel.predict(
            homeTeam=str(self._view.teamSelection1.currentText()),
            awayTeam=str(self._view.teamSelection2.currentText())
            )

        self._view.setOutputDisplay(str(result[0]))
        if (str(result[1])!="NULL"):
            self._view.updateOutputGraph(result[1][0])
    
    def _showHistoryWindow(self):
        # if self.historyWindow is None:
        self.historyWindow = HistoryWindow(self._predModel.history)
        self.historyWindow.show()

    def _connectSignalsAndSlots(self):
        self._view.outputButton.clicked.connect(partial(self._predictResult))
        self._view.historyButton.clicked.connect(partial(self._showHistoryWindow))


def main():
    ###Main function for app###
    predModel = PredModel()
    app = QApplication([])
    appWindow = AppWindow()
    appWindow.show()
    Controller(model=predModel, view=appWindow)
    sys.exit(app.exec())

### Entry point ###
if __name__ == "__main__":
    main()
    