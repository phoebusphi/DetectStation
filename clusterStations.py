import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, QuantileTransformer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
#from sklearn.metrics.cluster import adjusted_rand_score
#from sklearn.naive_bayes import GaussianNB, MultinomialNB
from itertools import combinations
from mpl_toolkits import mplot3d
import tabulate as tb

parametros = ['CO','PMCO','SO2','NO','NO2','NOX','O3','PM10','PM2.5',"RH","WSP","WDR","TMP"]
pol = pd.read_csv("./pollutants2018.csv")
met = pd.read_csv("./meteorology2018.csv")

pol.dateUTCShiftedDown = pd.to_datetime(pol.dateUTCShiftedDown)
met.dateUTCShiftedDown = pd.to_datetime(met.dateUTCShiftedDown)

def data(parameters):
    n = len(parameters)
    pol.dateUTCShiftedDown = pd.to_datetime(pol.dateUTCShiftedDown)
    met.dateUTCShiftedDown = pd.to_datetime(met.dateUTCShiftedDown)
    cols = ['dateUTCShiftedDown', 'id_station_id', 'original']
    stations = ['MER','TLA']
    df_parameter = {}
    df_inner = {}
    for parameter in parameters:
        df = pol if parameter in pol.id_parameter_id.unique() else met
        df_parameter[parameter] = df[(df.id_station_id.isin(stations))&
                                        (df.id_parameter_id==parameter)&
                                        (df.dateUTCShiftedDown.dt.hour.between(6,10))][cols]
        df_parameter[parameter].rename(columns = {'original':parameter}, inplace = True)
    
    df_inner = pd.merge(df_parameter[parameters[0]], df_parameter[parameters[1]], on=['dateUTCShiftedDown', 'id_station_id'],how='inner', validate='1:1')
    i=2
    while i < n:
        df_inner = pd.merge(df_inner, df_parameter[parameters[i]], on=['dateUTCShiftedDown', 'id_station_id'],how='inner', validate='1:1')
        i+=1
    df_inner.dropna(inplace=True)
    return df_inner

def clusStation(df, parameters):
    parameters = list(parameters)
    y = df.id_station_id.values
    le = LabelEncoder().fit(['TLA','MER'])
    y = le.transform(y)

    X = df[parameters].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    scaler = QuantileTransformer(n_quantiles=50, output_distribution='uniform').fit(X_train)
    X_train_transformed = scaler.transform(X_train)
    X_test_transformed = scaler.transform(X_test)
    clf = RandomForestClassifier(criterion='gini', n_estimators=3300, max_depth=12, min_samples_split=9)
    clf.fit(X_train_transformed, y_train)
    predict = clf.predict(X_test_transformed)
    score = accuracy_score(y_test, predict)
    f1 = f1_score(y_test,predict)
    cm = confusion_matrix(y_test, predict)
    s=""
    for i in parameters:
        s+=i+" "
    return {"parametros":s, "score":score, "f1":f1, "confusion_matrix":cm}

table = {"parametros":[], "score":[], "f1":[], "confusion_matrix":[]}

for k in range(2, len(parametros)):
    combination_wo_Rept = list(combinations(parametros,k))
    for combination in combination_wo_Rept:
        df = data(combination)
        classification = clusStation(df=df, parameters=combination)
        for key in table:
            table[key].append(classification[key])
        print(tb.tabulate(table,headers=table.keys(), tablefmt="pipe"))
        input()