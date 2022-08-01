from numpy import array, float16
from pytorch_forecasting.data import (
    TimeSeriesDataSet,
)
from datetime import timedelta
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
)
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_forecasting.models import RecurrentNetwork
from pytorch_forecasting.metrics import RMSE
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import requests
import json
import datetime
import ast
import time
from io import StringIO
import numpy as np
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 2000)


#Note: Usage Example at very bottom.

# ******************************************************************************************************************
# ********************************************** SMARD+MONTEL DATAS ************************************************
# ******************************************************************************************************************

def get_data_for_prediction(requestedTimeStamp,
                            numberOfDaysInPast=60):
    """
    :param requestedTimeStamp: Date and Time of the request. Should be a pandas datetime object
    :param numberOfDaysInPast: Int value of days in the past needed for prediction
    :return: Full dataset of required information
             Output Columns: (['Wind Onshore[MWh]', 'Steinkohle[MWh]', 'Erdgas[MWh]',
                              'Gesamt[MWh]', 'Value', 'Base', 'Peak']
    """

    endDate = requestedTimeStamp.strftime('%Y-%m-%d')
    startDate = requestedTimeStamp - datetime.timedelta(days=numberOfDaysInPast)
    montelStartDate = startDate.strftime('%Y-%m-%d')

    # Get MONTEL API DATA
    montelApiDf = getDataFromAPI_HourlyIntervals(startDate=montelStartDate, endDate=endDate)

    begin_timestamp = startDate  # From last Value of data
    end_timestamp = str(montelApiDf.iloc[-1].name)

    montelMissingData = montelApiDf.loc[begin_timestamp:end_timestamp]

    # GET SMARD DATA

    realizedPower = [1004071, 1004067, 1004069, 1004070]
    realizedConsumption = [5000410]
    #5000410
    modules_realized = realizedPower
    modules_consumed = realizedConsumption

    Days_behind = numberOfDaysInPast + 1

    EnergyProd = requestSmardData(modulIDs=modules_realized,
                                  timestamp_from_in_milliseconds=(int(time.time()) * 1000) - (
                                          Days_behind * 24 * 3600) * 1000)
    EnergyUsage = requestSmardData(modulIDs=modules_consumed,
                                   timestamp_from_in_milliseconds=(int(time.time()) * 1000) - (
                                           Days_behind * 24 * 3600) * 1000)




    # CLEAN UP DATA. REMOVE '-' from unknowns
    EnergyUsage['Datum'] = EnergyUsage['Datum'] + '-' + EnergyUsage['Uhrzeit']
    EnergyUsage = EnergyUsage.drop(columns=['Uhrzeit'])
    EnergyUsage['Datum'] = pd.to_datetime(EnergyUsage['Datum'], format='%d.%m.%Y-%H:%M')
    EnergyUsage = EnergyUsage.rename(columns={'Datum': 'Date', 'Gesamt (Netzlast)[MWh]': 'Gesamt[MWh]'})
    EnergyUsage['Gesamt[MWh]'] = (EnergyUsage['Gesamt[MWh]'].replace('-', np.nan)).astype(np.float64)

    EnergyProd['Datum'] = EnergyProd['Datum'] + '-' + EnergyProd['Uhrzeit']
    EnergyProd = EnergyProd.drop(columns=['Uhrzeit'])
    EnergyProd['Datum'] = pd.to_datetime(EnergyProd['Datum'], format='%d.%m.%Y-%H:%M')
    EnergyProd = EnergyProd.rename(columns={'Datum': 'Date'})
    EnergyProd['Wind Onshore[MWh]'] = (EnergyProd['Wind Onshore[MWh]'].replace('-', np.nan)).astype(np.float64)
    EnergyProd['Steinkohle[MWh]'] = (EnergyProd['Steinkohle[MWh]'].replace('-', np.nan)).astype(np.float64)
    EnergyProd['Erdgas[MWh]'] = (EnergyProd['Erdgas[MWh]'].replace('-', np.nan)).astype(np.float64)
    EnergyProd['Pumpspeicher[MWh]'] = (EnergyProd['Pumpspeicher[MWh]'].replace('-', np.nan)).astype(np.float64)


    EnergyUsage = EnergyUsage.resample('H', on='Date').mean()
    EnergyProd = EnergyProd.resample('H', on='Date').mean()

    # Remove Duplicates
    EnergyProd = EnergyProd.loc[~EnergyProd.index.duplicated(keep='first')]
    EnergyUsage = EnergyUsage.loc[~EnergyUsage.index.duplicated(keep='first')]
    montelMissingData = montelMissingData.loc[~montelMissingData.index.duplicated(keep='first')]

    MissingDataset = pd.concat([EnergyProd,EnergyUsage, montelMissingData], axis=1)

    MissingDataset = MissingDataset.dropna()


    return MissingDataset


# ******************************************************************************************************************
# ********************************************** SMARD DATA REQUEST ************************************************
# ******************************************************************************************************************
def requestSmardData(
        modulIDs=[8004169],
        timestamp_from_in_milliseconds=(int(time.time()) * 1000) - (3 * 3600) * 1000,
        timestamp_to_in_milliseconds=(int(time.time()) * 1000),
        region="DE",
        language="de",
        type="discrete"
):
    '''
    Requests and returns a dataframe of SMARD.de data

    :param modulIDs: ID of desired modules
    :param timestamp_from_in_milliseconds: Time from current
    :param timestamp_to_in_milliseconds:  Desired timepoint
    :param region: Region of data
    :param language: Language of data
    :param type: Type of data
    :return: Dataframe
    '''
    # http request content
    url = "https://www.smard.de/nip-download-manager/nip/download/market-data"
    body = json.dumps({
        "request_form": [
            {
                "format": "CSV",
                "moduleIds": modulIDs,
                "region": region,
                "timestamp_from": timestamp_from_in_milliseconds,
                "timestamp_to": timestamp_to_in_milliseconds,
                "type": type,
                "language": language
            }]})

    # http response
    data = requests.post(url, body)
    # create pandas dataframe out of response string (csv)
    df = pd.read_csv(StringIO(data.text), sep=';')

    return df


# ******************************************************************************************************************
# ********************************************** MONTEL API REQUEST ************************************************
# ******************************************************************************************************************

def getDataFromAPI_HourlyIntervals(startDate, endDate):
    """
    Input Data should be in the following form:
    year-month-day

    :param startDate: '2015-01-01'
    :param endDate: '2019-01-01'
    :return: Montel Api Dataframe in 15min intervals
    """

    def repeatlist(list_before, i):
        list_after = [val for val in list_before for k in range(i)]
        return list_after

    # Get Bearer Token
    page = requests.get('https://coop.eikon.tum.de/mbt/mbt.json')
    dictsoup = (ast.literal_eval(page.text))
    token = str((dictsoup['access_token']))
    # token = "Ju330josy0VMr1IJmuAfzphLF_TXOvh_jIgU1yzMpQVFgN_4l8RYPgOFRyZEOwDCSsqQR9Dxqv4oGU06P_7Fp3zeS-MYVdWjqtWuGqYVpaR7yWvMYlM19Ffhi4grc--ISDhYhND5Z-Ys3rvx9WPo40KolBFFwg2oD4KPOis9yMHd3OEk6Ol4BUKZfgzZ8jiAGhf4qhw7qa_Mw3x-C10rk80K3jdO7QRkyDnyBfQqiIMwTxriUOB0yiEwS_5uQLLVZ4dOXFMhEamHS2COWtbiYQm7lNq9iMximRrJoVEKdXcL_bBb7mNWupaN2s7gklINP0TThg9UJXpHnKg-Rtbcu8gPWSKT0fsljtnkgBE0KhX0EXlUpagHPx24oMhr8IVOS12DibzKmnmuqp0Mlr8-Bas0BJ4C2valHxPwSc3zQHo"
    url = 'http://api.montelnews.com/spot/getprices'
    headers = {"Authorization": 'Bearer ' + token}
    params = {
        'spotKey': '14',
        'fields': ['Base', 'Peak', 'Hours'],
        'fromDate': str(startDate),
        'toDate': str(endDate),
        'currency': 'eur',
        'sortType': 'Ascending'
    }

    response = requests.get(url, headers=headers, params=params)
    data = response.json()
    value = []
    Timespan = []
    date = []
    base = []
    peak = []

    for parts in data['Elements']:  # if we create extrenal, can hold data in data1
        date.append(parts['Date'])
        base.append(parts['Base'])
        peak.append(parts['Peak'])
        for df in parts['TimeSpans']:
            value.append(df['Value'])
            Timespan.append(df['TimeSpan'])

    date = repeatlist(date, 24)
    base = repeatlist(base, 24)
    peak = repeatlist(peak, 24)
    MontelData = pd.DataFrame(list(zip(date, Timespan, value, base, peak)),
                              columns=['Date', 'Timespan', 'Value', 'Base', 'Peak'])

    MontelData[['time', 'end']] = MontelData['Timespan'].str.split('-', 1, expand=True)
    MontelData = MontelData.drop(columns=['Timespan', 'end'])
    MontelData['Date'] = MontelData['Date'].str.replace('T00:00:00.0000000', '')
    MontelData['Date'] = MontelData['Date'] + '-' + MontelData['time']
    #MontelData['Date'] = MontelData[~MontelData['Date'].str.contains('dst')]
    MontelData = MontelData.drop(columns=['time'])
    MontelData['Date'] = pd.to_datetime(MontelData['Date'], format='%Y-%m-%d-%H:00')
    MontelData15 = MontelData.set_index('Date')
    MontelData15 = MontelData15.resample('H').mean()
    MontelData15 = MontelData15.interpolate(method='time')  # FINAL DATA
    MontelData15 = MontelData15.dropna()

    return MontelData15.loc[~MontelData15.index.duplicated(keep='first')]


# Three Datasets
# Electricity Price data from Montel
# Electricity production and consumption from Smard

# ******************************************************************************************************************
# ********************************************** MULTI VARIATE LSTM ************************************************
# ******************************************************************************************************************

# returns Dataframe of the following predicted variables inorder by hour. :
# Erdgas[MWh], Gesamt[MWh], Steinkohle[MWh], Wind Onshore[MWh], Value
# All Used variables are from either MONTELAPI or ENERGYPRODUCTION


def predict_price_LSTM(targetDatetime,
                       pathToCheckpoint,
                       historicalDays=60,
                       makePredicition=True,
                       loadFromCheckpoint=1,
                       trainingEnabled=0,
                       gpuEnabled=0,
                       batch_size=16,
                       loss_Function=RMSE(),
                       epochsNumber=90,
                       numberLayers=2,
                       hiddenSize=512,
                       numWorkers=8
                       ):
    """
    :param targetDatetime: Date and time of requested day to predict. Should be given as a pandas datetime object
    :param pathToCheckpoint: Computer Path to the LSTM model Checkpoint
    :param historicalDays: Number of days prior to the requested day to predict. Minimum number = 14. Default = 21
    :param makePredicition: Set Equal to True if you want a prediction at the output. Default = True
    :param loadFromCheckpoint: If activated, Checkpoint will be loaded into model. Default = 1
    :param trainingEnabled: If activated, training will be enabled. Default = 0
    :param gpuEnabled: If gpu available, Model will be trained with GPU at target position  Default = 0
    :param batch_size: For training. Default = 16
    :param loss_Function: Loss function for training. Default = RMSE
    :param epochsNumber: Number of epochs for training. Default = 90
    :param numberLayers: Number of layers in model to be created. Default = 2
    :param hiddenSize: Number of hidden states in lstm. Default = 512
    :param numWorkers: number of workers specified for dataloader. Default = 8
    :return: Returns a dataframe of predicted values 1 hour intervals.
             Also return individual steps of 1 hour, 1 day and 1 week ahead predictions
    """
    # ProcessFlags
    hourlyData = 1

    if loadFromCheckpoint == 1:
        chk_path = pathToCheckpoint

    if hourlyData == 1:
        max_prediction_length = 168  # forecast 1 week
        max_encoder_length = 168 * 2  # use 2 weeks of history

        data = get_data_for_prediction(targetDatetime, historicalDays)

        data['GroupID'] = 'A'
        data['time_idx'] = array(range(data.shape[0]))
        data.reset_index(level=0, inplace=True)
        Array1= np.array(data['Wind Onshore[MWh]'])
        Array2= np.array(data['Steinkohle[MWh]'])
        Array3= np.array(data['Erdgas[MWh]'])

        pos=0
        for i in Array1:
            if int(i)<10:
                Array1[pos] = i * 1000
            pos =pos+1

        pos=0
        for i in Array2:
            if int(i) < 10:
                Array2[pos] = i * 1000
            pos = pos + 1

        pos = 0
        for i in Array3:
            if int(i) < 10:
                Array3[pos] = i * 1000
            pos = pos + 1
        data.drop(columns={'Wind Onshore[MWh]','Steinkohle[MWh]','Erdgas[MWh]'})

        data['Wind Onshore[MWh]']= Array1
        data['Steinkohle[MWh]']= Array2
        data['Erdgas[MWh]']= Array3
        #print(data)




        # data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y %H:00')

    # **************************************************************************************************************
    # ********************************************* PREPROCESSING **************************************************
    # **************************************************************************************************************

    # fill in any missing values historical data may have

    training_cutoff = data["Date"].max() - timedelta(days=7)
    groupind = data['GroupID']
    groupind2 = data['time_idx']
    groupind3 = data['Date']
    data = data.drop(columns=['GroupID', 'time_idx', 'Date'])
    data = data.apply(lambda x: x.fillna(x.mean()))

    data = pd.concat([data, groupind], axis=1)
    data = pd.concat([data, groupind2], axis=1)
    data = pd.concat([data, groupind3], axis=1)
    data = data.dropna()


    # Preprocessing of only the important variables to be used
    # Erdgas[MWh], Gesamt[MWh], Steinkohle[MWh],Wind Onshore[MWh], Value

    groupind = data['GroupID']
    groupind2 = data['time_idx']
    groupind3 = data['Date']
    data = data.drop(columns=['GroupID', 'time_idx', 'Date', 'Base', 'Peak','Pumpspeicher[MWh]','Wind Onshore[MWh]',
                'Steinkohle[MWh]',
                'Erdgas[MWh]',
                'Gesamt[MWh]',])


    # scaled data
    scaler = MinMaxScaler(feature_range=(-1, 1))

    data = pd.DataFrame(scaler.fit_transform(data.astype(float16)))  # fix overflow error with dtype

    data = data.rename(columns={
                                0: 'Value',

                                })

    data = pd.concat([data, groupind], axis=1)
    data = pd.concat([data, groupind2], axis=1)
    data = pd.concat([data, groupind3], axis=1)

    data = data.dropna()
    # print(data)

    # **************************************************************************************************************
    # *******************************************CREATION OF TRAINING SET*******************************************
    # **************************************************************************************************************
    if trainingEnabled == 1:
        training = TimeSeriesDataSet(
            data[lambda x: x.Date <= training_cutoff],
            time_idx="time_idx",
            target='Value'

            ,
            group_ids=["GroupID"],
            min_encoder_length=max_encoder_length,
            max_encoder_length=max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=max_prediction_length,
            static_categoricals=["GroupID"],
            time_varying_known_reals=[
                "Date",
                "time_idx"

            ],

            time_varying_unknown_reals=['Value']
        )

        validation = TimeSeriesDataSet.from_dataset(
            training, data, predict=True, stop_randomization=True
        )

        # create Data-loader for model
        train_dataloader = training.to_dataloader(
            train=True, batch_size=batch_size, num_workers=numWorkers)
        val_dataloader = validation.to_dataloader(
            train=False, batch_size=1, num_workers=numWorkers)

        # **************************************************************************************************************
        # *********************************************** TRAINING *****************************************************
        # **************************************************************************************************************

        lr_logger = LearningRateMonitor()

        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=1e-4,
            patience=10,
            verbose=False,
            mode="min"
        )

        trainer = pl.Trainer(
            max_epochs=epochsNumber,
            gpus=gpuEnabled,
            gradient_clip_val=0.1,
            limit_train_batches=50,
            callbacks=[lr_logger, early_stop_callback],
        )

        model = RecurrentNetwork.from_dataset(
            training,
            learning_rate=0.03,
            hidden_size=hiddenSize,
            rnn_layers=numberLayers,
            dropout=0.2,
            loss=RMSE(),
            log_interval=20,
            reduce_on_plateau_patience=4,
        )
        if loadFromCheckpoint == 1:
            model = model.load_from_checkpoint(chk_path)

        trainer.fit(model,
                    train_dataloaders=train_dataloader,
                    val_dataloaders=val_dataloader
                    )
    else:

        training = TimeSeriesDataSet(
            data[lambda x: x.Date <= training_cutoff],
            time_idx="time_idx",
            target='Value'

            ,
            group_ids=["GroupID"],
            min_encoder_length=max_encoder_length,
            max_encoder_length=max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=max_prediction_length,
            static_categoricals=["GroupID"],
            time_varying_known_reals=[
                "Date",
                "time_idx"

            ],

            time_varying_unknown_reals=[
                'Value'
            ]
        )

        model = RecurrentNetwork.from_dataset(
            training,
            learning_rate=0.03,
            hidden_size=512,
            rnn_layers=4,
            dropout=0.2,
            loss=loss_Function,
            log_interval=20,
            reduce_on_plateau_patience=4,
        )

        model = model.load_from_checkpoint(chk_path)

    # **************************************************************************************************************
    # ********************************************** PREDICTION ****************************************************
    # **************************************************************************************************************

    if makePredicition:

        # Create Prediction_dataloader based of previous history
        encoder_data = data[lambda x: x.time_idx > x.time_idx.max() - max_encoder_length]

        # select last known data point and create decoder data from it by repeating it and incrementing the month
        # in a real world dataset
        last_data = data[lambda x: x.time_idx == x.time_idx.max()]

        decoder_data = pd.concat(
            [last_data.assign(Date=lambda x: x.Date + pd.offsets.Hour(i)) for i in
             range(1, max_prediction_length + 1)],
            ignore_index=True,
        )

        # add time index consistent with "data"
        timeindexDF = pd.DataFrame(array(range(max_prediction_length)))
        decoder_data["time_idx"] = timeindexDF[0] + 1 + decoder_data["time_idx"]


        # combine encoder and decoder data
        new_prediction_data = pd.concat([encoder_data, decoder_data], ignore_index=True)

        preds, index = model.predict(new_prediction_data,
                                     mode="raw",
                                     return_index=True,
                                     fast_dev_run=True
                                     )

        listOfPreds = preds['prediction']
        # print(listOfPreds.shape)
        Data_Pred = pd.DataFrame(array((listOfPreds.squeeze(2)))).T
        #for i in range(4):
        #    Temp = pd.DataFrame(array((listOfPreds[i + 1].squeeze(2)))).T
        #    Temp = Temp.rename(columns={0: i + 1})
        #    Data_Pred = pd.concat([Data_Pred, Temp], axis=1)

        Data_Pred = scaler.inverse_transform(Data_Pred)
        Data_Pred = pd.DataFrame(Data_Pred).rename(columns={
            0: 'Value'})

        DataTimePred = [(targetDatetime + datetime.timedelta(hours=1))
                        + datetime.timedelta(hours=x) for x in range(max_prediction_length)]

        Data_Pred['Date'] = (np.array(DataTimePred))

        return Data_Pred
    else:
        return None