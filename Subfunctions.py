import torch
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import requests

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
    # token='sGnXKyGQIuCs1zLTe3fYAzSqFUycJYJcj4sMQPma2VqmP8qVyyOk0mYuabF4FJdZK9PXdqj5waHiDnS_xe4bKyVzhRZGE8rB7Ovkx4gTi6KNcQ0U5eKAkIknLjLosaZD0zvh0oS0WcK9Y65BnW87LjavYyrhwAvgPz-okl3CPuxdVgL2wERM79E2Zo2GXxPiEfHTJB7udHiYIDCV2A4coCbyZBhqk2rst-DuwXi6HNI4n1SuLL0HRz796zuxHwmZa40OdGIIcUQeRxN9dtBm202fGCc9hPY6L8_LjgJIRQP22Gczj_auiDP45DCCD5erEuoFk98WRv6dnKcuq18zSYe_M2PwD0XAOSxsAl3yfIVuRqpeYAB_U43ZfEcgdLQTfEundeY_fOj6VXlJ3nD3ucDq0zWjMCLUYFtE0N5hQzM'
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
    MontelData['Date'] = MontelData[~MontelData['Date'].str.contains('dst')]
    MontelData = MontelData.drop(columns=['time'])
    MontelData['Date'] = pd.to_datetime(MontelData['Date'], format='%Y-%m-%d-%H:00')
    MontelData15 = MontelData.set_index('Date')
    MontelData15 = MontelData15.resample('H').mean()
    MontelData15 = MontelData15.interpolate(method='time')  # FINAL DATA
    MontelData15 = MontelData15.dropna()

    return MontelData15.loc[~MontelData15.index.duplicated(keep='first')]

def get_data(input_data, choose_hour,input_window, output_window, device):
    df = input_data
    df1 = df['Value']
    data = df1.values
    if choose_hour == 23:
        data = data[:]
    else:
        data = data[:choose_hour-24+1]

    amplitude = data[-input_window*3:]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    amplitude = scaler.fit_transform(amplitude.reshape(-1, 1)).reshape(-1)
    amplitude = amplitude.astype(np.float16)
    output_data = amplitude
    output_sequence = create_inout_sequences(output_data, input_window, output_window)
    output_sequence = output_sequence[:-output_window]
    return output_sequence.to(device), scaler

def create_inout_sequences(input_data, tw, output_window):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = np.append(input_data[i:i+tw][:-output_window] , output_window * [0])
        train_label = input_data[i:i+tw]
        inout_seq.append((train_seq ,train_label))
    return torch.FloatTensor(inout_seq)

def get_batch(source, i,batch_size,input_window):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i+seq_len]
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window,1)) # 1 is feature size
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window,1))
    return input, target
