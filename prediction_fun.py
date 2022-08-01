# The torch model I use is torch 1.8.1
# These are the lib for Transformer
from Transformermodel import TransAm
from Postional import PositionalEncoding
from Subfunctions import *

# These are the lib for LSTM
from Subfunction_LSTM import *

# Transformer model
# The input are the model, the date you want to do predict into the future, the hour you choose for the date and device.
# Since we train the model in cuda, so the device is automatically set into cuda
def predicition(
    load_trans_model = "96-48.pkl",
    load_lstm_model = "epoch_12-step_649.ckpt",
    choose_date = '2021-08-22',
    choose_hour = 0,
    devices = "cuda"
):
    # This is the transformer part
    input_window = 96
    output_window = 48
    if devices == 'cpu':
        model_transformer = torch.load(load_trans_model,map_location='cpu')
    elif devices == 'cuda':
        model_transformer = torch.load(load_trans_model)
    model_transformer.eval()
    data = getDataFromAPI_HourlyIntervals('2021-05-10',choose_date)

    devices = torch.device(devices)
    input_data, scaler = get_data(data,choose_hour,input_window,output_window,devices)

    steps = 4
    data = input_data[-1:]
    data = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window, 1))
    with torch.no_grad():
        for i in range(0, steps):
            output = model_transformer(data[-input_window:])
            # data_before = data
            data = torch.cat((data, output[-output_window:]))

    data = data.cpu().view(-1)
    data = scaler.inverse_transform(data.unsqueeze(1))
    trans_predict_7days = data[-192:-24]

  # # This is the LSTM part
    if choose_hour < 10:
        choose_hour = '0'+str(choose_hour)
    else:
        choose_hour = str(choose_hour)

    RequestedDatetime = pd.to_datetime(str(choose_date +' ' + choose_hour +':00'), format='%Y-%m-%d %H:00')
    pathToCheckpoint = load_lstm_model
    predictedDF = predict_price_LSTM(RequestedDatetime, pathToCheckpoint, historicalDays=180)
    predictedDF = predictedDF.set_index('Date')

    LSTM_predicted_7days = predictedDF
    LSTM_predicted_7days = LSTM_predicted_7days['Value'].values
    LSTM_predicted_7days = LSTM_predicted_7days.reshape(-1,1)
    combine_value = 0.5*LSTM_predicted_7days + 0.5*trans_predict_7days

    return combine_value, predictedDF
