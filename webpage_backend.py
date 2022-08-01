from flask import Flask, render_template, request
import datetime
import json
from Transformermodel import TransAm
from Postional import PositionalEncoding
# from main_yuchong import main
from prediction_fun import predicition
import numpy as np
app=Flask(__name__,template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
#delete unnecessary methods
#def home_page():

#return render_template('index.html', title='Home')'''
def getPrediction():
    #date=""
    price_1h=0
    price_1d=0
    price_1w=0
    values = [0,0,0,0,0,0,0,0]
    labels= ['2021-09-14 1:00','2021-09-14 2:00','2021-09-14 3:00','2021-09-14 4:00','2021-09-14 5:00','2021-09-14 6:00','2021-09-14 7:00','2021-09-14 8:00']


    if  request.method=='POST':
        date=request.form['date']
        if request.form['button']=='Predict':
            print(date)

            #call the function
            #prices=Yuchong's_function()

            #get price 1h ahead
            price_1h=5

            #get prices 1d ahead
            price_1d=8

            #get prices 1w ahead
            price_1w=11

    return render_template('index.html',values=values, labels=labels,  price_1h=price_1h, price_1d=price_1d, price_1w=price_1w)


@app.route('/getPrediction/')
def getPrediction_By_Date():
    # initial the value
    price_1h, price_1d, price_1w = 0,0,0
    date = datetime.datetime.now()

    if(request.args.get('date')):

        # get the chosen date
        date = request.args.get('date')
        # print(date)
        my_date = datetime.datetime.strptime(date, "%d/%m/%Y - %H:%M")
        date_=my_date.date().strftime('%Y-%m-%d')
        time = my_date.time().hour


        # get price 1h ahead
        prices, dataFrame =predicition(load_trans_model = "96-48.pkl",
               load_lstm_model = "epoch_12-step_649.ckpt",
               choose_date = date_,
               choose_hour = time,
               devices = "cpu")

        # np.set_printoptions(precision=2)
        prices = np.around(prices, 2)
        #array of all price values
        # get prices 1h ahead
        price_1h = prices[0]

        # get prices 1d ahead
        price_1d = prices[23]

        # get prices 1w ahead
        price_1w = prices[-1]

        final_price = prices.reshape(1,-1)
        final_price = final_price.squeeze()
        final_price = final_price.tolist()
        labels = [label.strftime('%Y-%m-%d %H:00') for label in dataFrame.index.tolist()]




    return render_template('index.html', values=final_price, labels=labels, date=date, price_1h=price_1h, price_1d=price_1d, price_1w=price_1w)

if __name__=='__main__':
    app.run(port=8888,host='0.0.0.0')
