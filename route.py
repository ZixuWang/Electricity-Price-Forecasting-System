from flask import render_template, request
def home_page():
#     date = "12.11.1994"
    price_1h = 0
    price_1d = 0
    price_1w = 0

    return render_template('index.html', price_1h=price_1h, price_1d=price_1d, price_1w=price_1w)

def getPrediction():
    price_1h = 0
    price_1d = 0
    price_1w = 0

    if(request.args.get('date')):

        date = request.args.get('date')

        # price_1h = model1(date)
        # price_1d = model2(date)
        # price_1w = model3(date)

        # get price 1h ahead
        price_1h = 3

        # get prices 1d ahead
        price_1d = 6

        # get prices 1w ahead
        price_1w = 11
    print(date)
    return render_template('index.html',date=date, price_1h=price_1h, price_1d=price_1d, price_1w=price_1w)
