# -*- coding: utf-8 -*-
import socket
import json
import sys
import time
from datetime import datetime
import pytz

########## Change only this

# Group secret
secret = "g078lcl"
# Group port, change the last two digits to your group number
port = 39007
# Path to your python file containing the prediction
sys.path.append("prediction_fun.py")
# Module(s) to perform prediction

from prediction_fun import predicition
from Transformermodel import TransAm
from Postional import PositionalEncoding
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Functions that are used to get your predictions
def predict_hour():
    Value, predict_df= predicition(choose_date = str(datetime.now().date()), choose_hour=int(datetime.now().hour))
    return Value[0][0]
def predict_day():
    Value, predict_df = predicition(choose_date = str(datetime.now().date()), choose_hour=int(datetime.now().hour))
    return Value[23][0]
def predict_week():
    Value, predict_df = predicition(choose_date = str(datetime.now().date()), choose_hour=int(datetime.now().hour))
    return Value[-1][0]

##########

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

host = socket.gethostbyname(socket.gethostname() + '.local') 

s.bind((host,port))
s.listen(5)

while True:
    conn, addr = s.accept()	# accept the connection
    conn.close()
    ts = str(datetime.now(pytz.timezone("Europe/Berlin")))[:-13]
    if addr[0] == "129.187.240.34":
        message = {"secret": secret, "time": ts, "hour": round(predict_hour(),4), "day": round(predict_day(),4), "week": round(predict_week(),4)}
        data = json.dumps(message)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            print("Connecting to server.")
            sock.connect(("129.187.240.34", 39000))
            print("Sending data.")
            sock.sendall(bytes(data,encoding="utf-8"))
            print("Sent: {}".format(data))
            print("Data successfully sent.")
        except:
            print("Failed, retrying.")
            time.sleep(1)
        finally:
            print("Closing connection.")
            sock.close()
