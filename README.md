# AMI_Project and prerequirement
This AMI_Project is the Project of Group_7.

In order to run the function without causing any error, you should garantee that you are in the university internet environment,
and the programm will get the token and doing the forecast automatically.

If you are not in university or in a studentenwerk, please commend line 23 and 24 in `Subfuntions.py` and line 171 and 172 in `Subfunction_LSTM.py`,
then type in the newest token and it should be fine.

In designing the web page, our group has already come up with the plotting value function (1h, 1d, and 1week) and also plotting the whole trend for 1 week, so there is no need for client.py. If you use `client.py`, it may cause error.   
## License
This project is released under the [MIT License](https://choosealicense.com/licenses/mit/)
    
## Documents
We will upload documents in the moodle.
    
## Getting Started

### Dataset
We use the followed dataset to train and test our model.
1. Electricity price data from [Montel](https://www.moodle.tum.de/pluginfile.php/2954489/mod_resource/content/2/Montel%20Web%20API%20V1.1.0.pdf)
2. Electricity production and consumption from [Smard](https://www.smard.de/home/downloadcenter/download-marktdaten#!?downloadAttributes=%7B%22selectedCategory%22:false,%22selectedSubCategory%22:false,%22selectedRegion%22:false,%22from%22:1622930400000,%22to%22:1623880799999,%22selectedFileType%22:false%7D)
3. Weather data from [DWD](https://www.dwd.de/DE/leistungen/klimadatendeutschland/klarchivstunden.html;jsessionid=7A6DEF0EA775E8F9C91B4095E4E85967.live11054?nn=16102)

    
### Download this project
```
git clone https://gitlab.ldv.ei.tum.de/ami2021/group07.git
```  

### Prerequisites
We use python3.7 and Pytorch 1.8.1, and all of packages we used are in requirements.txt.

Before using dockerfile to build a new image, make sure you are in uni or studentenwerk. Otherwise, you need to modify the token code according to the beginning of the readme, and then build the dockerfile on your computer to install our project:
```
docker build -t image_name 
```
Or you can easily use the following command to pull the image we created (make sure you are using uni or studentenwerk network)
```
docker pull eliasliu/electricity
```
### Running our model
After building a docker image, you can use the followed code to run our model
```
docker run -it -p 8888:8888 image_name
```
Then the model will run, and you should open your browser, run `localhost:8888` in your browser (don't copy and use the links generated in docker container)





