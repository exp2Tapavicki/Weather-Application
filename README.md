#Weather Forecast Application
####Proof of concept

Python3.6 is used.

Users often want to known what the weather will be like in their location, and to see recent temperature trends . The solution, will be an Ubuntu desktop application, which enables the user to specify location(s) retrieve forecasts and will hold historical data for trend analysis. The application consults services on the internet to retrieve a forecast and display it to the user.

Required Features :-
* Define at least one location
* Display weather for at least the next 24 hours
* Perform a refresh / update of weather data
* Keep a record of historical weather for each location
* Use the history to provide a temperature chart.

Technical Objectives :-

* Design and implement a GTK+ Desktop Interface
* Develop the supporting code in Python, Go, or C++
* Provide Tests for key functional components
* Identify and implement a suitable data store
* Share your code via source control e.g GitHub
* Provide appropriate documentation


DNNRegressor implementation example used from link below 

    https://stackabuse.com/using-machine-learning-to-predict-the-weather-part-3/


####Instalation instruction for Ubuntu 18.04.1 LTS

    sudo apt-get install libgtk-3-dev
    
    sudo apt-get install -y python3-venv python3-wheel python3-dev
    
    sudo apt-get install -y libgirepository1.0-dev build-essential \
      libbz2-dev libreadline-dev libssl-dev zlib1g-dev libsqlite3-dev wget \
      curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libcairo2-dev
    
    sudo apt-get install python3-tk 	
    
    pip3.6 install PyGObject   
    pip3.6 install gobject
    pip3.6 install Pillow
    pip3.6 install request
    pip3.6 install matplotlib
    pip3.6 install numpy
    pip3.6 install pandas
    pip3.6 install sklearn
    pip3.6 install tensorflow
    
Unpack city.list.json.gz with gunzip    
    
    gunzip city.list.json.gz

###Weather data provided by openweathermap.org

    appid=8747776c71eb8c06a27001b6d598ff2b

Get data for London
   
    http://api.openweathermap.org/data/2.5/forecast?id=2643743&APPID=8747776c71eb8c06a27001b6d598ff2b&units=metric
    https://openweathermap.org/forecast5#cityid5
    https://openweathermap.org/bulk
    http://bulk.openweathermap.org/sample/
    
Cities ids are downloaded from link below

    http://bulk.openweathermap.org/sample/
