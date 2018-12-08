import gi

import pandas as pd
import tensorflow as tf
from sklearn.metrics import explained_variance_score, \
    mean_absolute_error, \
    median_absolute_error
from sklearn.model_selection import train_test_split

gi.require_version('Gtk', '3.0')
from gi.repository import Gtk
from gi.repository.GdkPixbuf import Pixbuf

import http.client
import json
from PIL import Image
import requests
import datetime
import os
import glob

import matplotlib.pyplot as plt
import numpy as np
# Possibly this rendering backend is broken currently
# from matplotlib.backends.backend_gtk3agg import FigureCanvasGTK3Agg as FigureCanvas
import csv


class MyWindow(Gtk.Window):

    def on_match_selected(self, completion, treemodel, treeiter):
        connection = http.client.HTTPConnection('api.openweathermap.org', 80, timeout=10)
        print(treemodel[treeiter][2])
        self.location_id = treemodel[treeiter][2]
        self.location_city = treemodel[treeiter][completion.get_text_column()]
        connection.request("GET", "/data/2.5/forecast?id=" + str(self.location_id) + "&APPID=8747776c71eb8c06a27001b6d598ff2b&units=metric")
        response = connection.getresponse()
        json_data_array = json.loads(response.read().decode())
        self.save_json_data(json_data_array, self.location_city, str(datetime.datetime.today().strftime('%Y-%m-%d')), "json_data-0.json")

        self.pages = [None] * (len(json_data_array['list']) - 1)
        for child in self.notebook.get_children():
            self.notebook.remove(child)

        for i in range(0, len(json_data_array['list']) // 10):
            weather = json_data_array['list'][i]
            img = Image.open(requests.get('http://openweathermap.org/img/w/' + weather['weather'][0]['icon'] + ".png", stream=True).raw)
            img.save(weather['weather'][0]['icon'] + ".png", 'PNG')

            image = Gtk.Image()
            image.set_halign(Gtk.Align.CENTER)
            image.set_valign(Gtk.Align.CENTER)
            image.set_from_pixbuf(Pixbuf.new_from_file(weather['weather'][0]['icon'] + ".png"))

            page = Gtk.Box()
            page.set_border_width(10)
            page.add(Gtk.Label(json.dumps(weather, indent=4)))
            page.add(Gtk.Label(weather['dt_txt']))
            page.add(image)
            page.set_halign(Gtk.Align.CENTER)
            page.set_valign(Gtk.Align.CENTER)
            self.pages.append(page)
            self.notebook.append_page(page, Gtk.Label(weather['dt_txt']))
        self.notebook.show_all()

    def default_location(self):
        connection = http.client.HTTPConnection('api.openweathermap.org', 80, timeout=10)
        connection.request("GET", "/data/2.5/forecast?id=" + str(self.location_id) + "&APPID=8747776c71eb8c06a27001b6d598ff2b&units=metric")
        response = connection.getresponse()
        json_data_array = json.loads(response.read().decode())
        self.save_json_data(json_data_array, self.location_city, str(datetime.datetime.today().strftime('%Y-%m-%d')), "json_data-0.json")

        self.pages = [None] * (len(json_data_array['list']) - 1)
        for child in self.notebook.get_children():
            self.notebook.remove(child)

        for i in range(0, len(json_data_array['list']) // 10):
            weather = json_data_array['list'][i]
            img = Image.open(requests.get('http://openweathermap.org/img/w/' + weather['weather'][0]['icon'] + ".png", stream=True).raw)
            img.save(weather['weather'][0]['icon'] + ".png", 'PNG')

            image = Gtk.Image()
            image.set_halign(Gtk.Align.CENTER)
            image.set_valign(Gtk.Align.CENTER)
            image.set_from_pixbuf(Pixbuf.new_from_file(weather['weather'][0]['icon'] + ".png"))

            page = Gtk.Box()
            page.set_border_width(10)
            page.add(Gtk.Label(json.dumps(weather, indent=4)))
            page.add(Gtk.Label(weather['dt_txt']))
            page.add(image)
            page.set_halign(Gtk.Align.CENTER)
            page.set_valign(Gtk.Align.CENTER)
            self.pages.append(page)
            self.notebook.append_page(page, Gtk.Label(weather['dt_txt']))
        self.notebook.show_all()
        self.entry.set_text(self.location_city)

    def flattenjson(self, json_data, delim):
        val = {}
        for key in json_data.keys():
            if isinstance(json_data[key], dict):
                get = self.flattenjson(json_data[key], delim)
                for keyTmp in get.keys():
                    val[key + delim + keyTmp] = get[keyTmp]
            elif isinstance(json_data[key], list):
                for i in range(0, len(json_data[key])):
                    if isinstance(json_data[key][i], dict):
                        get = self.flattenjson(json_data[key][i], delim)
                        for j in range(0, len(list(get.keys()))):
                            val[key + delim + str(i) + delim + str(list(get.keys())[j])] = get[list(get.keys())[j]]
                    elif isinstance(json_data[key], list):
                        for k in range(0, len(json_data[key])):
                            if isinstance(json_data[key][k], dict):
                                get = self.flattenjson(json_data[key][k], delim)
                                for l in range(0, len(list(get.keys()))):
                                    val[key + delim + str(k) + delim + str(list(get.keys())[l])] = get[list(get.keys())[l]]
                            else:
                                val[key] = json_data[key]
                    else:
                        val[key] = json_data[key]
            else:
                val[key] = json_data[key]

        return val

    def save_json_data(self, json_data, city_name, directory_name, file_name):
        cwd = os.getcwd()  # Get the current working directory (cwd)
        path = cwd + "/../data/" + city_name + "/" + directory_name + "/"
        if not os.path.exists(path):
            os.makedirs(path)

        max = -1
        for file in glob.glob(path + "*.json"):
            file = file.replace(path + "json_data-", "")
            file = file.replace(".json", "")
            if (max < int(file)):
                max = int(file)

        if (max == -1):
            max = 0
        else:
            max = max + 1

        file_name = "json_data-" + str(max) + ".json"

        with open(path + file_name, 'w+') as outfile:
            json.dump(json_data, outfile)

        file_name_csv = "data-" + str(max) + ".csv"

        f = csv.writer(open(path + file_name_csv, "w"))
        d = dict()

        for i in range(0, len(json_data['list'])):
            dictAsList = None
            if ('rain' not in json_data['list'][i].keys() or len(json_data['list'][i]['rain']) == 0):
                if ('rain' in json_data['list'][i].keys()):
                    del json_data['list'][i]['rain']
                dictAsList = [(k, v) for k, v in json_data['list'][i].items()]
                for j in range(len(dictAsList)):
                    if dictAsList[j][0] == 'wind':
                        dictAsList.insert(j+1, ('rain', {'0h':0.0}))
                        break

            if (dictAsList != None):
                d = dict.copy(d)
                for x, y in dictAsList:
                    d.update({x:y})
                json_data['list'][i] = d

            flatten_element = self.flattenjson(json_data['list'][i], "_")
            if (i == 0):
                f.writerow(list(self.flattenjson(json_data['list'][0], "_").keys()))
            f.writerow(list(flatten_element.values()))

    def on_refresh_data_clicked(self, button):
        self.default_location()

    def on_trained_clicked(self, button):
        file_name = None
        cwd = os.getcwd()  # Get the current working directory (cwd)
        path = cwd + "/../data/" + self.location_city + "/" + str(datetime.datetime.today().strftime('%Y-%m-%d')) + "/"
        if os.path.exists(path):
            max = -1
            for file in glob.glob(path + "*.json"):
                file = file.replace(path + "json_data-", "")
                file = file.replace(".json", "")
                if (max < int(file)):
                    max = int(file)
            file_name = "data-" + str(max) + ".csv"
        # read in the csv data into a pandas data frame and set the date as the index
        df = pd.read_csv(path + file_name).set_index('dt')

        # execute the describe() function and transpose the output so that it doesn't overflow the width of the screen
        df.describe().T

        df.info()

        # First drop the maxtempm and mintempm from the dataframe
        df = df.drop(['sys_pod', 'dt_txt', 'weather_0_icon', 'weather_0_description', 'weather_0_main'], axis=1)

        # X will be a pandas dataframe of all columns except main_temp
        X = df[[col for col in df.columns if col != 'main_temp']]

        # y will be a pandas series of the main_temp
        y = df['main_temp']

        # split data into training set and a temporary set using sklearn.model_selection.traing_test_split
        X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.2, random_state=23)

        # take the remaining 20% of data in X_tmp, y_tmp and split them evenly
        self.X_test, X_val, self.y_test, y_val = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=23)

        X_train.shape, self.X_test.shape, X_val.shape
        print("Training instances   {}, Training features   {}".format(X_train.shape[0], X_train.shape[1]))
        print("Validation instances {}, Validation features {}".format(X_val.shape[0], X_val.shape[1]))
        print("Testing instances    {}, Testing features    {}".format(self.X_test.shape[0], self.X_test.shape[1]))

        feature_cols = [tf.feature_column.numeric_column(col) for col in X.columns]

        self.regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols,
                                              hidden_units=[50, 50],
                                              model_dir='tf_wx_model')

        def wx_input_fn(X, y, num_epochs=None, shuffle=True, batch_size=40):
            return tf.estimator.inputs.pandas_input_fn(x=X,
                                                       y=y,
                                                       num_epochs=num_epochs,
                                                       shuffle=shuffle,
                                                       batch_size=batch_size)

        evaluations = []
        STEPS = 40
        for i in range(10):
            self.regressor.train(input_fn=wx_input_fn(X_train, y_train), steps=STEPS)
            evaluations.append(self.regressor.evaluate(input_fn=wx_input_fn(X_val,
                                                                       y_val,
                                                                       num_epochs=1,
                                                                       shuffle=False)))

        # manually set the parameters of the figure to and appropriate size
        plt.rcParams['figure.figsize'] = [14, 10]

        loss_values = [ev['loss'] for ev in evaluations]
        training_steps = [ev['global_step'] for ev in evaluations]

        plt.scatter(x=training_steps, y=loss_values)
        plt.xlabel('Training steps (Epochs = steps / 2)')
        plt.ylabel('Loss (SSE)')
        plt.show()

    def on_predict_clicked(self, button):
        def wx_input_fn(X, y=None, num_epochs=None, shuffle=True, batch_size=100):
            return tf.estimator.inputs.pandas_input_fn(x=X,
                                                       y=y,
                                                       num_epochs=num_epochs,
                                                       shuffle=shuffle,
                                                       batch_size=batch_size)

        pred = self.regressor.predict(input_fn=wx_input_fn(self.X_test,
                                                      num_epochs=1,
                                                      shuffle=False))
        predictions = np.array([p['predictions'][0] for p in pred])

        print("The Explained Variance: %.2f" % explained_variance_score(
            self.y_test, predictions))
        print("The Mean Absolute Error: %.2f degrees Celcius" % mean_absolute_error(
            self.y_test, predictions))
        print("The Median Absolute Error: %.2f degrees Celcius" % median_absolute_error(
            self.y_test, predictions))


    def on_draw_chart_clicked(self, button):
        # fig = mpl.Figure(figsize=(5, 5), dpi=100)
        # ax = fig.add_subplot(111, projection='polar')
        #
        # N = 20
        # theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
        # radii = 10 * np.random.rand(N)
        # width = np.pi / 4 * np.random.rand(N)
        #
        # bars = ax.bar(theta, radii, width=width, bottom=0.0)
        #
        # for r, bar in zip(radii, bars):
        #     bar.set_facecolor(cm.jet(r / 10.))
        #     bar.set_alpha(0.5)
        #
        # ax.plot()

        file_name = None
        cwd = os.getcwd()  # Get the current working directory (cwd)
        path = cwd + "/../data/" + self.location_city + "/" + str(datetime.datetime.today().strftime('%Y-%m-%d')) + "/"
        if os.path.exists(path):
            max = -1
            for file in glob.glob(path + "*.json"):
                file = file.replace(path + "json_data-", "")
                file = file.replace(".json", "")
                if (max < int(file)):
                    max = int(file)
            file_name = "data-" + str(max) + ".csv"

        data = np.genfromtxt(path + file_name, delimiter=',', skip_header=1, dtype=None, encoding='utf-8', names=['dt', 'main_temp', 'main_temp_min', 'main_temp_max', 'main_pressure', 'main_sea_level', 'main_grnd_level', 'main_humidity', 'main_temp_kf', 'weather_0_id', 'weather_0_main', 'weather_0_description', 'weather_0_icon', 'clouds_all', 'wind_speed', 'wind_deg', 'rain_0h', 'sys_pod', 'dt_txt'])

        plt.title("Temperature chart")
        plt.xlabel("date")
        plt.ylabel("degrees Celsius")
        plt.bar(data['dt_txt'], data['main_temp'], color="blue")
        plt.show()

        # self.scrolled_window = Gtk.ScrolledWindow()
        # self.grid.attach(self.scrolled_window, 1, 1, 2, 2)
        # canvas = FigureCanvas(fig)
        # canvas.set_size_request(400, 400)
        # self.scrolled_window.add_with_viewport(canvas)
        self.scrolled_window.set_size_request(400, 400)
        self.show_all()

    def init_buttons(self):
        self.buttonRefresh = Gtk.Button.new_with_label("Refresh Data")
        self.buttonRefresh.connect("clicked", self.on_refresh_data_clicked)
        self.buttonDrawChart = Gtk.Button.new_with_label("Draw Chart")
        self.buttonDrawChart.connect("clicked", self.on_draw_chart_clicked)
        self.buttonTrainChart = Gtk.Button.new_with_label("Train DNNRegressor Model")
        self.buttonTrainChart.connect("clicked", self.on_trained_clicked)
        self.buttonPredictChart = Gtk.Button.new_with_label("Predict")
        self.buttonPredictChart.connect("clicked", self.on_predict_clicked)

    def init_tree_store(self):
        treestore = Gtk.TreeStore(str, Pixbuf, int)

        img = Gtk.IconTheme.get_default().load_icon("web-browser", Gtk.IconSize.MENU, 0)

        json_data = None
        with open('../openweathermap/city.list.json', encoding='utf-8') as json_file:
            json_data = json.load(json_file)

        for i in range(0, len(json_data) // 20):
            element = json_data[i]
            treestore.append(None, [element['name'] + ' - ' + element['country'], img, element['id']])

        completion = Gtk.EntryCompletion()
        completion_img = Gtk.CellRendererPixbuf()
        completion.set_model(treestore)
        completion.pack_start(completion_img, True)
        completion.add_attribute(completion_img, "pixbuf", 1)
        completion.set_text_column(0)

        completion.connect('match-selected', self.on_match_selected)

        self.entry = Gtk.Entry()
        self.entry.set_completion(completion)

    def __init__(self):
        # help(Gtk.Window.set_position)
        Gtk.Window.__init__(self, title="Weather")
        self.set_border_width(3)
        self.location_id = 2643743  # London id
        self.location_city = "London - GB"

        self.init_tree_store()
        self.init_buttons()

        layout = Gtk.VBox(spacing=5)
        layout.pack_start(self.entry, False, True, 2)

        self.notebook = Gtk.Notebook()
        # self.add(self.notebook)
        self.notebook.set_scrollable(True)
        self.notebook.set_size_request(500, 600)

        self.grid = Gtk.Grid()
        self.grid.attach(layout, 0, 0, 1, 1)
        self.grid.attach(self.notebook, 0, 1, 1, 10)
        self.grid.attach(self.buttonRefresh, 1, 0, 1, 1)
        self.grid.attach(self.buttonDrawChart, 1, 4, 1, 1)
        self.grid.attach(self.buttonTrainChart, 1, 5, 1, 1)
        self.grid.attach(self.buttonPredictChart, 1, 6, 1, 1)

        self.add(self.grid)

        self.set_default_size(500, 600)
        self.set_size_request(500, 600)
        self.set_position(Gtk.WindowPosition.CENTER)
        if (self.location_id is not None):
            self.default_location()


window = MyWindow()
window.connect("destroy", Gtk.main_quit)
window.show_all()
Gtk.main()
