from fileinput import filename
from flask import Flask, flash, render_template, request, redirect, url_for 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import librosa
import io
import base64
import numpy as np
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
import librosa
import sklearn
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import root
import re
import librosa.display
from os import path
from pydub import AudioSegment
from createdb import *

app = Flask(__name__)



try:
    con = sqlite3.connect("db1.db") 
    df = pd.read_sql_query("SELECT * FROM results", con)
except Exception as E:
    print("Error while reading .db file as dataframe\n", E)




@app.route("/add")  
def add():  
    create_db()  
    return render_template("add.html")  
 



@app.route("/savedetails",methods = ["POST","GET"])  
def saveDetails():  
    msg = "msg"  
    if request.method == "POST":  
        try:  
            name = request.form["name"]  
            email = request.form["email"]  
            audio_file = request.form["audio_file"]
            print(os.path.split(audio_file))
            prediction = request.form["prediction"]
            truth_value = request.form["truth_value"]
            with sqlite3.connect("db1.db") as con:
  
                print('inside con')
                cur = con.cursor()  
                cur.execute("INSERT into results (name, email, audio_file, prediction, truth_value) values (?,?,?,?,?)",(name, email, audio_file, prediction,truth_value))  
                con.commit()  
                msg = "data successfully Added"  
        except:  
            con.rollback()  
            msg = "We can not add the data to the list"  
        finally:  
            return ('', 204)
            #return render_template("view.html") 
            
            con.close()  
 



@app.route("/view")  
def view():  
    con = sqlite3.connect("db1.db")  
    con.row_factory = sqlite3.Row  
    cur = con.cursor()  
    cur.execute("select * from results")  
    rows = cur.fetchall()  
    return render_template("view.html",rows = rows)  
 


@app.route("/delete")  #
def delete():  
    return render_template("delete.html")  
 




@app.route("/deleterecord",methods = ["POST"])  
def deleterecord():  
    id = request.form["id"]  
    with sqlite3.connect("db1.db") as con:  
        try:  
            cur = con.cursor()  
            cur.execute("delete from results where id = ?",id)  
            msg = "record successfully deleted"  
        except:  
            msg = "can't be deleted"  
        finally:  
            return render_template("delete_record.html",msg = msg) 




def model_init():
    global model 

    #load model
    model = tf.keras.models.load_model('MM5_2.3_66.h5')
    



#Route for recorder.html
@app.route("/recorder", methods=["GET", "POST"])
def recorder():
    return render_template('recorder.html')


#route for index.html
@app.route("/", methods=["GET", "POST"])

def index():
    create_db()  
    gauge_value=''
    prediction=''

    if request.method=="POST":
        print("data received")
 

        file = request.files['file']
        # global filename
        filename = secure_filename(file.filename)


        #Check of the file is of 'mp3' format
        if filename[-3:] == 'mp3':
            file.save(os.path.join('audio', filename))
            output_file = filename[:-4] + ".wav"
    
            # convert mp3 file to wav file
            sound = AudioSegment.from_mp3("audio/"+filename)
            #save the file directly in the path "audio/..."
            file=sound.export("audio/"+ output_file, format="wav")
            
            #remove the .mp3 file from the path as we only require .wav file
            os.remove("audio/"+ filename[:-4]+".mp3")
            filename= filename[:-4] +".wav" 

        # If a .wav file is input than the following will be executed
        else:
            file.save(os.path.join('audio', filename))
            print("FORM DATA RECEIVED")

            if "file" not in request.files:
                return redirect(request.url)

            file = request.files["file"]
            if file.filename == "":
                return redirect(request.url)

        # If file is available 

        if file:
            global path
            path = os.path.join('audio', filename)


            prod_encd= root.prod_Y
            prod_std = root.prod_std

            # Adding noise to the audio files 
            def noise(data):
                noise_amp = 0.035*np.random.uniform()*np.amax(data)
                data = data + noise_amp*np.random.normal(size=data.shape[0])
                return data

            # Stretching
            def stretch(data, rate=0.8):
                return librosa.effects.time_stretch(data, rate)

            #Shifting 
            def shift(data):
                shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
                return np.roll(data, shift_range)

            #Pitching
            def pitch(data, sampling_rate, pitch_factor=0.7):
                return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)

            # Loading the audio file
            
            data, sample_rate = librosa.load(path) 


            def extract_features(data):
                # ZCR
                result = np.array([])
                zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
                result=np.hstack((result, zcr)) # stacking horizontally

                # Chroma_shft
                stft = np.abs(librosa.stft(data))
                chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
                result = np.hstack((result, chroma_stft)) # stacking horizontally

                # MFCC
                mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
                result = np.hstack((result, mfcc)) # stacking horizontally

                # Root Mean Square Value
                rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
                result = np.hstack((result, rms)) # stacking horizontally

                # MelSpectogram
                mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
                result = np.hstack((result, mel)) # stacking horizontally
                
                return result

            def get_features(path):
                # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
                data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
                
                # without augmentation
                res1 = extract_features(data)
                result = np.array(res1)
                print(result)
                
                
                return result


            feature = get_features(path)

            feature= np.append(feature,[0,1])

            feature= np.reshape(feature,(1,164))
            prod_input= feature.copy()

            #Performing Standard Scaler

            prod_scaler= StandardScaler()
            _ = prod_scaler.fit_transform(prod_std)
            prod_input = prod_scaler.transform(prod_input)

            prod_input = np.expand_dims(prod_input, axis=-1)

            pred_sample = model.predict(prod_input)

            # inverse encoding for production
            encoder= OneHotEncoder()
            _ = encoder.fit_transform(np.array(prod_encd).reshape(-1,1)).toarray()

            prediction = encoder.inverse_transform(pred_sample)
            print(prediction)
            prediction=str(prediction)
    
            prediction = " ".join(re.findall("[a-zA-Z]+", prediction)).upper()

            #waveplot(data, sample_rate)
            #spectrogram(data, sample_rate)
            if prediction=="NEUTRAL":
                gauge_value=0

            elif prediction=="SAD":
                gauge_value=1
            elif prediction=="ANGRY":
                gauge_value=2
            elif prediction=="FEAR":
                gauge_value=3
            else:
                gauge_value=4

            
            save_plots( path, 2.5, 0.6)
        
    os.getcwd()    
  
    return render_template('main.html', prediction = gauge_value )
    



def waveplot(data, sr):


    plt.figure(figsize=(10, 3))
    #plt.title('Waveplot: {} [{}]'.format(path,e), size=15)
    librosa.display.waveplot(data, sr=sr)
    plt.axis('off')
    plt.savefig(r'static\image\waveplot.jpg')

def spectrogram(data, sr):
    # stft function converts the data into short term fourier transform
    X = librosa.stft(data)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(12, 3))
#     plt.title('Spectrogram: {} [{}]'.format(path, e), size=15)
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')   
    #librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
#     plt.colorbar()
    plt.axis('off')
    plt.savefig(r'static\image\spectrogram.jpg')


def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)


def zcr(data,sr):

    spectral_centroids = librosa.feature.spectral_centroid(data+0.01, sr=sr)[0]
    #librosa.display.waveplot(data, sr=sr, alpha=0.4)
    frames = range(len(spectral_centroids))
    t = librosa.frames_to_time(frames)
    plt.plot(t, normalize(spectral_centroids), color='r') # normalize for visualization purposes
    plt.savefig(r'static\image\zcr.jpg')


def mfcc_plot(data,sr):


    mfcc = librosa.core.power_to_db(librosa.feature.mfcc(data, sr=sr))
    fig, ax = plt.subplots(figsize=(15,9))
    img = librosa.display.specshow(mfcc, x_axis='time',y_axis='mel', sr=sr,fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='MFCCs')
    plt.savefig(r'static\image\mfcc_plot.jpg')


def spect_centroid(data,sr):

    S = librosa.stft(data)
    spec_bw = librosa.feature.spectral_bandwidth(data, sr=sr)

    times = librosa.times_like(spec_bw)
    centroid = librosa.feature.spectral_centroid(S=np.abs(S))
    S_dB = librosa.amplitude_to_db(S, ref=np.max)
    
    fig, ax = plt.subplots(figsize=(15,9))
    img = librosa.display.specshow(S_dB, x_axis='time',
                         y_axis='log', sr=sr,
                         fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Spectral centroid plus/minus spectral bandwidth')
    ax.fill_between(times, centroid[0] - spec_bw[0], centroid[0] + spec_bw[0],
                alpha=0.5, label='Centroid +- bandwidth')
    ax.plot(times, centroid[0], label='Spectral centroid', color='w')
    ax.legend(loc='lower right')
    plt.savefig(r'static\image\spect_centroid_plot.jpg')

def stft(data, sr):
    S =   librosa.stft(data)
    S_dB = librosa.amplitude_to_db(S, ref=np.max)
    fig, ax = plt.subplots(figsize=(15,9))
    img = librosa.display.specshow(S_dB, x_axis='time',
                         y_axis='log', sr=sr,
                         fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='STFT (amplitude to DB scaled) log scale spectrogram')
    plt.savefig(r'static\image\STFT.jpg')



'''
@app.cache
def get_melspec(audio):
  y, sr = librosa.load(audio, sr=44100)
  X = librosa.stft(y)
  Xdb = librosa.amplitude_to_db(abs(X))
  img = np.stack((Xdb,) * 3,-1)
  img = img.astype(np.uint8)
  grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  grayImage = cv2.resize(grayImage, (224, 224))
  rgbImage = np.repeat(grayImage[..., np.newaxis], 3, -1)
  return (rgbImage, Xdb)
'''
@app.route('/plot')
def build_plot():

    img = io.BytesIO()

    y = [1,2,3,4,5]
    x = [0,2,1,3,4]
    plt.plot(x,y)
    plt.savefig(img, format='png')
    img.seek(0)

    plot_url = base64.b64encode(img.getvalue()).decode()

    return '<img src="data:../static/image/png;base64,{}">'.format(plot_url)

def save_plot():
    import numpy as np
    import os
    import sqlite3
    import pandas as pd
    from bokeh.io import output_notebook, push_notebook, show
    from bokeh.plotting import figure, output_file, show, save
    from bokeh.plotting import output_notebook
    from bokeh.io import export_png
    import math

    try:
        con = sqlite3.connect("db1.db") 
        df = pd.read_sql_query("SELECT * FROM results", con)
    except Exception as E:
        print("Error while reading .db file as dataframe\n", E)

    df_h = df[df['truth_value']=='Happy']
    df_n = df[df['truth_value']=='Neutral']
    df_f = df[df['truth_value']=='Fear']
    df_s = df[df['truth_value']=='Sad']
    df_a = df[df['truth_value']=='Angry']
    dfs = [df_h,df_n,df_f,df_s,df_a]

    def get_acc(df_temp, final=0):
        accurate = 0
        for index, row in df_temp.iterrows():
            if row['prediction'] == row['truth_value']:
                accurate+=1
        if not final:
            try:
                return (df_temp['truth_value'][0], round((accurate/df_temp.shape[0])*100,2))
                
            except Exception as E:
                # print('No emotion found in truth_values')
                return ('NA', -1)
        if final:
            return (round((accurate/df_temp.shape[0])*100,2))

    try:
        con = sqlite3.connect("db1.db") 
        df = pd.read_sql_query("SELECT * FROM results", con)
    except Exception as E:
        print("Error while reading .db file as dataframe\n", E)

        
    df_all = df.copy()
    corr = get_acc(df_all,final=1)
    percentages = [corr, 100-corr]
    # file to save the model 
    output_file("static/image/pie.html") 
            
    # instantiating the figure object 
    graph = figure(title = 'Accuracy: '+str(corr), plot_width = 500, plot_height = 500) 
    
    # name of the sectors
    sectors = ["Correct","Incorrect"]
    
    # % tage weightage of the sectors

    
    # converting into radians
    radians = [math.radians((percent / 100) * 360) for percent in percentages]
    
    # starting angle values
    start_angle = [math.radians(0)]
    prev = start_angle[0]
    for i in radians[:-1]:
        start_angle.append(i + prev)
        prev = i + prev
    
    # ending angle values
    end_angle = start_angle[1:] + [math.radians(0)]
    
    # center of the pie chart
    x = 0
    y = 0
    
    # radius of the glyphs
    radius = 1
    
    # color of the wedges
    color = [ (255, 196, 81), (68,68,68)]
    
    # plotting the graph
    for i in range(len(sectors)):
        graph.wedge(x, y, radius,
                    start_angle = start_angle[i],
                    end_angle = end_angle[i],
                    color = color[i],
                    legend_label = sectors[i])
    # displaying the graph

    graph.xgrid.grid_line_color = None
    graph.ygrid.grid_line_color = None
    graph.xaxis.visible = False
    graph.yaxis.visible = False
    graph.toolbar.logo = None
    #graph.toolbar_location = None
    #graph.outline_line_color=None

    export_png(graph, filename = "static\image\pie.png")

    # show(graph)
    #save(graph)

    temp = {'Correct':0,
        'Incorrect':0}
    dic = {'Angry': temp.copy(),
            'Neutral': temp.copy(),
            'Fear': temp.copy(),
            'Sad': temp.copy(), 
            'Angry':temp.copy()
    }

    #output_file("static/image/bar.html") 

    def get_list(df_temp):
            acc = 0
            for index, row in df_temp.iterrows():
                    if row['prediction'] == row['truth_value']:
                            acc+=1
            return [acc, df_temp.shape[0]-acc]


    # print('Happy',get_list(df_h))
    # print('Neutral',get_list(df_n))
    # print('Fear',get_list(df_f))
    # print('Sad',get_list(df_s))
    # print('Angry',get_list(df_a))

    results = [get_list(df_h), get_list(df_n), get_list(df_f), get_list(df_s), get_list(df_a)]

    val1 = []
    val2 = []
    for lst in results:
            val1.append(lst[0])
            val2.append(lst[1])


    # print(val1)
    # print(val2)

    labs = ['Happy', 'Neutral', 'Fear', 'Sad', 'Angry']
    vals = ['val_1','val_2']
    my_data = {'labs':labs,
    'val_1':val1,
    'val_2':val2
    }
    cols = [(255, 196, 81), (68,68,68)]
    fig = figure(x_range = labs, plot_width = 1080, plot_height = 620)

    fig.vbar_stack(vals, x = 'labs', source = my_data, color = cols,width = 0.5, legend_label=['Correct','Incorrect'])

    fig.xaxis.axis_label = "Emotions"
    fig.xaxis.axis_line_width = 2
    fig.xaxis.axis_line_color = "black"
    fig.yaxis.minor_tick_line_color = None

    fig.yaxis.axis_label = "Count"
    fig.yaxis.axis_line_width = 2
    fig.yaxis.axis_line_color = "black"

    fig.xgrid.grid_line_color = None
    fig.ygrid.grid_line_color = None
    fig.toolbar.logo = None
    fig.toolbar_location = None

    
    export_png(fig, filename = "static\image\Bar.png")
    #save(fig)


def save_plots(path, duration=2.5, offset=0.6):
    data, sr = librosa.load(path, duration=duration, offset=offset)
    
    waveplot(data, sr)
    spectrogram(data, sr)
    zcr(data, sr)
    mfcc_plot(data,sr)
    spect_centroid(data, sr)
    stft(data, sr)
    save_plot()

if __name__ == "__main__":
    model_init()
    app.run(debug=True, threaded=True, host="0.0.0.0", port=5000)