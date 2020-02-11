import pyaudio
import wave
from keras.models import load_model
import librosa
import numpy as np
import warnings
import osascript
import webbrowser
import os
import cv2

warnings.filterwarnings(action='ignore',category=FutureWarning)

model_1 = load_model('commands_model_spect.h5')
print('* model loaded')

CHUNK = 256
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 22050
RECORD_SECONDS = 2
WAVE_OUTPUT_FILENAME = "test.wav"
print("* recording")

while(True):


    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)



    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK, exception_on_overflow = False)
        frames.append(data)


    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    

    data, sf = librosa.load('test.wav')
    D = np.abs(librosa.stft(data))


    D = D.reshape((1,1025,87,1))

    output_1 = model_1.predict(D)

    class_pred = np.argmax(output_1)
    confidence = max(output_1[0])

    print(class_pred)
    print(confidence)


    os.remove('test.wav')

    if(class_pred == 0 and confidence > 0.99):
        vol = osascript.osascript('get volume settings')
        cur_vol = int(vol[1].split(':')[1].split(',')[0])
        cur_vol = cur_vol - 20
        if(cur_vol < 0):
            cur_vol = 0
        osascript.osascript("set volume output volume "+str(cur_vol))

    elif(class_pred == 1 and confidence > 0.99):
        vol = osascript.osascript('get volume settings')
        cur_vol = int(vol[1].split(':')[1].split(',')[0])
        cur_vol = cur_vol + 20
        if(cur_vol > 100):
            cur_vol = 100
        osascript.osascript("set volume output volume "+str(cur_vol))

    elif(class_pred == 2 and confidence > 0.9999):
        webbrowser.open('http://google.com')

    elif(class_pred == 3 and confidence > 0.99):
        os.system('top')

    elif(class_pred == 4 and confidence > 0.99):
        cam = cv2.VideoCapture(0)

        cv2.namedWindow("take a picture") 

        img_counter = 0

        while True:
            ret, frame = cam.read()
            cv2.imshow("test", frame)
            if not ret:
                break
            k = cv2.waitKey(1)

            if k%256 == 27:
                print("Escape hit, closing...")
                break
            elif k%256 == 32:
                img_name = "opencv_frame_{}.png".format(img_counter)
                cv2.imwrite(img_name, frame)
                print("{} written!".format(img_name))
                img_counter += 1

        cam.release()

        cv2.destroyAllWindows()







