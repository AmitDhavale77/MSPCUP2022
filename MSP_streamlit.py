#install the following packages prior to the running of code in streamlit
!pip install synthesizer
!pip install streamlit
!pip install inference-tools
!pip install sounddevice
!sudo apt-get install libportaudio2
#code
import streamlit as st
import sounddevice as sd
%%writefile app.py
import streamlit as st
duration = 5  # seconds
fs = 48000
sd.default.samplerate = fs
uploaded_file = st.file_uploader("Choose an audio...", type=["wav"])

if uploaded_file is not None:
  print("Playing audio file")
  sd.play(uploaded_file, fs) #st
  print("Done! Saved sample as myvoice.mp3")

  if st.button('PREDICT'):
    Categories = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21']    
    st.write('Result.....')
    myaudio = AudioSegment.from_file(uploaded_file)
    chunk_length_ms = 1000
    chunks = make_chunks(myaudio, chunk_length_ms)
    i=0
    for j, chunk in enumerate(chunks):
             chunk_name = "gdrive/My Drive/colab_input_1/temp_f{0}.wav".format(i)
             print("exporting", chunk_name)
             chunk.export(chunk_name, format="wav")
             i=i+1
    folder_dir1='gdrive/My Drive/colab_input_1'
    audiolist1 = []
    filelist1=[]
    labellist1=[]
    featurelist1=[]
    faudiolist1=[]
    imagelist=[]
    normlist=[]
    for files in os.listdir(folder_dir1):
 
    # check if the image ends with wav
        if (files.endswith(".wav")):
            x , sr = librosa.load(folder_dir1+'/'+files) 
            S = librosa.feature.melspectrogram(x, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
            S_DB = librosa.power_to_db(S, ref=np.max)
            img1=cv2.resize(S_DB, (224, 224),interpolation = cv2.INTER_NEAREST)#The INTER_NEAREST method uses the nearest neighbor concept for interpolation. This is one of the simplest methods, using only one neighboring pixel from the image for interpolation.
            norm_image = cv2.normalize(img1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)    
            imagelist.append(norm_image)
    xx = np.array(imagelist)
    xx_1 = np.expand_dims(xx, axis=3)
    y_pred=saved_model.predict(xx_1)
    y_pred1=np.round(y_pred)
    temp=[]
    temp1=[]
    
    for i in y_pred1:
          temp.append(i.argmax(axis=0))# to extract position where output is 1
    temp=list(temp)
    from collections import Counter
 
    def most_frequent(temp):
        occurence_count = Counter(temp)
        return occurence_count.most_common(1)[0][0]
   
    f=most_frequent(temp)
    st.title( 'PREDICTED OUTPUT')
    if f==0:
       print("English")
    elif f==1:
       print("Hindi")
    else:
       print("Marathi")

st.text("")
st.text('Made by Amit Dhavale')
