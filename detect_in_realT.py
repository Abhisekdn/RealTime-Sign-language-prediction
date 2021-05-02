import numpy as np
from keras.models import model_from_json
import operator
import cv2
import sys, os
from gtts import gTTS
from playsound import playsound

s=""

# Loading the model

json_file = open("model-e13-abhi-data(old).json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)
# load weights into new model
loaded_model.load_weights("model-e12-abhi-data(old).h5")
print("Loaded model from disk")

cap = cv2.VideoCapture(0)
pix =10
c=0
# Category dictionary
categories = {
    0: 'ZERO ', 'A': 'a', 'B': 'b', 'C': 'c', 'D': 'd', 'E': 'e','F':'f','G':'g','H':'h','I':'i','J':'j','K':'k','L':'l','M':'m','N':'n','O':'o','P':'p','Q':'q','R':'r','S':'s','T':'t','U':'u','V':'v','W':'w','X':'x','Y':'y','Z':'z'}
while True:
    _, frame = cap.read()
    # Simulating mirror image
    frame = cv2.flip(frame, 1)
    
    # Got this from collect-data.py
    # Coordinates of the ROI
    x1 = int(0.5*frame.shape[1])
    y1 = 10
    x2 = frame.shape[1]-10
    y2 = int(0.5*frame.shape[1])
    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
    # Extracting the ROI
    roi = frame[y1:y2, x1:x2]
    
    # Resizing the ROI so it can be fed to the model for prediction
    roi = cv2.resize(roi, (128, 128)) 
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(roi,(5,5),2)
    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)

    _, test_image = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cv2.imshow("test", test_image)
    # Batch of 1
    result = loaded_model.predict(test_image.reshape(1, 128, 128, 1))
    prediction = {'ZERO': result[0][0], 
                  'a': result[0][1], 
                  'b': result[0][2],
                  'c': result[0][3],
                  'd': result[0][4],
                  'e': result[0][5],
                  'f': result[0][6],
                  'g': result[0][7],
                  'h': result[0][8],
                  'i': result[0][9],
                  'j': result[0][10],
                  'k': result[0][11],
                  'l': result[0][12],
                  'm': result[0][13],
                  'n': result[0][14],
                  'o': result[0][15],
                  'p': result[0][16],
                  'q': result[0][17],
                  'r': result[0][18],
                  's': result[0][19],
                  't': result[0][20],
                  'u': result[0][21],
                  'v': result[0][22],
                  'w': result[0][23],
                  'x': result[0][24],
                  'y': result[0][25],
                  'z': result[0][26]
                  }
    # Sorting based on top prediction
    prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
    
    # Displaying the predictions
    cv2.putText(frame, "Append: Space",(970,30), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0), 1)
    cv2.putText(frame, "Delete: d",(970,50), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0), 1)
    cv2.putText(frame, "Speak: s",(970,70), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0), 1)
    cv2.putText(frame, prediction[0][0], (10, 120), cv2.FONT_HERSHEY_PLAIN, 4, (255,255,0), 2)  
    if c>0:
        cv2.putText(frame,s,(pix,350),cv2.FONT_HERSHEY_PLAIN, 2.5, (25,0,255), 2)
    cv2.imshow("Frame", frame)
    
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27: # esc key
        break
    if interrupt & 0xFF in (13,ord(' ')): #enter key or space
        if prediction[0][0] == 'ZERO':
            s+=" "
            c+=1
            pix+=4
        else:
            c+=1 
            s+=prediction[0][0]
            pix+=4
    if interrupt & 0xFF == ord('d'):
        s=s[:-1]
    if interrupt & 0xFF == ord('s'):
        # The text that you want to convert to audio
        # Language in which you want to convert
        language = 'en'

        # Passing the text and language to the engine, 
        # here we have marked slow=False. Which tells 
        # the module that the converted audio should 
        # have a high speed
        myobj = gTTS(text=s, lang=language, slow=False)

        # Saving the converted audio in a mp3 file named
        # welcome 
        myobj.save("speech1.mp3")
        os.system("afplay speech1.mp3")

cap.release()        
cv2.destroyAllWindows()
print(s)