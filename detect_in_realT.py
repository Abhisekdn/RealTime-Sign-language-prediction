import numpy as np
from keras.models import model_from_json
import operator
import cv2
import sys, os
#for text to speech conversion
from gtts import gTTS
from playsound import playsound
#for google image search and download
import requests
from bs4 import BeautifulSoup    #a library to parse Html
from PIL import Image   #To open downloaded image
s=""


# Loading the model(27 classes)

json_file = open("Models/model-e12-abhi-data(old).json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)
# load weights into new model
loaded_model.load_weights("Models/model-e12-abhi-data(old).h5")

#loading the model_dru
json_file_dru = open("Models/model-e12data-dru.json","r")
model_json_dru = json_file_dru.read()
json_file_dru.close()
loaded_model_dru = model_from_json(model_json_dru)
loaded_model_dru.load_weights("Models/model-e12data-dru.h5")

#loading the model_ktr
json_file_ktr = open("Models/model-e12data-ktr.json","r")
model_json_ktr = json_file_ktr.read()
json_file_ktr.close()
loaded_model_ktr = model_from_json(model_json_ktr)
loaded_model_ktr.load_weights("Models/model-e12data-ktr.h5")

#loading the model_se
json_file_se = open("Models/model-e12data-se.json","r")
model_json_se = json_file_se.read()
json_file_se.close()
loaded_model_se = model_from_json(model_json_se)
loaded_model_se.load_weights("Models/model-e12data-se.h5")

#loading the model_smnt
json_file_smnt = open("Models/model-e12data-smnt.json","r")
model_json_smnt = json_file_smnt.read()
json_file_smnt.close()
loaded_model_smnt = model_from_json(model_json_smnt)
loaded_model_smnt.load_weights("Models/model-e12data-smnt.h5")

#loading the model_ij
# json_file_ij = open("Models/model-e12data-ij.json","r")
# model_json_ij = json_file_ij.read()
# json_file_ij.close()
# loaded_model_ij = model_from_json(model_json_ij)
# loaded_model_ij.load_weights("Models/model-e12data-ij.h5")



print("Loaded all models from disk")

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
    #results of all models loaded 
    #sending the ROI Frame to the models to predict

    result = loaded_model.predict(test_image.reshape(1, 128, 128, 1))  
    result_dru = loaded_model_dru.predict(test_image.reshape(1, 128, 128, 1))
    result_se = loaded_model_se.predict(test_image.reshape(1, 128, 128, 1))
    result_ktr = loaded_model_ktr.predict(test_image.reshape(1, 128, 128, 1))
    result_smnt = loaded_model_smnt.predict(test_image.reshape(1, 128, 128, 1))
#     result_ij = loaded_model_ij.predict(test_image.reshape(1, 128, 128, 1))
    
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
    current_sym = prediction[0][0]
    
   
    #Addressing letters with similar gestures
    if (current_sym == 'd' or current_sym == 'r' or current_sym == 'u'):
        prediction1 = {}
        prediction1['d'] = result_dru[0][0]
        prediction1['r'] = result_dru[0][1]
        prediction1['u'] = result_dru[0][2]
        prediction1 = sorted(prediction1.items(), key=operator.itemgetter(1), reverse=True)
        current_sym = prediction1[0][0]
    
    if (current_sym == 'k' or current_sym == 't' or current_sym == 'r'):
        prediction2 = {}
        prediction2['k'] = result_ktr[0][0]
        prediction2['r'] = result_ktr[0][1]
        prediction2['t'] = result_ktr[0][2]
        prediction2 = sorted(prediction2.items(), key=operator.itemgetter(1), reverse=True)
        current_sym = prediction2[0][0]
        
    if (current_sym == 's' or current_sym == 'e'):
        prediction2 = {}
        prediction2['e'] = result_se[0][0]
        prediction2['s'] = result_se[0][1]
        prediction2 = sorted(prediction2.items(), key=operator.itemgetter(1), reverse=True)
        current_sym = prediction2[0][0]
        
    if (current_sym == 'm' or current_sym == 'm' or current_sym == 'n' or current_sym == 't'):
        prediction2 = {}
        prediction2['m'] = result_smnt[0][0]
        prediction2['n'] = result_smnt[0][1]
        prediction2['s'] = result_smnt[0][2]
        prediction2['t'] = result_smnt[0][3]
        prediction2 = sorted(prediction2.items(), key=operator.itemgetter(1), reverse=True)
        current_sym = prediction2[0][0]
    
#     if (current_sym == 'i' or current_sym == 'j'):
#         prediction2 = {}
#         prediction2['i'] = result_smnt[0][0]
#         prediction2['j'] = result_smnt[0][1]
#         prediction2 = sorted(prediction2.items(), key=operator.itemgetter(1), reverse=True)
#         current_sym = prediction2[0][0]

    
    # Displaying the legend
    cv2.putText(frame, "Append: Space",(970,30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,240), 1)
    cv2.putText(frame, "Delete: d",(970,50), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,240), 1)
    cv2.putText(frame, "Speak: s",(970,70), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,240), 1)
    cv2.putText(frame, "Images: i",(970,90), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,240), 1)
    cv2.putText(frame, "Quit: q",(970,110), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,240), 1)
        
    # Displaying the predictions
    cv2.putText(frame, current_sym, (10, 120), cv2.FONT_HERSHEY_PLAIN, 4, (255,255,0), 2)  
    if c>0:
        cv2.putText(frame,s,(pix,350),cv2.FONT_HERSHEY_PLAIN, 2.5, (0,0,0), 2)
    cv2.imshow("Frame", frame)
    
    interrupt = cv2.waitKey(10)
    
    if interrupt & 0xFF == 27: # esc key
        break

    #Append Key
    if interrupt & 0xFF in (13,ord(' ')): #enter key or space
        if current_sym == 'ZERO':
            s+=" "
            c+=1
            pix+=4
        else:
            c+=1 
            s+=current_sym
            pix+=4

    #Delete key
    if interrupt & 0xFF == ord('d'):
        s=s[:-1]       #to delete the string by character
        
    #Speak Key
    if interrupt & 0xFF == ord('s'):
        
        # Language in which you want to convert
        language = 'en'

        # Passing the text and language to the engine
        myobj = gTTS(text=s, lang=language, slow=False)
        # Saving the converted audio in a mp3 file named
        # welcome 
        myobj.save("speech1.mp3")
        os.system("afplay speech1.mp3")
       
    
    
    if interrupt & 0xFF == ord('i'):
        #google images url
        Google_Image = \
        'https://www.google.com/search?site=&tbm=isch&source=hp&biw=1873&bih=990&'
        
        #Request Header needed for google ---> info about which OS we are using ad its versions
        usr_agent = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Safari/537.36',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                        'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
                        'Accept-Encoding': 'none',
                        'Accept-Language': 'en-US,en;q=0.8',
                        'Connection': 'keep-alive',
        } 
        save_folder = 'images'
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        
        #data is the string (appended letter by letter after prediction)
        data = s
        
        #we only need the first image of the google image search
        num_images = 1
        
        #Adding our String(data) to the Google images URL
        search_url = Google_Image + 'q=' + data
        
        #request the html script of the google image search result
        response = requests.get(search_url, headers=usr_agent)
        html = response.text
           
        #BuestifulSoup is a html Parser
        b_soup = BeautifulSoup(html, 'html.parser')
        
        #the images are inside the class= rg_i Q4LuWd in html script
        results = b_soup.findAll('img', {'class': 'rg_i Q4LuWd'})
        count = 0

        #Saving the image to our Directory images
        imagelinks= []
        for res in results:
            try:
                link = res['data-src']
                imagelinks.append(link)
                count = count + 1
                if (count >= num_images):
                    break

            except KeyError:
                continue

        for i, imagelink in enumerate(imagelinks):

            response = requests.get(imagelink)
            imagename = save_folder + '/' + data + str(i+1) + '.png'
            with open(imagename, 'wb') as file:
                file.write(response.content)
#         os.system(imagename)
        im = Image.open(imagename) 
  
        # This method will show image in any image viewer 
        im.show()
        print('Image Download Successful')


cap.release()        
cv2.destroyAllWindows()
print(s)
