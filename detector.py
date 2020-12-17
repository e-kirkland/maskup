# Standard Library Imports
import subprocess
import sys

# Third Party Imports
from keras.models import load_model
import cv2
import numpy as np
import checkpointe as check

def main(audioEnabled=False):

    # Start checkpointe
    check.start(summary=True, verbose=True, memory=True)

    # Load model
    model = load_model('model-011.model')
    check.point('MODEL LOADED')

    # Instantiate classifier
    face_clsfr = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    check.point('FACE CLASSIFIER INSTANTIATED')

    # Start video capture
    source = cv2.VideoCapture(0)
    check.point('VIDEO CAPTURE STARTED')

    # Build dictionary for mask detection
    labels_dict = {0:'Thank you!',1:'Mask up!'}
    color_dict = {0:(0,255,0),1:(0,0,255)}

    # Install audio file
    no_mask = "./audio/error.mp3"
    masked = "./audio/ping.mp3"

    # Monitor camera
    while(True):

        # Read camera
        ret,img = source.read()
        # Grayscale image for analysis
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Detect face objects (multiple allowed)
        faces = face_clsfr.detectMultiScale(gray,1.3,5)  


        for x,y,w,h in faces:
            
            # Get coordinates for facial object
            face_img=gray[y:y+w,x:x+w]
            # Resize/shape for analysis
            resized=cv2.resize(face_img,(100,100))
            normalized=resized/255.0
            reshaped=np.reshape(normalized,(1,100,100,1))
            # Make prediction
            result=model.predict(reshaped)

            # Extracting label from prediction
            label=np.argmax(result,axis=1)[0]
        
            # Building rectangle for each result
            cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],5)
            cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
            cv2.putText(
            img, labels_dict[label], 
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

            print("LABEL: ", label)

            # Play audio if enabled
            if audioEnabled:
                if label == 1:
                    # Play airhorn
                    subprocess.call(["afplay", no_mask], stdin=None, stdout=None, stderr=None)
                elif label == 0:
                    subprocess.call(["afplay", masked], stdin=None, stdout=None, stderr=None)
                else:
                    pass
            else:
                pass
            
            
        cv2.imshow('LIVE',img)
        key=cv2.waitKey(1)
        
        if(key==27):
            break
            
    cv2.destroyAllWindows()
    source.release()

    check.stop()

    return

if __name__=="__main__":
    if str(sys.argv[1])=='audio-on':
        main(audioEnabled=True)
    elif str(sys.argv[1])=='audio-off':
        main(audioEnabled=False)
    else:
        print("No audio argument [audio-on, audio-off] provided, defaulting to audio-off.")
        main()