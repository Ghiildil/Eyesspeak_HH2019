from tkinter import *
import time
from threading import Thread, Lock, Condition



# USAGE
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat --video blink_detection_demo.mp4
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat

# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils

# linker le code d'affichage en fils
#from prise_info import *

import numpy as np
import argparse
import imutils
import time
import dlib
import cv2



mutex = Lock()
SYNC = Condition()


class Interface(Frame, Thread):
    def __init__(self, fenetre, **kwargs):
        Frame.__init__(self, fenetre, **kwargs)
        Thread.__init__(self)
        self.pack(fill=BOTH)

        # interface
        self.BLINK_TYPE=0

        self.photo_selectionner = 0

        photo1 = PhotoImage(file='clavier1.gif')
        photo2 = PhotoImage(file='clavier2.gif')
        photo3 = PhotoImage(file='clavier3.gif')
        photo4 = PhotoImage(file='clavier4.gif')
        photo5 = PhotoImage(file='clavier5.gif')
        photo6 = PhotoImage(file='clavier6.gif')
        photo7 = PhotoImage(file='clavier7.gif')
        photo8 = PhotoImage(file='clavier8.gif')
        photo9 = PhotoImage(file='clavier9.gif')
        photo10 = PhotoImage(file='clavier10.gif')
        photo11 = PhotoImage(file='clavier11.gif')

        self.fenetre=fenetre

        self.mes_photos = [photo1, photo2, photo3, photo4, photo5, photo6, photo7, photo8, photo9, photo10, photo11]

        self.can = Canvas(self.fenetre, width=400, height=600, bg='white')
        self.can.pack()

        self.id_im = self.can.create_image(200, 300)
        self.changer_image2()  ## Exécute la fonction
        
        # variables
        f = open("dico.txt", "r")  # le fichier dico.txt se trouve dans le dossier data
        self.dico = f.read()
        f.close()
        self.phrase=""
        self.mots=[]
        

    def find_word(self, code, combinaisons):
        if (code == '1'):
            tmp = "eja"
        elif (code == '2'):
            tmp = "sin"
        elif (code == '3'):
            tmp = "tul"
        elif (code == '4'):
            tmp = "rom"
        elif (code == '5'):
            tmp = "dpc"
        elif (code == '6'):
            tmp = "qfbg"
        elif (code == '7'):
            tmp = "hvf"
        else :
            tmp = "xzkw"
        # faire les combinaison possible
        if len(combinaisons) == 0:
            for i in tmp:
                combinaisons.append(i)
        else:
            tmp_list = []
            for i in combinaisons:
                for j in tmp:
                    tmp_list.append(i + j)
            combinaisons = tmp_list.copy()

        # supprimer ce qui n'existe pas
        to_del = []
        for i in range(len(combinaisons)):
            mot = '\n' + combinaisons[i]
            if (mot not in self.dico):
                to_del.append(i)

        N = len(to_del)
        for i in range(N):
            del combinaisons[to_del[-1]]
            del to_del[-1]

        if len(combinaisons) == 0:
            print("le mot n'existe pas.")
            return [], False

        elif (len(combinaisons) == 1):
            index = self.dico.find('\n' + combinaisons[0]) + 1
            mot = ''
            i = 0
            while (self.dico[index + i] != '\n'):
                mot += self.dico[index + i]
                i += 1
            return mot, True
        return combinaisons, False


    def main(self, i):
        tmp = str(i)
        self.mots, fin = self.find_word(tmp, self.mots)
        print(self.mots)
        if fin == True:
            self.phrase += self.mots + ' '
            self.mots=[]
            fin=False
            return 0

    def say_it(self):
        fichier = open('test3.txt', "w")
        fichier.write(self.phrase)
        fichier = open('test3.txt', "r")
        x = fichier.read()
        language = 'fr'
        audio = gTTS(text=x, lang=language, slow=False)
        audio.save("motjeune.mp3")
        os.system("motjeune.mp3")
        fichier.close()

    def validation(self):
        self.say_it()
        self.fenetre.destroy()

    def numero(self, i):
        self.photo_selectionner = 0
        self.main(str(i))

    def changer_image2(self):
        self.can.itemconfig(self.id_im, image=self.mes_photos[self.photo_selectionner])
        self.photo_selectionner += 1

        if self.photo_selectionner >= len(self.mes_photos):
            self.photo_selectionner = 0  ## Remet à zéro
        ## Re-exécute la fonction, à chaque 5 secondes.
        self.fenetre.after(1300, self.changer_image2)


    def set_blink(self, data):
        self.BLINK_TYPE=data

    def run(self):
        while(1):
            SYNC.acquire()
            SYNC.wait()
            SYNC.release()

            mutex.acquire()
            if self.BLINK_TYPE>0:
                #print(self.BLINK_TYPE)
                self.numero(self.photo_selectionner)
            mutex.release()

            SYNC.acquire()
            SYNC.notify()
            SYNC.release()



#/!\ Bonne configuration :
#       env 30cm de la cam
#       Pas d'éclairage néon
class Vision(Thread):
    def __init__(self):
        Thread.__init__(self)

        self.BLINK_TYPE=0
 
        # construct the argument parse and parse the arguments
        self.ap = argparse.ArgumentParser()
        self.ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
        self.ap.add_argument("-v", "--video", type=str, default="",
                help="path to input video file")
        self.args = vars(self.ap.parse_args())
         
        # define two constants, one for the eye aspect ratio to indicate
        # blink and then a second constant for the number of consecutive
        # frames the eye must be below the threshold
        self.EYE_AR_THRESH = 0.25 #default 0.3 // 0.23 ça à l'air bien
        self.EYE_AR_CONSEC_FRAMES = 15 #13
        self.C_UN_LONG = 30

        # initialize the frame counters and the total number of blinks
        self.COUNTER = 0
        self.TOTAL = 0
        self.compteur = 0

        # initialize dlib's face detector (HOG-based) and then create
        # the facial landmark predictor
        print("[INFO] loading facial landmark predictor...")
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.args["shape_predictor"])

        # grab the indexes of the facial landmarks for the left and
        # right eye, respectively
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        # start the video stream thread
        print("[INFO] starting video stream thread...")
        self.vs = FileVideoStream(self.args["video"]).start()
        self.fileStream = True
        # utiliser une camera externe ou la webcam integree
        self.vs = VideoStream(src=0).start()
        #vs = VideoStream(usePiCamera=True).start()
        self.fileStream = False
        time.sleep(1.0)


    def send_data(self, data):
        mutex.acquire()
        INTERFACE.set_blink(data)
        mutex.release()
 
    def eye_aspect_ratio(self, eye):
        # compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])

        # compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        C = dist.euclidean(eye[0], eye[3])

        # compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)

        # return the eye aspect ratio
        return ear
        
    def run(self):
        time.sleep(1)
        i=0
        while(1): 
            # if this is a file video stream, then we need to check if
            # there any more frames left in the buffer to process
            if self.fileStream and not vs.more():
                    break

            # grab the frame from the threaded video file stream, resize
            # it, and convert it to grayscale
            # channels)
            frame = self.vs.read()
            frame = imutils.resize(frame, width=720)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # detect faces in the grayscale frame
            rects = self.detector(gray, 0)

            # loop over the face detections
            for rect in rects:
                # magic    
                shape = self.predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                leftEye = shape[self.lStart:self.lEnd]
                rightEye = shape[self.rStart:self.rEnd]
                leftEAR = self.eye_aspect_ratio(leftEye)
                rightEAR = self.eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
                
                # decision : blink
                if(ear < self.EYE_AR_THRESH and self.COUNTER>self.EYE_AR_CONSEC_FRAMES):
                    if(self.COUNTER>self.C_UN_LONG):
                        #print("long")
                        self.BLINK_TYPE=2
                    else:
                        #print("court")
                        self.BLINK_TYPE=1
                    self.COUNTER=0
                    self.send_data(self.BLINK_TYPE)
                    self.BLINK_TYPE=0
                elif(ear<self.EYE_AR_THRESH):
                    self.COUNTER+=1
                
                SYNC.acquire()
                SYNC.notify()
                SYNC.release()
                SYNC.acquire()
                SYNC.wait()
                SYNC.release()

                self.send_data(0)

                # draw (useless?)
                cv2.putText(frame, "Clignements: {}".format(self.TOTAL), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "Coeff.: {:.2f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
            # show the frame (useless ?)
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
 
            if key == ord("q"):
                break


if __name__=="__main__":
    fenetre=Tk()

    INTERFACE = Interface(fenetre)
    CAM = Vision()

    CAM.start()
    INTERFACE.start()

    fenetre.mainloop()
    fenetre.destroy()
    
    INTERFACE.join()
    CAM.join()
