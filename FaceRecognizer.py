import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))                   #To bring the location path
image_dir = os.path.join(BASE_DIR, "images")                            #To define the path of the images

    
def images():                       #To add images in the database
    
    import cv2

    #calling the cascades
    face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
    eye_cascade= cv2.CascadeClassifier('cascades/haarcascade_eye.xml')

    #Creating Folder and image file based on the given name
    name=input('Enter name: ')
    path=str("images/"+ name)
    if os.path.exists(path)==True:
        print("This will add data to the existing Face")
    else:
        os.mkdir(path)

    #To show the video by looping in the frames   
    while True:                 
        
        cap = cv2.VideoCapture(0)           #To start the camera
        
        while(True):
            ret,frame = cap.read()
            img=frame
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, "Bring your face in the frame and press q",
                        (70, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            faces= face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=3)
            for (x,y,w,h) in faces:
                
                roi_gray = gray[y:y+h, x:x+h]
                roi_color = img[y:y+h-30, x:x+w-30]
                cropped =img[y:y+h+60, x:x+w+60]
                cv2.rectangle(img, (x,y),(x+w,y+h), (0,12,12), 2)
                eyes=eye_cascade.detectMultiScale(gray,1.5, minNeighbors=3)
                
                #print(eyes)                    #To add rectangles around the eyes
                for (ex,ey,ew,eh) in eyes:
                    cv2.rectangle(img, (ex,ey),(ex+ew,ey+eh), (0,255,0), 1)

            #To show the frame
            cv2.imshow('frame',img)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                for (x,y,w,h) in faces:
                    
                    file_count = sum([len(files) for r, d, files in os.walk(path)])
                    path="images/"+ name + "/" + str(file_count+1) + ".jpg"
                    cv2.imwrite(path, cropped)
                    
                break
                            
            elif cv2.waitKey(20) & 0xFF == ord('z'):        #To break the loop for camera using 'z' key
                break
            else:
                continue

        cap.release()
        cv2.destroyAllWindows()                             

        #Continuing the loop if user wants to add another image 
        print("Do You Want To Add Another Image of "+name+"?")
        will=input("yes/no")
        if will=="yes":
            continue
        else:
            break

        #Updating the SQL Database
        images()

def train():                                    #Trains the program by collecting the given data
    
    import cv2
    import numpy as np
    from PIL import Image
    import pickle


    #calling the cascade files
    #To recognize face/faces in a frame
    face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    current_id = 0
    label_ids = {}
    y_labels = []
    x_train = []

    #searching for images and serialising, resizing them based on faces
    for root, dirs, files in os.walk(image_dir):
            for file in files:
                    if file.endswith("png") or file.endswith("jpg"):
                            path = os.path.join(root, file)
                            label = os.path.basename(root).replace(" ", "-").lower()
                            
                            if not label in label_ids:
                                    label_ids[label] = current_id
                                    current_id += 1
                            id_ = label_ids[label]
                            pil_image = Image.open(path).convert("L") # grayscale
                            size = (550, 550)
                            final_image = pil_image.resize(size, Image.ANTIALIAS)
                            image_array = np.array(final_image, "uint8")
                            faces = face_cascade.detectMultiScale(image_array,
                                                            scaleFactor=1.5, minNeighbors=3)

                            for (x,y,w,h) in faces:
                                    roi = image_array[y:y+h, x:x+w]
                                    x_train.append(roi)
                                    y_labels.append(id_)

    #dumping the data in the pickle file in the serial order
    with open("labels.pickle", 'wb') as f:
            pickle.dump(label_ids, f)

    #saving the trained data
    recognizer.train(x_train, np.array(y_labels))
    recognizer.save("trainner.yml")


def recognizer():
    
    train()                     #Update the program database by appending the latest images
    
    import numpy as np
    import cv2
    import pickle


    #calling the cascade files for recognition
    face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt.xml')  
    eye_cascade= cv2.CascadeClassifier('cascades/haarcascade_eye.xml')
    #smile_cascade = cv2.CascadeClassifier('cascades/haarcascade_smile.xml')

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainner.yml")         #Read the trained data


    #Use the data from the pickle file
    labels = {"person_name": 1}             
    with open("labels.pickle",'rb') as f:
        og_labels = pickle.load(f)
        labels = {v:k for k,v in og_labels.items()}

    #Starting the Camera
    cap = cv2.VideoCapture(0)
    print("Press q to quit")

    while(True):
        ret,frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces= face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=3)
        
        for (x,y,w,h) in faces:
            end_cord_x = x+w
            end_cord_y = y+h
            roi_gray = gray[y:y+h, x:x+h]
            roi_color = frame[y:y+h, x:x+w]
            cv2.rectangle(frame, (x,y),(end_cord_x,end_cord_y), (255,255,255), 2)

        #recognize?  deep learned model predict keras tensorflow pytorch scikit learn
            id_, conf = recognizer.predict(roi_gray)
            
            print(conf)
            
            #Only show the recognition if has given amount of confirmation
            #Lesser the conf, better the recognition
            
            #if conf>=4 and conf <= 60:          
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255,255,255)
            stroke = 2
            cv2.putText(frame, name + ' ' + str( int (conf)), (x,y), font, 1 , color, stroke, cv2.LINE_AA)

            color=(255,255,255)
            stroke = 2
            end_cord_x = x+w
            end_cord_y = y+h
            cv2.rectangle(frame, (x,y),(end_cord_x,end_cord_y), color, stroke)
            eyes = eye_cascade.detectMultiScale(roi_gray,scaleFactor=1.5,minNeighbors=3)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        #Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


#To add names of the faces according to stored images
def name_a():

    import os

    #List all the folders of different faces
    folders = []

    # r=root, d=directories, f = files
    for r, d, f in os.walk(BASE_DIR+'/images'):
        for folder in d:
            folders.append(folder)
    print(folders)

#To view stored images by the program that are stored as the training data
def view_images():
    
    import cv2
    import os

    #Printing the names of the stored facedata
    name_a()
    name=input("Enter the name to be shown : ")
    img_path=os.path.join(image_dir,name)

    #Assigning the image path and then showing the image using opencv
    for file in os.listdir(img_path): 
        file_path = os.path.join(img_path, file)
        print(file_path)
        img = cv2.imread(file_path)
        cv2.imshow('image',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


#To Remove someone's facedata from the database
def remove():

    name_a()

    name=input("Enter the name to be deleted : ")

    print("Facedata in the database")

    #Removing the images from the disk
    import shutil
        
    dir_path=os.path.join(image_dir,name)
    shutil.rmtree(dir_path)

#***************************************************************************************************

                                        #OTHER FUNCTIONS

def click():
    import cv2
    import numpy           

    cap = cv2.VideoCapture(0)           #To start the camera
    
    while(True):
        
        ret,frame = cap.read()
        img=frame

        #To show the frame
        cv2.imshow('frame',img)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, "q- capture\nz- close",
                    (70, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        if cv2.waitKey(20) & 0xFF == ord('q'):              
                name=input('save as : ')
                path=os.path.join(BASE_DIR,'others')
                cv2.imwrite(os.path.join(path,name+'.jpg'), frame)  
                break

        else:
            continue

    cap.release()
    cv2.destroyAllWindows()

def photos():
    
    import cv2
    import numpy as np
    import os

    path=os.path.join(BASE_DIR,'others')
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        print(file_path)
        img = cv2.imread(file_path)
        print("Press q to go to next image")
        cv2.imshow('image',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
             
                
#***************************************************************************************************
            
                        #RUNNING THE MENU-DRIVEN PROGRAM USING ALL THE DEFINED ABOVE
    
while True:             #Running an infinite loop
    
    print("*"*50)
    print("*"*25,"WELCOME TO FACE RECOGNITION PROGRAM","*"*25)
    print("")
    print("1.Add Face")
    print("")
    print("2.Recognize Face")
    print("")
    print('3.Check names and number of faces')
    print("")
    print('4.View the images in the database')
    print('')
    print('5.Remove facedata of a user')
    print("")
    print('6.For Others...')
    print("")
    
    choice=int(input("Enter your choice:"))
    
    if choice==1:
        images()
        
    elif choice==2:
        recognizer()
        
    elif choice==3:
        name_a()

    elif choice==4:
        view_images()
        
    elif choice==5:
        remove()
                
    elif choice==6:
        print("*"*50)
        print("Other functions:")
        print("1.Click Photos")
        print("2.Show Images")
        print("3.Go back to previous menu")
        
        choice=int(input("Enter your choice"))

        if choice==1:
            click()

        if choice==2:
            photos()

        if choice==3:
            continue
        
    else:
        print("Wrong choice")

        
    print("Do you want to run the program again?")
    run=input("yes or no")
    
    if run=='yes':
        continue
    
    elif run=='no':
        print("BYE-BYE")
        break
    
    else:
        print("BYE-BYE")
        break