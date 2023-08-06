import cv2
# haarcascade_frontalface_default.xml is a file that contains a pre-trained Haar cascade classifier for detecting frontal faces in images or video streams. 
# The Haar cascade algorithm is a machine learning-based approach for object detection
# The below line is used to create an instance of the CascadeClassifier class and load the trained cascade classifier for frontal face detection from the XML file named "haarcascade_frontalface_default.xml".
cascade_face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") 
# haarcascade_smile.xml is another XML file that contains a pre-trained Haar cascade classifier, for detecting smiles in images or video streams
cascade_smile = cv2.CascadeClassifier("haarcascade_smile.xml")
# used to create a VideoCapture object that allows you to access the video stream from a specified device. 
# In this case, 0 represents the index of the default camera device.
cap = cv2.VideoCapture(0)
# we use infinite while loop so that it detects the smile every time we smile
while True:
# The below line is used in OpenCV to read a frame from the video capture source, such as a camera or a video file
# It returns two values 
# ret: A Boolean value indicating whether the frame was successfully read or not. It will be True if the frame was read successfully and False if there was an error or the video capture source has reached the end.
# img: The frame that was read, represented as an image (NumPy array) in BGR format.
    ret, img = cap.read()
# the cv2.cvtColor() function is used to convert the img image from BGR(blue,green,red) to grayscale using the cv2.COLOR_BGR2GRAY conversion flag
# converting to grayscale means losing color information
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# The line of code performs face detection using a cascade classifier on a grayscale image
# scaleFactor: This parameter specifies how much the image size is reduced at each image scale. A value of 1.4 means reducing the image by 40% at each scale
# minNeighbors: This parameter defines the minimum number of neighboring rectangles required for a region to be considered as a face
# minSize: This parameter sets the minimum size of the detected face region. Any detected face region smaller than this size will be ignored
    f = cascade_face.detectMultiScale(g, scaleFactor=1.4, minNeighbors=5, minSize=(30,30))
    for (x,y,w,h) in f:
    # used to draw rectangles around the detected face regions on the original image
    # For each detected face, the variables x, y, w, and h represent the coordinates and dimensions of the face region
    # 255,0,0 these are BGR colors values
    # The last argument 2 represents the thickness of the rectangle's border
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
    # The code you provided appears to be indexing a variable g using the variables x, y, w, and h to obtain a subarray gray_r.
    # By using the slicing notation x : x+w, y : y+h, you are specifying a range of indices along each dimension to extract a subarray from g. 
    # The resulting subarray, gray_r, will contain the elements from g that fall within the specified range.
        gray_r = g[y : y+h, x : x+w]
    # s = cascade_smile.detectMultiScale(grey_r, scaleFactor=1.5, minNeighbors=15, minSize=(25,25)): This line detects smiles in the grey_r image using a smile cascade classifier (cascade_smile). 
    #The detectMultiScale function applies the classifier to the image and returns a list of rectangles that represent the detected smiles. 
    #The scaleFactor parameter specifies how much the image size is reduced at each scale, minNeighbors specifies how many neighbors each candidate rectangle should have to retain it, and minSize sets the minimum size of the detected smile region.
        s = cascade_smile.detectMultiScale(gray_r, scaleFactor=1.5, minNeighbors=15, minSize=(25,25))
    #for i in s: This line starts a loop that iterates over each rectangle in the list s, which contains the detected smiles.
        for i in s:
    # if len(s) > 1:: This condition checks if there is more than one smile detected in the image.
            if len(s) > 1:
        # If the condition in the previous line is satisfied, this line adds a text label, "Smiling", to the original image (img). 
        # The text is positioned above each detected smile rectangle, with the (x, y-30) coordinates specifying the position. 
        # The remaining arguments define the font type, size, color, thickness, and line type for the text
                cv2.putText(img, "Smiling",(x,y-30),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3,cv2.LINE_AA)
    #  This line displays the image img in a window named 'video'
    cv2.imshow('video',img)
    # This line waits for a key event. The function cv2.waitKey() waits for a specified number of milliseconds (in this case, 30 milliseconds) for a keyboard event. 
    # The return value k contains the ASCII value of the pressed key bitwise-ANDed with 0xff to extract the lower 8 bits
    k = cv2.waitKey(30) & 0xff
    # This line checks if the key pressed has an ASCII value of 27, 
    # which corresponds to the 'Esc' key. If the 'Esc' key is pressed, the loop is broken, and the program will exit
    if k == 27:
        break
# This line releases the video capture object cap. It's used to free up system resources related to video capturing
cap.release()
#  To close all OpenCV windows that were opened for displaying images
cv2.destroyAllWindows()

            
            
    
    
    
    
