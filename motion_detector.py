"""
Course : ENGI 9805 Computer Vision
Project : Motion detector
Team members : Utkarsh Trivedi , Sona Pujari , Shujhana Mostafa
Date Last Modified : 02/04/2020
Version : 1.9

"""

#Importing libraries for usage in python
import imutils
import cv2
import numpy as np
import time
import pandas
from datetime import datetime


# =============================================================================
# User Set Parameters
# =============================================================================

#Initializing first_frame to variable none
first_frame=None

#Initializing status_list
status_list=[None,None]

#Initializing times
times=[]

# Minimum length of time where no motion is detected it should take
#(in program cycles) for the program to declare that there is no movement
MOVEMENT_DETECTED_PERSISTENCE = 100

font = cv2.FONT_HERSHEY_SIMPLEX
delay_counter = 0
movement_persistent_counter = 0

df=pandas.DataFrame(columns=["Start","End"])

# =============================================================================
# Core Program
# =============================================================================

# Video Capture in opencv2
video = cv2.VideoCapture(5) # Flush the stream
video.release()
video=cv2.VideoCapture(0)

# Loop
while True:

    # Set transient motion detected as false
    transient_movement_flag = False

    # After the first frame is done the next frame is continued for capture
    check, frame = video.read()
    text = "Unoccupied"

    # If there's an error in capturing
    if not check:
        print("CAPTURE ERROR")
        continue

    # Starting of frame so status is assigned 0
    status=0

    # Resizing the frame inorder to get a better view of all windows
    frame = imutils.resize(frame, width = 500)
    # Assigning and Conversion to gray scale
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

# Applying a GaussianBlur inorder to remove noise in gray scale
# value is set to (21,21),0 in order to remove camera noise (reducing false positives)
    gray=cv2.GaussianBlur(gray,(21,21),0)

# In the while loop when first_frame is none it would be assigned to gray to initialise it
    if first_frame is None:
        first_frame=gray
        continue

    delay_counter += 1


    # Otherwise, set the first frame to compare as the previous frame
    # But only if the counter reaches the appriopriate value
    # The delay is to allow relatively slow motions to be counted as large
    # motions if they're spread out far enough
    if delay_counter > 10:
        delay_counter = 0
        first_frame = gray

# Delta frame compares current frame(gray) with background/first frame
    delta_frame=cv2.absdiff(first_frame,gray)

# Compare the two frames, find the difference using THRESH_BINARY method
# Value initially was chosen 30 but made 25 for better accuracy
#[1] is to access the second methid of tupple of THRESH_BINARY
    thresh_frame=cv2.threshold(delta_frame, 25, 255, cv2.THRESH_BINARY)[1]

# Fill in holes via dilate() - smoothens the white areas
    thresh_frame=cv2.dilate(thresh_frame, None, iterations=2)

# Assigning contours in opencv2. Change to (_,cnts,_) if using opencv3/python3
# RETR_EXTERNAL method is used to retrieve external of the contours in frame
# CHAIN_APPROX_SIMPLE method is used for retrieving the contours in opencv2
    (cnts,_)=cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# loop over the contours
    for contour in cnts:
        #10000 is for big objects being closer to camera
        #10000 = 100x100 pixels window
        # can be set to a lower value like 2000 for micro movements
        if cv2.contourArea(contour) > 2000 :
            transient_movement_flag = True
            continue
        # Status is assigned 1
        status=1

        # Applying rectangle over found contours
        (x, y, w, h)=cv2.boundingRect(contour)

        # Draw a rectangle around big enough movements
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)


    # The moment something moves momentarily, reset the persistent
    # movement timer.
    if transient_movement_flag == True:
        movement_persistent_flag = True
        movement_persistent_counter = MOVEMENT_DETECTED_PERSISTENCE

    # As long as there was a recent transient movement, say a movement
    # was detected
    if movement_persistent_counter > 0:
        text = "Movement Detected " + str(movement_persistent_counter)
        movement_persistent_counter -= 1
    else:
        text = "No Movement Detected"

    # Print the text on the screen, and display the raw and processed video
    # feeds
    cv2.putText(frame, str(text), (10,35), font, 0.75, (255,255,255), 2, cv2.LINE_AA)


    # Appending status to status_list
    status_list.append(status)

    # To capture the last two items of status list
    status_list=status_list[-2:]

    # we want to record datetime if status_list changes from 1 to 0
    if status_list[-1]==1 and status_list[-2]==0:
        times.append(datetime.now())
    # we want to record datetime if status_list changes from 0 to 1
    if status_list[-1]==0 and status_list[-2]==1:
        times.append(datetime.now())

    # Gray Frame - Shows gray scaled video
    # Delta Frame - Shows difference from original
    # Threshold Frame - shows black and white threshold
    # Color Frame - regular video frame capture

    """
    => The below was used in previous version 1.5 but showed 4 different windows
     so modified to different enhanced imshow for better functionality in version 1.9

    # cv2.imshow("Gray Frame",gray)
    # cv2.imshow("Delta Frame",delta_frame)
    # cv2.imshow("Threshold Frame",thresh_frame)
    # cv2.imshow("Color Frame",frame)
    """
    # Convert the delta frame to color for splicing
    delta_frame = cv2.cvtColor(delta_frame, cv2.COLOR_GRAY2BGR)

    # Splice the two video frames together to make one long horizontal one
    cv2.imshow("Delta Frame & Color Frame", np.hstack((delta_frame, frame)))
    cv2.imshow("Gray Frame & Threshold Frame", np.hstack((gray, thresh_frame)))

    key=cv2.waitKey(1)

    if key==ord('q'):
        if status==1:
            times.append(datetime.now())
        break

print(status_list)
print(times)

for i in range(0,len(times),2):
    df=df.append({"Start":times[i],"End":times[i+1]},ignore_index=True)

df.to_csv("Times.csv")

# Cleanup when closed
cv2.destroyAllWindows()
video.release()
