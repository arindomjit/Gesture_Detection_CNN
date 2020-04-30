import cv2
import os
import sys

try:
    num_of_images = int(sys.argv[1])
    image_label = sys.argv[2]
except:
    print("\nInvalid syntax. Please pass the number of images to be captured and label name as arguments.")
    print("Sample Command: python get_training_images.py 100 mute")
    exit(-1)

font = cv2.FONT_HERSHEY_PLAIN
click = False
#All the captured categories will have a sub-folder placed inside the 'training_img_folder'
training_img_folder = 'training_images'

#The label_name is the name of our category (Eg: - up, down, chrome etc.)
label_name = os.path.join(training_img_folder, image_label)

#count of images to be captured
count = image_name = 0

try:
    os.mkdir(training_img_folder)
except FileExistsError:
    pass
try:
    os.mkdir(label_name)
except FileExistsError:
    #If any images are already present, updating the image name starting number
    image_name=len(os.listdir(label_name))

    
#Begin Capturing Images from webcam
video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 2000)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 2000)

while True:
    ret, image = video.read()
    image = cv2.flip(image, 1)
    
    if not ret:
        continue

    #Stop capturing images once the count is reached
    if count == num_of_images:
        break

    #Drawing a square with white border. Anything inside this square box will be captured as training image.
    cv2.rectangle(image, (200, 200), (550, 550), (255, 255, 255), 2)

    #Start capturing pictures when user presses 's' key
    if click:
        region_of_interest = image[200:550, 200:550]
        save_path = os.path.join(label_name, '{}.jpg'.format(image_name + 1))
        cv2.imwrite(save_path, region_of_interest)
        image_name += 1
        count += 1

    #putText() method is used here to display message inside the webcam feed. It takes the following parameters
    #<image> : the image where the text is to be displayed
    #<text> : text to be displayed
    #(x,y) : position of the text
    #<font> : the font name of the text
    #<font_size>: size of the font
    #(BGR) : the color of the text in BGR format
    #<font_thickness> : thickness of the text characters
    cv2.putText(image, "Fit the gesture inside the white box and Press 's' key to start clicking pictures",
            (20, 30), font, 0.8, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, "Press 'q' to exit.",
            (20, 60), font, 0.8, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, "Image Count: {}".format(count),
            (20, 100), font, 1, (12, 20, 200), 2, cv2.LINE_AA)
    cv2.imshow("Get Training Images", image)

    k = cv2.waitKey(10)
    if k==ord('q'):
            break
    if k == ord('s'):
        click = not click

print("\n\nDone\n\n")
video.release()
cv2.destroyAllWindows()
