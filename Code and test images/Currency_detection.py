"""
Digital Image Processing - 1
Mini - Project
Title: Currency detection and verification using image processing
Team members: Arya R Adkoli (PES1UG20EC034)
              Aryan R (PES1UG20EC036)
Procedure to run this code:
    Download all images from 
                https://drive.google.com/drive/folders/1aPTaw4HA5RGHp3AIowAApLO_UcJjp7iS?usp=sharing 
    and this Currency_detection.py file into the same folder.
    While entering the currency note file location, choose the file location of one of the notes downloaded from the above google drive.
"""

import cv2

fileName = input("Enter file location of the note: ")
note = cv2.imread(fileName)
cv2.imshow('Input image', note)
cv2.waitKey(0)
comparision_note = ''


# Function which detects the value of the currency note based on the hue value of the input note
def detectCurrency(image):
    h, w, c = image.shape
    hpart = h//6
    wpart = w//6
    RegionForDet = image[:, (3*wpart):(5*wpart)]
    cv2.imshow('Region of note used to detect currency type', RegionForDet)
    cv2.waitKey(0)
    hsv = cv2.cvtColor(RegionForDet, cv2.COLOR_BGR2HSV)
    [h, s, v] = hsv[100, 100]
    print('Hue value in input image is = ', h)
    print('Value of the note is:')
    if h < 16:
        print('10 Rupee note')
        comparision_note = "Real_10.jpeg"
    elif h < 30:
        print("200")
        comparision_note = "Real_200.jpeg"
    elif h < 35:
        print("20")
        comparision_note = "Real_20.jpeg"
    elif h < 50:
        print("500")
        comparision_note = "Real_500.jpeg"
    elif h < 100:
        print("50")
        comparision_note = "Real_50.jpeg"
    else:
        print("Unfortunately, this model was not trained for this currency note.")
    return comparision_note, h


# Resizing the image into a standard format to make computation easy
def resiziedImage(img):
    h, w, channels = img.shape
    print('Height and width of the input image are: ', h, w)
    dim = (1080, 512)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    cv2.imshow('Resized image', resized)
    cv2.waitKey(0)
    print('Dimensions of the resized image are: ', dim)
    return resized


# Applying Median filter to remove noise
def medianFilter(img):
    img = cv2.medianBlur(img, 5)
    cv2.imshow('Median Blur', img)
    cv2.waitKey(0)
    return img


# Converting image to Binary image to verify the note
def binaryImage(img):
    ret, bw_img = cv2.threshold(img, 150, 250, cv2.THRESH_BINARY)
    cv2.imshow("Binary Image", bw_img)
    cv2.waitKey(0)
    return bw_img


# Converting RGB image to Gray scale
def rgbtogray(img):
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray scale image', img1)
    cv2.waitKey(0)
    return img1


# Using the canny filter to get all the edges in the note
def canny(img):
    edged = cv2.Canny(img, 30, 200)
    cv2.imshow('Image after canny filter', edged)
    cv2.waitKey(0)
    return edged


# Using orb similarity function to get the similarity between input image and the ideal currency note of the respective value
def orb_sim(img1, img2):
    orb = cv2.ORB_create()
    kp_a, desc_a = orb.detectAndCompute(img1, None)
    kp_a, desc_b = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc_a, desc_b)
    similar_regions = [i for i in matches if i.distance < 50]
    if len(matches) == 0:
        return 0
    return len(similar_regions) / len(matches)


# Applying all above functions on the input image
img = resiziedImage(note)
comparision_note, hue = detectCurrency(img)
img = medianFilter(img)
img = rgbtogray(img)
img = binaryImage(img)
img = canny(img)

# Applying all above functions on the ideal currency note
real_note = cv2.imread(comparision_note)
img1 = resiziedImage(real_note)
img1 = medianFilter(img1)
img1 = rgbtogray(img1)
img1 = binaryImage(img1)
img1 = canny(img1)

# Finding the similarity between the above two processed images
orb_similarity = orb_sim(img, img1)
print("Similarity using orb = ", orb_similarity)

# Printing the final result
print("Verdict: ")
if orb_similarity < 0.8:
    print("Fake note")
else:
    print("Real note")
