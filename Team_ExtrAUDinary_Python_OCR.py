from PIL import Image, ImageEnhance
from pylab import *
import pandas as pd
import os
import nltk
import cv2
import pytesseract
import numpy as np
import nltk  # For tokenization and stemming
from nltk.corpus import stopwords  # For stopwords removal
from gensim.parsing.preprocessing import remove_stopwords
import warnings

warnings.filterwarnings("ignore")

# add tesseracts to path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Assign directory names from the book images
directory = '5 books'

# iterate over files in that directory
books = []
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        books.append(f.replace('\\', '/'))
# open images
im = []
for i in books:
    im.append(Image.open(i))

# sharpen
im_sharpen = []
for i in im:
    enhancer = ImageEnhance.Sharpness(i)
    im_sharpen.append(enhancer.enhance(5.5))

# image-to-text
im_str = []
# Adding custom options
custom_config = r'--oem 3 --psm 6'
for i in im_sharpen:
    im_str.append(pytesseract.image_to_string(i, config=custom_config))

# tokenize and remove stopwords
im_token = []
for i in range(len(im_str)):
    im_token.append(nltk.word_tokenize(im_str[i]))
im_temp = []
converted_string = []
for i in im_token:
    im_temp.append([token for token in i if token not in stopwords.words('english') if token.isalpha()])
for i in im_temp:
    converted_string.append(' '.join(map(str, i)))
print(converted_string)
for x, y in zip(im, im_sharpen):
    figure()
    plt.subplot(1, 2, 1)
    imshow(x)
    plt.subplot(1, 2, 2)
    imshow(y)
    plt.show()
