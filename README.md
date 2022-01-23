# CraigslistBooksClassification
Developing models to classify Craigslist books data into genres based on scraped listings and using image processing to retrieve missing titles

This project is a part of MGMT 59000-066 - Analyzing Unstructured Data course.
This project is done by Team ExtrAUDinary - Shreyansh Jain, Vineet Suhas Soni, Adithya Bharadwaj Umamahesh, Vishal Shokeen and Sanjana Santhanakrishnan


Python Files List
--------------------------
1. Team_ExtrAUDinary_Scraping.py

This code scrapes the data for all book listings in Seattle from Craigslist and saves it to ScrapedBooks.csv file. The images are saved in BookImages subfolder.

2. Team_ExtrAUDinary_Modeling.py

This file has the code for the text classification models to predict the genre for books using titles.

3. Team_ExtrAUDinary_OCR.py

This file has the code for OCR of book titles from images in the '5 books' subfolder using pytesseract.


CSV Files List
--------------
1. ScrapedBooks.csv

This file has the scraped Craigslist Seattle book listings data.

2. craigs_listings.csv

This file has the Craigslist Seattle book listings data with manually tagged genres.

3. book32-listing.zip

This zipped file has the Amazon book listings data in csv format used for training.


Subfolders
----------

1. BookImages

This folder contains a sample of the scraped images from Seattle Craigslist books section.

2. 5 books

This folder contains the input images for OCR used in the Team_ExtrAUDinary_OCR.py file.


Dependencies
------------
1. Team_ExtrAUDinary_Scraping.py

	(1) Install Selenium using pip (pip install selenium)
	(2) Next, install chromedriver on your system:
		https://chromedriver.chromium.org/downloads
	Select the appropriate file based on Chrome version of your system and operating system.
	(3) Place the downloaded chromedriver.exe file in the same folder as this python file. (This executable is included in the submitted zip folder.)
	(4) Create a subfolder called BookImages if it doesn't already exist to store the scraped images.


2. Team_ExtrAUDinary_Modeling.py

	craigs_listings.csv and book32-listing.csv files are used as inputs to this python file. Place them in the same folder as this python file.


3. Team_ExtrAUDinary_OCR.py

	(1) Install pytesseract using pip (pip install pytesseract)
	(2) Next, install pytesseract on your system:
		1. For Windows, there are 2 links:
			https://digi.bib.uni-mannheim.de/tesseract/
			https://github.com/UB-Mannheim/tesseract/wiki
		2. For Mac/Linux:
			https://tesseract-ocr.github.io/tessdoc/Installation.html
	(3) Note down the save path, since you’ll be needing it in the next steps
	(4) Import pytesseract in your program
	(5) Add pytesseract to your path using command:
		pytesseract.pytesseract.tesseract_cmd=r'C:\Users\USER\AppData\Local\Tesseract-OCR\tesseract.exe'
	Now, you can run the tesseract image-to-string function used in this python file.

	(6) Place the '5 books' folder which has the input images in the same folder as this python file.




[Extraudinary.pptx](https://github.com/sanjana-s11/CraigslistBooksClassification/files/7919862/Extraudinary.pptx)
[Team_ExtrAUDinary_Report.pdf](https://github.com/sanjana-s11/CraigslistBooksClassification/files/7919863/Team_ExtrAUDinary_Report.pdf)
