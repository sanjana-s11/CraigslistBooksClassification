# importing libraries
import numpy as np
import pandas as pd
import time
from bs4 import BeautifulSoup as bs
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import urllib
import requests
import random
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

# setting selenium chromedriver options
chrome_options = Options()
# chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')

# setting up user agents for scraping
headers_list = [
    # Firefox 77 Mac
    {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:77.0) Gecko/20100101 Firefox/77.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://www.google.com/",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    },
    # Chrome 92.0 Win10
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.google.com/",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    },
    # Chrome 91.0 Win10
    {
        "Connection": "keep-alive",
        "DNT": "1",
        "Upgrade-Insecure-Requests": "1",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Dest": "document",
        "Referer": "https://www.google.com/",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8"
    },
    # Firefox 90.0 Win10
    {
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-User": "?1",
        "Sec-Fetch-Dest": "document",
        "Referer": "https://www.google.com/",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.9"
    }
]

# opening the Seattle Craigslist book listings using Selenium
url = 'https://seattle.craigslist.org/search/bka'
driver = webdriver.Chrome('chromedriver.exe', options=chrome_options)
driver.get(url)

# initializing lists for book attributes
booktitles = []
prices = []
descriptions = []
conditions = []
count = 0
bookcount = 0

# iterating through each page of listings
while True:

    try:
        # getting the html response of page
        time.sleep(5)
        html = driver.page_source
        soup = bs(html, 'html.parser')

        # iterating through each listing in the page
        for row in soup.find_all('div', attrs={"class": "result-info"}):

            # scraping book listing title
            btitle = row.find('a', attrs={"class": "result-title hdrlnk"}).getText()
            booktitles.append(btitle)
            # scraping listing price
            price = row.find('span', attrs={"class": "result-price"}).getText()
            prices.append(price)

            # opening each listing to get more data
            listing_url = row.find('a', attrs={"class": "result-title hdrlnk"})['href']
            headers = random.choice(headers_list)
            r = requests.Session()
            r.headers = headers
            listing_html = r.get(listing_url).text
            listing_soup = bs(listing_html, 'html.parser')

            # scraping book listing description
            desc = listing_soup.find('section', attrs={"id": "postingbody"})
            if desc == None:
                descriptions.append('')
            else:
                descriptions.append(desc.getText().replace('QR Code Link to This Post', '').strip())

            # scraping book condition
            bookcondition = listing_soup.find('p', attrs={"class": "attrgroup"})
            if bookcondition == None:
                conditions.append('')
            elif bookcondition.find('b') == None:
                conditions.append('')
            elif bookcondition.find('b').getText().strip() not in ['excellent', 'fair', 'good', 'like new', 'new',
                                                                   'salvage']:
                conditions.append('')
            else:
                conditions.append(bookcondition.find('b').getText().strip())

            # scraping listing images and saving to BookImages folder
            if listing_soup.find_all('a', attrs={"class": "thumb"}) == []:
                image_tag = listing_soup.find('img')
                if image_tag != None:
                    image_url = image_tag['src']
                    img = Image.open(requests.get(image_url, stream=True).raw)
                    imgname = 'listing' + str(bookcount + 1) + "_image" + str(1) + '.jpg'
                    img.save('BookImages/' + imgname)
            else:
                for i, image in enumerate(listing_soup.find_all('a', attrs={"class": "thumb"})):
                    image_url = image["href"]
                    img = Image.open(requests.get(image_url, stream=True).raw)
                    imgname = 'listing' + str(bookcount + 1) + "_image" + str(i + 1) + '.jpg'
                    img.save('BookImages/' + imgname)

            bookcount += 1

        # finding next button and clicking to move to next page
        driver.find_elements_by_xpath("//span[@class='buttons']/a[@class='button next']")[1].click()

        # if count == 1:
        #     break

        count += 1

    except Exception as e:
        print(e)
        break

# closing Selenium chromedriver
driver.quit()

#converting book listing data to a dataframe and saving to a csv file
df = pd.DataFrame(list(zip(*[booktitles, prices, descriptions, conditions])),
                  columns=['Title', 'Price', 'Description', 'Condition'], dtype=str)
df.to_csv('ScrapedBooks.csv', index=False)
print(df)
