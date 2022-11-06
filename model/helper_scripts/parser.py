'''Parser for google search images'''

import requests
import os
from bs4 import BeautifulSoup
from selenium import webdriver
from time import time


folder = r'PATH'   # insert output folder path

if not os.path.isdir(folder):
    os.makedirs(folder)


def download(image, folder, index):
    reponse = requests.get(image)

    if reponse.status_code == 200:
        with open(os.path.join(folder, f'image{index}.jpg'), 'wb') as file:
            file.write(reponse.content)


driver = webdriver.Chrome(r'chromedriver.exe')   # insert PATH to a driver
url = r'URL'    # insert page link
driver.get(url)

input('Scroll down the page to determine the number of images and then enter something: ')
driver.execute_script('window.scrollTo(0, 0);')

soup = BeautifulSoup(driver.page_source, 'html.parser')
images = soup.find_all('div', class_='isv-r PNCib MSM1fd BUooTd')
print(f'Found {len(images)} images.')

downloaded = 0
fails = 0
for index in range(1, len(images) + 1):
    # every 25th index in a google image search is a link to another page

    if index % 25 != 0:
        poor_quality_image = driver.find_element(
            'xpath', f'//*[@id="islrg"]/div[1]/div[{index}]/a[1]/div[1]/img').get_attribute('src')

        driver.find_element(
            'xpath', f'//*[@id="islrg"]/div[1]/div[{index}]').click()

        start = time()
        while time() - start <= 15:
            # good quality image doesn't appear immediately, I'll give 15 seconds for this

            image = driver.find_element(
                'xpath', '//*[@id="Sva75c"]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div[3]/div/a/img').get_attribute('src')

            if image != poor_quality_image:
                break

        try:
            download(image, folder, index)
            print(f'{index} / {len(images) - len(images) // 25} downloaded')
            downloaded += 1

        except:
            print(f'{index} / {len(images) - len(images) // 25} failed')
            fails += 1


print(f'Done!\nDownloaded: {downloaded}\nFailed: {fails}')
