"""
IMAGE CRAWLER WITH SELENIUM, CHROME, YAHOO.

2020 02 04 JJH 

referred from : https://pythonspot.com/selenium-get-images/ , GOOGLE IMAGE SELENIUM CRAWLER 2020 01 30 JJH, https://j-remind.tistory.com/61 , https://engkimbs.tistory.com/896 , https://mochalatte.me/programming/py-crawling .

Since my main internet browser is chrome, I'm going to automate web crawling with chrome by default. The code is optimized for Yahoo page(Because of page renewal method)
Crawls images by url, not search term. IT WON'T BE RUN AS LONG AS YOU MODIFY PARAMETRES.

PARAMETRES

user_header : dictionary. d{'User-Agent' : check chrome://version:user_agent_tab(personalised. You must fill it as your own)}
set_title : string. It is recommended that matching its title with url.
image_type : string. set as 'png', 'jpg', 'jpeg'. png for default.
driver : string. To run this code, you must set where the CHROMEDRIVER is. go https://chromedriver.chromium.org/downloads to download it. the chromedriver requires EXACTLY matched version of chrome you use.

"""

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time, json, os, urllib.request

"""PARAMETRES"""
user_header = {'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36"}
set_title = 'Yahoo Japan Doggies'
image_type='png'
driver = webdriver.Chrome('C:/Users/Azerates/Downloads/chromedriver.exe')
#set chrome driver for automated webpage control.
driver.get('https://search.yahoo.co.jp/image/search?ei=UTF-8&p=%E7%8A%AC')
#input url which we'd like to investigate

driver.implicitly_wait(0.5)
#automated bot will wait for 0.5 second by every renewal. 

options = webdriver.ChromeOptions()
options.add_argument('headless')
options.add_argument('window-size=1920x1080')
options.add_argument('disable-gpu')

body = driver.find_element_by_tag_name('body')

number_of_page_down = 15 # Must be a natural number. Determines how many times automated bot will scroll down the pages.

if not os.path.exists(set_title): os.mkdir(set_title)  # if there is no folder named 'set_title'(if set_title='Yahoo Japan Kitties, then it means if there is no 'Yahoo Japan Kitties' folder where this code is running), then create it.

for i in range(number_of_page_down):
    print(f'Page Down Count :{i}')
    body.send_keys(Keys.PAGE_DOWN)  # Has same effect when you pushed PAGE DOWN function key.
    time.sleep(0.25) # wait for 0.25 second.
    try:
        driver.find_element_by_xpath("""//*[@id="autopagerMore"]""").click()
        # to obtain XPath, you can use copy-> copy XPath. //*[@id="autopagerMore"] is XPath of <li id="autopagerMore">.
    except:
        # if click 'see more' failed, do nothing.
        pass

images = driver.find_elements_by_tag_name('img') # Since all images have same method 'img src' to represent themselves, find elements by 'tag' is valid.

cnt = 1  # success count
f_cnt = 1 # failure count

for x in images:
    target_image_url = x.get_attribute('src') # and itt attribute is undoubtedly src.
    print(cnt, target_image_url) # prints where the target image is.
    try:
        send_request = urllib.request.Request(target_image_url, headers=user_header)  # send a request to server that we want to obtain the url of the image, verifying that we aren't fully automated robot.
        raw_image = urllib.request.urlopen(send_request).read() # open and read it.
        save_file = open(os.path.join(set_title, set_title+'_'+str(cnt)+'.'+image_type), 'wb') # now define filename to store to your own local HDD, where the folder named with 'set_title' is. e.g. kitty_1.jpg by write binary mode. 
        save_file.write(raw_image) # create file there
        save_file.close() # and close it. 
        print(cnt,' downloaded image. \n', target_image_url)
        cnt += 1
    except:
        print(f'unable to download image {f_cnt}')
        f_cnt += 1

print(f'TOTAL COUNT : SUCCESS : {cnt}, FAILURE : {f_cnt}')
print('----------END OF CRAWLING----------')


# -> scroll all pages. Since yahoo requires to push button to expand scroll.

