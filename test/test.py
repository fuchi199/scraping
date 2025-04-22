# pip install beautifullsoup4
# pip install pandas
# pip install requests
# pip install selenium

from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup as bs
import pandas as pd
import requests
from datetime import datetime
import time
import re
import warnings
warnings.filterwarnings("ignore")

#urlのres化
def res_data(url):
    res = requests.get(url)
    res_bs = bs(res.content , "html.parser")
    return res_bs

#pandasに格納
def dfData(res_bs):
    item_list = res_bs.findAll(class_ =re.compile(r""))
    output = []
    for num , item in enumerate(item_list):
        item_url = item.find(class_ = "").get("href")
        title = item.find(class_ = "").text
        if item.find(class_ = ""):
            price = item.find(class_ = "").text
        else:
            price = "none"
        
        output.append({
            "item_url":item_url,
            "title":title,
            "price":price})
    df = pd.DataFrame(output)
    return df

def next_url(res_bs):
    domain = ""
    return None

def alldf(url):
    all_df = pd.DataFrame()
    for _ in range(10):
        res_bs = res_data(url)
        df = dfData(res_bs)
        all_df = all_df.append(df)
        print("all_df:",len(all_df))
        print(f"sleepin...{_}回目")
        time.sleep(5)
        if url is None:
            break

# login
def login(login_url,USER,PASS):
    browser = webdriver.Chrome()
    browser.get(login_url)
    elem_username  = browser.find_element(By.NAME,"").send_keys(USER)
    elem_password = browser.find_element(By.ID,"").send_keys(PASS)
    browser_from = browser.find_element(By.ID,"").click()
    time.sleep(15)
    
    current_url = browser.current_url
    print("ログイン後のURL:", current_url)
    
    url = next_url(1)
    browser.get(url)

    urls = {}
    
    for url in urls:
        browser.get(url)
        time.sleep(1)  # 必要に応じて調整

        page_data = browser.page_source
        print(f"\n--- {url} ---")
        print(page_data[:300])  # 長いので一部だけ表示

    html = browser.page_source
    
    browser.quit()


