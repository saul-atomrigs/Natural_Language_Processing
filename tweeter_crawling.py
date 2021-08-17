from selenium import webdriver
from bs4 import BeautifulSoup as bs
import time
import datetime as dt

# driver.implicitly_wait(3)
# driver.get('https: // twitter.com/search?q=aespa&src=typed_query')

# 검색할 날짜를 적어줍니다.
startdate = dt.date(year=2021, month=8, day=1)  # 시작날짜
untildate = dt.date(year=2021, month=8, day=2)  # 시작날짜 +1
enddate = dt.date(year=2021, month=8, day=10)  # 끝날짜


# 1) 크롬 드라이버를 사용해서 웹 브라우저를 실행
query_txt = 'bts'

driver = webdriver.Chrome('/Users/sollee/Desktop/chromedriver')

driver.get(f"https://twitter.com/search?q={query_txt}&src=typed_query")
time.sleep(2)  # 위 페이지가 모두 열릴 때 까지 2초 기다립니다.

# 검색창 검색
# element = driver.find_element_by_id("bts")

full_html = driver.page_source
soup = bs(full_html, 'html.parser')
content_list = soup.select(
    'css-1dbjc4n r-18u37iz r-1wbh5a2')
# content_list = driver.find_element_by_id("bts")
for i in content_list:
    print(i.text.strip())
    print()
