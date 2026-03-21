from bs4 import BeautifulSoup
import requests

url = "https://www.timesjobs.com/job-search?keywords=%22Ai%22%2C%22c%2B%2B%22%2C%22c%22%2C&location=&experience=&refreshed=true"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

page = requests.get(url, headers=headers)
# print(page)

soup = BeautifulSoup(page.text, "lxml")
jobs = soup.find_all("div", class_ = "mt-3 text-sm text-gray-700 leading-tight line-clamp-2")
print(jobs)