# scraper.py

import requests
from bs4 import BeautifulSoup as bs
from sentences import split

def scrapeBBC(url):
    r = requests.get(url)
    soup = bs(r.text, 'html.parser')
    # find article body
    body = soup.find('article')
    # find all paragraphs
    text = [split(p.text) for p in body.find_all('p')]
    # split article into sentences
    sents = [item for sublist in text for item in sublist]
    return sents


if __name__ == '__main__':
    scrapeBBC("https://www.bbc.co.uk/news/world-middle-east-62574102")
