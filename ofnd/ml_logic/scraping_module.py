# from newspaper import Article
# url = 'www.eeas.europa.eu/eeas/upholding-and-strengthening-international-non-proliferation-and-disarmament-architecture_en'
# article = Article(url)

# article.download()
# article.html
# article.parse
# article.text
import pandas as pd
import trafilatura
from trafilatura import extract

def scraping(url: str):

    down = trafilatura.fetch_url(url)
    test_sample = pd.DataFrame({'news': [extract(down)]})

    return test_sample
