import pandas as pd
import trafilatura
from trafilatura import extract

def scraping(url: str):

    down = trafilatura.fetch_url(url)
    test_sample = pd.DataFrame({'news': [extract(down)]})

    return test_sample
