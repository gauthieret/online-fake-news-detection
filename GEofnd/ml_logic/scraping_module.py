import pandas as pd
import trafilatura
from trafilatura import extract
from trafilatura.settings import use_config


def scraping(url: str):
    # Solution for signal / thread error
    config = use_config()
    config.set("DEFAULT", "EXTRACTION_TIMEOUT", "0")
    down = trafilatura.fetch_url(url)
    scraped_df = pd.DataFrame({'news': [extract(down, config=config)]})

    return scraped_df

if __name__ == '__main__':
    print(scraping('https://www.cnbc.com/2022/12/05/russia-ukraine-live-updates.html'))
