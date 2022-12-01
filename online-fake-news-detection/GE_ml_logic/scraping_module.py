# from newspaper import Article
# url = 'www.eeas.europa.eu/eeas/upholding-and-strengthening-international-non-proliferation-and-disarmament-architecture_en'
# article = Article(url)

# article.download()
# article.html
# article.parse
# article.text

import trafilatura

def get_new_data(url):
    downloaded = trafilatura.fetch_url(f'{url}')
    article = trafilatura.extract(downloaded)
    return article
