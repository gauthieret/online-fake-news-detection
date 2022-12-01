# from newspaper import Article
# url = 'www.eeas.europa.eu/eeas/upholding-and-strengthening-international-non-proliferation-and-disarmament-architecture_en'
# article = Article(url)

# article.download()
# article.html
# article.parse
# article.text

import trafilatura
downloaded = trafilatura.fetch_url('https://edition.cnn.com/europe/live-news/russia-ukraine-war-news-11-29-22/index.html')
article = trafilatura.extract(downloaded)
print(article)
