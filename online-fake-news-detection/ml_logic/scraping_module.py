# from newspaper import Article
# url = 'www.eeas.europa.eu/eeas/upholding-and-strengthening-international-non-proliferation-and-disarmament-architecture_en'
# article = Article(url)

# article.download()
# article.html
# article.parse
# article.text

import trafilatura
from trafilatura import extract
down = trafilatura.fetch_url('https://edition.cnn.com/2022/11/30/uk/china-embassy-uk-king-charles-gbr-intl/index.html')
test_sample = pd.DataFrame({'news': [extract(down)]})
