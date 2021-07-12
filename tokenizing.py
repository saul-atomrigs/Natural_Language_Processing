import nltk
from nltk.stem import PorterStemmer

text=""" Airbnb (NASDAQ: ABNB) is in a tizzy anticipating the end of the controversial eviction moratorium, which has been extended to July 31. This vacation-rental platform, in collaboration with city governments, wants to ban from its site any landlord who evicted a tenant for nonpayment of rent. Although landlords shouldn't evict if they're under an eviction moratorium, if they aren't receiving rent or rental relief, many would fall on hard times themselves. """
sentences = nltk.sent_tokenize(text)  # list of sentence tokens / 문장 단위로 끊어서 리스트로 반환
words = nltk.word_tokenize(text) # list of word tokens / 단어 단위로 끊어서 리스트로 반환

print(words[0]) 

# Stemming the words / 영문 단어들을 원형으로 전처리 한다
stemmer = PorterStemmer()
stemmed = []
for w in words:
  stemmed.append(stemmer.stem(w.lower()))
print(stemmed)


# Preprocessing in Korean / 한글 전처리
clean_review=re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]","", reviews[])

# 한글 토크나이징
okt=Okt()
review=okt.morphs(review,stem=True)

# 한글 불용어 (직접 설정)
stopwords=set(['은','는','이','가','하','아','것','들','의','있','되','수','보','주','등','한','앗'])
clean_review=[r for r in reviews if r not in stopwords]
