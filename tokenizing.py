import nltk

text=""" Airbnb (NASDAQ: ABNB) is in a tizzy anticipating the end of the controversial eviction moratorium, which has been extended to July 31. This vacation-rental platform, in collaboration with city governments, wants to ban from its site any landlord who evicted a tenant for nonpayment of rent. Although landlords shouldn't evict if they're under an eviction moratorium, if they aren't receiving rent or rental relief, many would fall on hard times themselves. """
sentences = nltk.sent_tokenize(text)  # list of sentence tokens
words = nltk.word_tokenize(text) # list of word tokens

print(words[0]) 
