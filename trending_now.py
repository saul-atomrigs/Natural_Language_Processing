# pip install pytrends
import pandas as pd
from pytrends.request import TrendReq
pytrends = TrendReq(hl=’en-US’, tz=360)

# Create function to get trending keywords from Google over time / 정해진 시간에 따른 구글 검색 키워드를 반환해주는 함수 
def gtrends_overtime(full_list, key_ref, save_name="", directory="", category=0, time='all', loc=''):
    #iterate every 4 item in list plus a keyword reference as the relative comparison / 
    #구글 트렌드는 최대 5개 키워드 제한이 있으므로 리스트 안에 4개의 키워드와 비교 대상이 되는 1개의 키워드 설정
    i = 0
    while i < len(kw_list):
        l = kw_list[i:(i+4)]
        l.append(key_ref)
        pytrends.build_payload(l, cat=category, timeframe=time, geo=loc, gprop='')
        df_time = pytrends.interest_over_time()
        df_time.reset_index(inplace=True)
        df_time_name = "gtrends_overtime"+str(save_name)+str((i+4)//4)+".csv"
        df_time.to_csv(directory+df_time_name, index = False)
        i += 4
        
# Check 'Sunscreen' searches worldwide / 전 세계 선크림 검색량을 살핀다
gtrends_overtime(kw_list, 'Sunscreen', "_worldwide_", directory,
                 category=71, time='all', loc='')


def normalise(df, n_file, key_ref, col='date'):
    li = []
    # Checking the relative popularity between comparisons
    for i in range(n_file-1):    
        df = df.drop(columns=col+"."+str(i+1)+"")
        # Appending the list if relative popularity of the keyword reference is different
        if df[key_ref+"."+str(i+1)+""][0] == df[key_ref][0]:
            pass
        else:
            li.append(i+1)
    
    # Normalizing relative popularity when the relative popularity of the keyword reference is different         
    for l in li:
        k = df.columns.get_loc(key_ref+"."+str(l)+"")
        for n in range(len(df.index)):
            # Computing relative popularity by normalizing according to the reference
            if df.iloc[n,(k)] > 0:
                for m in range(5):
                    df.iloc[n,(k-4+m)] = (df.iloc[n,(k-4+m)] * (df[key_ref][n]/df.iloc[n,(k)]))
            else:
                for m in range(5):
                    df.iloc[n,(k-4+m)] = (df.iloc[n,(k-4+m)] * (df[key_ref][n]/0.01))
    return df

normalised = normalise(combined, n_file=25, key_ref="Noodles", col='date')


def tidy(df, n_file, key_ref, kw_file, col='date'):
    for i in range(n_file-1):    
        df = df.drop(columns=key_ref+"."+str(i+1)+"")
    df=pd.melt(df,id_vars=[col],var_name='Keywords', value_name='RelativePopularity')
    df = df.merge(kw_file, on="Keywords")
    return df
  
overtime = tidy(normalised, 25, "Noodles", kw_file, col='date')
