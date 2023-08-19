import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
from newspaper import Article
from tqdm import tqdm

#get links
def get_news(name):
    def find_elements_with_class(url, class_name):
        # Send a GET request to the URL
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code != 200:
            print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
            return []

        # Parse the webpage content with BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all elements with the specified class and extract the 'href' attribute
        elements = soup.find_all('div',class_=class_name)
        urls = [elem for elem in elements]

        return urls

    url = "https://news.google.com/search?q="+name+"&hl=en-IN&gl=IN&ceid=IN%3Aen"
    class_name = "xrnccd"
    element_urls = find_elements_with_class(url, class_name)
    
    name,source,link,date = [],[],[],[]
    
    for i in element_urls:
        name.append(i.find_all('a', class_='RZIKme')[0].text)
        source.append(i.find_all('a', class_='wEwyrc')[0].text)
        link.append(i.find_all('a', class_='RZIKme')[0]['href'].replace(".","news.google.com"))
        date.append(i.find('time',class_='slhocf')['datetime'])
        
    x =  pd.DataFrame({'title':name,'source':source,'link':link,'date':date})
    
    date = []
    for i in x.index:
        date.append(datetime.strptime(x['date'].values[i], '%Y-%m-%dT%H:%M:%SZ').date())
    x['date'] = date
    
    x = x[x['date'] >  datetime.today().date() - timedelta(days=14)].sort_values(by='date',ascending=False).head(10)
    
    return x

#get summaries
def get_details(df):
#A new article from TOI
    text,summaries,title = [],[],[]

    for i in df.index:
        try:
            url = "http://"+df.link[i]
            #For different language newspaper refer above table
            toi_article = Article(url, language="en") # en for English
            #To download the article
            toi_article.download()
            #To parse the article
            toi_article.parse()
            toi_article.nlp()
            #To extract text
            text.append(toi_article.text)
            #To extract summary
            summaries.append(toi_article.summary)
            title.append(toi_article.title)
            
        except:
            text.append(None)
            summaries.append(None)
            title.append(None)            

    df['summary'] = summaries
    df['text'] = text
    df['title_retrieved'] = title

    return df

names = pd.read_csv("db_firmo.csv").Name.unique()
for n in tqdm(names):
    #print(n.replace(" ","_"))
    links = (get_news(n+" Company India"))
    #news = get_details(links)
    links.to_csv("news_articles/"+n.replace(" ","_")+".csv")
