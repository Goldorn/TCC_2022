#!/usr/bin/env python3
import requests
import datetime

from bs4 import BeautifulSoup

# Session connection
session = requests.session()
session.proxies["http"] = "socks5h://localhost:9050"
session.proxies["https"] = "socks5h://localhost:9050"

count = 1

with open('URL_list9.txt') as f:
  for line in f:
     url = line.rstrip()
     response = session.get(url)

     # URL content extraction
     #raw_content = response.content

     # URL content transformation into BeautifulSoup object type - HTML
     html_content = BeautifulSoup(response.content, "html.parser")

     # Initial post data finder
     post_html = html_content.find("div", attrs={"class": "qa-q-view"})
     # HTML Preparation
     user_html = post_html.find("span", attrs={"class": "qa-q-view-who-data"})
     date_html = post_html.find("span", attrs={"class": "qa-q-view-when-data"})
     # Text Only
     user_initialpost = user_html.find("span", attrs={"itemprop": "name"}).text		# Initial post user
     comment_initialpost = post_html.find("div", attrs={"itemprop": "text"}).text	# Initial post comment
     date_initialpost = date_html.find("time", attrs={"itemprop": "dateCreated"}).text	# Initial post creation date

     # Answers data finder
     answers_html = html_content.find("div", attrs={"class": "qa-part-a-list"})
     # HTML Preparation
     users_answer_html = answers_html.find_all("span", attrs={"class": "qa-a-item-who-data"})
     comments_answer_html = answers_html.find_all("div", attrs={"class": "qa-a-item-content qa-post-content"})
     date_answer_html = answers_html.find_all("span", attrs={"class": "qa-a-item-when-data"})
     #print(date_answer_html.text)
     # Text Only
     userFinal = []
     userFinal.append(user_initialpost)
     for user in users_answer_html:
       users_answer = user.find("span", attrs={"itemprop": "name"}).text
       userFinal.append(users_answer)

     dateFinal = []
     dateFinal.append(date_initialpost)
     for date in date_answer_html:
       date_answer = date.find("time", attrs={"itemprop": "dateCreated"})
       if date_answer:
         #print(date_answer.text)
         dateFinal.append(date_answer.text)

     commentFinal = []
     commentFinal.append(comment_initialpost)
     for comment in comments_answer_html:
       comment_answer = comment.find("div", attrs={"itemprop": "text"}).text
       if len(comment_answer) > 40:
         commentFinal.append(comment_answer)

     # Concatenate lists
     #list0 = list(zip(userFinal,commentFinal,dateFinal))
     list0 = list(zip(userFinal,commentFinal))
     list1 = "\n".join(map(str, list0))
     list2 = list1.replace("(","")
     list3 = list2.replace(")","")
     #print(list3)

     basename = '/home/kali/TCC/Resultado Final/respostas_ocultas'
     suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
     filename = "_".join([basename, suffix, '.txt'])

     # Salvar conteudo em um arquivo txt
     text_file =  open(filename, 'w')
     n = text_file.write(list1)

     # Fechando arquivo
     text_file.close()

     count += 1

