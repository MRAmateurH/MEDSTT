
# from selenium import webdriver

import os
import re

class Spider:

    def __init__(self, content='') -> None:
        self.content = content

    # def web_crawler(url):
    #     options = webdriver.ChromeOptions()
    #     options.add_argument('headless')
    #     driver = webdriver.Chrome(options=options)
    #     driver.get(url)

    #     driver.quit()

    def remove_extra_spaces(self, text: str):

        self.content = " ".join(text.split())

    def remove_parenthesis_content(self, text: str):

        pattern = r'\([^)]*\)'

        self.content = re.sub(pattern, '', text)

    def remove_html_tag(self, text): 

        text = re.sub('<[^>]*>', ' ', str(text))
        self.content = " ".join(text.split())

    def remove_url(self, text: str):
        url_regex = re.compile(r'https?://\S+')

        url_in_text = url_regex.findall(text)

        if url_in_text:
            # 將url加入句子中，進行刪除時可以刪除包含url之完整句子
            sentence_regex = re.compile(r'([^.!?]*{})[^.!?]*[.!?]'.format('|'.join(url_in_text)))
            self.content = sentence_regex.sub('', text)

            if self.content == text:
                sentence_regex = re.compile(r'([^.!?]*{})[^.!?]*'.format('|'.join(url_in_text)))
                self.content = sentence_regex.sub('', text)
        
    
    def normalize_quotes(self, text: str):
        text = text.replace('n"t', 'n\'t')
        table = text.maketrans('‘’', '\'\'')
        text = text.translate(table)
        
        self.content = text






    
