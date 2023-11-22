import re
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from markdown import markdown
from unidecode import unidecode


class TextCleaner:
    def __init__(self):
        self.url_pattern = re.compile(r"https?://\S+")
        self.username_pattern = re.compile(r"@[\w]+")
        self.emoticon_pattern = re.compile(r":[!#@\w\-]+:")

    def _strip_markdown(self, text):
        html = markdown(text)
        return unidecode(BeautifulSoup(html, "html.parser").get_text().strip())

    def _replace_urls(self, text):
        return self.url_pattern.sub(lambda m: urlparse(m.group()).netloc or "url", text)

    def _clean_text(self, text):
        tokens = text.split()
        cleaned_tokens = []
        for token in tokens:
            if self.username_pattern.match(token):
                token = re.sub(r"('s)?\W*$", "", token)
                cleaned_tokens.append(token)
            elif self.emoticon_pattern.match(token):
                cleaned_tokens.append(token.replace("!", "").replace("#", ""))
            else:
                token = re.sub(r"\W+", "", token)
                cleaned_tokens.append(token)
        return " ".join(cleaned_tokens)

    def tokenize(self, text):
        text = text.lower()
        text = self._strip_markdown(text)
        text = self._replace_urls(text)
        text = text.replace("::", ": :")
        text = self._clean_text(text)
        return [token for token in text.split() if token and len(token) < 50]

