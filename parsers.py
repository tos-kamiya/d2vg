import bs4


def parse(file_name):
    i = file_name.rfind('.')
    if i < 0:
        return None
    extension = file_name[i:]
    if extension in ['.html', 'htm']:
        return html_parse(file_name)
    else:
        with open(file_name) as inp:
            text = inp.read()
        return text


def html_parse(file_name):
    with open(file_name) as inp:
        html_doc = inp.read()
        soup = bs4.BeautifulSoup(html_doc, 'html.parser')
        for script in soup(["script", "style"]):
            script.decompose()
        texts = soup.find_all(text=True)
    return '\n'.join(texts)
