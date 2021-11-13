from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET

def remove_code_from_post(soup):
    for script in soup(['code']):
        script.decompose()

# ref: https://meta.stackexchange.com/questions/2677/database-schema-documentation-for-the-public-data-dump-and-sede
# PostTypeId (listed in the PostTypes table)
# 1 = Question
# 2 = Answer
# 3 = Orphaned tag wiki
# 4 = Tag wiki excerpt
# 5 = Tag wiki
# 6 = Moderator nomination
# 7 = "Wiki placeholder" (seems to only be the election description)
# 8 = Privilege wiki

tree = ET.parse('Posts.xml')

root = tree.getroot()

question_to_accepted_answer = dict()
for c in root:
    if c.tag == 'row':
        post_id = c.attrib['Id']
        post_type_id = c.attrib['PostTypeId']
        if post_type_id == "1":
            try:
                accepted_answer_id = c.attrib['AcceptedAnswerId']
                question_to_accepted_answer[post_id] = accepted_answer_id
            except KeyError:
                pass

post_data = dict()  # str(accepted_answer_id) -> str(text)
for c in root:
    if c.tag == 'row':
        post_id = c.attrib['Id']
        post_type_id = c.attrib['PostTypeId']
        if post_type_id == '1' and post_id in question_to_accepted_answer:
            try:
                tags_str = c.attrib['Tags']
                assert tags_str[0] == '<' and tags_str[-1] == '>'
                tags = tags_str[1:-1].split('><')
                tags_lines = ["tags: %s\n" % ', '.join(tags)] if tags else []
            except KeyError:
                tags_lines = []
            text = c.attrib['Body']
            soup = BeautifulSoup(text, 'html.parser')
            remove_code_from_post(soup)
            texts = soup.find_all(text=True)

            if texts and not texts[-1].endswith('\n'):
                texts[-1] = texts[-1] + '\n'

            post_data[question_to_accepted_answer[post_id]] = texts + tags_lines
        elif post_type_id == '2' and post_id in post_data:
            text = c.attrib['Body']
            soup = BeautifulSoup(text, 'html.parser')
            remove_code_from_post(soup)
            texts = soup.find_all(text=True)

            pdi = post_data[post_id]
            pdi.append("---\n")
            pdi.extend(texts)
            # print("---\npost_id=%s, texts=%s" % (post_id, ''.join(pdi)))
            with open("post-answer-id-%s.txt" % post_id, 'w') as outp:
                print(''.join(pdi), file=outp)
            
            del post_data[post_id]
