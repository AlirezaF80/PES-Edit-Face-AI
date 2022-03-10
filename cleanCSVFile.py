import unidecode

with open('data.csv', 'r', encoding='UTF-8') as file:
    text = file.read()
    text = text.replace(';', ',')
unaccented_text = unidecode.unidecode(text)
with open('data-cleaned.csv', 'w') as file:
    file.write(unaccented_text)
