

def merge_simliar_languages(id_lang_path, in_path, out_path, allowed_langs):
    with open(in_path, 'r', encoding='utf-8', newline='') as f_in:
        with open(out_path, 'w', encoding='utf-8', newline='') as f_out:
            reader = DictReader(f_in, delimiter=',')
            writer = DictWriter(f_out, fieldnames=['id', 'lang', 'text'])
            writer.writeheader()
            for row in reader:
                lang = row['lang']
                text = row['text'].replace('\r', '').replace('\n', '')

                if lang in ['sr', 'hr', 'bs']: # merge serbian, croatian, bosnian
                    lang = 'scb'
                elif lang in ['no', 'da', 'sv']: # merge norwegian, danish, swedish
                    lang = 'nds'
                elif lang in ['hi-Latn', 'ur']: # merge (latinized) hindu, urdu
                    lang = 'hl'

                writer.writerow({
                    'id': id,
                    'lang': lang,
                    'text': text
                })