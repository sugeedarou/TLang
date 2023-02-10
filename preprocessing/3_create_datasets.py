from csv import DictReader, DictWriter
import html

def get_id_to_lang_dict(path):
    id_to_lang = {}
    with open(path, 'r', encoding='utf-8', newline='') as f_in:
        for line in f_in:
            line_split = line.split('\t')
            id = line_split[1].strip()
            id_to_lang[id] = line_split[0]
    return id_to_lang

def get_langs_70():
    langs_70_str = 'am (Amharic), ar (Arabic), bg (Bulgarian), bn (Bengali), bo (Tibetan), bs (Bosnian), ca (Catalan), ckb (Sorani Kurdish), cs (Czech), cy (Welsh), da (Danish), de (German), dv (Maldivian), el (Greek), en (English), es (Spanish), et (Estonian), eu (Basque), fa (Persian), fi (Finnish), fr (French), gu (Gujarati), he (Hebrew), hi (Hindi), hi-Latn (Latinized Hindi), hr (Croatian), ht (Haitian Creole), hu (Hungarian), hy (Armenian), id (Indonesian), is (Icelandic), it (Italian), ja (Japanese), ka (Georgian), km (Khmer), kn (Kannada), ko (Korean), lo (Lao), lt (Lithuanian), lv (Latvian), ml (Malayalam), mr (Marathi), ms (Malay), my (Burmese), ne (Nepali), nl (Dutch), no (Norwegian), pa (Panjabi), pl (Polish), ps (Pashto), pt (Portuguese), ro (Romanian), ru (Russian), sd (Sindhi), si (Sinhala), sk (Slovak), sl (Slovenian), sr (Serbian), sv (Swedish), ta (Tamil), te (Telugu), th (Thai), tl (Tagalog), tr (Turkish), ug (Uyghur), uk (Ukrainian), ur (Urdu), vi (Vietnamese), zh-CN (Simplified Chinese), zh-TW (Traditional Chinese)'
    langs_70 = set([s.split(' ')[0] for s in langs_70_str.split(', ')])
    return langs_70

def get_shared_langs(id_lang_train, data_train, id_lang_test, data_test):
    def get_langs_in_dataset(id_lang_path, data_path):
        used_langs = set()
        id_to_lang = get_id_to_lang_dict(id_lang_path)
        reader = DictReader(open(data_path, 'r', encoding='utf-8', newline=''), delimiter=',')
        for row in reader:
            id = row['id']
            lang = id_to_lang[id]
            used_langs.add(lang)
        return used_langs
    
    langs_train = get_langs_in_dataset(id_lang_train, data_train)
    langs_test = get_langs_in_dataset(id_lang_test, data_test)
    return langs_train & langs_test

def get_allowed_langs():
    langs_70 = get_langs_70()
    shared_langs = get_shared_langs('data/raw/uniform_precision.tsv', 'data/raw/uniform_precision_data.csv', 'data/raw/recall_oriented.tsv', 'data/raw/recall_data.csv')
    allowed_langs = shared_langs & langs_70
    return allowed_langs

def create_clean_dataset(id_lang_path, in_path, out_path, allowed_langs):
    id_to_lang = get_id_to_lang_dict(id_lang_path)
    with open(in_path, 'r', encoding='utf-8', newline='') as f_in:
        with open(out_path, 'w', encoding='utf-8', newline='') as f_out:
            reader = DictReader(f_in, delimiter=',')
            writer = DictWriter(f_out, fieldnames=['id', 'lang', 'text'])
            writer.writeheader()
            for row in reader:
                id = row['id']
                text = row['text'].replace('\r', '').replace('\n', '') 
                for _ in range(3): # unescape multiple times to fix all entities
                    text = html.unescape(text)
                lang = id_to_lang[id]

                if not lang in allowed_langs or text == '':
                    continue

                writer.writerow({
                    'id': id,
                    'lang': lang,
                    'text': text
                })

allowed_langs = get_allowed_langs()
create_clean_dataset('data/raw/uniform_precision.tsv', 'data/raw/uniform_precision_data.csv', 'data/raw/train_val.csv', allowed_langs)
create_clean_dataset('data/raw/recall_oriented.tsv', 'data/raw/recall_data.csv', 'data/raw/test.csv', allowed_langs)
print('done')