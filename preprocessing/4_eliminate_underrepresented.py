from csv import DictReader, DictWriter

def get_shared_langs(f_train, f_test):
    def get_langs_in_dataset(f):
        used_langs = set()
        reader = DictReader(f, delimiter='\t')
        for row in reader:
            used_langs.add(row['lang'])
        return used_langs
    
    langs_train = get_langs_in_dataset(f_train)
    langs_test = get_langs_in_dataset(f_test)
    return langs_train & langs_test

def count_samples_per_language(f):
    samples_per_language = {}
    reader = DictReader(f, delimiter='\t')
    for row in reader:
        lang = row['lang']
        if not lang in samples_per_language:
            samples_per_language[lang] = 1
        else:
            samples_per_language[lang] += 1
    return samples_per_language

def delete_langs_with_less_than_x(samples_per_language, x):
    newlangs = []
    deleted_langs_count = 0
    for lang in samples_per_language:
        count = samples_per_language[lang]
        if count >= MIN_SAMPLES_COUNT:
            newlangs.append(lang)
        else:
            print(f'removed {lang} which has {count} < {MIN_SAMPLES_COUNT} samples')
            deleted_langs_count += 1

    return set(newlangs)

def delete_not_lang_70(langs):
    langs_70_str = 'am (Amharic), ar (Arabic), bg (Bulgarian), bn (Bengali), bo (Tibetan), bs (Bosnian), ca (Catalan), ckb (Sorani Kurdish), cs (Czech), cy (Welsh), da (Danish), de (German), dv (Maldivian), el (Greek), en (English), es (Spanish), et (Estonian), eu (Basque), fa (Persian), fi (Finnish), fr (French), gu (Gujarati), he (Hebrew), hi (Hindi), hi-Latn (Latinized Hindi), hr (Croatian), ht (Haitian Creole), hu (Hungarian), hy (Armenian), id (Indonesian), is (Icelandic), it (Italian), ja (Japanese), ka (Georgian), km (Khmer), kn (Kannada), ko (Korean), lo (Lao), lt (Lithuanian), lv (Latvian), ml (Malayalam), mr (Marathi), ms (Malay), my (Burmese), ne (Nepali), nl (Dutch), no (Norwegian), pa (Panjabi), pl (Polish), ps (Pashto), pt (Portuguese), ro (Romanian), ru (Russian), sd (Sindhi), si (Sinhala), sk (Slovak), sl (Slovenian), sr (Serbian), sv (Swedish), ta (Tamil), te (Telugu), th (Thai), tl (Tagalog), tr (Turkish), ug (Uyghur), uk (Ukrainian), ur (Urdu), vi (Vietnamese), zh-CN (Simplified Chinese), zh-TW (Traditional Chinese)'
    langs_70 = set([s.split(' ')[0] for s in langs_70_str.split(', ')])
    newlangs = langs & langs_70
    return newlangs

def delete_langs_not_in_train_and_test(langs, train_path, test_path):
    shared_langs = get_shared_langs(train_path, test_path)
    return langs & shared_langs

def write_output_file(f_in, f_out, newlangs):
    f_in.seek(0)
    reader = DictReader(f_in, delimiter='\t')
    writer = DictWriter(f_out, fieldnames=['id', 'lang', 'text'], delimiter='\t')
    writer.writeheader()
    for row in reader:
        lang = row['lang']
        if lang in newlangs:
            writer.writerow(row)

    with open('data/langs.tsv', 'w', encoding='utf-8', newline='') as f:
        f.write('\n'.join(newlangs)+'\n')


MIN_SAMPLES_COUNT = 100

train_val_in = open('data/input/train_val.tsv', 'r', encoding='utf-8', newline='')
train_val_out = open('data/processed/train_val.tsv', 'w', encoding='utf-8', newline='')
test_in = open('data/input/test.tsv', 'r', encoding='utf-8', newline='')
test_out = open('data/processed/test.tsv', 'w', encoding='utf-8', newline='')

print('counting samples')
samples_per_language = count_samples_per_language(train_val_in)
train_val_in.seek(0)
newlangs = delete_langs_with_less_than_x(samples_per_language, MIN_SAMPLES_COUNT)
newlangs = delete_not_lang_70(newlangs)
newlangs = delete_langs_not_in_train_and_test(newlangs, train_val_in, test_in)
newlangs = list(newlangs)
newlangs.sort()

print('writing output')
write_output_file(train_val_in, train_val_out, newlangs)
write_output_file(test_in, test_out, newlangs)

train_val_in.close()
train_val_out.close()
test_in.close()
test_out.close()
print('done')