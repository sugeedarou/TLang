


used_langs = ['is', 'uk', 'fa', 'ka', 'ru', 'es', 'pa', 'nl', 'hy', 'tr', 'zh-CN', 'et', 'bs', 'it', 'vi', 'lo', 'mr', 'no', 'eu', 'my', 'ur', 'el', 'tl', 'kn', 'en', 'ko', 'lv', 'sr', 'sv', 'ml', 'fi', 'ps', 'th', 'pt', 'si', 'id', 'zh-TW', 'ht', 'bn', 'ta', 'gu', 'cs', 'lt', 'ca', 'sl', 'hi', 'ne', 'ar', 'hu', 'ro', 'dv', 'sd', 'de', 'ja', 'km', 'pl', 'hr', 'te', 'cy', 'ckb', 'he', 'fr', 'da', 'sk', 'bg', 'am', 'hi-Latn']

langs = {}
with open('../data/train_val.csv', 'r', encoding='utf-8', newline='') as f_in:
    next(f_in)
    for line in f_in:
        line_split = line.split(',')
        lang = used_langs[int(line_split[1])]
        if not lang in langs:
            langs[lang] = 1
        else:
            langs[lang] += 1

with open('../data/raw/lang_distribution.tsv', 'w', encoding='utf-8', newline='') as f_out:
    for lang in langs:
        f_out.write(f'{lang}\t{langs[lang]}\n')