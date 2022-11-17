from csv import reader

s = """am (Amharic), ar (Arabic), bg (Bulgarian), bn (Bengali), bo (Tibetan), bs (Bosnian), ca (Catalan), ckb (Sorani Kurdish), cs (Czech), cy (Welsh), da (Danish), de (German), dv (Maldivian), el (Greek), en (English), es (Spanish), et (Estonian), eu (Basque), fa (Persian), fi (Finnish), fr (French), gu (Gujarati), he (Hebrew), hi (Hindi), hi-Latn (Latinized Hindi), hr (Croatian), ht (Haitian Creole), hu (Hungarian), hy (Armenian), id (Indonesian), is (Icelandic), it (Italian), ja (Japanese), ka (Georgian), km (Khmer), kn (Kannada), ko (Korean), lo (Lao), lt (Lithuanian), lv (Latvian), ml (Malayalam), mr (Marathi), ms (Malay), my (Burmese), ne (Nepali), nl (Dutch), no (Norwegian), pa (Panjabi), pl (Polish), ps (Pashto), pt (Portuguese), ro (Romanian), ru (Russian), sd (Sindhi), si (Sinhala), sk (Slovak), sl (Slovenian), sr (Serbian), sv (Swedish), ta (Tamil), te (Telugu), th (Thai), tl (Tagalog), tr (Turkish), ug (Uyghur), uk (Ukrainian), ur (Urdu), vi (Vietnamese), zh-CN (Simplified Chinese), zh-TW (Traditional Chinese)"""

langs = []
for lang in s.split(', '):
    short = lang.split(' ')[0]
    langs.append(short)

out = open('data/sample_clean.tsv', 'w')
with open('data/uniformly_sampled.tsv', 'r') as f:
     reader = reader(f, delimiter='\t')
     for line in reader:
         lang = line[0]
         if lang in langs:
            out.write(' '.join(line) + '\n')
out.close()
