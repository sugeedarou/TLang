def count_samples_per_language(f):
    samples_per_language = {}
    next(f)
    for line in f:
        line_split = line.split(',')
        lang = line_split[1]
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

    newlangs.sort()
    print(f'deleted {deleted_langs_count} languages')
    print(newlangs)
    print(len(newlangs))
    return newlangs

def write_output_file(input_file, output_file, newlangs):
    next(input_file)
    output_file.write('id,lang,text\n')
    for line in input_file:
        line_split = line.split(',')
        lang = line_split[1]
        if lang in newlangs:
            output_file.write(line)


MIN_SAMPLES_COUNT = 100

train_val_in = open('data/raw/train_val.csv', 'r', encoding='utf-8', newline='')
train_val_out = open('data/raw/train_val_eliminated.csv', 'w', encoding='utf-8', newline='')
test_in = open('data/raw/test.csv', 'r', encoding='utf-8', newline='')
test_out = open('data/raw/test_eliminated.csv', 'w', encoding='utf-8', newline='')

print('counting samples')
samples_per_language = count_samples_per_language(train_val_in)
train_val_in.seek(0)
newlangs = delete_langs_with_less_than_x(samples_per_language, MIN_SAMPLES_COUNT)

print('writing output')
write_output_file(train_val_in, train_val_out, newlangs)
write_output_file(test_in, test_out, newlangs)

train_val_in.close()
train_val_out.close()
test_in.close()
test_out.close()
print('done')