from create_datasets import create_dataset
from eliminate_underrepresented import eliminate_underrepresented
from create_characters_list import create_characters_list
from optional.chars_to_lowercase import chars_to_lowercase
from optional.merge_similiar_languages import merge_simliar_languages
from optional.reduce_distinct_characters import reduce_distinct_characters
from optional.reduce_repetitive_characters import reduce_repetitive_characters
from optional.reduce_zn_ko_jp import reduce_zn_ko_jp
from optional.remove_emojis import remove_emojis
from optional.remove_hashtags import remove_hashtags
from optional.remove_urls import remove_urls
from optional.remove_user_refs import remove_user_refs
from split_train_val import split_train_val
from encode_datasets import load_characters_and_languages, encode_dataset

MIN_SAMPLES_COUNT = 100 # languages with less than MIN_SAMPLES count in training set will be removed
VAL_PERCENTAGE = 0.1 # VAL_PERCENTAGE% of of train_val data used for validation
OPTIONAL_PREPROCESSING = [
    chars_to_lowercase, # letters to lowercase
    merge_simliar_languages, # merge languages that can be considered dialects of the same language
    reduce_distinct_characters, # reduce characters that are distinct to a single language into a single character
    reduce_zn_ko_jp, # map all chinese, korean and japanese -exclusive characters to a single character respectively
    remove_emojis,
    remove_hashtags,
    remove_urls,
    remove_user_refs,
    reduce_repetitive_characters, # reduce multiple repeats of a character in a string to only 3 repeats
]

print('creating datasets')
create_dataset('data/input/uniform_precision.tsv', 'data/input/uniform_precision_data.csv', 'data/input/train_val.tsv')
create_dataset('data/input/recall_oriented.tsv', 'data/input/recall_data.csv', 'data/input/test.tsv')

print('eliminating underrepresented languages')
eliminate_underrepresented(MIN_SAMPLES_COUNT)

print('creating characters list')
create_characters_list()

for op_fun in OPTIONAL_PREPROCESSING:
    print('applying ' + op_fun.__name__)
    op_fun()
    
print('splitting train / validation')
split_train_val(VAL_PERCENTAGE)

print('encoding datasets')
load_characters_and_languages()
encode_dataset('data/processed/train.tsv', 'data/train.tsv')
encode_dataset('data/processed/val.tsv', 'data/val.tsv')
encode_dataset('data/processed/test.tsv', 'data/test.tsv')

print('done')