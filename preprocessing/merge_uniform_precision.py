f1_in = open('../data/raw/precision_oriented.tsv', 'r', encoding='utf-8', newline='')
f2_in = open('../data/raw/uniformly_sampled.tsv', 'r', encoding='utf-8', newline='')
test_in = open('../data/raw/recall_oriented.tsv', 'r', encoding='utf-8', newline='')
f_out = open('../data/raw/uniform_precision.tsv', 'w', encoding='utf-8', newline='')

test_ids = [l[1].strip() for l in test_in]
newlines1 = set([l for l in f1_in if not l.split('\t')[1].strip() in test_ids])
newlines2 = set([l for l in f2_in if not l.split('\t')[1].strip() in test_ids])
newlines = newlines1.union(newlines2)
f_out.writelines(newlines)

f1_in.close()
f2_in.close()
test_in.close()
f_out.close()