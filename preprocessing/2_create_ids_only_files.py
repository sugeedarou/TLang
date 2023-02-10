def write_ids(path_in, path_out):
    ids = []
    with open(path_in, 'r', encoding='utf-8', newline='') as f_in:
        for line in f_in:
            line_split = line.split('\t')
            id = line_split[1]
            if 'E+' in line:
                continue
            ids.append(id)
    with open(path_out, 'w', encoding="utf-8") as f_out:
        f_out.writelines(ids)

write_ids('data/input/uniform_precision.tsv', 'data/input/uniform_precision_ids.tsv')
write_ids('data/input/recall_oriented.tsv', 'data/input/recall_oriented_ids.tsv')
print('done')