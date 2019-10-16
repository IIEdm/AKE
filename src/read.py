def read_dict(dir):
    ent2id={}
    id2ent={}
    with open(dir, 'r', encoding='utf-8') as f:
        for i in f:
            line = i.rstrip('\n').split()
            ent = line[0]
            id = line[1]
            ent2id[ent] = id
            id2ent[id] = ent
    return ent2id, id2ent

def read_trainfile(dir):
    head_list = []
    rel_list = []
    tail_list = []
    with open(dir, 'r', encoding='utf-8') as f:
        for i in f:
            line = i.rstrip('\n').split()
            if len(line) == 3:
                head = int(line[0])
                rel = int(line[1])
                tail = int(line[2])
                head_list.append(head)
                rel_list.append(rel)
                tail_list.append(tail)
            else:
                continue
        return head_list, rel_list, tail_list

def read_crossgraph(dir):
    lang1_entities = []
    lang2_entities = []
    with open(dir, 'r', encoding='utf-8') as f:
        for i in f:
            line = i.rstrip('\n').split()
            lang1_id = int(line[0])
            lang2_id = int(line[1])
            lang1_entities.append(lang1_id)
            lang2_entities.append(lang2_id)
    return lang1_entities, lang2_entities