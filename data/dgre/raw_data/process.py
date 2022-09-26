import json
import codecs

data_path = "train.json"

with codecs.open(data_path, 'r', encoding="utf-8") as fp:
    data = fp.readlines()
output_file = open("../mid_data/doccnao_train.json', 'w', encoding='utf-8')

for did, d in enumerate(data):
    d = eval(d)
    tmp = {}
    tmp["id"] = d['ID']
    tmp['text'] = d['text']
    tmp['relations'] = []
    tmp['entities'] = []
    ent_id = 0
    for rel_id,spo in enumerate(d['spo_list']):
        rel_tmp = {}
        ent_tmp = {}
        rel_tmp['id'] = rel_id
        ent_tmp['id'] = ent_id
        h = spo['h']
        ent_tmp['start_offset'] = h['pos'][0]
        ent_tmp['end_offset'] = h['pos'][1]
        ent_tmp['label'] = "主体"
        if ent_tmp not in tmp['entities']:
            tmp['entities'].append(ent_tmp)
            from_id = ent_id
            ent_id += 1
        else:
            ind = tmp['entities'].index(ent_tmp)
            from_id = tmp['entities'][ind]['id']
            ent_id = len(tmp['entities']) + 1

        t = spo['t']
        ent_tmp = {}
        ent_tmp['id'] = ent_id
        ent_tmp['start_offset'] = t['pos'][0]
        ent_tmp['end_offset'] = t['pos'][1]
        ent_tmp['label'] = "客体"
        if ent_tmp not in tmp['entities']:
            tmp['entities'].append(ent_tmp)
            to_id = ent_id
            ent_id += 1
        else:
            ind = tmp['entities'].index(ent_tmp)
            to_id = tmp['entities'][ind]['id']
            ent_id = len(tmp['entities']) + 1

        rel_tmp['from_id'] = from_id
        rel_tmp['to_id'] = to_id
        rel_tmp['type'] = spo['relation']

        tmp['relations'].append(rel_tmp)
    output_file.write(json.dumps(tmp, ensure_ascii=False) + "\n")