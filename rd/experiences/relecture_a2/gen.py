import json, pathlib
here = pathlib.Path(__file__).resolve().parent
exp = here.parent
fam = {}
for l in open(exp / 'data_c2' / 'familles.jsonl', encoding='utf-8'):
    f = json.loads(l)
    if f['split'] == 'train':
        fam[f['famille']] = f
ex = {}
for l in open(exp / 'data_c2' / 'pilote_train.jsonl', encoding='utf-8'):
    r = json.loads(l)
    ex.setdefault(r['famille'], {}).setdefault(r['variante'], r)
flags = json.load(open(here / 'flags_a2.json', encoding='utf-8'))
cases = []
for fid in sorted(flags):
    f = fam[fid]
    vs = []
    for v in ('gold', 'sans_preuve', 'contrefactuel'):
        r = ex[fid][v]
        vs.append({'v': v, 'u': r['messages'][0]['content'], 'm': r['messages'][1]['content'], 'p': r['preuve'] or ''})
    cases.append({'id': fid, 'gab': f['gabarit'], 'ent': f['entite'], 'val': f['valeur'], 'cf': f['valeur_contrefactuelle'], 'vs': vs, 'flag': ' ; '.join(flags[fid])})
tpl = (here / 'tpl.html').read_text(encoding='utf-8')
html = tpl.replace('__DATA__', json.dumps(cases, ensure_ascii=False))
(here / 'relecture_a2.html').write_text(html, encoding='utf-8')
print(len(cases), len(html))
