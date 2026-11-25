import json,re,collections,itertools
fs={}
for l in open('../data_c2/familles.jsonl',encoding='utf-8'):
    f=json.loads(l)
    if f['split']=='train': fs[f['famille']]=f
pil=[json.loads(l) for l in open('../data_c2/pilote_train.jsonl',encoding='utf-8')]
flags=collections.defaultdict(list)
def ed(a,b):
    a,b=a.lower(),b.lower()
    d=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        p=d[:];d[0]=i
        for j,cb in enumerate(b,1):
            d[j]=min(p[j]+1,d[j-1]+1,p[j-1]+(ca!=cb))
    return d[-1]
voc=json.load(open('../../../tokenizer_fr32k.json',encoding='utf-8'))['model']['vocab']
voc=set(voc) if isinstance(voc,dict) else {v[0] for v in voc}
def realword(n):
    n=n.lower()
    return any(k in voc for k in (n,'Ġ'+n,'▁'+n,' '+n))
kind_name={'capitale','fondateur'}
for fid,f in fs.items():
    ents=[f['entite']]+[d['entite'] for d in f['distracteurs']]
    vals=[f['valeur'],f['valeur_contrefactuelle']]+[d['valeur'] for d in f['distracteurs']]
    if f['valeur']==f['valeur_contrefactuelle']: flags[fid].append('valeur=contrefactuelle')
    if len(set(ents))<len(ents): flags[fid].append('entites dupliquees')
    for a,b in itertools.combinations(ents,2):
        if ed(a,b)<=2 or a.lower() in b.lower() or b.lower() in a.lower(): flags[fid].append(f'entites proches {a}/{b}')
    if f['gabarit'] in kind_name:
        names=[f['valeur'],f['valeur_contrefactuelle']]+[d['valeur'] for d in f['distracteurs']]
        for a,b in itertools.combinations(names,2):
            if ed(a,b)<=2: flags[fid].append(f'valeurs proches {a}/{b}')
        for e in ents:
            for v in names:
                if e==v: flags[fid].append(f'entite=valeur {e}')
        for n in set(ents+names):
            if realword(n): flags[fid].append(f'mot reel? {n}')
    elif f['gabarit']=='metier':
        pass
    else:
        for n in set(ents):
            if realword(n): flags[fid].append(f'mot reel? {n}')
    if f['gabarit'] in ('habitants','annee'):
        if f['valeur'] in [d['valeur'] for d in f['distracteurs']] or f['valeur_contrefactuelle'] in [d['valeur'] for d in f['distracteurs']]: flags[fid].append('valeur egale a un distracteur')
        if abs(int(f['valeur'])-int(f['valeur_contrefactuelle']))<=3: flags[fid].append('valeurs trop proches')
    if f['gabarit']=='couleur' or f['gabarit']=='metier':
        if f['valeur']==f['valeur_contrefactuelle']: pass
# exemples
for r in pil:
    fid=r['famille'];f=fs[fid]
    u=r['messages'][0]['content'];a=r['messages'][1]['content']
    docs=re.findall(r'^\[\d+\] (.*)$',u,re.M)
    with_e=[d for d in docs if f['entite'] in d]
    q=u.split('Question :')[1]
    if r['variante']=='sans_preuve':
        if with_e: flags[fid].append('sans_preuve: entite presente dans un doc')
        if f['entite'] not in q: flags[fid].append('question sans entite')
    else:
        if len(with_e)!=1: flags[fid].append(f"{r['variante']}: {len(with_e)} docs avec l'entite")
        val=f['valeur'] if r['variante']=='gold' else f['valeur_contrefactuelle']
        if val not in a: flags[fid].append('reponse sans valeur')
        if f['valeur'] in a and r['variante']=='contrefactuel': flags[fid].append('cf: reponse contient valeur gold')
        if f['valeur_contrefactuelle'] in a and r['variante']=='gold': flags[fid].append('gold: reponse contient valeur cf')
        others=[d for d in docs if f['entite'] not in d]
        if f['gabarit'] not in ('couleur','metier'):
            if any(val in d for d in others): flags[fid].append(f"{r['variante']}: valeur de la preuve repetee dans un distracteur")
    if len(docs)!=4: flags[fid].append('nb docs != 4')
print(len(flags))
c=collections.Counter(x.split(' ')[0]+' '+x.split(' ')[1] if ' ' in x else x for v in flags.values() for x in set(v))
for k,v in c.most_common(): print(v,k)
json.dump({k:sorted(set(v)) for k,v in flags.items()},open('flags_a2.json','w'),ensure_ascii=False,indent=1)
