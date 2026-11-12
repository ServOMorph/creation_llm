import json,re,sys
from pathlib import Path
sys.path.insert(0,'.')
from bench_ollama import chat, post, check
T=[("politesse","Reformule cette phrase de manière plus polie"),("passe_compose","Mets cette phrase au passé composé"),
("futur","Mets cette phrase au futur simple"),("pluriel","Mets cette phrase au pluriel"),("question","Transforme cette phrase en question fermée"),
("negation","Mets cette phrase à la forme négative"),("simplifier","Simplifie cette phrase pour un enfant de 7 ans"),("paraphrase","Dis la même chose avec d'autres mots"),
("present","Mets cette phrase au présent de l'indicatif"),("formel","Reformule cette phrase dans un registre plus soutenu")]
S=[re.sub(r'\[[^\]]*\]','',s).replace('»','').strip() for s in json.loads(Path('corpus_sents.json').read_text(encoding='utf-8'))]
items=[(T[(2*i+k)%len(T)],s) for i,s in enumerate(S) for k in (0,1)]
for model in sys.argv[1:]:
    rows=[]
    for (kind,instr),src in items:
        p=(f'{instr} : « {src} »\nSi la transformation est impossible ou absurde pour cette phrase, réponds {{"sortie": null}}. '
           'Sinon réponds au format {"sortie": "..."}, avec uniquement la phrase transformée.')
        raw,sp,_=chat(model,p)
        try: o=json.loads(raw).get('sortie')
        except Exception: o='INVALIDE '+raw
        rows.append({'type':kind,'entree':src,'sortie':o,'flags':check(kind,src,o or '') if o else ['null']})
    Path(f'bench_results/b2_{model.replace(":","_")}.json').write_text(json.dumps(rows,ensure_ascii=False,indent=1),encoding='utf-8')
    print(model, 'null',sum(r['sortie'] is None for r in rows),'flags',sum(1 for r in rows if r['flags'] and r['sortie']),flush=True)
    post("generate", {"model": model, "keep_alive": 0})
