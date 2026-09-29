"""Números del informe docs/analisis_estabilidad.md (solo-stdlib).

Lee viz/crossrun/crossrun.json (scripts/build_crossrun.py) y, para el cruce con la
validación externa, models/cluster_v2/desmonte_eval*.csv (scripts/eval_desmonte.py).
Imprime todas las tablas del informe, para los dos sets.

Convenciones: las medidas agregadas sobre pares se suman sobre los 30 pares ORDENADOS
(simétrico); el -1 se excluye de los dos lados salvo donde se indica; la cobertura
modal es una propiedad de cada cluster medida sobre sus miembros del pool dinámico,
así que en pooled_subsampled solo cambian los pesos.

Uso, desde la raíz del repo:
    python scripts/build_crossrun.py
    python scripts/analisis_estabilidad.py
"""
import json, csv, glob, collections
from itertools import combinations, permutations
P = json.load(open('viz/crossrun/crossrun.json'))
NP = P['noise_pid']
PL = [p['label'] for p in P['processes']] + ['sin tipificar']
PK = [p['key'] for p in P['processes']]
SH = ['fina/H','fina/P','media/H','media/P','gruesa/H','gruesa/P']
SUF = ['','_parametric','_medium','_medium_parametric','_coarse','_coarse_parametric']

def fl(B, i, j, lvl):
    t = B['pairs'][f"{min(i,j)}-{max(i,j)}"][lvl]
    if i < j: return t
    o = [0]*len(t)
    for x in range(0, len(t), 3): o[x], o[x+1], o[x+2] = t[x+1], t[x], t[x+2]
    return o
def trip(f): return ((f[x], f[x+1], f[x+2]) for x in range(0, len(f), 3))
def m(B, i, j): return B['pairs'][f"{min(i,j)}-{max(i,j)}"]['m']

for sname, B in P['sets'].items():
    N = B['n']; R = B['runs']
    print(f"\n######## SET {sname}  N={N}")
    print("corridas:", [(SH[i], r['k'], round(100*r['noise'],2)) for i, r in enumerate(R)])
    for key, lab in [('acu_proc','acuerdo proceso sin -1'), ('acu_proc11','acuerdo proceso -1 como cat'),
                     ('ari','ARI cluster sin -1'), ('cob','cobertura')]:
        print(f"\n[{lab}]")
        for i in range(6):
            print(SH[i].ljust(9), ' '.join('   —  ' if i==j else f"{m(B,i,j)[key]:.3f} " for j in range(6)))
    print("\n[anidamiento pur fila->col]")
    for i in range(6):
        row=[]
        for j in range(6):
            if i==j: row.append('   —  '); continue
            mm=m(B,i,j); ab=i<j
            row.append(f"{(mm['pur_ab'] if ab else mm['pur_ba']):.4f}")
        print(SH[i].ljust(9), ' '.join(row))
    print("\n[escaleras]")
    for i,j in [(0,2),(0,4),(2,4),(1,3),(1,5),(3,5)]:
        mm=m(B,i,j); print(f"  {SH[i]}->{SH[j]}: pur {mm['pur_ab']:.4f}  parten {mm['spl_ab']}/{mm['k_ab']}  cob {mm['cob']:.3f}")
    # clusters 100% absorbidos por -1 destino
    for i,j in [(0,2),(0,4),(2,4)]:
        f=fl(B,i,j,'cluster'); src=set(); keep=set()
        for a,b,n in trip(f):
            if a==-1: continue
            src.add(a)
            if b!=-1: keep.add(a)
        print(f"  {SH[i]}->{SH[j]}: {len(src-keep)} de {len(src)} clusters quedan enteros en el -1 destino")
    print("\n[misma granularidad, entre familias]")
    for h,p in [(0,1),(2,3),(4,5)]:
        mm=m(B,h,p); print(f"  {SH[h]}<->{SH[p]}: ARI {mm['ari']:.3f}  H->P {mm['pur_ab']:.4f} ({mm['spl_ab']}/{mm['k_ab']})  P->H {mm['pur_ba']:.4f} ({mm['spl_ba']}/{mm['k_ba']})  cob {mm['cob']:.3f}")
    # robustez por proceso sobre 30 pares ordenados, sin -1
    stay=collections.Counter(); tot=collections.Counter()
    for i,j in permutations(range(6),2):
        for a,b,n in trip(fl(B,i,j,'proceso')):
            if a==NP or b==NP: continue
            tot[a]+=n; stay[a]+= n if a==b else 0
    print("\n[robustez por proceso, 30 pares ordenados, sin -1]")
    for p_ in sorted(tot, key=lambda x:-tot[x]): print(f"  {PL[p_]:45} {tot[p_]:>9}  {stay[p_]/tot[p_]:.3f}")
    # urbanización con/sin gruesa/P
    U=PK.index('urbanizacion')
    c=[0,0]; s=[0,0]
    for i,j in permutations(range(6),2):
        for a,b,n in trip(fl(B,i,j,'proceso')):
            if a!=U or b==NP: continue
            t=c if 5 in (i,j) else s; t[1]+=n; t[0]+= n if b==U else 0
    print(f"  URB con gruesa/P {c[0]}/{c[1]}   sin gruesa/P {s[0]}/{s[1]} = {s[0]/s[1]:.3f}")
    print("  clusters de urbanizacion por corrida:", [(SH[i], sum(1 for nd in r['nodes'] if nd['id']!=-1 and nd['p']==U), sum(nd['n'] for nd in r['nodes'] if nd['id']!=-1 and nd['p']==U)) for i,r in enumerate(R)])
    conf=collections.Counter()
    for i,j in permutations(range(6),2):
        for a,b,n in trip(fl(B,i,j,'proceso')):
            if a!=NP and b!=NP and a!=b: conf[tuple(sorted((a,b)))]+=n
    T=sum(conf.values())
    print("\n[confusiones dominantes: pares NO ordenados de procesos, share de la masa fuera de la diagonal]")
    for (a,b),n in conf.most_common(8): print(f"  {PL[a]} <-> {PL[b]}: {n//2} por sentido ({n/T:.3f})")
    # destino del -1 media/H -> media/P
    f=fl(B,2,3,'proceso'); out=[(b,n) for a,b,n in trip(f) if a==NP]; t=sum(n for _,n in out)
    print(f"\n[-1 de media/H ({t}, {t/N:.3f}) bajo media/P]")
    for b,n in sorted(out,key=lambda x:-x[1])[:6]: print(f"  {PL[b]}: {n} ({n/t:.3f})")
    # cobertura modal
    print("\n[cobertura modal ponderada]")
    for i,r in enumerate(R):
        nd=[n for n in r['nodes'] if n['id']!=-1 and 'cov' in n]; tt=sum(n['n'] for n in nd)
        print(f"  {SH[i]} k={len(nd)} pond {sum(n['cov']*n['n'] for n in nd)/tt:.3f}  masa cov<0.5 {sum(n['n'] for n in nd if n['cov']<0.5)/tt:.3f}")

# externo
cov={}
B=P['sets']['dynamic']
for i,r in enumerate(B['runs']):
    nd=[n for n in r['nodes'] if n['id']!=-1 and 'cov' in n]; tt=sum(n['n'] for n in nd)
    cov[SUF[i]]=sum(n['cov']*n['n'] for n in nd)/tt
rows=[]
for i,s in enumerate(SUF):
    for r in csv.DictReader(open(f'models/cluster_v2/desmonte_eval{s}.csv')):
        if r['zone']=='chaco_santiago_frontier' and r['nivel'] not in ('r0','r1','permutacion'):
            rows.append((SH[i], cov[s], float(r['mcc_neg1_neg']), float(r['mcc_neg1_neg_ci_lo']), float(r['mcc_neg1_neg_ci_hi'])))
print("\n[externo chaco]")
for x in rows: print(" ", x)
def rk(v): s=sorted(range(len(v)),key=lambda i:v[i]); r=[0]*len(v); [r.__setitem__(i,p) for p,i in enumerate(s)]; return r
a=rk([x[1] for x in rows]); b=rk([x[2] for x in rows]); n=len(rows)
print("  spearman", 1-6*sum((a[i]-b[i])**2 for i in range(n))/(n*(n*n-1)))
