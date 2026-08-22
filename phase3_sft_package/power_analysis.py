import _paths  # noqa: F401  (core/ on sys.path, cwd=root)

import json, math, itertools, numpy as np
from wordle_solver import SolverConfig, load_artifacts, make_solver, play_game
from tree_search import TreeSearchConfig, TreeSearchSolver
b=load_artifacts("artifacts"); cfg=SolverConfig(max_guesses=6,seed=20260817,guess_pool="full")
ans=[json.loads(l)["answer"].lower() for l in open("sft_package/eval/val_answers.jsonl",encoding="utf-8")]
S={}
sv=make_solver("entropy",b.fb,cfg,b.model); sv.reset(); op=sv.opening_guess()
S["entropy"]=[play_game(sv,a,first_guess=op).score for a in ans]
for lbl,op in [("tree_soare","soare"),("tree_salet","salet")]:
    tc=TreeSearchConfig(depth=6,top_k=100,endgame_top_k=60,endgame_threshold=10,
                        opening_guess=op,max_guesses=6)
    sv=TreeSearchSolver(b.fb,cfg,tc)
    S[lbl]=[play_game(sv,a,first_guess=op).score for a in ans]
out={"n":len(ans),"per_game":S,"pairs":{}}
lines=[f"{'pair':<28}{'diff':>9}{'pairedSE':>10}{'t':>7}   verdict"]
for a,bn in itertools.combinations(S,2):
    d=np.array(S[a],float)-np.array(S[bn],float)
    se=d.std(ddof=1)/math.sqrt(len(d)); t=d.mean()/se if se else 0.0
    out["pairs"][f"{a}-{bn}"]={"diff":d.mean(),"se":se,"t":t,
        "significant":bool(abs(t)>1.96)}
    lines.append(f"{a+' - '+bn:<28}{d.mean():>+9.4f}{se:>10.4f}{t:>7.2f}   "
                 f"{'significant' if abs(t)>1.96 else 'INDISTINGUISHABLE'}")
d=np.array(S["entropy"],float)-np.array(S["tree_salet"],float)
need=(1.96*d.std(ddof=1)/0.0432)**2
out["games_needed_for_true_gap"]=need
lines.append(f"\npaired sd = {d.std(ddof=1):.4f}; to resolve the true 0.0432 gap "
             f"needs ~{need:.0f} paired games (you have {len(ans)})")
open("artifacts/power_analysis.json","w").write(json.dumps(out,indent=2))
print("\n".join(lines))
