import numpy as np
import pandas as pd
from scipy import stats

GINF_CSV = "ginf.csv"
COUNTRIES = ["england", "france", "germany", "italy", "spain"]
SEASONS = [2012, 2013, 2014, 2015, 2016, 2017]
N_SIMS = 10000
rng = np.random.default_rng(42)

g = pd.read_csv(GINF_CSV)
sub = g[(g.country.isin(COUNTRIES)) & (g.season.isin(SEASONS))].copy()
N = len(sub)
observed_21 = int((((sub.fthg==2)&(sub.ftag==1)) | ((sub.fthg==1)&(sub.ftag==2))).sum())

def prob_2_1(h, a):
    return (stats.poisson.pmf(2,h)*stats.poisson.pmf(1,a)
          + stats.poisson.pmf(1,h)*stats.poisson.pmf(2,a))

# SIMPLE MODEL
h_s, a_s = sub.fthg.mean(), sub.ftag.mean()
p_s = prob_2_1(h_s, a_s); exp_s = p_s * N
sim_s = rng.binomial(N, p_s, size=N_SIMS)
p_boot_s = (np.abs(sim_s - exp_s) >= abs(observed_21 - exp_s)).mean()

# TEAM-STRENGTH MODEL
overall = pd.concat([sub.fthg, sub.ftag]).mean()
scored, conceded, games = {}, {}, {}
for _, r in sub.iterrows():
    for t, gs, gc in [(r["ht"], r["fthg"], r["ftag"]), (r["at"], r["ftag"], r["fthg"])]:
        scored[t]=scored.get(t,0)+gs; conceded[t]=conceded.get(t,0)+gc; games[t]=games.get(t,0)+1
p_match=[]
for _, r in sub.iterrows():
    atk_h=(scored[r["ht"]]/games[r["ht"]])/overall; def_a=(conceded[r["at"]]/games[r["at"]])/overall
    atk_a=(scored[r["at"]]/games[r["at"]])/overall; def_h=(conceded[r["ht"]]/games[r["ht"]])/overall
    p_match.append(prob_2_1(h_s*atk_h*def_a, a_s*atk_a*def_h))
p_match=np.array(p_match); exp_t=p_match.sum()
sim_t=np.array([rng.binomial(1, p_match).sum() for _ in range(N_SIMS)])
p_boot_t=(np.abs(sim_t - exp_t) >= abs(observed_21 - exp_t)).mean()

print(f"Observed 2-1: {observed_21} (N={N})")
print(f"SIMPLE: expected {exp_s:.1f}, bootstrap p = {p_boot_s:.4f}")
print(f"TEAM-STRENGTH: expected {exp_t:.1f}, bootstrap p = {p_boot_t:.4f}")
