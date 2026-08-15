"""
analysis.py  --  2-1 Football Scoreline Study (v2, corrected for data coverage)
===============================================================================
IMPORTANT DATA NOTE
-------------------
The Football Events dataset only contains goal-by-goal timing (needed to work
out the ORDER of goals) for the later seasons. Final scores exist for all six
seasons, but goal minutes exist only for 2013/14 (partial) through 2016/17.

So this script splits the work correctly:
  * RQ1 (how OFTEN 2-1 happens) uses ALL six seasons, from final scores.
  * RQ2 (the ORDER of goals) uses only the seasons that have goal timing.

HOW TO RUN (in Google Colab):
  1. Upload events.csv, ginf.csv, and this file.
  2. Set USE_SIMULATED_DATA = False below (already set).
  3. Run:  !python analysis.py
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt

# ================================ CONFIG ================================
USE_SIMULATED_DATA = False
EVENTS_CSV = "events.csv"
GINF_CSV   = "ginf.csv"
COUNTRIES = ["england","france","germany","italy","spain"]  # all five leagues
SEASONS = [2012, 2013, 2014, 2015, 2016, 2017]
FEATURED_SEASON = 2013          # 2012/13 -- the record season (for RQ1 only)
ALPHA   = 0.05
FIG_DIR = "figures"

# season label (ending year) -> readable name
SEASON_NAMES = {2012:"2011/12", 2013:"2012/13", 2014:"2013/14",
                2015:"2014/15", 2016:"2015/16", 2017:"2016/17"}

mpl.rcParams.update({"font.family":"DejaVu Sans","font.size":12,
    "axes.spines.top":False,"axes.spines.right":False,
    "axes.grid":True,"grid.alpha":0.25,"figure.dpi":120})
BLUE, ORANGE, GREY, RED = "#0072B2", "#E69F00", "#999999", "#D55E00"


# ============================ 1. LOAD ============================
def load_data():
    ginf = pd.read_csv(GINF_CSV)
    ginf = ginf[(ginf["country"].isin(COUNTRIES)) & (ginf["season"].isin(SEASONS))].copy()
    events = pd.read_csv(EVENTS_CSV)
    return events, ginf


# ============ 2. RQ1 + scoreline dist: from FINAL SCORES (all seasons) ============
def prob_2_1(h, a):
    return (stats.poisson.pmf(2,h)*stats.poisson.pmf(1,a)
          + stats.poisson.pmf(1,h)*stats.poisson.pmf(2,a))

def is_2_1(row):
    return {row["fthg"], row["ftag"]} == {2, 1}

def run_rq1(ginf):
    N = len(ginf)
    obs = int(ginf.apply(is_2_1, axis=1).sum())
    h, a = ginf["fthg"].mean(), ginf["ftag"].mean()

    # simple Poisson (one league-wide rate)
    p_simple = prob_2_1(h, a)
    # team-strength Poisson (attack/defence per team; in the spirit of Maher 1982)
    scored, conceded, games = {}, {}, {}
    for _, r in ginf.iterrows():
        for t, gs, gc in [(r["ht"], r["fthg"], r["ftag"]),
                          (r["at"], r["ftag"], r["fthg"])]:
            scored[t]=scored.get(t,0)+gs; conceded[t]=conceded.get(t,0)+gc
            games[t]=games.get(t,0)+1
    overall = pd.concat([ginf["fthg"], ginf["ftag"]]).mean()
    league_h, league_a = h, a
    probs=[]
    for _, r in ginf.iterrows():
        atk_h=(scored[r["ht"]]/games[r["ht"]])/overall
        def_a=(conceded[r["at"]]/games[r["at"]])/overall
        atk_a=(scored[r["at"]]/games[r["at"]])/overall
        def_h=(conceded[r["ht"]]/games[r["ht"]])/overall
        probs.append(prob_2_1(league_h*atk_h*def_a, league_a*atk_a*def_h))
    p_team = float(np.mean(probs))

    print("="*60)
    print(f"RQ1  --  frequency of 2-1 results (all six seasons)")
    print("="*60)
    print(f"  Total matches         : {N}")
    print(f"  Observed 2-1 matches  : {obs}  ({100*obs/N:.1f}%)")
    print(f"  Featured season {SEASON_NAMES[FEATURED_SEASON]}: "
          f"{int(ginf[ginf.season==FEATURED_SEASON].apply(is_2_1,axis=1).sum())} of "
          f"{len(ginf[ginf.season==FEATURED_SEASON])} matches were 2-1")
    for label, p in [("simple Poisson", p_simple), ("team-strength Poisson", p_team)]:
        exp = p*N
        chi2, pv = stats.chisquare(f_obs=[obs, N-obs], f_exp=[exp, N-exp])
        print(f"  [{label}] expected {exp:.1f} ({100*p:.1f}%)  "
              f"chi2={chi2:.3f}, df=1, p={pv:.4f} "
              f"-> {'SIGNIFICANT' if pv<ALPHA else 'not significant'}")
    return N, obs, h, a


# ============ 3. RQ2 + sequences: from EVENT DATA (later seasons only) ============
def build_sequences(events, ginf):
    """Return a dataframe of 2-1 matches that HAVE valid goal timing,
    with their LWW/WLW/WWL sequence, plus goal-minute records."""
    valid = set(ginf["id_odsp"])
    goals = events[(events["is_goal"]==1) & (events["id_odsp"].isin(valid))].copy()
    goals = goals.sort_values(["id_odsp","sort_order","time"])
    ginf_i = ginf.set_index("id_odsp")

    rows, minute_rows = [], []
    for mid, g in goals.groupby("id_odsp"):
        info = ginf_i.loc[mid]
        hg, ag = int(info["fthg"]), int(info["ftag"])
        if {hg, ag} != {2, 1}:
            continue
        sides = g["side"].tolist()
        if len(sides) != 3:            # need exactly 3 recorded goals
            continue
        win_side = 1 if hg > ag else 2
        seq = "".join("W" if s==win_side else "L" for s in sides)
        if seq not in ("LWW","WLW","WWL"):
            continue
        rows.append(dict(id_odsp=mid, season=int(info["season"]),
                         league=str(info["country"]), sequence=seq))
        for _, gr in g.iterrows():
            minute_rows.append(dict(minute=gr["time"],
                                    who=("winner" if gr["side"]==win_side else "loser")))
    return pd.DataFrame(rows), pd.DataFrame(minute_rows)

def run_rq2(seq_df):
    order = ["LWW","WLW","WWL"]
    counts = seq_df["sequence"].value_counts().reindex(order, fill_value=0)
    n = int(counts.sum())
    exp = np.array([n/3]*3)
    obs = counts.values.astype(float)
    chi2, p = stats.chisquare(f_obs=obs, f_exp=exp)
    V = np.sqrt(chi2/(n*2)) if n else 0.0
    res = (obs-exp)/np.sqrt(exp)
    seasons_used = sorted(seq_df["season"].unique())
    print("\n" + "="*60)
    print(f"RQ2  --  goal order in 2-1 matches (event-data seasons only)")
    print("="*60)
    print(f"  Seasons used : {', '.join(SEASON_NAMES[s] for s in seasons_used)}")
    print(f"  2-1 matches with goal timing : {n}")
    for o,c,e,r in zip(order,obs,exp,res):
        print(f"    {o}: observed {int(c):3d} | expected {e:5.1f} | std.residual {r:+.2f}")
    print(f"  chi2={chi2:.3f}, df=2, p={p:.4f} "
          f"-> {'SIGNIFICANT' if p<ALPHA else 'not significant'}")
    print(f"  Cramer's V (effect size) = {V:.3f}")

    # focused test of the DIRECTIONAL prediction (H2): loser scores last vs earlier
    wwl = int(counts["WWL"]); earlier = int(counts["LWW"] + counts["WLW"])
    exp_wwl = n/3; exp_earlier = 2*n/3
    chi2b, pb = stats.chisquare(f_obs=[wwl, earlier], f_exp=[exp_wwl, exp_earlier])
    print(f"\n  Focused test (H2 direction): 'loser scores last' vs 'earlier'")
    print(f"    WWL {wwl} vs earlier {earlier}  (expected {exp_wwl:.0f} vs {exp_earlier:.0f})")
    print(f"    chi2={chi2b:.3f}, df=1, p={pb:.4f} "
          f"-> {'SIGNIFICANT' if pb<ALPHA else 'not significant'}")

    # per-league breakdown
    print("\n  Per-league goal-order breakdown:")
    for lg, sub in seq_df.groupby("league"):
        cc = sub["sequence"].value_counts().reindex(order, fill_value=0)
        nn = int(cc.sum())
        if nn == 0:
            continue
        c2, pp = stats.chisquare(f_obs=cc.values, f_exp=[nn/3]*3)
        print(f"    {lg:9s} n={nn:4d}  LWW {int(cc['LWW']):3d} "
              f"WLW {int(cc['WLW']):3d} WWL {int(cc['WWL']):3d}  "
              f"p={pp:.3f} {'SIG' if pp<ALPHA else 'ns'}")
    return counts, exp


# ============================ 4. FIGURES ============================
def fig_scorelines(ginf, path):
    sl = (ginf["fthg"].astype(str)+"-"+ginf["ftag"].astype(str)).value_counts()
    sl = sl.sort_values(ascending=False).head(12)
    colours=[ORANGE if s in ("2-1","1-2") else BLUE for s in sl.index]
    fig,ax=plt.subplots(figsize=(9,4.5)); ax.bar(sl.index,sl.values,color=colours)
    ax.set_xlabel("Final scoreline"); ax.set_ylabel("Number of matches")
    ax.set_title("Frequency of final scorelines (five leagues, 2011/12–2016/17)")
    plt.xticks(rotation=45,ha="right"); fig.tight_layout()
    fig.savefig(path,bbox_inches="tight"); plt.close(fig)

def fig_sequences(counts, exp, path):
    labels=list(counts.index); x=np.arange(3)
    fig,ax=plt.subplots(figsize=(7,4.5))
    ax.bar(x,exp,width=0.67,color=GREY,alpha=0.3,label="Expected (even)")
    b=ax.bar(x,counts.values,width=0.55,color=BLUE,label="Observed")
    ax.axhline(np.mean(exp),color=RED,ls="--",lw=1.3,label="One-third line")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("Goal sequence (W = winner's goal, L = loser's goal)")
    ax.set_ylabel("Number of 2-1 matches")
    ax.set_title("Observed vs expected goal sequences in 2-1 matches (five leagues)")
    for bar,v in zip(b,counts.values):
        ax.text(bar.get_x()+bar.get_width()/2,v+0.5,str(int(v)),ha="center",fontsize=10)
    ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(path,bbox_inches="tight"); plt.close(fig)

def fig_minutes(min_df, path):
    bins=np.arange(0,100,10); fig,ax=plt.subplots(figsize=(8,4.5))
    ax.hist(min_df[min_df.who=="winner"]["minute"],bins=bins,color=BLUE,alpha=0.75,label="Winner's goals")
    ax.hist(min_df[min_df.who=="loser"]["minute"],bins=bins,color=ORANGE,alpha=0.75,label="Loser's goals")
    ax.set_xlabel("Minute of match"); ax.set_ylabel("Number of goals")
    ax.set_title("When goals are scored in 2-1 matches (five leagues)"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(path,bbox_inches="tight"); plt.close(fig)

def fig_seasons(seq_df, path):
    order=["LWW","WLW","WWL"]
    tab=(seq_df.groupby("season")["sequence"].value_counts()
         .unstack().reindex(columns=order,fill_value=0))
    names=[SEASON_NAMES[s] for s in tab.index]; x=np.arange(len(names)); w=0.26
    fig,ax=plt.subplots(figsize=(9,4.5))
    ax.bar(x-w,tab["LWW"],w,label="LWW",color=BLUE)
    ax.bar(x,  tab["WLW"],w,label="WLW",color=ORANGE)
    ax.bar(x+w,tab["WWL"],w,label="WWL",color=GREY)
    ax.set_xticks(x); ax.set_xticklabels(names,rotation=30,ha="right")
    ax.set_xlabel("Season"); ax.set_ylabel("Number of 2-1 matches")
    ax.set_title("Goal sequences by season (five leagues)")
    ax.legend(frameon=False,ncol=3); fig.tight_layout()
    fig.savefig(path,bbox_inches="tight"); plt.close(fig)


# ============================ MAIN ============================
def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    events, ginf = load_data()

    # data coverage report -- honest and important for the paper
    cov = ginf.groupby("season")["adv_stats"].apply(lambda s: int(s.sum()))
    print("Event-data (goal timing) coverage by season:")
    for s in SEASONS:
        got = int(cov.get(s, 0)); tot = int((ginf.season==s).sum())
        print(f"  {SEASON_NAMES[s]}: {got}/{tot} matches have goal timing")
    print()

    run_rq1(ginf)
    seq_df, min_df = build_sequences(events, ginf)
    counts, exp = run_rq2(seq_df)

    fig_scorelines(ginf, f"{FIG_DIR}/fig2_scorelines.png")
    fig_sequences(counts, exp, f"{FIG_DIR}/fig4_sequences.png")
    fig_minutes(min_df, f"{FIG_DIR}/fig5_minutes.png")
    fig_seasons(seq_df, f"{FIG_DIR}/fig6_seasons.png")
    print(f"\nFigures saved to ./{FIG_DIR}/  --  Done.")

if __name__ == "__main__":
    main()
