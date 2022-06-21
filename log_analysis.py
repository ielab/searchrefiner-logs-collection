import difflib
import os
from collections import Counter

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import pearsonr

from sr_log import load_sr_logs, load_sr_sessions

# %%
logs = list(load_sr_logs("searchrefiner.log.jsonl"))
logs_with_seed = [x for x in logs if x.has_seed_studies]
# %%
logical_sessions = {}
mission_sessions = {}
unconsidered_logical_logs = []
unconsidered_mission_logs = []
for k, v in load_sr_sessions("searchrefiner.logical.log.json").items():
    if v[0].has_seed_studies and len(v) > 1:
        logical_sessions[k] = v
    else:
        for log in v:
            unconsidered_logical_logs.append(log)
for k, v in load_sr_sessions("searchrefiner.mission.log.json").items():
    if v[0].has_seed_studies and len(v) > 1:
        mission_sessions[k] = v
    else:
        for log in v:
            unconsidered_mission_logs.append(log)
# %%
pd.Series([v[0].num_rel for k, v in logical_sessions.items()]).describe()
# %%
pd.Series([v[0].num_rel for k, v in mission_sessions.items()]).describe()
# %%
rel_ret_data = [[v[0].num_rel_ret, v[-1].num_rel_ret] for k, v in logical_sessions.items()]
start_higher = [x for x in rel_ret_data if x[0] > x[1]]
end_higher_no_change = [x for x in rel_ret_data if x[0] <= x[1]]
# %%
f, ax = plt.subplots()
for results in end_higher_no_change:
    xy = [(i, v) for i, v in enumerate(results)]
    x = [x[0] for x in xy]
    y = [x[1] for x in xy]
    sns.lineplot(x=x, y=y, ax=ax)
# ax.set(yscale="log")
plt.show()

# %%
log_summary = pd.DataFrame(pd.Series({"Logs": len(logs),
                                      "Sessions": len(logical_sessions),
                                      "Sessions (End Higher)": len(end_higher_no_change),
                                      "Sessions (Start Higher)": len(start_higher),
                                      })).reset_index()
log_summary.columns = ["Name", "Count"]
print(log_summary.to_latex(escape=False, index=False))
# %%
rel_ret_props = [[(x.num_rel_ret / x.num_rel, (x.num_rel_ret + 1) / (x.num_ret + 1)) for x in v] for k, v in logical_sessions.items()]
# %%
kind = "logical"
improvements = [(v[0], v[-1]) for k, v in logical_sessions.items()]  # if v[0].num_rel_ret <= v[-1].num_rel_ret]
axis_size = 36
tick_size = 28
# %%
f, ax = plt.subplots()
sns.set()
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
diff = [(x[0].num_ret, x[-1].num_ret) for x in improvements]
x = [x[0] for x in diff]
y = [x[1] for x in diff]
df = pd.DataFrame({"Docs Retrieved": x + y, "Position in the Session": ["Start" if i < len(x) else "End" for i, _ in enumerate(x + y)]})
sns.boxplot(data=df, x="Position in the Session", y="Docs Retrieved", color="lightgrey", showfliers=True)
ax.set(yscale="log")
plt.rc("axes", labelsize=axis_size)
plt.rc("ytick", labelsize=tick_size)
plt.rc("xtick", labelsize=tick_size)
plt.tight_layout()
plt.savefig(f"figures/{kind}_numret.pdf")
plt.show()
# %%
f, ax = plt.subplots()
sns.set()
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
diff = [(x[0].num_rel_ret / x[0].num_rel, x[-1].num_rel_ret / x[-1].num_rel) for x in improvements]
x = [x[0] for x in diff]
y = [x[1] for x in diff]
df = pd.DataFrame({"Recall (Seed)": x + y, "Position in the Session": ["Start" if i < len(x) else "End" for i, _ in enumerate(x + y)]})
sns.boxplot(data=df, x="Position in the Session", y="Recall (Seed)", color="lightgrey")
# ax.set(yscale="log")
# ax.yaxis.set_ticks(np.linspace(1,0,5))
plt.rc("axes", labelsize=axis_size)
plt.rc("ytick", labelsize=tick_size)
plt.rc("xtick", labelsize=tick_size)
plt.tight_layout()
plt.savefig(f"figures/{kind}_recall.pdf")
plt.show()
# %%
f, ax = plt.subplots()
sns.set()
plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
sns.set_style("whitegrid")
diff = [((x[0].num_rel_ret + 1) / (x[0].num_ret + 1), (x[-1].num_rel_ret + 1) / (x[-1].num_ret + 1)) for x in improvements]
x = [x[0] for x in diff]
y = [x[1] for x in diff]
df = pd.DataFrame({"Precision (Seed)": x + y, "Position in the Session": ["Start" if i < len(x) else "End" for i, _ in enumerate(x + y)]})
sns.boxplot(data=df, x="Position in the Session", y="Precision (Seed)", color="lightgrey", showfliers=True)
# ax.set(yscale="log")
ax.set_ylim(0, .1)
plt.rc("axes", labelsize=axis_size)
plt.rc("ytick", labelsize=tick_size)
plt.rc("xtick", labelsize=tick_size)
plt.tight_layout()
plt.savefig(f"figures/{kind}_precision.pdf")
plt.show()
# print(boxplot_stats(df["Precision (Seed)"]).pop(0)['fliers'])
# %%
axis_size = 28
tick_size = 22
f, ax = plt.subplots(2, 1)
sns.set()
plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
sns.set(rc={"figure.figsize": (8, 8)})
sns.set_style("whitegrid")
f1_score = lambda log: 2 * ((((log.num_rel_ret + 1) / (log.num_ret + 1)) * ((log.num_rel_ret + 1) / (log.num_ret + 1))) / (((log.num_rel_ret + 1) / (log.num_ret + 1)) + ((log.num_rel_ret + 1) / (log.num_ret + 1))))
logical_session_f1 = [f1_score(v[0]) - f1_score(v[-1]) for _, v in logical_sessions.items()]
mission_session_f1 = [f1_score(v[0]) - f1_score(v[-1]) for _, v in mission_sessions.items()]
logical_session_lengths = [len(v) for k, v in logical_sessions.items()]
mission_session_lengths = [len(v) for k, v in mission_sessions.items()]

session_lengths = []
session_f1 = []
tmp_arr = []
prev_length = 0
box_size = 5
current_box = 0
for length, f1 in sorted(list(zip(mission_session_lengths, mission_session_f1))):
    if prev_length != length:
        current_box += 1
        if current_box == box_size:
            session_lengths.append([x[0] for x in tmp_arr])
            session_f1.append([x[1] for x in tmp_arr])
            tmp_arr = []
            current_box = 0
    tmp_arr.append((length, f1))
    prev_length = length
session_lengths[-1] += [x[0] for x in tmp_arr]
session_f1[-1] += [x[1] for x in tmp_arr]
x = []
y = []
for list_session, list_f1 in zip(session_lengths, session_f1):
    x += [f1 for f1 in list_f1]
    y += [f"{list_session[0]}-{list_session[-1]}"] * len(list_session)

df = pd.DataFrame({"F1 (Start$-$End)": x, "Session Length": y})  # , "Session Type": ["Logical" if i < len(logical_session_lengths) else "Mission" for i, _ in enumerate(logical_session_lengths + mission_session_lengths)]})
sns.lineplot(data=df, x="Session Length", y="F1 (Start$-$End)", color="grey", ax=ax[0])
sns.countplot(data=df, x="Session Length", color="grey", ax=ax[1])
# ax.set(yscale="log")
# ax.set_ylim(0, 100)
# ax.yaxis.set_ticks(np.linspace(50,0,5))
ax[1].set_ylabel("Count")
plt.rc("axes", labelsize=axis_size)
plt.rc("ytick", labelsize=tick_size)
plt.rc("xtick", labelsize=tick_size)
ax[0].tick_params(axis="x", labelrotation=45)
ax[1].tick_params(axis="x", labelrotation=45)
plt.tight_layout()
plt.savefig(f"figures/mission_session_length.pdf")
plt.show()
# %%
same_improved_sessions = []
same_sessions = []
for _, session in mission_sessions.items():
    if session[0].query == session[-1].query:
        same_sessions.append(session)
        found_better = False
        for query in session[1:-1]:
            if query.num_ret < session[0].num_ret and query.num_rel_ret > session[0].num_rel_ret:
                found_better = True
        if found_better:
            same_improved_sessions.append(session)
print(len(same_sessions))
print(len(same_improved_sessions))
# %%
axis_size = 16
tick_size = 14
f, ax = plt.subplots()
sns.set()
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
sns.set(rc={"figure.figsize": (8, 2)})
sns.set_style("whitegrid")
df = pd.DataFrame({"Date": [f"{log.time.tm_year}-{log.time.tm_mon}" for log in logs]})
sns.countplot(data=df, x="Date", color="grey")
ax.set_ylabel("Queries")
ax.set_xlabel("")
plt.axhline(pd.Series(Counter([f"{log.time.tm_year}-{log.time.tm_mon}" for log in logs])).describe()["mean"], color="black", linestyle="--")
ax.annotate("Average", xy=(0, 520), xycoords='data')
plt.rc("axes", labelsize=axis_size)
plt.rc("ytick", labelsize=tick_size)
plt.rc("xtick", labelsize=tick_size)
ax.tick_params(axis="x", labelrotation=45)
plt.tight_layout()
plt.savefig(f"figures/dates.pdf")
plt.show()
# %%
for i, log in enumerate(logs):
    with open(f"code/queries/{i}", "w") as f:
        f.write(log.query.replace('\\"', '"').replace("\\n", " ").replace("\\r", " "))
# ./extract_terms
# %%
terms = []
individual_terms = []
mon = -1
j = -1
k = 0
for i, log in enumerate(logs):
    if log.has_seed_studies:
        with open(f"code/terms/{i}", "r") as f:
            individual_terms.append([])
            if log.time.tm_mon != mon:
                j += 1
                terms.append([])
                mon = log.time.tm_mon
            read_terms = [x.replace("\n", "") for x in f.readlines()]
            terms[j] += read_terms
            individual_terms[k] = read_terms
            k += 1

flat_terms = [x for y in terms for x in y]
counted_flat_terms = Counter(flat_terms)
# %%
counted_terms = [Counter(x) for x in terms]
most_common_counted_terms = [[y[0] for y in x.most_common(20)] for x in counted_terms]
# %%
improvements = []
improved_sessions = []
best_sessions = []
best_queries = []
worsened_sessions = []
better_logical_sessions = []
better_logical_queries = []
last_logical_queries = []
for _, session in logical_sessions.items():
    found_better = False
    found_best = False
    found_last = False
    last_logical_queries.append(session[-1])
    best_query = session[0]
    for query in session[1:]:
        if session[0].num_ret > query.num_ret > 0 and query.num_rel_ret > session[0].num_rel_ret:
            found_best = True
            best_query = query
        elif session[0].num_ret > query.num_ret > 0 and query.num_rel_ret >= session[0].num_rel_ret:
            found_better = True
            best_query = query
    #
    #     if query.num_rel_ret >= last_query.num_rel_ret and query.num_rel_ret > 0:
    #         last_query = query
    #         found_last = True
    #
    # improvements.append((session[0], best_query))
    # better_logical_sessions.append(best_query)
    # if found_last and best_query.query != last_query.query:
    #     better_logical_queries.append(last_query)

    if found_better and not found_best:
        improved_sessions.append(session)
        better_logical_queries.append(best_query)
    elif found_best:
        best_sessions.append(session)
        best_queries.append(best_query)
        better_logical_queries.append(best_query)
        improved_sessions.append(session)
    else:
        worsened_sessions.append(session)
print(len(improved_sessions))
# %%
for i, query in enumerate(improved_sessions[13]):
    fname = f"code/case-study/{query.logical_hash}"
    os.makedirs(fname, exist_ok=True)
    with open(f"{fname}/{i}", "w") as f:
        f.write(query.query)
        # print(query.num_ret, query.num_rel_ret, query.num_rel, query.query)


# %%


def normalise_query(q):
    return q.replace("\\r\\n", "").replace("\\n", "").replace("\\\"", '\"')


# key = 7
key = 0
NUM=len(best_sessions[key])
print(NUM-1)
print("\\begin{Verbatim}[frame=single,framesep=.5pt,breakautoindent=false,breaklines,breakanywhere,commandchars=\\\\\\{\\}]")
for i in range(3, NUM):
    # print(1+i-1,1+i)
    # print(normalise_query(best_sessions[key][i].query))
    toks = []
    for tok in difflib.ndiff(normalise_query(best_sessions[key][i-1].query), normalise_query(best_sessions[key][i].query)):
        token = tok[-1]
        token = token.replace("\\", "").replace("“", "\\textquotedblleft{}").replace("”", "\\textquotedblright{}")
        if tok[0] == "+":
            toks.append(f"\\diffadd{{{token}}}")
        elif tok[0] == '-':
            # toks.append(f"\\xout{{\\textcolor{{red}}{token}}}}}")
            toks.append(f"\\diffsub{{{token}}}")
            # toks.append(f"{token}")
        elif token != ' ':
            # toks.append(f"{{\\textcolor{{gray}}{token}}}")
            toks.append(f"\\diffnul{{{token}}}")
            # toks.append(f"{token}")
        else:
            toks.append(f"\\diffnul{{{token}}}")
    print(f"\\circled{{{i}}}{''.join(toks)}")
print("\\end{Verbatim}")

# %%

# key = 5
key = 0
for i in range(1, len(best_sessions[key])):
    toks = []
    for tok in difflib.ndiff(normalise_query(best_sessions[key][i].query), normalise_query(best_sessions[key][i + 1].query)):
        token = tok[-1]
        token = token.replace("\\", "")#.replace("[", "\\[").replace("]", "\\]")
        if len(token) == 0:
            continue
        if tok[0] == "+":
            toks.append(f"<b>{token}</b>")
        elif tok[0] == '-':
            toks.append(f"<s>{token}</s>")
        elif token != ' ':
            toks.append(f"{token}")
        else:
            toks.append(token)
    print(f" {i-1}) <pre style='font-size: small'>{''.join(toks)}</pre>")

# \\circled{{{i+1}}}
# %%
kind = "logical"
axis_size = 36
tick_size = 28
# %%
f, ax = plt.subplots()
sns.set()
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
diff = [(x[0].num_ret, x[-1].num_ret) for x in improvements]
x = [x[0] for x in diff]
y = [x[1] for x in diff]
df = pd.DataFrame({"Docs Retrieved": x + y, "Position in the Session": ["Start" if i < len(x) else "End" for i, _ in enumerate(x + y)]})
sns.boxplot(data=df, x="Position in the Session", y="Docs Retrieved", color="lightgrey", showfliers=True)
ax.set(yscale="log")
plt.rc("axes", labelsize=axis_size)
plt.rc("ytick", labelsize=tick_size)
plt.rc("xtick", labelsize=tick_size)
plt.tight_layout()
plt.savefig(f"figures/{kind}_retrospective_numret.pdf")
plt.show()
# %%
f, ax = plt.subplots()
sns.set()
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
diff = [(x[0].num_rel_ret / x[0].num_rel, x[-1].num_rel_ret / x[-1].num_rel) for x in improvements]
x = [x[0] for x in diff]
y = [x[1] for x in diff]
df = pd.DataFrame({"Recall (Seed)": x + y, "Position in the Session": ["Start" if i < len(x) else "End" for i, _ in enumerate(x + y)]})
sns.boxplot(data=df, x="Position in the Session", y="Recall (Seed)", color="lightgrey")
# ax.set(yscale="log")
# ax.yaxis.set_ticks(np.linspace(1,0,5))
plt.rc("axes", labelsize=axis_size)
plt.rc("ytick", labelsize=tick_size)
plt.rc("xtick", labelsize=tick_size)
plt.tight_layout()
plt.savefig(f"figures/{kind}_retrospective_recall.pdf")
plt.show()
# %%
f, ax = plt.subplots()
sns.set()
plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
sns.set_style("whitegrid")
diff = [((x[0].num_rel_ret + 1) / (x[0].num_ret + 1), (x[-1].num_rel_ret + 1) / (x[-1].num_ret + 1)) for x in improvements]
x = [x[0] for x in diff]
y = [x[1] for x in diff]
df = pd.DataFrame({"Precision (Seed)": x + y, "Position in the Session": ["Start" if i < len(x) else "End" for i, _ in enumerate(x + y)]})
sns.boxplot(data=df, x="Position in the Session", y="Precision (Seed)", color="lightgrey", showfliers=True)
# ax.set(yscale="log")
ax.set_ylim(0, .1)
plt.rc("axes", labelsize=axis_size)
plt.rc("ytick", labelsize=tick_size)
plt.rc("xtick", labelsize=tick_size)
plt.tight_layout()
plt.savefig(f"figures/{kind}_retrospective_precision.pdf")
plt.show()
# print(boxplot_stats(df["Precision (Seed)"]).pop(0)['fliers'])
#%%

axis_size = 28
tick_size = 22

f, ax = plt.subplots()
sns.set()
plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
sns.set(rc={"figure.figsize": (7, 5)})
sns.set_style("whitegrid")

key = 0
start = 1
end = -1
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
df = pd.DataFrame({"Document Retrieved": [x.num_ret for x in best_sessions[key][start:]], "Seed Studies Retrieved": [x.num_rel_ret for x in best_sessions[key][start:]], "Session Number": [i+1 for i in range(len(best_sessions[key][start:]))]})
sns.barplot(data=df, x="Session Number", y="Seed Studies Retrieved", color="grey", ax=ax)
ax.tick_params(axis="y", labelsize=tick_size)
plt.rc("axes", labelsize=axis_size)
plt.rc("xtick", labelsize=tick_size)
ax.tick_params(labelbottom=True)
ax.annotate("Total Seed Studies", xy=(0.1, best_sessions[key][0].num_rel + 0.1), xycoords='data', fontsize=tick_size)
plt.axhline(best_sessions[key][0].num_rel, color="grey", linestyle="--")
ax2 = ax.twinx()

sns.pointplot(data=df, x="Session Number", y="Document Retrieved", color="black", ax=ax2)
ax2.set(yscale="log")
ax2.grid(False)

ax.set_ylim(0, best_sessions[key][0].num_rel + 1)
# ax.yaxis.set_ticks(np.linspace(50,0,5))

ax2.tick_params(axis="y", labelsize=tick_size)
ax2.set_ylabel("Documents Retrieved", fontsize=axis_size)
ax.tick_params(axis="x", labelrotation=0)
plt.tight_layout()
plt.savefig(f"figures/session_{best_sessions[key][0].logical_hash[:8]}.pdf")
plt.show()

# %%
axis_size = 28
tick_size = 22

best_sessions = dict([(i, v) for i, v in enumerate(logical_sessions.values())])

for key in range(len(best_sessions)):
    f, ax = plt.subplots()
    sns.set()
    plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
    sns.set(rc={"figure.figsize": (7, 5)})
    sns.set_style("whitegrid")

    # key = 3
    start = 0
    end = -1
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    df = pd.DataFrame({"Document Retrieved": [x.num_ret for x in best_sessions[key][start:]], "Seed Studies Retrieved": [x.num_rel_ret for x in best_sessions[key][start:]], "Session Number": [i for i in range(len(best_sessions[key][start:]))]})
    sns.barplot(data=df, x="Session Number", y="Seed Studies Retrieved", color="grey", ax=ax)
    ax.tick_params(axis="y", labelsize=tick_size)
    plt.rc("axes", labelsize=axis_size)
    plt.rc("xtick", labelsize=tick_size)
    ax.tick_params(labelbottom=False)
    plt.title(best_sessions[key][0].logical_hash[:9], fontdict={"fontsize":tick_size}, loc="left", pad=2)
    # ax.annotate("Total Seed Studies", xy=(0.1, best_sessions[key][0].num_rel + 0.1), xycoords='data', fontsize=tick_size)
    plt.axhline(best_sessions[key][0].num_rel, color="grey", linestyle="--")
    ax2 = ax.twinx()

    sns.pointplot(data=df, x="Session Number", y="Document Retrieved", color="black", ax=ax2)
    ax2.set(yscale="log")
    ax2.grid(False)

    ax.set_ylim(0, best_sessions[key][0].num_rel + 1)
    # ax.yaxis.set_ticks(np.linspace(50,0,5))

    ax2.tick_params(axis="y", labelsize=tick_size)
    ax2.set_ylabel("Documents Retrieved", fontsize=axis_size)
    # ax.tick_params(axis="x", labelrotation=0)
    plt.tight_layout()
    plt.savefig(f"figures/session_{best_sessions[key][0].logical_hash[:8]}.pdf")
    # plt.show()
print("done")
# %%
axis_size = 28
tick_size = 22
f, ax = plt.subplots()
plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
sns.set(rc={"figure.figsize": (7, 5)})
sns.set_style("whitegrid")
df = pd.DataFrame({"Documents Retrieved": [x.num_ret for x in better_logical_queries], "Seed Studies Retrieved": [x.num_rel_ret for x in better_logical_queries]})
# df = pd.DataFrame({"Documents Retrieved": [x.num_ret for x in last_logical_queries], "Seed Studies Retrieved": [x.num_rel_ret for x in last_logical_queries]})
sns.regplot(data=df, seed=3397, ci=90, n_boot=100, fit_reg=True, x="Seed Studies Retrieved", y="Documents Retrieved", color="black")
ax.set(yscale="log")
ax.tick_params(labelbottom=True)
ax.yaxis.tick_left()
ax.tick_params(axis="y", labelsize=tick_size, which="minor")
plt.rc("axes", labelsize=axis_size)
plt.rc("xtick", labelsize=tick_size)
plt.tight_layout()
r = pearsonr(df['Documents Retrieved'], df['Seed Studies Retrieved'])
pos = (17, 10e5)
# pos = (18, 10e4*1.7)
ax.annotate(f"Pearson's r = {round(r[0], 4)}{'*' if r[1] < 0.05 else ''}", xy=pos, xycoords='data', fontsize=tick_size)
plt.savefig(f"figures/retrieved_seed_corr_118.pdf")
# plt.savefig(f"figures/retrieved_seed_corr_274.pdf")
plt.show()
# %%
