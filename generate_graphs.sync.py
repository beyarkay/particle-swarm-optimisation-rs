# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import datetime
import scipy
import os
import ipywidgets as widgets
from ipywidgets import interact, interact_manual

sns.set()
# %% [md] key="value"
# # Preprocess
# %%
%%time
# CPU times: user 28 s, sys: 146 ms, total: 28.1 s
# Wall time: 27.6 s

# stdpso = pd.read_csv(f'data/raw/std_pso_{datetime.date.today()}.csv').sort_values('benchmark')
stdpso = pd.read_csv(f'data/raw/std_pso_2022-03-20.csv').sort_values('benchmark')

prefs = [
    {
        'key': 'perc_oob',
        'in_parens': 'Fraction out of bounds',
        'save_as': 'standard_pso_perc_oob',
    }, {
        'key': 'gdiversity',
        'in_parens': 'Swarm diversity',
        'save_as': 'standard_pso_gdiversity',
    }, {
        'key': 'gbest_fit',
        'in_parens': 'Global best fitness',
        'save_as': 'standard_pso_gbest_fit',
    },
]
pref = prefs[2]
for pref in prefs:

    g = sns.FacetGrid(
        data=stdpso,
        col="benchmark",
        col_wrap=3,
        height=3.5,
        aspect=1.5,
        sharey=False,
        palette="viridis",
    )

    g.map_dataframe(
        sns.lineplot,
        x="iter_num",
        y=pref['key'],
    #     estimator=None,
    #     units='rep_num',
    #     ci=None
    )
    g.add_legend()
    g.set_titles(
        col_template="{col_name}",
    #     row_template="{row_name}"
    )

    g.set_axis_labels("Iteration Number", pref['in_parens'])
#     plt.savefig(f"../report/figs/{pref['save_as']}.pdf")
    plt.show()


# %% [md]
# Optimise Control Parameters per Benchmark

# %% [md]
## Heatmaps of Control Params

# %%


# min_per_bench = opt_gb.groupby('benchmark').gbest_fit.min()
# rng_per_bench = opt_gb.groupby('benchmark').gbest_fit.max() - min_per_bench
# rng_per_bench
sorted(opt.benchmark.unique())


# %% --------------------------------------------------------------------------------------
%%time
use_mld_files = True
# CPU times: user 31.9 s, sys: 10.8 s, total: 42.8 s
# Wall time: 46.1 s


prefs = [
#     {
#         'key': 'perc_oob',
#         'in_parens': 'Fraction out of bounds',
#         'save_as': 'opt_heatmaps_perc_oob',
#     }, {
#         'key': 'gdiversity',
#         'in_parens': 'Swarm diversity',
#         'save_as': 'opt_heatmaps_gdiversity',
#     }, {
#         'key': 'log_diversity',
#         'in_parens': '$\log_{{10}}$ of swarm diversity',
#         'save_as': 'opt_heatmaps_log_diversity',
#     },
    {
        'key': 'gbest_fit',
        'in_parens': 'Global best fitness',
        'save_as': 'opt_heatmaps_gbest_fit',
    },
]
# pref = prefs[2]

VAL = 'gbest_std'

pp_df_path = 'data/opt.csv'
for pref in prefs:
#     if False and os.path.exists(pp_df_path):
#         opt_gb = pd.read_csv(pp_df_path)
#     else:
    files = [
        'data/raw/opt_2022-05-28_rep_0.csv',
        'data/raw/opt_2022-05-28_rep_1.csv',
        'data/raw/opt_2022-05-28_rep_2.csv',
        'data/raw/opt_2022-05-28_rep_3.csv',
        'data/raw/opt_2022-05-28_rep_4.csv',
        'data/raw/opt_2022-05-29_rep_0.csv',
        'data/raw/opt_2022-05-29_rep_1.csv',
        'data/raw/opt_2022-05-29_rep_2.csv',
        'data/raw/opt_2022-05-29_rep_3.csv',
        'data/raw/opt_2022-05-29_rep_4.csv',
    ]
    dfs = []

    for i, file in enumerate(files, 1):
        df = pd.read_csv(file)
        df['rep'] = i
        df = df[~df.benchmark.isin([
            'Generalized Paviani Function',
            'Deb 3 Function',
            'Generalized Price 2 Function',
            'Mishra 1 Function',
            'Exponential Function'
        ])]
        df = df[df['w'].between(-1.0, 1.0)]
        dfs.append(df)
    opt = pd.concat(dfs)

    opt['c'] = np.round(opt['c1'] + opt['c2'], 4)
    opt['w'] = np.round(opt['w'], 4)
    opt['log_diversity'] = np.log10(opt['gdiversity'])
    opt_gb = opt[opt.iter_num == opt.iter_num.max()].groupby(['benchmark', 'w', 'c'])[pref['key']] \
                .mean() \
                .reset_index() \
                .sort_values('benchmark')
    min_per_bench = opt_gb.groupby('benchmark').gbest_fit.min()
    rng_per_bench = opt_gb.groupby('benchmark').gbest_fit.max() - min_per_bench
    opt_gb['gbest_std'] = opt_gb.apply(
        lambda row: (row['gbest_fit'] - min_per_bench[row['benchmark']]) / rng_per_bench[row['benchmark']],
        axis=1
    )
    opt_gb.to_csv(pp_df_path)

#     break
    w = np.arange(-1.1, 1.1, 0.05)
    c = (24 * (1 - w*w)) / (7 - 5 * (-w))
    x_offset = 0
    x_factor = 9.75
    y_offset = 10.5
    y_factor = 10
    def draw_heatmap(*args, **kwargs):
        data = kwargs.pop('data')
        d = data.pivot(columns=args[0], index=args[1], values=args[2])
        d = d.reindex(index=sorted(d.index)[::-1])
        d = d.reindex(columns=sorted(d.columns))
        ax = sns.heatmap(d, **kwargs)
#         plt.plot(
#             (c  * x_factor) + x_offset,
#             (w  * y_factor) + y_offset,
#             c='white'
#         )

    g = sns.FacetGrid(opt_gb,
                      col='benchmark',
                      col_wrap=3,
                      height=4,
                      aspect=1.9
                     )

    g.map_dataframe(draw_heatmap, 'c', 'w', VAL,
                    cmap="viridis",
                    yticklabels=True,
                    xticklabels=True)

    g.set_titles(
        col_template=f"{{col_name}} ({VAL})",
    )
    g.set_axis_labels("$c=c_1+c_2$", "$w$")

#     plt.savefig(f"../report/figs/{pref['save_as']}.pdf")
#     plt.savefig(f"../../../assignment3/report/imgs/heatmaps.pdf")

    plt.show()


# %%


opt_gb = pd.read_csv("data/opt.csv")


# %%


get_ipython().run_line_magic("matplotlib", "")
d = opt_gb[opt_gb.benchmark == "Schwefel 1 Function"].pivot(
    columns="c", index="w", values="gbest_fit"
)
shp = d.shape
xx = np.tile(d.index.to_numpy(), (shp[1], 1)).T
yy = np.tile(d.columns.to_numpy(), (shp[0], 1))
zz = d.values
xx.shape, yy.shape, zz.shape
# ax = plt.figure().add_subplot(projection='3d')

# ax.plot_surface(xx, yy, zz, cmap='autumn')
# plt.show()


# %%


ax = plt.figure().add_subplot(projection="3d")

ax.plot_surface(
    xx, yy, zz, edgecolor="royalblue", lw=0.5, rstride=8, cstride=8, alpha=0.3
)
plt.show()


# %%
%%time

VAL = 'gbest_std'

pp_df_path = 'data/opt.csv'
# for pref in prefs:
# if False and os.path.exists(pp_df_path):
opt_gb = pd.read_csv(pp_df_path)
# else:
#     opt = pd.read_csv(f'data/raw/opt_2022-05-28_rep_0.csv')
# opt_gb = opt_gb[opt.benchmark.isin([
#     'Cosine Mixture Function',
#     'Pathological Function',
#     'Schwefel 1 Function',
# ])]
#     opt = opt[opt['w'].between(-1.0, 1.0)]

#     opt['c'] = np.round(opt['c1'] + opt['c2'], 4)
#     opt['w'] = np.round(opt['w'], 4)
#     opt['log_diversity'] = np.log10(opt['gdiversity'])
#     opt_gb = opt[opt.iter_num == 3900].groupby(['benchmark', 'w', 'c'])[pref['key']] \
#                 .mean() \
#                 .reset_index() \
#                 .sort_values('benchmark')
#     min_per_bench = opt_gb.groupby('benchmark').gbest_fit.min()
#     rng_per_bench = opt_gb.groupby('benchmark').gbest_fit.max() - min_per_bench
#     opt_gb['gbest_std'] = opt_gb.apply(
#         lambda row: (row['gbest_fit'] - min_per_bench[row['benchmark']]) / rng_per_bench[row['benchmark']],
#         axis=1
#     )
#     opt_gb.to_csv(pp_df_path)
q=0.25
gbest_std_quant = opt_gb.gbest_std.quantile(q)
opt_gb['gbest_std'] = opt_gb.apply(
    lambda row: row.gbest_std if row.gbest_std < gbest_std_quant else np.nan,
    axis=1
)

# opt_gb[opt_gb.gbest_std < opt_gb.gbest_std.quantile(0.5)]

w = np.arange(-1.0, 1.0, 0.05)
c = (24 * (1 - w*w)) / (7 - 5 * (-w))
x_offset = 0
x_factor = 9.75
y_offset = 10.5
y_factor = 10
def draw_heatmap(*args, **kwargs):
    data = kwargs.pop('data')
    d = data.pivot(columns=args[0], index=args[1], values=args[2])
    d = d.reindex(index=sorted(d.index)[::-1])
    d = d.reindex(columns=sorted(d.columns))
    ax = sns.heatmap(d, **kwargs)
#     plt.plot(
#         (c  * x_factor) + x_offset,
#         (w  * y_factor) + y_offset,
#         c='white'
#     )

g = sns.FacetGrid(
    opt_gb,
    col='benchmark',
    col_wrap=3,
    height=4,
    aspect=1.9
)

g.map_dataframe(
    draw_heatmap, 'c', 'w', VAL,
    cmap="viridis",
    yticklabels=True,
    xticklabels=True,
    vmin=0.0,
    vmax=1.0,
#     square=True
)

g.set_titles(
    col_template=f"{{col_name}} ({VAL})",
)
g.set_axis_labels("$c=c_1+c_2$", "$w$")
plt.savefig(f"../../../assignment3/report/imgs/thresh_{q}.pdf")

# plt.savefig(f"../report/figs/{pref['save_as']}.pdf")
plt.show()

# %%


pp_df_path = "data/opt.csv"

opt_gb = pd.read_csv(pp_df_path)


@interact(q=(0, 1, 0.01))
def widget(q=0.25):
    opt_gb = pd.read_csv(pp_df_path)
    gbest_std_quant = opt_gb.gbest_std.quantile(q)
    opt_gb["gbest_std"] = opt_gb.apply(
        lambda row: row.gbest_std if row.gbest_std < gbest_std_quant else 1.0, axis=1
    )
    opt_gb = opt_gb.groupby(["w", "c"]).gbest_std.sum().reset_index()
    opt_gb["gbest_std"] = opt_gb["gbest_std"].max() - opt_gb["gbest_std"]
    opt_gb["gbest_std"] /= opt_gb.gbest_std.sum()
    d = opt_gb.pivot(columns="c", index="w", values="gbest_std")
    d = d.reindex(index=sorted(d.index)[::-1])
    d = d.reindex(columns=sorted(d.columns))
    ax = sns.heatmap(d, vmin=0, square=True, cbar=False, cmap="viridis")
    #     print("return vec![")
    #     for i, row in opt_gb.iterrows():
    #         if row.gbest_std > 0:
    #             print(f"    ( ControlParams {{ w: {row.w}, c1: {row.c}, c2: {row.c} }}, {row.gbest_std} ),")
    #     print("];")
    #     plt.tight_layout()
    plt.savefig(
        f"../../../assignment3/report/imgs/quantile{q}.pdf", bbox_inches="tight"
    )


# %%


df


# %%


df = pd.read_csv("data/raw/et_pso_2022-05-29_rep_4.csv")
df["rep"] = i
df = df[
    ~df.benchmark.isin(
        [
            "Generalized Paviani Function",
            "Deb 3 Function",
            "Generalized Price 2 Function",
            "Mishra 1 Function",
            "Exponential Function",
        ]
    )
]
df = df[df["w"].between(-1.0, 1.0)]
df = df[df["iter_num"] > 0]
df = pd.concat([df])
df["log10_iter_num"] = np.log10(df["iter_num"])

df = (
    df.groupby(["benchmark", "strat", "log10_iter_num"])["gbest_fit"]
    .mean()
    .reset_index()
    .sort_values("benchmark")
)

df


# %%


# df = pd.read_csv('data/raw/et_pso_2022-05-29_rep_0.csv')
# df['strat'] = df.apply(lambda row: 'PSO-RAC' if row['max_stagnent_iters'] == 1 else 'ET-PSO', axis=1)
files = [
    "data/raw/et_pso_2022-05-29_rep_4.csv",
    "data/raw/et_pso_2022-05-29_rep_3.csv",
    "data/raw/et_pso_2022-05-29_rep_2.csv",
    "data/raw/et_pso_2022-05-29_rep_1.csv",
    "data/raw/et_pso_2022-05-29_rep_0.csv",
]
dfs = []

for i, file in enumerate(files, 1):
    df = pd.read_csv(file)
    df["rep_num"] = i
    df = df[
        ~df.benchmark.isin(
            [
                "Generalized Paviani Function",
                "Deb 3 Function",
                "Generalized Price 2 Function",
                "Mishra 1 Function",
                "Exponential Function",
            ]
        )
    ]
    df = df[df["w"].between(-1.0, 1.0)]
    df = df[df["iter_num"] > 0]
    df["log10_iter_num"] = np.log10(df["iter_num"])
    df = df[["benchmark", "log10_iter_num", "rep_num", "gbest_fit", "strat"]]
    dfs.append(df)

df = pd.concat(dfs).reset_index()

g = sns.FacetGrid(
    data=df,
    col="benchmark",
    hue="strat",
    col_wrap=3,
    height=3.5,
    aspect=1.5,
    sharey=False,
    col_order=np.sort(df.benchmark.unique()),
    palette="viridis",
)

g.map_dataframe(
    sns.lineplot,
    x="log10_iter_num",
    y="gbest_fit",
    #     units="rep"
)
g.add_legend(title="Strategy")
g.set_titles(
    col_template="{col_name}",
    #     row_template="{row_name}"
)

g.set_axis_labels("$\log_{10}$(Iteration Number)", "Fitness")
plt.savefig(f"../../../assignment3/report/imgs/strategies.pdf")
plt.show()


# %%


# %%


sns.lineplot(
    data=df.reset_index(),
    x="log10_iter_num",
    y="gbest_fit",
    hue="benchmark",
    style="strat",
    units="rep_num",
    estimator=None,
)


# %%


df = pd.read_csv("data/raw/et_pso_2022-05-29_rep_0.csv")
# df['strat'] = df.apply(lambda row: 'PSO-RAC' if row['max_stagnent_iters'] == 1 else 'ET-PSO', axis=1)

df = df[df["iter_num"] == df["iter_num"].max()]
# df['log10_iter_num'] = np.log10(df['iter_num'])

df = df[
    ~df.benchmark.isin(
        [
            "Generalized Paviani Function",
            "Deb 3 Function",
            "Generalized Price 2 Function",
            "Mishra 1 Function",
            "Exponential Function",
        ]
    )
]

values = (
    df.groupby(["benchmark", "strat"])["gbest_fit"]
    .max()
    .reset_index()
    .pivot(columns="strat", index="benchmark", values="gbest_fit")
)
with open(f"../../../assignment3/report/table.tex", "w") as f:
    tex = values.to_latex(caption="Optimal strategies", label="tab:best_strat")
    f.write(
        tex.replace(" Function", "")
        .replace("benchmark", "Benchmark Function")
        .replace("strat", "PSO Variant")
        .replace("EmpiricallyTuned", "ET-PSO")
        .replace("RandomAccelerationCoefficients", "PSO-RAC")
    )
values


# ## Calculate the latex table

# %%


opt = pd.read_csv(f"data/raw/opt_2022-03-20.csv")
opt = opt[
    ~opt.benchmark.isin(
        [
            "Deb 3 Function",
            "Generalized Price 2 Function",
            "Mishra 7 Function",
            "Generalized Paviani Function",
        ]
    )
]

opt["c"] = np.round(opt["c1"] + opt["c2"], 4)
opt["w"] = np.round(opt["w"], 4)

opt_gb = (
    opt[opt.iter_num == 4900]
    .groupby(["benchmark", "w", "c"])["gbest_fit"]
    .mean()
    .reset_index()
    .sort_values("benchmark")
)

opt_gb["1%"] = opt_gb.groupby("benchmark").gbest_fit.transform(
    lambda x: x.quantile(0.01)
)
best_cps = (
    opt_gb[opt_gb.gbest_fit < opt_gb["1%"]]
    .sort_values(["benchmark", "gbest_fit", "w", "c"])
    .set_index(["benchmark", "w", "c"])[["gbest_fit"]]
)
best_cps["gbest_fit"] = np.round(best_cps["gbest_fit"], 3)

with open("../report/table.tex", "w") as f:
    tex = best_cps.to_latex(
        caption="Optimal control parameter values for all tested functions",
        label="tab_best_control_params",
    )
    f.write(
        tex.replace(" Function", "")
        .replace("benchmark", "Benchmark Function")
        .replace(" c ", " $c$ ")
        .replace(" w ", " $w$ ")
        .replace("gbest\\_fit", "Global Best Fitness")
    )
best_cps


# ## Best fitness value over time

# %%
%%time
# CPU times: user 39min 4s, sys: 26.2 s, total: 39min 30s
# Wall time: 39min 31s
# start at 10h35

prefs = [
    {
        'key': 'perc_oob',
        'legend': 'c',
        'in_parens': 'Fraction out of bounds',
        'save_as': 'opt_perc_oob',
    }, {
        'key': 'gdiversity',
        'in_parens': 'Swarm diversity',
        'save_as': 'opt_gdiversity',
    }, {
        'key': 'gbest_fit',
        'in_parens': 'Global best fitness',
        'save_as': 'opt_gbest_fitness',
    },
]
pref = prefs[2]
for pref in prefs:
#     opt = pd.read_csv(f'data/raw/opt_{datetime.date.today()}.csv')
    opt = pd.read_csv(f'data/raw/opt_2022-03-20.csv')
    opt = opt[~opt.benchmark.isin([
        'Deb 3 Function',
        'Generalized Price 2 Function',
        'Mishra 7 Function',
        'Generalized Paviani Function',
    ])].sort_values('benchmark')

    opt['c'] = np.round(opt['c1'] + opt['c2'], 4)
    opt['w'] = np.round(opt['w'], 4)

    for param in ['c', 'w']:
        opt_ot = opt.groupby(['benchmark', 'w', 'c', 'iter_num']) \
            .gbest_fit.mean() \
            .reset_index() \
            .sort_values('benchmark')

        g = sns.FacetGrid(
            data=opt,
            col="benchmark",
            hue=param,
            col_wrap=3,
            height=3.5,
            aspect=1.5,
            sharey=False,
            palette="viridis"
        )
        g.map(
            sns.lineplot,
            "iter_num",
            pref['key'],
        #     ci=None
        )
        g.add_legend()
        g.set_titles(
            col_template="{col_name}",
        )
        g.set_axis_labels("Iteration Number", pref['in_parens'])
        plt.savefig(f"../report/figs/{pref['save_as']}_{param}.pdf")
        plt.show()

# %% [md]
# Resample from Poli

# %%

%%time
# CPU times: user 13min 22s, sys: 3.66 s, total: 13min 26s
# Wall time: 13min 25s
prefs = [
    {
        'key': 'log_diversity',
        'in_parens': '$\log_{{10}}$ of swarm diversity',
        'save_as': 'resample_log_diversity_max_stagnent_iters',
    }, {
        'key': 'perc_oob',
        'in_parens': 'Fraction out of bounds',
        'save_as': 'resample_perc_oob_max_stagnent_iters',
    }, {
        'key': 'gdiversity',
        'in_parens': 'Swarm diversity',
        'save_as': 'resample_gdiversity_max_stagnent_iters',
    }, {
        'key': 'gbest_fit',
        'in_parens': 'Global best fitness',
        'save_as': 'resample_gbest_fit_max_stagnent_iters',
    },
]
pref = prefs[0]
for pref in prefs:
#     res = pd.read_csv(f'data/raw/resample_{datetime.date.today()}.csv').sort_values('benchmark')
    res = pd.read_csv(f'data/raw/resample_2022-03-20.csv').sort_values('benchmark')
    res = res[~res.benchmark.isin([
        'Deb 3 Function',
        'Generalized Price 2 Function',
        'Mishra 7 Function',
        'Generalized Paviani Function',
    ])]
    res['log_diversity'] = np.log10(res['gdiversity'])

    g = sns.FacetGrid(
        data=res,
        col="benchmark",
        hue='max_stagnent_iters',
        col_wrap=3,
        height=3.5,
        aspect=1.5,
        sharey=False,
        palette="viridis" # Different colour pallete to distinguish colours better
    )
    g.map(
        sns.lineplot,
        "iter_num", pref['key'],
    #     ci=None,
    )
    g.add_legend(title='Maximum stagnent iterations')
    g.set_titles(
        col_template=f"{{col_name}}",
    )
    g.set_axis_labels("Iteration number", pref['in_parens'])

    plt.savefig(f"../report/figs/{pref['save_as']}.pdf")
    plt.show()

# %%
