from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

from os.path import exists
import os
from matplotlib.figure import Figure


numCols = [
    "Neuroticism",
    "Extraversion",
    "Openness",
    "Agreeableness",
    "Conscientiousness",
    "Impulsiveness",
    "Sensationness",
]

drugs = [
    "Alcohol",
    "Amphet",
    "Amyl",
    "Benzos",
    "Caff",
    "Cannabis",
    "Choc",
    "Coke",
    "Crack",
    "Ecstasy",
    "Heroin",
    "Ketamine",
    "Legalh",
    "LSD",
    "Meth",
    "Mushrooms",
    "Nicotine",
    # "Semer",
    "VSA",
]

# Substâncias com menor relevância de análise.
lesser_drugs = ["Choc", "Alcohol", "Caff"]

# k substâncias escolhidas para análise profunda
best_k = [
    "Amphet",
    "Cannabis",
    "Ecstasy",
    "Legalh",
    "LSD",
    "Mushrooms",
    "Nicotine",
]

classifiers = [
    (NearestCentroid(), None),
    (GaussianNB(), None),
    (BernoulliNB(), None),
    (
        RandomForestClassifier(n_jobs=-1),
        dict(max_depth=[9, 10, 11, 12]),
    ),
    (KNeighborsClassifier(n_jobs=-1), dict(n_neighbors=list(range(7, 16, 2)))),
]

order_ = {
    0: "Never Used",
    1: "Used over a Decade Ago",
    2: "Used in Last Decade",
    3: "Used in Last Year",
    4: "Used in Last Month",
    5: "Used in Last Week",
    6: "Used in Last Day",
}


def create_dirs():
    if not exists("figures"):
        os.mkdir("figures")

    if not exists("results"):
        os.mkdir("results")


def run_or_load(path: str, func) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Roda o experimento ou carrega uma execução salva anteriormente.

    Args:
        path (str): caminho do arquivo com formatação de "{a}" para preencher com "results" ou "cm".
        func (_type_): função para execução do experimento.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: scores, matriz de confusão.
    """
    r_path = path.format(a="results")
    cm_path = path.format(a="cm")

    if func or (not exists(r_path) and not exists(cm_path)):
        results, cm = func()
        results.to_csv(r_path)
        cm.to_csv(cm_path)
    else:
        results = pd.read_csv(r_path)
        cm = pd.read_csv(cm_path)

    return results, cm


def build_dataset(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retorna o dataset com as filtragens feitas em "classification.ipynb".

    Args:
        data (pd.DataFrame): dataset pré-processado.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: X, labels
    """
    data_filtered = data[data["Semer"] == 0]

    ignore_cols = (
        ["Country", "Age_", "Education_", "Semer", "Semer_", "Ethnicity"]
        + [x + "_" for x in drugs]
        + drugs
    )

    X = data_filtered.drop(columns=ignore_cols)
    y = data_filtered[drugs]

    return X, y


def metric_preprocessor():
    return ColumnTransformer(
        [("numerical", StandardScaler(), numCols)],
        verbose_feature_names_out=False,
        remainder="passthrough",
    )


def test_classifiers(
    X: pd.DataFrame,
    labels: pd.DataFrame,
    classifiers: list = classifiers,
    pipe: tuple = (metric_preprocessor(),),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Realiza testes de validação cruzada para cada classificador e substância dada.

    Args:
        X (pd.DataFrame):.
        labels (pd.DataFrame): todos os labels de substâncias.
        pipe (tuple, optional): pipeline de transformação de dados. Defaults to (metric_preprocessor(),).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: _description_
    """
    all_results = {}
    all_df = pd.DataFrame(columns=drugs)
    conf_df = None

    n_classes = len(np.unique(labels))
    conf_df = pd.DataFrame(
        columns=["Model", "Substância"]
        + [f"cm_{i}" for i in range(n_classes * n_classes)]
    )

    for c, param_grid in classifiers:
        name = c.__class__.__name__
        for d in drugs:
            estimator = c if param_grid is None else GridSearchCV(c, param_grid, cv=4)
            pipe_ = make_pipeline(*pipe, estimator)

            y = labels[d].to_numpy()
            results = []

            cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3)
            for train, test in cv.split(X, y):
                pipe_ = clone(pipe_)

                # fit & predict
                pipe_.fit(X.iloc[train], y[train])
                preds = pipe_.predict(X.iloc[test])

                # append score
                results.append(balanced_accuracy_score(y[test], preds))

                # draw confusion matrix
                cm = confusion_matrix(y[test], preds)
                conf_df.loc[len(conf_df)] = [name, d] + [y for x in cm for y in x]

            all_results[d] = results

        df_ = pd.DataFrame(all_results)
        df_["Model"] = name
        all_df = pd.concat([all_df, df_], ignore_index=True)

    return all_df.melt("Model", var_name="Substância", value_name="Score"), conf_df


def get_k_highest_mean(df: pd.DataFrame, k: int = 5) -> list[str]:
    """Retorna as k maiores médias no dataframe.

    Args:
        df (pd.DataFrame): dataframe com as substâncias e pontuações.
        k (int, optional): número de itens a retornar. Defaults to 5.

    Returns:
        list[str]: lista de nomes de substâncias.
    """

    per_subs = df.groupby(by=["Substância"]).mean()
    subs = per_subs.sort_values(by=["Score"], ascending=False).index[:k]
    return list(subs)


def boxplot(
    df: pd.DataFrame,
    title="",
    lims=(0, 1),
    refs=None,
    substance_filter: None | list[str] = None,
) -> Figure:
    """Plota o boxplot de pontuações para cada substância e os modelos.

    Args:
        df (pd.DataFrame): dataframe de pontuações com "Model", "Substância" e "Score".
        title (str, optional): título do boxplot. Defaults to "".
        lims (tuple, optional): limites da figura. Defaults to (0, 1).
        refs (list, optional): linhas para referência de pontuações. Defaults to None.
        substance_filter (None | list[str], optional): lista de substância a apresentar. Defaults to None.

    Returns:
        Figure: a figura que contém os boxplots.
    """
    data = df[df["Substância"].isin(substance_filter)] if substance_filter else df
    subs = data["Substância"].nunique()
    classifs = data["Model"].nunique()

    figsize = (2 * subs, 2 * subs if 2 * subs <= 10 else 10)
    fig, ax = plt.subplots(1, 1, figsize=figsize, sharex=True, sharey=True)
    ax.grid(True, axis="y")

    if lims is not None:
        ax.set_ylim(*lims)

    if refs is not None:
        for i in refs:
            plt.axhline(i, linestyle="--")

    plt.title(title)
    sns.boxplot(
        data=data,
        x="Substância",
        y="Score",
        hue="Model" if classifs > 1 else "Substância",
        ax=ax,
        dodge=classifs > 1,
    )

    if classifs == 1:
        ax.legend({})
        ax.set_xlabel("")

    return fig


def confusion(cm: pd.DataFrame) -> None:
    """Plota a matriz de confusão dado um dataframe com colunas "Model", "Substância" e colunas da matriz no formato "cm_i".

    Args:
        cm (pd.DataFrame): dataframe de matriz de confusão.
    """
    cms = cm.groupby(by=["Model", "Substância"], as_index=False).sum()

    models = np.unique(cms.Model)
    subs = np.unique(cms["Substância"])
    mat_cols = [c for c in cm.columns if c.startswith("cm_")]
    size = int(math.sqrt(len(mat_cols)))

    for m in models:
        fig, ax = plt.subplots(3, 6, figsize=(14, 6))
        fig.suptitle(m)
        for a, s in zip(ax.flatten(), subs):
            c = cms.loc[(cms["Model"] == m) & (cms["Substância"] == s), mat_cols]
            a.set_title(s)
            sns.heatmap(
                np.reshape(c.to_numpy(), (size, size)),
                ax=a,
                annot=True,
                fmt=".10g",
                cmap="viridis",
            )


def show_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Reformata o DataFrame para um formato mais legível.

    Args:
        df (pd.DataFrame): dataframe com modelos, substâncias e pontuações.

    Returns:
        pd.DataFrame: dataframe formatado com substâncias como colunas e (média, std) como subcolunas.
    """
    # Agrupa por classificador e substância e tira média e desvio padrão.
    g = df.groupby(by=["Model", "Substância"], as_index=False).agg(
        {"Score": ["mean", "std"]}
    )

    # Inverte a tabela de forma que as substância virem colunas
    g = g.pivot_table(
        values="Score", index="Model", columns="Substância", aggfunc="first"
    )

    # Coloca importância nas substâncias
    g = g.swaplevel(0, 1, axis=1)

    # Ordena para que média e desvio padrão fiquem lado a lado.
    g = g.sort_index(level=0, axis=1)

    return g


def threshold(x: int, t: list[int] = [2]) -> int:
    """Realiza a re-rotulação do label de acordo com o threshold dado. O threshold é feito com "<", ou seja, não inclusivo.

    Args:
        x (int): label
        t (list[int], optional): valores de threshold para cada classe (em ordem). Defaults to [2].

    Returns:
        int: rótulo novo.
    """
    for i, j in enumerate(t):
        if x < j:
            return i
    return i + 1
