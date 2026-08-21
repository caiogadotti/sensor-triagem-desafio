"""Carrega o dataset autoral do desafio (triagem de residuos) e separa treino/teste.

51 itens sinteticos gerados por script proprio (numpy.random.default_rng),
17 por classe, com quatro leituras numericas cada: peso, densidade,
condutividade eletrica e opacidade visual.
"""

import os

import pandas as pd
from sklearn.model_selection import train_test_split

CAMINHO_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset_desafio.csv")
COLUNA_ALVO = "classe"
TEST_SIZE = 0.25
RANDOM_STATE = 42


def carregar():
    """Retorna (X, y, feature_names, target_names) do dataset completo."""
    dados = pd.read_csv(CAMINHO_CSV)
    X = dados.drop(columns=COLUNA_ALVO)
    y = dados[COLUNA_ALVO]
    feature_names = list(X.columns)
    target_names = sorted(y.unique())
    return X, y, feature_names, target_names


def carregar_treino_teste():
    """Split estratificado, reproduzivel (mesma semente do notebook do desafio)."""
    X, y, feature_names, target_names = carregar()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test, feature_names, target_names
