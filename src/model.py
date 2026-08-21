"""Treino e comparacao de profundidades da arvore de decisao."""

from sklearn.tree import DecisionTreeClassifier

MAX_DEPTH_PADRAO = 3
RANDOM_STATE = 42

PROFUNDIDADES_COMPARADAS = [1, 2, 3, 5, 10, None]


def criar_modelo(max_depth=MAX_DEPTH_PADRAO):
    return DecisionTreeClassifier(max_depth=max_depth, random_state=RANDOM_STATE)


def treinar(X_train, y_train, max_depth=MAX_DEPTH_PADRAO):
    return criar_modelo(max_depth).fit(X_train, y_train)


def comparar_profundidades(X_train, y_train, X_test, y_test):
    """Acuracia de treino/teste para cada max_depth. Replica o experimento do notebook."""
    resultados = []
    for depth in PROFUNDIDADES_COMPARADAS:
        candidato = criar_modelo(depth).fit(X_train, y_train)
        resultados.append(
            {
                "max_depth": "sem limite" if depth is None else depth,
                "profundidade_obtida": candidato.get_depth(),
                "folhas": candidato.get_n_leaves(),
                "acuracia_treino": candidato.score(X_train, y_train),
                "acuracia_teste": candidato.score(X_test, y_test),
            }
        )
    return resultados
