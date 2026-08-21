"""Traduz o caminho percorrido na arvore em passos legiveis.

sklearn marca folhas com feature == -2 (TREE_UNDEFINED). Percorrer
tree_.feature/threshold/children_* manualmente reproduz exatamente o que
model.predict() faz internamente, mas devolve cada pergunta no caminho
em vez de so a resposta final.
"""

import numpy as np

FOLHA = -2


def caminho_decisao(modelo, amostra, feature_names):
    """amostra: array 1D com os atributos, na ordem de feature_names."""
    arvore = modelo.tree_
    no = 0
    passos = []

    while arvore.feature[no] != FOLHA:
        idx_atributo = arvore.feature[no]
        limite = arvore.threshold[no]
        valor = amostra[idx_atributo]
        vai_esquerda = valor <= limite

        passos.append(
            {
                "atributo": feature_names[idx_atributo],
                "valor": float(valor),
                "limite": float(limite),
                "comparacao": "<=" if vai_esquerda else ">",
            }
        )
        no = arvore.children_left[no] if vai_esquerda else arvore.children_right[no]

    contagem_classes = arvore.value[no][0]
    classe = modelo.classes_[int(np.argmax(contagem_classes))]
    pureza_no = 1 - sum((c / contagem_classes.sum()) ** 2 for c in contagem_classes)
    return passos, classe, pureza_no


def passos_para_frase(passos, classe):
    """'Se condutividade > 42.30, entao o material previsto e metal.' """
    if not passos:
        return f"A raiz ja e uma folha: o material previsto e {classe}."

    condicoes = " e ".join(
        f"{p['atributo']} {p['comparacao']} {p['limite']:.2f}" for p in passos
    )
    return f"Se {condicoes}, entao o material previsto e {classe}."
