"""Sensor de triagem de residuos com arvore de decisao -- interface Streamlit.

Desafio autoral (Aula 3, LCML): classifica residuos em papel, plastico ou
metal a partir de quatro leituras de sensor (peso, densidade, condutividade,
opacidade).

Rodar:
    streamlit run app.py
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score
from sklearn.tree import export_text, plot_tree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset import carregar_treino_teste
from src.explicacao import caminho_decisao, passos_para_frase
from src.model import MAX_DEPTH_PADRAO, comparar_profundidades, treinar

st.set_page_config(
    page_title="Sensor de Triagem",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ESTILO = """
<style>
:root {
    --fundo: #17140f;
    --superficie: #1f1b14;
    --superficie-alta: #2a251b;
    --borda: #3a3327;
    --texto: #f5f1e6;
    --texto-suave: #b8ae98;
    --texto-fraco: #746b58;
    --destaque: #e8834a;
    --destaque-claro: #f4ad82;
    --metal: #9fb8d9;
    --papel: #d9b877;
    --plastico: #5fd0c0;
}

.stApp { background: var(--fundo); }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2.2rem 3rem 4rem; max-width: 1240px; }

.marca {
    display: block;
    color: var(--destaque); font-size: 0.7rem; font-weight: 700;
    letter-spacing: 0.16em; text-transform: uppercase; margin-bottom: 0.7rem;
}

h1.titulo {
    color: var(--texto); font-size: 2.7rem; font-weight: 700;
    letter-spacing: -0.035em; line-height: 1.05; margin: 0 0 0.6rem;
}
.chamada { color: var(--texto-suave); font-size: 1rem; line-height: 1.6; max-width: 66ch; }
.chamada strong { color: var(--texto); font-weight: 600; }

.faixa-metricas { display: flex; gap: 2.4rem; margin: 1.8rem 0 2.2rem; flex-wrap: wrap; }
.metrica-valor {
    font-family: ui-monospace, "Cascadia Code", Consolas, monospace;
    color: var(--destaque); font-size: 1.6rem; font-weight: 600; line-height: 1;
}
.metrica-rotulo {
    color: var(--texto-fraco); font-size: 0.7rem; text-transform: uppercase;
    letter-spacing: 0.1em; margin-top: 0.35rem;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--superficie) !important; border: 1px solid var(--borda) !important;
    border-radius: 14px !important;
}
.painel-titulo {
    color: var(--texto-fraco); font-size: 0.7rem; font-weight: 700;
    letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.9rem;
}

.especie {
    font-size: 3rem; font-weight: 700; line-height: 1; text-align: center;
    color: var(--destaque); text-shadow: 0 0 60px rgba(232,131,74,0.28);
    margin: 0.6rem 0 0.2rem; text-transform: capitalize;
}
.especie-rotulo {
    text-align: center; color: var(--texto-fraco); font-size: 0.7rem;
    text-transform: uppercase; letter-spacing: 0.14em; margin-bottom: 1.4rem;
}

.barra-linha { display: flex; align-items: center; gap: 0.8rem; margin-bottom: 0.5rem; }
.barra-nome {
    width: 6.5rem; text-align: right;
    color: var(--texto); font-weight: 600; font-size: 0.82rem; text-transform: capitalize;
}
.barra-trilho {
    flex: 1; height: 8px; background: var(--superficie-alta);
    border-radius: 99px; overflow: hidden;
}
.barra-preenchida {
    height: 100%; border-radius: 99px;
    background: linear-gradient(90deg, var(--destaque), var(--destaque-claro));
}
.barra-preenchida.secundaria { background: var(--borda); }
.barra-pct {
    font-family: ui-monospace, monospace; width: 3rem; text-align: right;
    color: var(--texto-suave); font-size: 0.78rem;
}

.frase-regra {
    background: var(--superficie-alta); border: 1px solid var(--borda);
    border-radius: 10px; padding: 0.9rem 1.1rem; margin-top: 1.2rem;
    color: var(--texto-suave); font-size: 0.85rem; line-height: 1.55;
}
.frase-regra strong { color: var(--destaque); }

.passo-linha {
    display: flex; justify-content: space-between; gap: 0.6rem;
    padding: 0.4rem 0; border-bottom: 1px dashed var(--borda);
    font-size: 0.82rem;
}
.passo-linha:last-child { border-bottom: none; }
.passo-atributo { color: var(--texto); }
.passo-condicao { color: var(--texto-fraco); font-family: ui-monospace, monospace; }

.nota { color: var(--texto-fraco); font-size: 0.78rem; line-height: 1.6; }

div[data-testid="stExpander"] {
    border: 1px solid var(--borda); border-radius: 12px; background: var(--superficie);
}
div[data-testid="stExpander"] summary { color: var(--texto-suave); font-size: 0.85rem; }

.stSlider label p { color: var(--texto-suave) !important; font-size: 0.85rem !important; }

.rodape {
    color: var(--texto-fraco); font-size: 0.72rem; text-align: center;
    margin-top: 3rem; padding-top: 1.6rem; border-top: 1px solid var(--borda);
}
</style>
"""

st.markdown(ESTILO, unsafe_allow_html=True)

CORES_CLASSE = {"metal": "#9fb8d9", "papel": "#d9b877", "plastico": "#5fd0c0"}
ROTULOS_ATRIBUTO = {
    "peso_g": "peso (g)",
    "densidade_g_cm3": "densidade (g/cm³)",
    "condutividade": "condutividade",
    "opacidade_pct": "opacidade (%)",
}


@st.cache_resource(show_spinner="Treinando a arvore no dataset do desafio...")
def preparar():
    X_train, X_test, y_train, y_test, feature_names, target_names = carregar_treino_teste()
    modelo = treinar(X_train, y_train)
    acuracia = accuracy_score(y_test, modelo.predict(X_test))
    comparacao = comparar_profundidades(X_train, y_train, X_test, y_test)
    return modelo, X_train, y_train, X_test, y_test, feature_names, target_names, acuracia, comparacao


modelo, X_train, y_train, X_test, y_test, feature_names, target_names, acuracia, comparacao = preparar()


def serializar_arvore(modelo, feature_names):
    """Estrutura da arvore como lista de nos, para o mesmo percurso rodar em JS."""
    arvore = modelo.tree_
    nos = []
    for i in range(arvore.node_count):
        if arvore.feature[i] == -2:
            contagem = arvore.value[i][0]
            classe = modelo.classes_[int(np.argmax(contagem))]
            nos.append({"folha": True, "classe": str(classe)})
        else:
            nos.append(
                {
                    "folha": False,
                    "atributo": feature_names[arvore.feature[i]],
                    "limite": round(float(arvore.threshold[i]), 2),
                    "esquerda": int(arvore.children_left[i]),
                    "direita": int(arvore.children_right[i]),
                }
            )
    return nos


def num(valor, casas=0):
    texto = f"{valor:,.{casas}f}"
    return texto.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


st.markdown('<div class="marca">LCML · Desafio autoral, Aula 3</div>', unsafe_allow_html=True)
st.markdown('<h1 class="titulo">Sensor de<br>triagem de resíduos</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="chamada"><strong>Arraste um item até o sensor</strong> e veja a árvore de '
    "decisão classificar o material em tempo real, usando as mesmas quatro leituras "
    "(peso, densidade, condutividade, opacidade) e o mesmo modelo treinado em "
    "<code>dataset_desafio.csv</code>, 51 itens sintéticos, 17 por classe.</p>",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="faixa-metricas">
        <div>
            <div class="metrica-valor">{num(acuracia * 100, 1)}%</div>
            <div class="metrica-rotulo">acurácia no teste</div>
        </div>
        <div>
            <div class="metrica-valor">{num(len(X_train))}</div>
            <div class="metrica-rotulo">itens no treino</div>
        </div>
        <div>
            <div class="metrica-valor">{len(feature_names)}</div>
            <div class="metrica-rotulo">leituras de sensor</div>
        </div>
        <div>
            <div class="metrica-valor">max_depth={MAX_DEPTH_PADRAO}</div>
            <div class="metrica-rotulo">profundidade máxima</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Zona de arrastar e soltar -------------------------------------------

ITENS_ARRASTAVEIS = [
    {"id": "metal1", "emoji": "\U0001f96b", "rotulo": "Lata de alumínio",
     "classe_real": "metal", "amostra": [38.8, 9.78, 87.9, 98.6]},
    {"id": "papel1", "emoji": "\U0001f4c4", "rotulo": "Folha de papel",
     "classe_real": "papel", "amostra": [15.3, 0.81, 6.6, 89.7]},
    {"id": "plastico1", "emoji": "\U0001f964", "rotulo": "Copo plástico",
     "classe_real": "plastico", "amostra": [17.1, 0.82, 6.9, 40.1]},
    {"id": "papel2", "emoji": "\U0001f9fb", "rotulo": "Guardanapo usado",
     "classe_real": "papel", "amostra": [10.7, 0.83, 4.7, 71.5]},
]

TREE_NODES = serializar_arvore(modelo, feature_names)

componente_html = f"""
<style>
  * {{ box-sizing: border-box; font-family: 'Segoe UI', system-ui, sans-serif; }}
  .wrap {{ padding: 4px 2px 2px; }}
  .fila {{ display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 18px; }}
  .item {{
    flex: 1 1 140px; cursor: grab; user-select: none;
    background: #1f1b14; border: 1px solid #3a3327; border-radius: 12px;
    padding: 14px 10px; text-align: center; transition: all .15s ease;
  }}
  .item:hover {{ border-color: #e8834a; transform: translateY(-2px); }}
  .item.selecionado {{ border-color: #e8834a; box-shadow: 0 0 0 2px rgba(232,131,74,.35); }}
  .item .emoji {{ font-size: 2.1rem; display: block; margin-bottom: 6px; }}
  .item .rotulo {{ color: #f5f1e6; font-size: 0.78rem; font-weight: 600; }}
  .zona {{
    border: 2px dashed #3a3327; border-radius: 16px; padding: 26px 18px;
    text-align: center; transition: all .18s ease; background: #17140f;
  }}
  .zona.sobre {{ border-color: #e8834a; background: #2a251b; }}
  .zona .rotulo-zona {{
    color: #746b58; font-size: 0.7rem; text-transform: uppercase;
    letter-spacing: 0.12em; margin-bottom: 6px;
  }}
  .zona .instrucao {{ color: #b8ae98; font-size: 0.85rem; }}
  .resultado {{ display: none; margin-top: 16px; }}
  .resultado.ativo {{ display: block; }}
  .resultado .classe {{
    font-size: 2.1rem; font-weight: 700; text-align: center;
    text-transform: capitalize; margin: 4px 0 2px;
  }}
  .resultado .conferencia {{ text-align: center; font-size: 0.78rem; margin-bottom: 12px; }}
  .resultado .conferencia.ok {{ color: #5fd0c0; }}
  .resultado .conferencia.erro {{ color: #e8834a; }}
  .caminho {{
    background: #1f1b14; border: 1px solid #3a3327; border-radius: 10px;
    padding: 10px 14px; font-size: 0.8rem; color: #b8ae98; line-height: 1.7;
  }}
  .caminho b {{ color: #f5f1e6; }}
  .toque-nota {{ color: #746b58; font-size: 0.72rem; text-align: center; margin-top: 10px; }}
</style>
<div class="wrap">
  <div class="fila" id="fila"></div>
  <div class="zona" id="zona">
    <div class="rotulo-zona">Sensor</div>
    <div class="instrucao" id="instrucao">Arraste um item até aqui (ou toque no item e depois aqui)</div>
    <div class="resultado" id="resultado">
      <div class="classe" id="classe-prevista"></div>
      <div class="conferencia" id="conferencia"></div>
      <div class="caminho" id="caminho"></div>
    </div>
  </div>
  <div class="toque-nota">Em celular/tablet: toque no item para selecionar, depois toque no sensor.</div>
</div>
<script>
  const ITENS = {json.dumps(ITENS_ARRASTAVEIS)};
  const ARVORE = {json.dumps(TREE_NODES)};
  const CORES = {json.dumps(CORES_CLASSE)};
  const ROTULOS_ATRIBUTO = {json.dumps(ROTULOS_ATRIBUTO)};

  function classificar(amostra) {{
    let no = 0;
    const passos = [];
    while (!ARVORE[no].folha) {{
      const n = ARVORE[no];
      const idx = {json.dumps(feature_names)}.indexOf(n.atributo);
      const valor = amostra[idx];
      const vaiEsquerda = valor <= n.limite;
      passos.push({{atributo: n.atributo, valor: valor, limite: n.limite, comparacao: vaiEsquerda ? '<=' : '>'}});
      no = vaiEsquerda ? n.esquerda : n.direita;
    }}
    return {{classe: ARVORE[no].classe, passos: passos}};
  }}

  const fila = document.getElementById('fila');
  const zona = document.getElementById('zona');
  const instrucao = document.getElementById('instrucao');
  const resultado = document.getElementById('resultado');
  let selecionado = null;

  ITENS.forEach(function(item) {{
    const el = document.createElement('div');
    el.className = 'item';
    el.draggable = true;
    el.id = 'card-' + item.id;
    el.innerHTML = '<span class="emoji">' + item.emoji + '</span><span class="rotulo">' + item.rotulo + '</span>';
    el.addEventListener('dragstart', function(e) {{
      e.dataTransfer.setData('text/plain', item.id);
    }});
    el.addEventListener('click', function() {{
      document.querySelectorAll('.item').forEach(function(c) {{ c.classList.remove('selecionado'); }});
      el.classList.add('selecionado');
      selecionado = item.id;
    }});
    fila.appendChild(el);
  }});

  function soltar(itemId) {{
    const item = ITENS.find(function(i) {{ return i.id === itemId; }});
    if (!item) return;
    const r = classificar(item.amostra);
    instrucao.style.display = 'none';
    resultado.classList.add('ativo');
    const cor = CORES[r.classe];
    document.getElementById('classe-prevista').textContent = item.emoji + ' ' + r.classe;
    document.getElementById('classe-prevista').style.color = cor;
    document.getElementById('classe-prevista').style.textShadow = '0 0 40px ' + cor + '55';

    const conf = document.getElementById('conferencia');
    if (r.classe === item.classe_real) {{
      conf.textContent = '✓ previsto pela árvore = classe real (' + item.classe_real + ')';
      conf.className = 'conferencia ok';
    }} else {{
      conf.textContent = '✗ árvore previu ' + r.classe + ', classe real era ' + item.classe_real;
      conf.className = 'conferencia erro';
    }}

    const linhas = r.passos.map(function(p, i) {{
      const rot = ROTULOS_ATRIBUTO[p.atributo] || p.atributo;
      return (i + 1) + '. ' + rot + ' = <b>' + p.valor.toFixed(1) + '</b> ' + p.comparacao + ' ' + p.limite.toFixed(2);
    }});
    document.getElementById('caminho').innerHTML = linhas.length ? linhas.join('<br>') : 'A raiz já é uma folha.';

    zona.style.transform = 'scale(1.015)';
    setTimeout(function() {{ zona.style.transform = 'scale(1)'; }}, 160);
  }}

  zona.addEventListener('dragover', function(e) {{ e.preventDefault(); zona.classList.add('sobre'); }});
  zona.addEventListener('dragleave', function() {{ zona.classList.remove('sobre'); }});
  zona.addEventListener('drop', function(e) {{
    e.preventDefault();
    zona.classList.remove('sobre');
    const itemId = e.dataTransfer.getData('text/plain');
    soltar(itemId);
  }});
  zona.addEventListener('click', function() {{
    if (selecionado) soltar(selecionado);
  }});
</script>
"""

st.markdown('<div class="painel-titulo">Arraste até o sensor</div>', unsafe_allow_html=True)
st.iframe(componente_html, height="content")

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

# --- Modo manual: sliders --------------------------------------------------

st.markdown('<div class="painel-titulo">Modo manual: ajuste as leituras do sensor</div>', unsafe_allow_html=True)

coluna_entrada, coluna_saida = st.columns([1, 1], gap="large")

with coluna_entrada:
    with st.container(border=True):
        valores = {}
        for atributo in feature_names:
            coluna = X_train[atributo]
            valores[atributo] = st.slider(
                ROTULOS_ATRIBUTO.get(atributo, atributo),
                min_value=float(coluna.min()),
                max_value=float(coluna.max()),
                value=float(coluna.mean()),
                step=0.1,
            )

amostra = np.array([valores[atributo] for atributo in feature_names])
vetor = pd.DataFrame([amostra], columns=feature_names)

previsto = modelo.predict(vetor)[0]
probabilidades = modelo.predict_proba(vetor)[0]
passos, classe_prevista, pureza_folha = caminho_decisao(modelo, amostra, feature_names)

with coluna_saida:
    with st.container(border=True):
        st.markdown(f'<div class="especie">{previsto}</div>', unsafe_allow_html=True)
        st.markdown('<div class="especie-rotulo">material previsto</div>', unsafe_allow_html=True)

        classes_ordenadas = list(modelo.classes_)
        ordem = sorted(
            range(len(classes_ordenadas)),
            key=lambda c: (-probabilidades[c], classes_ordenadas[c] != previsto),
        )
        for posicao, classe_idx in enumerate(ordem):
            pct = probabilidades[classe_idx] * 100
            estilo = "" if posicao == 0 else " secundaria"
            st.markdown(
                f'<div class="barra-linha">'
                f'<div class="barra-nome">{classes_ordenadas[classe_idx]}</div>'
                f'<div class="barra-trilho"><div class="barra-preenchida{estilo}" '
                f'style="width:{max(pct, 1.5)}%"></div></div>'
                f'<div class="barra-pct">{pct:.0f}%</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown(
            f'<div class="frase-regra">{passos_para_frase(passos, classe_prevista)}<br>'
            f'Impureza de Gini na folha: <strong>{pureza_folha:.2f}</strong></div>',
            unsafe_allow_html=True,
        )

st.markdown('<div class="painel-titulo" style="margin-top:1.8rem;">Caminho percorrido na árvore</div>', unsafe_allow_html=True)
with st.container(border=True):
    if passos:
        for i, passo in enumerate(passos, start=1):
            st.markdown(
                f'<div class="passo-linha">'
                f'<span class="passo-atributo">{i}. {ROTULOS_ATRIBUTO.get(passo["atributo"], passo["atributo"])}</span>'
                f'<span class="passo-condicao">valor = {passo["valor"]:.2f} {passo["comparacao"]} {passo["limite"]:.2f}</span>'
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown('<p class="nota">A raiz já é uma folha para esta configuração de profundidade.</p>', unsafe_allow_html=True)

with st.expander("Ver a árvore inteira e todas as regras"):
    plt.rcParams.update(
        {
            "figure.facecolor": "#1f1b14",
            "axes.facecolor": "#1f1b14",
        }
    )
    fig, ax = plt.subplots(figsize=(16, 7))
    plot_tree(
        modelo,
        feature_names=feature_names,
        class_names=list(modelo.classes_),
        filled=True,
        rounded=True,
        fontsize=9,
        ax=ax,
    )
    fig.patch.set_facecolor("#1f1b14")
    st.pyplot(fig, clear_figure=True)

    st.markdown('<p class="nota" style="margin-top:0.8rem;">Regras em texto (mesma árvore):</p>', unsafe_allow_html=True)
    st.code(export_text(modelo, feature_names=feature_names), language=None)

with st.expander("Profundidade vs. desempenho: por que max_depth=3"):
    st.markdown(
        '<p class="nota">O mesmo experimento do notebook: treinar a árvore com '
        "profundidades diferentes e comparar acurácia de treino e teste. Uma árvore "
        "sem limite pode memorizar o treino (100%) e ainda assim generalizar pior "
        "que uma árvore mais rasa, sinal de overfitting.</p>",
        unsafe_allow_html=True,
    )
    linhas = "".join(
        f'<tr><td>{r["max_depth"]}</td><td>{r["profundidade_obtida"]}</td>'
        f'<td>{r["folhas"]}</td><td>{r["acuracia_treino"]:.1%}</td>'
        f'<td>{r["acuracia_teste"]:.1%}</td></tr>'
        for r in comparacao
    )
    st.markdown(
        f"""
        <table style="width:100%; border-collapse:collapse; font-size:0.82rem; color:var(--texto-suave);">
        <thead style="color:var(--texto-fraco); text-transform:uppercase; font-size:0.68rem; letter-spacing:0.08em;">
        <tr><th style="text-align:left; padding:0.4rem 0;">max_depth</th><th>profundidade obtida</th>
        <th>folhas</th><th>acurácia treino</th><th>acurácia teste</th></tr>
        </thead>
        <tbody>{linhas}</tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="rodape">Sensor de triagem · árvore de decisão · scikit-learn · Streamlit &nbsp;|&nbsp; '
    "desafio autoral, LCML, Aula 3</div>",
    unsafe_allow_html=True,
)
