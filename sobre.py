"""Blocos de explicação (escopo e glossário) com componentes nativos, que seguem o tema do app."""
import streamlit as st


def _t(texto: str) -> str:
    return texto.replace("$", "\\$").replace("|", "\\|")


def escopo(problema: str, dentro: list[str], fora: list[str], titulo: str = "Sobre o projeto: problema, escopo e limites"):
    with st.expander(titulo, icon=":material/info:"):
        c1, c2, c3 = st.columns(3, gap="large")
        c1.markdown("**O problema**\n\n" + _t(problema))
        c2.markdown("**Dentro do escopo**\n\n" + "\n".join(f"- {_t(x)}" for x in dentro))
        c3.markdown("**Fora do escopo**\n\n" + "\n".join(f"- {_t(x)}" for x in fora))


def como_ler(itens: list[tuple[str, str, str]], titulo: str = "Como ler os resultados: o que cada valor significa"):
    with st.expander(titulo, icon=":material/menu_book:"):
        linhas = ["| Valor | Unidade | O que significa |", "|---|---|---|"]
        linhas += [f"| **{_t(n)}** | {_t(u)} | {_t(e)} |" for n, u, e in itens]
        st.markdown("\n".join(linhas))
