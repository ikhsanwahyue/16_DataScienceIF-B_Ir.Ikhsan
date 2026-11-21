"""Komponen tampilan bersama."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from settings import ACCENT, ALERT, NEUTRAL, ROOT


def _id_format(text):
    return text.replace(",", "\0").replace(".", ",").replace("\0", ".")


def fmt_num(x, decimals=2):
    return _id_format(f"{x:,.{decimals}f}")


def fmt_usd(x, decimals=2):
    return "$" + fmt_num(x, decimals)


def pretty_name(name):
    """Nama fitur dari get_feature_names_out() tanpa awalan transformer."""
    from schema import LABELS
    short = name.split("__", 1)[-1]
    return LABELS.get(short, short)


def missing_artifact(label, key, err=None):
    if err:
        st.error(f"{label} gagal dibaca dari `models/model.joblib`: {err}")
        return
    st.info(f"{label} belum ada di `models/model.joblib` (kunci `{key}`). Simpan dari sel terakhir main.ipynb.")


def missing_state(label, path, err=None):
    try:
        shown = path.relative_to(ROOT)
    except ValueError:
        shown = path
    if err:
        st.error(f"{label} gagal dibaca dari `{shown}`: {err}")
    else:
        st.info(f"{label} belum tersedia. Aplikasi mencarinya di `{shown}`.")


def style_fig(fig, height=380):
    fig.update_layout(
        height=height, template="plotly_white", margin=dict(l=10, r=10, t=40, b=10),
        colorway=[ACCENT, NEUTRAL, ALERT], font=dict(color="#1F2933"),
    )
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch", theme=None)


def empty_figure(message):
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False)
    return style_fig(fig, 200)