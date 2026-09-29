"""Cache successful public data; errors propagate and remain retryable."""
import streamlit as st

from market_data import load_constituents, load_history


@st.cache_data(ttl=600, max_entries=400, show_spinner=False)
def get_history(ticker, kind="stock", provider="自动切换"):
    return load_history(ticker, kind, provider)


@st.cache_data(ttl=21600, max_entries=2, show_spinner=False)
def get_constituents():
    return load_constituents()


@st.cache_data(ttl=60, max_entries=16, show_spinner=False)
def get_index_history(ticker, provider="自动切换"):
    return load_history(ticker, "index", provider)


def clear_data_cache():
    get_history.clear()
    get_constituents.clear()
    get_index_history.clear()
