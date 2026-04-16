import streamlit as st

from common.utils.utils import load_config
from pages import monitor, collection, inference


st.set_page_config(page_title="M.AX", layout="wide")


@st.cache_resource
def load_cfg():
    return load_config(fname="config/server_config.yaml", project="gt_kitting")


# cfg를 session_state에 저장해 각 페이지 함수에서 참조
if "cfg" not in st.session_state:
    st.session_state.cfg = load_cfg()


def page_monitor():
    monitor.show(st.session_state.cfg)


def page_collection():
    collection.show(st.session_state.cfg)


def page_inference():
    inference.show(st.session_state.cfg)


pg = st.navigation({
    "Monitor": [
        st.Page(page_monitor,    title="Monitor",         icon="📷"),
    ],
    "Operation": [
        st.Page(page_collection, title="Data Collection", icon="💾"),
        st.Page(page_inference,  title="Inference",       icon="🤖"),
    ],
})

pg.run()
