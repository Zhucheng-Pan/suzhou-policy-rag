import sys
from pathlib import Path
import streamlit as st
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from loader import load_and_split
from reranker import RerankRetriever
from generator import Generator

st.set_page_config(page_title="苏州人才政策问答", page_icon=":robot_face:", layout="wide")

@st.cache_resource(show_spinner = "正在加载模型和建立索引，请耐心等待...")

def load_system():
    chunks = load_and_split()
    retriever = RerankRetriever()
    retriever.build(chunks)
    generator = Generator()
    return chunks, retriever, generator

chunks, retriever, generator = load_system()

METHODS = {
    "混合检索 + 重排序（推荐）": lambda q, k: retriever.search(q, top_k=k),
    "混合检索": lambda q, k: retriever.hybrid.search(q, top_k=k),
    "向量检索": lambda q, k: retriever.hybrid.vector.search(q, top_k=k)
}

##### side bar #####

with st.sidebar:
    st.header("⚙ 设置")
    method = st.radio("检索方式", list(METHODS))
    top_k = st.slider("交给大模型的资料块数", min_value=3, max_value=10, value=5)
    st.divider()

    st.header("💡 知识库")
    sources = sorted({c["source"] for c in chunks})
    st.caption(f"{len(sources)} 份文件, {len(chunks)} 个文本块")
    for s in sources:
        st.markdown(f"- {s}")

    if st.button("清空对话"):
        st.session_state.messages = []
        st.rerun()

def show_sources(results):
    with st.expander(f"📎 参考来源（共 {len(results)} 个）"):
        for i, r in enumerate(results, start=1):
            st.markdown(f"**[{i}] {r['source']}** * 第{r['chunk_index']}块 * 分数{r['score']:.3f}")
            st.text(r["text"])

st.title("📄 苏州人才政策问答")
st.caption("基于苏州、南京、上海等地的人才政策文件，使用混合检索与重排序找到相关条文，由大模型生成带出处的回答。政策可能更新，请以官方最新文件为准。")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            show_sources(msg["sources"])

query = st.chat_input("请输入问题，按回车发送")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        try: 
            with st.spinner("正在检索，请稍候..."):
                results = METHODS[method](query, top_k)

            with st.spinner("正在生成答案，请稍候..."):
                answer = generator.generate_answer(query, results)

        except Exception as e:
            st.error(f"发生错误: {e}")
            st.stop()

        st.markdown(answer)
        show_sources(results)
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": results})