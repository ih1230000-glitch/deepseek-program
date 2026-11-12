import streamlit as st
from openai import OpenAI

# 페이지 기본 설정
st.set_page_config(page_title="DeepSeek AI 챗봇", page_icon="💬", layout="centered")
st.title("💬 DeepSeek AI 챗봇")
st.caption("이전 대화의 맥락과 지시사항을 기억하여 연속적인 대화를 수행합니다.")

# 1. 사이드바: API 및 대화 설정
st.sidebar.header("⚙️ 챗봇 설정")

provider = st.sidebar.radio(
    "API 발급 플랫폼 선택",
    ["텐센트 클라우드 (TokenHub)", "DeepSeek 공식 플랫폼"],
    help="발급받은 API Key의 사이트와 일치해야 404 오류가 발생하지 않습니다."
)

if provider == "텐센트 클라우드 (TokenHub)":
    base_url = "https://api.tencentcloud.com/tokenhub/v1"
    model_name = st.sidebar.selectbox(
        "모델 선택",
        ["deepseek-v4-flash", "deepseek-v3", "deepseek-r1"],
        index=0
    )
else:
    base_url = "https://api.deepseek.com"
    model_name = st.sidebar.selectbox(
        "모델 선택",
        ["deepseek-chat", "deepseek-reasoner"],
        index=0
    )

# Streamlit Secrets에서 API Key 자동 로드 (없으면 사이드바에서 직접 입력 가능)
default_key = (
    st.secrets.get("API_KEY", "")
    or st.secrets.get("TOKENHUB_API_KEY", "")
    or st.secrets.get("DEEPSEEK_API_KEY", "")
)
api_key = st.sidebar.text_input(
    "API Key (Secrets 미설정 시 직접 입력)",
    value=default_key,
    type="password"
)

# 대화 초기화 버튼
if st.sidebar.button("🗑️ 이전 대화 기억 지우기", use_container_width=True):
    st.session_state.messages = [
        {
            "role": "system",
            "content": "너는 사용자의 이전 대화 내용과 맥락, 앞서 내린 조건과 지시사항을 정확히 기억하고 일관되게 수행하는 유능한 AI 어시스턴트야."
        }
    ]
    st.rerun()

# 2. 대화 기록(세션 상태) 초기화 — 이 리스트에 모든 대화가 누적되어 기억됨
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": "너는 사용자의 이전 대화 내용과 맥락, 앞서 내린 조건과 지시사항을 정확히 기억하고 일관되게 수행하는 유능한 AI 어시스턴트야."
        }
    ]

# 3. 지금까지 나눈 이전 대화 화면에 출력
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# 4. 새 메시지 입력 및 AI 응답 생성
if user_input := st.chat_input("메시지를 입력하세요... (이전 대화를 이어서 질문할 수 있습니다)"):
    if not api_key:
        st.error("⚠️ 왼쪽 사이드바 또는 Streamlit Secrets에 API Key를 입력해 주세요.")
    else:
        # 사용자 질문을 대화 기록에 추가 (기억 저장)
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # 누적된 전체 대화 기록(st.session_state.messages)을 모델에 함께 전달
        try:
            client = OpenAI(api_key=api_key.strip(), base_url=base_url)
            with st.chat_message("assistant"):
                with st.spinner("이전 대화 내용을 바탕으로 답변을 생성 중입니다..."):
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=st.session_state.messages
                    )
                    reply = response.choices[0].message.content
                    st.markdown(reply)

            # AI 답변도 대화 기록에 추가 (다음 턴에 기억하도록 저장)
            st.session_state.messages.append({"role": "assistant", "content": reply})

        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")
