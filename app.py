import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="DeepSeek AI 챗봇", page_icon="💬", layout="centered")
st.title("💬 DeepSeek AI 챗봇")
st.caption("이전 대화의 맥락과 지시사항을 기억하여 연속적인 대화를 수행합니다.")

# Streamlit Secrets에서 API Key 자동 로드
def get_api_key():
    try:
        return (
            st.secrets.get("API_KEY", "")
            or st.secrets.get("OPENROUTER_API_KEY", "")
            or st.secrets.get("DEEPSEEK_API_KEY", "")
        )
    except Exception:
        return ""

default_key = get_api_key()

with st.expander("⚙️ 모델 및 설정 (클릭해서 열기/닫기)", expanded=False):
    api_key = st.text_input(
        "OpenRouter API Key",
        value=default_key,
        type="password",
        help="Streamlit Secrets에 등록했다면 자동으로 입력됩니다."
    )

    model_name = st.selectbox(
        "DeepSeek 무료 모델 선택",
        [
            "deepseek/deepseek-r1:free",
            "deepseek/deepseek-chat-v3-0324:free",
            "openrouter/free"
        ],
        index=0,
        help="무료 서버에 사용자가 몰려 일시적으로 응답이 지연되면 다른 모델로 바꿔보세요."
    )

    if st.button("🗑️ 이전 대화 기억 지우기", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "system",
                "content": "너는 사용자의 이전 대화 내용과 맥락, 앞서 내린 조건과 지시사항을 정확히 기억하고 일관되게 수행하는 유능한 AI 어시스턴트야."
            }
        ]
        st.rerun()

# 이전 대화 기록(세션 상태) 초기화 — 모든 대화가 여기에 누적되어 기억됨
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": "너는 사용자의 이전 대화 내용과 맥락, 앞서 내린 조건과 지시사항을 정확히 기억하고 일관되게 수행하는 유능한 AI 어시스턴트야."
        }
    ]

# 지금까지 나눈 대화 화면에 출력
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# 새 메시지 입력 및 AI 응답 생성
if user_input := st.chat_input("메시지를 입력하세요... (이전 대화를 이어서 질문할 수 있습니다)"):
    if not api_key:
        st.error("⚠️ Streamlit Secrets 또는 상단 설정창에 OpenRouter API Key를 입력해 주세요.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        try:
            client = OpenAI(
                api_key=api_key.strip(),
                base_url="https://openrouter.ai/api/v1",
                default_headers={
                    "HTTP-Referer": "https://streamlit.app",
                    "X-Title": "DeepSeek Streamlit Chatbot"
                }
            )
            with st.chat_message("assistant"):
                with st.spinner("이전 대화 내용을 바탕으로 답변을 생성 중입니다..."):
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=st.session_state.messages
                    )
                    reply = response.choices[0].message.content
                    st.markdown(reply)

            st.session_state.messages.append({"role": "assistant", "content": reply})

        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")
