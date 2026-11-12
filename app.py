import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="DeepSeek AI 챗봇", page_icon="💬", layout="centered")
st.title("💬 DeepSeek AI 챗봇")
st.caption("이전 대화의 맥락과 지시사항을 기억하여 연속적인 대화를 수행합니다.")

BASE_URL = "https://tokenhub-intl.tencentcloudmaas.com/v1"

# Streamlit Secrets에서 API Key 자동 로드
default_key = (
    st.secrets.get("API_KEY", "")
    or st.secrets.get("TOKENHUB_API_KEY", "")
    or st.secrets.get("DEEPSEEK_API_KEY", "")
)

@st.cache_data(ttl=300, show_spinner=False)
def fetch_available_models(key: str):
    if not key:
        return []
    try:
        temp_client = OpenAI(api_key=key.strip(), base_url=BASE_URL)
        models = temp_client.models.list()
        return [m.id for m in models.data]
    except Exception:
        return []

with st.expander("⚙️ 모델 및 API 설정 (모델 ID 선택/변경)", expanded=True):
    api_key = st.text_input(
        "API Key",
        value=default_key,
        type="password",
        help="Streamlit Secrets에 등록되어 있으면 자동으로 입력됩니다."
    )

    fetched_models = fetch_available_models(api_key)
    fallback_models = [
        "deepseek-v3",
        "deepseek-r1",
        "deepseek-v3-0324",
        "deepseek-r1-0528",
    ]
    model_choices = (fetched_models if fetched_models else fallback_models) + ["직접 입력"]

    if fetched_models:
        st.success(f"✅ 계정에서 사용 가능한 모델 {len(fetched_models)}개를 자동으로 불러왔습니다.")

    selected_model = st.selectbox("사용할 모델 ID 선택", model_choices, index=0)
    if selected_model == "직접 입력":
        model_name = st.text_input("콘솔의 Model / Service ID 직접 입력", placeholder="예: deepseek-v3")
    else:
        model_name = selected_model

    if st.button("🗑️ 이전 대화 기억 지우기", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "system",
                "content": "너는 사용자의 이전 대화 내용과 맥락, 앞서 내린 조건과 지시사항을 정확히 기억하고 일관되게 수행하는 유능한 AI 어시스턴트야."
            }
        ]
        st.rerun()

# 이전 대화 기록(세션 상태) 초기화
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": "너는 사용자의 이전 대화 내용과 맥락, 앞서 내린 조건과 지시사항을 정확히 기억하고 일관되게 수행하는 유능한 AI 어시스턴트야."
        }
    ]

# 지금까지 나눈 대화 내역 화면에 출력
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# 새 메시지 입력 및 AI 응답 생성
if user_input := st.chat_input("메시지를 입력하세요... (이전 대화를 이어서 질문할 수 있습니다)"):
    if not api_key:
        st.error("⚠️ Streamlit Secrets 또는 상단 설정창에 API Key를 입력해 주세요.")
    elif not model_name:
        st.error("⚠️ 사용할 모델 ID를 선택하거나 입력해 주세요.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        try:
            client = OpenAI(
                api_key=api_key.strip(),
                base_url=BASE_URL
            )
            with st.chat_message("assistant"):
                with st.spinner("이전 대화 내용을 바탕으로 답변을 생성 중입니다..."):
                    response = client.chat.completions.create(
                        model=model_name.strip(),
                        messages=st.session_state.messages
                    )
                    reply = response.choices[0].message.content
                    st.markdown(reply)

            st.session_state.messages.append({"role": "assistant", "content": reply})

        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")
