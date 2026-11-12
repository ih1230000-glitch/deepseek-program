import json
import urllib.request
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="OpenRouter", page_icon="💬", layout="centered")
st.title("💬 OpenRouter")
st.caption("이전 대화의 맥락과 지시사항을 기억하여 연속적인 대화를 수행합니다.")

def get_api_key():
    try:
        return (
            st.secrets.get("API_KEY", "")
            or st.secrets.get("OPENROUTER_API_KEY", "")
            or st.secrets.get("DEEPSEEK_API_KEY", "")
        )
    except Exception:
        return ""

# 기본값인 openrouter/free를 맨 앞에 두고, 현재 무료로 제공 중인 모델을 함께 조회
@st.cache_data(ttl=600, show_spinner=False)
def fetch_active_free_models():
    try:
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/models",
            headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=5) as res:
            models = json.loads(res.read().decode("utf-8")).get("data", [])

        free_models = []
        for m in models:
            mid = m.get("id", "")
            pricing = m.get("pricing", {})
            is_free = mid.endswith(":free") or (
                str(pricing.get("prompt", "1")) == "0"
                and str(pricing.get("completion", "1")) == "0"
            )
            if "deepseek" in mid.lower() and is_free and mid != "openrouter/free":
                free_models.append(mid)

        return ["openrouter/free"] + free_models
    except Exception:
        return ["openrouter/free", "deepseek/deepseek-r1:free"]

default_key = get_api_key()
available_models = fetch_active_free_models()

with st.expander("⚙️ 모델 및 설정 (클릭해서 열기/닫기)", expanded=True):
    api_key = st.text_input(
        "OpenRouter API Key",
        value=default_key,
        type="password",
        help="Streamlit Secrets에 등록했다면 자동으로 입력됩니다."
    )

    model_name = st.selectbox(
        "현재 사용 가능한 무료 모델 선택",
        available_models,
        index=0,
        help="기본값으로 openrouter/free가 선택되며, 목록에서 다른 무료 모델로 변경할 수 있습니다."
    )

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

# 지금까지 나눈 대화 화면에 출력
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(
