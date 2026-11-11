import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from openai import OpenAI

st.set_page_config(page_title="인공신경망 & DeepSeek AI 플랫폼", page_icon="🤖", layout="wide")

# 챗봇 세션 상태 초기화 (이전 대화 맥락 기억)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "너는 데이터 분석과 인공신경망(ANN) 학습을 보조하는 친절한 AI 챗봇이야. 이전 대화 내용을 기억하고 피드백을 줘."}
    ]

tab1, tab2 = st.tabs(["📊 인공신경망 & 회귀 예측기", "💬 DeepSeek AI 챗봇"])

# ==========================================
# TAB 1: 데이터 예측 플랫폼
# ==========================================
with tab1:
    st.title("🧠 인공신경망(ANN) 및 회귀 데이터 예측 플랫폼")
    st.markdown("오픈소스 CSV 데이터를 업로드하거나 기본 샘플 데이터를 활용하여 모델을 학습하고 시각화합니다.")
    
    st.sidebar.header("🛠️ 1. 데이터 관리")
    uploaded_file = st.sidebar.file_uploader("CSV 파일 업로드 (선택사항)", type=["csv"], key="ann_file")
    
    if uploaded_file is None:
        st.sidebar.info("💡 기본 샘플 데이터가 자동 적용되어 파일 업로드 없이도 즉시 테스트할 수 있습니다.")
        sample_data = """X,Y
1.0,2.1
2.0,3.8
3.0,7.9
4.0,15.2
5.0,24.8
6.0,36.1
7.0,49.5
8.0,64.2
9.0,81.0
10.0,100.5
11.0,121.2
12.0,143.8
13.0,168.1
14.0,196.0
15.0,225.2"""
        df = pd.read_csv(io.StringIO(sample_data))
    else:
        df = pd.read_csv(uploaded_file)
        
    st.subheader("📋 데이터셋 미리보기")
    st.dataframe(df.head(), use_container_width=True)
    
    columns = df.columns.tolist()
    
    st.sidebar.header("📊 2. 모델 & 하이퍼파라미터")
    x_col = st.sidebar.selectbox("X축 독립 변수", columns, index=0)
    y_col = st.sidebar.selectbox("Y축 종속 변수", columns, index=len(columns)-1 if len(columns)>1 else 0)
    
    model_choice = st.sidebar.selectbox(
        "분석 모델 선택",
        ["1. 선형 회귀 (y = ax + b)", "2. 2차 비선형 회귀 (y = w2·x² + w1·x + b)", "3. 인공신경망 (ANN - 다층 퍼셉트론)"]
    )
    
    learning_rate = st.sidebar.number_input("학습률 (Learning Rate)", value=0.01, format="%.4f")
    epochs = st.sidebar.slider("학습 횟수 (Epochs)", 100, 10000, 2000, step=100)
    
    if "인공신경망" in model_choice:
        activation_name = st.sidebar.selectbox("활성화 함수 (Activation)", ["Sigmoid", "ReLU", "Tanh"])
        hidden_units = st.sidebar.slider("은닉층 뉴런 수 (Hidden Neurons)", 1, 16, 4, step=1)
        
    try:
        temp_df = df[[x_col, y_col]].copy()
        temp_df[x_col] = pd.to_numeric(temp_df[x_col], errors='coerce')
        temp_df[y_col] = pd.to_numeric(temp_df[y_col], errors='coerce')
        temp_df = temp_df.dropna()
        
        X = temp_df[x_col].values.astype(float)
        y_true = temp_df[y_col].values.astype(float)
        N = len(X)
        
        if N < 2:
            st.warning("⚠️ 최소 2개 이상의 유효한 숫자 데이터가 필요합니다.")
        else:
            loss_history = []
            
            if "1. 선형" in model_choice:
                w, b = 0.0, 0.0
                for _ in range(epochs):
                    y_pred = w * X + b
                    error = y_true - y_pred
                    loss = np.mean(error ** 2)
                    loss_history.append(loss)
                    
                    dw = (-2 / N) * np.sum(X * error)
                    db = (-2 / N) * np.sum(error)
                    w -= learning_rate * dw
                    b -= learning_rate * db
                    
                x_dense = np.linspace(X.min(), X.max(), 100)
                y_dense = w * x_dense + b
                formula = f"y = {w:.4f}x + {b:.4f}"
                
            elif "2. 2차" in model_choice:
                w2, w1, b = 0.0, 0.0, 0.0
                for _ in range(epochs):
                    y_pred = w2 * (X**2) + w1 * X + b
                    error = y_true - y_pred
                    loss = np.mean(error ** 2)
                    loss_history.append(loss)
                    
                    dw2 = (-2 / N) * np.sum((X**2) * error)
                    dw1 = (-2 / N) * np.sum(X * error)
                    db = (-2 / N) * np.sum(error)
                    w2 -= learning_rate * dw2
                    w1 -= learning_rate * dw1
                    b -= learning_rate * db
                    
                x_dense = np.linspace(X.min(), X.max(), 100)
                y_dense = w2 * (x_dense**2) + w1 * x_dense + b
                formula = f"y = {w2:.4f}x² + {w1:.4f}x + {b:.4f}"
                
            else: # ANN
                X_mean, X_std = X.mean(), X.std() if X.std() != 0 else 1.0
                y_mean, y_std = y_true.mean(), y_true.std() if y_true.std() != 0 else 1.0
                
                X_scaled = (X - X_mean) / X_std
                y_scaled = (y_true - y_mean) / y_std
                
                np.random.seed(42)
                W1 = np.random.randn(1, hidden_units) * 0.1
                b1 = np.zeros((1, hidden_units))
                W2 = np.random.randn(hidden_units, 1) * 0.1
                b2 = np.zeros((1, 1))
                
                def activate(z, name):
                    if name == "Sigmoid":
                        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
                    elif name == "ReLU":
                        return np.maximum(0, z)
                    elif name == "Tanh":
                        return np.tanh(z)
                        
                def activate_deriv(a, name):
                    if name == "Sigmoid":
                        return a * (1 - a)
                    elif name == "ReLU":
                        return (a > 0).astype(float)
                    elif name == "Tanh":
                        return 1 - a**2
                        
                for _ in range(epochs):
                    Z1 = np.outer(X_scaled, np.ones(hidden_units)) * W1 + b1
                    A1 = activate(Z1, activation_name)
                    y_pred_scaled = np.dot(A1, W2) + b2
                    
                    error_scaled = y_pred_scaled.flatten() - y_scaled
                    loss = np.mean(error_scaled ** 2)
                    loss_history.append(loss)
                    
                    dY = (2 / N) * error_scaled.reshape(-1, 1)
                    dW2 = np.dot(A1.T, dY)
                    db2 = np.sum(dY, axis=0, keepdims=True)
                    
                    dA1 = np.dot(dY, W2.T)
                    dZ1 = dA1 * activate_deriv(A1, activation_name)
                    dW1 = np.outer(X_scaled, np.ones(hidden_units)) * dZ1
                    db1 = np.sum(dZ1, axis=0, keepdims=True)
                    
                    W1 -= learning_rate * dW1
                    b1 -= learning_rate * db1
                    W2 -= learning_rate * dW2
                    b2 -= learning_rate * db2
                    
                x_dense = np.linspace(X.min(), X.max(), 100)
                x_dense_scaled = (x_dense - X_mean) / X_std
                Z1_dense = np.outer(x_dense_scaled, np.ones(hidden_units)) * W1 + b1
                A1_dense = activate(Z1_dense, activation_name)
                y_dense_scaled = np.dot(A1_dense, W2) + b2
                y_dense = y_dense_scaled.flatten() * y_std + y_mean
                formula = f"인공신경망 (은닉층 뉴런 {hidden_units}개, {activation_name})"
                
            st.markdown("---")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📈 학습 결과 시각화")
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                ax1.scatter(X, y_true, color='crimson', s=60, label='실제 데이터', zorder=5)
                ax1.plot(x_dense, y_dense, color='navy', linewidth=2.5, label='모델 예측선', zorder=6)
                ax1.set_xlabel(x_col)
                ax1.set_ylabel(y_col)
                ax1.set_title(model_choice)
                ax1.legend()
                ax1.grid(True, linestyle='--', alpha=0.5)
                
                ax2.plot(loss_history, color='purple', linewidth=1.5)
                ax2.set_title("손실 함수 (MSE Loss) 추이")
                ax2.set_xlabel("Epochs")
                ax2.set_ylabel("Loss")
                ax2.grid(True, linestyle='--', alpha=0.5)
                
                st.pyplot(fig)
                
            with col2:
                st.subheader("📌 학습 결과 요약")
                st.success(f"**적용 모델/수식:**\n\n${formula}$")
                st.metric("최종 손실 (MSE)", f"{loss_history[-1]:.6f}")
                st.metric("학습 데이터 수", f"{N}개")
                
    except Exception as e:
        st.error(f"⚠️ 데이터 처리 중 오류 발생: {e}")

# ==========================================
# TAB 2: DeepSeek / TokenHub AI 챗봇 (기억 기능)
# ==========================================
with tab2:
    st.title("💬 DeepSeek AI 데이터 분석 도우미")
    st.caption("이 챗봇은 이전 대화 맥락을 기억하며 학습 및 분석 결과에 대한 피드백을 제공합니다.")
    
    # Secrets에서 TOKENHUB_API_KEY 또는 DEEPSEEK_API_KEY 자동 로드
    api_key = st.secrets.get("TOKENHUB_API_KEY", "") or st.secrets.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        api_key = st.text_input("API Key를 입력하세요 (선택사항):", type="password", key="key_input")
        
    for msg in st.session_state.messages:
        if msg["role"] != "system":
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                
    if user_input := st.chat_input("질문이나 피드백 요청을 입력하세요..."):
        if not api_key:
            st.error("⚠️ API Key가 필요합니다. Streamlit Secrets에 등록해 주세요.")
        else:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)
                
            try:
                # API Key 접두사(thsk- / sk-)와 상관없이 텐센트 클라우드 TokenHub 엔드포인트 및 모델 지정
                base_url = "https://api.tencentcloud.com/tokenhub/v1"
                model_name = "deepseek-v4-flash"

                client = OpenAI(api_key=api_key, base_url=base_url)
                with st.chat_message("assistant"):
                    with st.spinner("AI가 답변을 생성 중입니다..."):
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=st.session_state.messages
                        )
                        reply = response.choices[0].message.content
                        st.write(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")
