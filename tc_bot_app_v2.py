import streamlit as st
import os
import zipfile
import tempfile
import pandas as pd
import requests

# ✅ OpenRouter API KEY (보안 주의!)
API_KEY = "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"

st.set_page_config(page_title="TC-Bot v3", layout="wide")
st.title("🧪 TC-Bot v3: 테스트케이스 자동 생성기")

# ✅ 사이드바 입력
with st.sidebar:
    st.header("⚙️ 설정")
    model = st.selectbox("🤖 사용할 LLM 모델", ["qwen/qwen-max", "mistral"])
    role = st.selectbox("👤 QA 역할", ["기능 QA", "보안 QA", "성능 QA"])

# ✅ 세션 초기화
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None
if "last_model" not in st.session_state:
    st.session_state.last_model = None
if "last_role" not in st.session_state:
    st.session_state.last_role = None
if "llm_result" not in st.session_state:
    st.session_state.llm_result = None
if "parsed_df" not in st.session_state:
    st.session_state.parsed_df = None

uploaded_file = st.file_uploader("📂 소스코드 zip 파일 업로드", type=["zip"])


def need_llm_call(uploaded_file, model, role):
    # 이전 세션 상태와 비교
    return (uploaded_file is not None
            and (st.session_state.last_uploaded_file != uploaded_file.name
                 or st.session_state.last_model != model
                 or st.session_state.last_role != role))


# ✅ LLM 호출 조건 확인
if uploaded_file and need_llm_call(uploaded_file, model, role):
    with st.spinner("🔍 LLM 호출 중입니다. 잠시만 기다려 주세요..."):
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, uploaded_file.name)
            with open(zip_path, "wb") as f:
                f.write(uploaded_file.read())

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)

            full_code = ""
            for root, _, files in os.walk(tmpdir):
                for file in files:
                    if file.endswith(
                        (".py", ".java", ".js", ".ts", ".cpp", ".c", ".cs")):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path,
                                      "r",
                                      encoding="utf-8",
                                      errors="ignore") as f:
                                code = f.read()
                                full_code += f"\n\n# FILE: {file}\n{code}"
                        except:
                            continue

        # ✅ Prompt 구성
        prompt = f"""
        너는 시니어 QA 엔지니어이며, 현재 '{role}' 역할을 맡고 있다.
        아래에 제공된 소스코드를 분석하여 기능 단위의 테스트 시나리오 기반 테스트케이스를 생성하라.

        📌 출력 형식은 아래 마크다운 테이블 형태로 작성하되,
        우선순위는 반드시 High / Medium / Low 중 하나로 작성할 것:

        | TC ID | 기능 설명 | 입력값 | 예상 결과 | 우선순위 |
        |-------|-----------|--------|------------|---------|

        소스코드:
        {full_code}
        """

        # ✅ LLM 호출
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={
                "model": model,
                "messages": [{
                    "role": "user",
                    "content": prompt
                }]
            })

        result = response.json()["choices"][0]["message"]["content"]
        st.session_state.llm_result = result

        # ✅ 결과 파싱
        rows = []
        for line in result.splitlines():
            if "|" in line and "TC" in line:
                parts = [p.strip() for p in line.strip().split("|")[1:-1]]
                if len(parts) == 5:
                    rows.append(parts)

        if rows:
            df = pd.DataFrame(
                rows, columns=["TC ID", "기능 설명", "입력값", "예상 결과", "우선순위"])
            st.session_state.parsed_df = df

        # ✅ 세션 상태 업데이트
        st.session_state.last_uploaded_file = uploaded_file.name
        st.session_state.last_model = model
        st.session_state.last_role = role

# ✅ 결과 렌더링
if st.session_state.llm_result:
    st.success("✅ 테스트케이스 생성 완료!")
    st.markdown("## 📋 생성된 테스트케이스")
    st.markdown(st.session_state.llm_result)

# ✅ 엑셀 다운로드
if st.session_state.parsed_df is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        st.session_state.parsed_df.to_excel(tmp.name, index=False)
        tmp.seek(0)
        st.download_button("⬇️ 엑셀 다운로드",
                           data=tmp.read(),
                           file_name="테스트케이스.xlsx")

