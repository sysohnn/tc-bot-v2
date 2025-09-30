import streamlit as st
import os, zipfile, tempfile
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="TC-Bot v2", layout="wide")
st.title("🧪 TC-Bot v2: 테스트케이스 자동 생성기 (고도화 버전)")

# 사이드바 입력
with st.sidebar:
    st.header("⚙️ 설정")
    api_key = st.text_input("🔑 OpenRouter API Key", type="password")
    model = st.selectbox("🤖 사용할 LLM 모델",
                         ["qwen/qwen-max", "mistral", "openai/gpt-4"])
    role = st.selectbox("👤 QA 역할", ["기능 QA", "보안 QA", "성능 QA"])
    project_name = st.text_input("📁 프로젝트명", value="MyProject")
    show_heatmap = st.checkbox("📊 테스트 커버리지 Heatmap 보기", value=True)

uploaded_file = st.file_uploader("📂 소스코드 zip 파일 업로드", type=["zip"])

if uploaded_file and api_key:
    with st.spinner("🔍 LLM 호출 중입니다. 잠시만 기다려 주세요..."):
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, uploaded_file.name)
            with open(zip_path, "wb") as f:
                f.write(uploaded_file.read())
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)

            full_code = ""
            file_line_counts = {}

            for root, _, files in os.walk(tmpdir):
                for file in files:
                    if file.endswith(
                        (".py", ".java", ".js", ".ts", ".cpp", ".c", ".cs")):
                        filepath = os.path.join(root, file)
                        try:
                            with open(filepath,
                                      "r",
                                      encoding="utf-8",
                                      errors="ignore") as f:
                                code = f.read()
                                lines = code.count("\n") + 1
                                file_line_counts[file] = lines
                                full_code += f"\n\n# FILE: {file}\n" + code
                        except:
                            continue

        prompt = f"""
        너는 시니어 QA 엔지니어이며, '{project_name}' 프로젝트의 {role}이다.
        아래에 제공된 소스코드를 분석하여 화면/기능 단위의 테스트 시나리오 기반 테스트케이스를 생성하라.

        📌 출력 형식은 아래 마크다운 테이블 형태로 작성하되,
        우선순위는 반드시 High / Medium / Low 중 하나로 작성할 것:

        | TC ID | 기능 설명 | 입력값 | 예상 결과 | 우선순위 |
        |-------|-----------|--------|------------|---------|

        소스코드:
        {full_code}
        """

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [{
                    "role": "user",
                    "content": prompt
                }]
            })

    result = response.json()["choices"][0]["message"]["content"]
    st.success("✅ 테스트케이스 생성 완료!")
    st.markdown("## 📋 생성된 테스트케이스")
    st.markdown(result)

    # 마크다운 파싱 및 Excel 저장
    rows = []
    for line in result.splitlines():
        if "|" in line and "TC" in line:
            parts = [p.strip() for p in line.strip().split("|")[1:-1]]
            if len(parts) == 5:
                rows.append(parts)

    if rows:
        df = pd.DataFrame(rows,
                          columns=["TC ID", "기능 설명", "입력값", "예상 결과", "우선순위"])
        out_path = os.path.join(tempfile.gettempdir(),
                                f"{project_name}_테스트케이스.xlsx")
        df.to_excel(out_path, index=False)
        with open(out_path, "rb") as f:
            st.download_button("⬇️ 엑셀 다운로드",
                               data=f,
                               file_name=os.path.basename(out_path))

    # Heatmap
    if show_heatmap and file_line_counts:
        st.markdown("## 🔥 테스트 커버리지 Heatmap (라인 수 기준)")
        coverage_df = pd.DataFrame(list(file_line_counts.items()),
                                   columns=["파일명", "라인수"])
        fig, ax = plt.subplots(figsize=(6, len(coverage_df) * 0.4))
        sns.barplot(x="라인수",
                    y="파일명",
                    data=coverage_df,
                    palette="Blues_d",
                    ax=ax)
        st.pyplot(fig)