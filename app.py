from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import streamlit as st
from openai import OpenAI


# ============================================================
# Parsing helpers
# ============================================================

def normalize_spaces(s: str) -> str:
    return " ".join(s.replace("\n", " ").replace("\r", " ").split())


def strip_outer_parens(s: str) -> str:
    s = s.strip()
    while s.startswith("(") and s.endswith(")"):
        depth = 0
        in_quote = False
        ok = True
        for i, ch in enumerate(s):
            if ch == '"' and (i == 0 or s[i - 1] != "\\"):
                in_quote = not in_quote
            if in_quote:
                continue
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and i != len(s) - 1:
                    ok = False
                    break
        if ok:
            s = s[1:-1].strip()
        else:
            break
    return s


def split_top_level_or(expr: str) -> List[str]:
    expr = strip_outer_parens(expr.strip())
    parts: List[str] = []
    buf: List[str] = []
    depth = 0
    in_quote = False

    i = 0
    n = len(expr)

    def flush():
        chunk = "".join(buf).strip()
        if chunk:
            parts.append(strip_outer_parens(chunk))
        buf.clear()

    while i < n:
        ch = expr[i]

        if ch == '"' and (i == 0 or expr[i - 1] != "\\"):
            in_quote = not in_quote
            buf.append(ch)
            i += 1
            continue

        if not in_quote:
            if ch == "(":
                depth += 1
                buf.append(ch)
                i += 1
                continue
            if ch == ")":
                depth -= 1
                buf.append(ch)
                i += 1
                continue

            if depth == 0 and expr[i:i + 4] == " OR ":
                flush()
                i += 4
                continue

        buf.append(ch)
        i += 1

    flush()
    return parts


FIELD_PATTERNS = {
    "TITLE-ABS": re.compile(r'TITLE-ABS\(\s*"([^"]+)"\s*\)'),
    "AUTHKEY": re.compile(r'AUTHKEY\(\s*"([^"]+)"\s*\)'),
}


def extract_terms(clause: str, field: str) -> List[str]:
    return FIELD_PATTERNS[field].findall(clause)


@dataclass
class ClauseInfo:
    clause: str
    title_terms: List[str]
    auth_terms: List[str]


def parse_query_to_clauses(query_text: str) -> List[ClauseInfo]:
    expr = normalize_spaces(query_text)
    clauses = split_top_level_or(expr)
    out: List[ClauseInfo] = []
    for c in clauses:
        out.append(
            ClauseInfo(
                clause=c,
                title_terms=extract_terms(c, "TITLE-ABS"),
                auth_terms=extract_terms(c, "AUTHKEY"),
            )
        )
    return out


def load_all_sdgs(base_dir: Path) -> Dict[int, Dict]:
    sdg_data: Dict[int, Dict] = {}
    for sdg in range(1, 18):
        p = base_dir / f"SDG{sdg:02d}.txt"
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        clause_infos = parse_query_to_clauses(txt)

        # keyword pool (union, preserve order)
        terms_all: List[str] = []
        seen = set()
        for ci in clause_infos:
            for t in ci.title_terms + ci.auth_terms:
                if t not in seen:
                    seen.add(t)
                    terms_all.append(t)

        sdg_data[sdg] = {
            "query_text": txt,
            "clauses": clause_infos,
            "terms_all": terms_all,
        }
    return sdg_data


# ============================================================
# Combination check logic (rule-based)
# ============================================================

def build_combinations_for_keyword(
    clauses: List[ClauseInfo],
    keyword: str,
    max_results: int = 10,
    max_terms_per_sentence: int = 10,
) -> List[Dict]:
    combos: List[Dict] = []
    seen = set()

    for ci in clauses:
        if (keyword not in ci.title_terms) and (keyword not in ci.auth_terms):
            continue

        title = tuple(sorted(set(ci.title_terms)))
        auth = tuple(sorted(set(ci.auth_terms)))
        key = (title, auth)
        if key in seen:
            continue
        seen.add(key)

        def sentence(field_label: str, items: List[str]) -> str:
            if not items:
                return ""
            if len(items) > max_terms_per_sentence:
                shown = items[:max_terms_per_sentence]
                return f"{field_label}에 {', '.join(shown)} 등(총 {len(items)}개)을 함께 사용하세요."
            return f"{field_label}에 {', '.join(items)}를 함께 사용하세요."

        title_s = sentence("제목/초록(TITLE-ABS)", list(title))
        auth_s = sentence("저자키워드(AUTHKEY)", list(auth))

        combos.append(
            {
                "title_terms": list(title),
                "auth_terms": list(auth),
                "title_sentence": title_s if title_s else None,
                "auth_sentence": auth_s if auth_s else None,
                "clause_excerpt": ci.clause[:900] + ("..." if len(ci.clause) > 900 else ""),
            }
        )

        if len(combos) >= max_results:
            break

    return combos


# ============================================================
# LLM explanation (ChatGPT via OpenAI API)
# ============================================================

def build_llm_prompt(sdg: int, keyword: str, combos: List[Dict], full_query: str) -> str:
    """
    1순위: keyword 포함 clause/조합(=combos) 요약
    2순위: 전체 쿼리(full_query) 훑은 뒤, 사용자가 이해할 안내문 생성
    """
    # combos에서 근거를 몇 개만 넣기(너무 길면 비용/품질 둘 다 악화)
    top_blocks = []
    for i, c in enumerate(combos[:5], start=1):
        top_blocks.append(
            f"[근거 {i}]\n"
            f"- TITLE-ABS terms: {c['title_terms']}\n"
            f"- AUTHKEY terms: {c['auth_terms']}\n"
            f"- clause excerpt: {c['clause_excerpt']}\n"
        )

    evidence = "\n".join(top_blocks) if top_blocks else "(근거 조합을 찾지 못함)"

    return f"""
당신은 Elsevier SDG 매핑 쿼리를 연구자에게 설명하는 도우미입니다.
사용자가 선택한 SDG: {sdg}
사용자가 선택한 키워드: "{keyword}"

요구사항:
- 1순위로 아래 '근거'를 읽고, 키워드가 어떤 필드(TITLE-ABS/AUTHKEY)에 어떤 조합으로 쓰이는지 요약하세요.
- 2순위로 전체 쿼리(아래 '전체 쿼리')를 훑고, 사용자가 실무에서 어떻게 검색어를 넣어야 SDG {sdg} 연구로 인식되는지 안내문을 작성하세요.
- 출력은 한국어로, 간결한 문단 3~6줄 + 필요하면 '추천 입력 예시' 1개(짧게)만 제공하세요.
- 목록을 너무 길게 나열하지 말고(최대 8개), 핵심만 말하세요.
- "SDG로 인식된다"는 표현은 '가능성이 높다/매핑될 수 있다'처럼 과도한 확정 표현을 피하세요.

[근거]
{evidence}

[전체 쿼리]
{full_query[:6000]}
""".strip()


def call_openai_explainer(prompt: str) -> str:
    """
    배포 환경(Streamlit Cloud 등)과 로컬 환경을 모두 지원:
    1) st.secrets["OPENAI_API_KEY"] 우선
    2) 환경변수 OPENAI_API_KEY fallback
    """
    api_key = None

    # 1) Streamlit secrets 우선 시도
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        api_key = None

    # 2) 환경변수 fallback
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")

    # 3) 키 없으면 사용자/관리자 모두 이해 가능한 메시지로 실패
    if not api_key:
        raise RuntimeError(
            "OpenAI API 키가 설정되지 않았습니다. "
            "배포 환경의 Secrets 또는 환경변수(OPENAI_API_KEY)를 설정하세요."
        )

    # 4) 명시적으로 api_key 전달 (배포 환경에서 안전)
    client = OpenAI(api_key=api_key)

    resp = client.responses.create(
        model="gpt-5",
        input=prompt,
    )

    # SDK 버전에 따라 output_text가 없을 가능성까지 대비
    if hasattr(resp, "output_text") and resp.output_text:
        return resp.output_text

    # fallback: 최대한 안전하게 텍스트 추출
    try:
        texts = []
        for item in getattr(resp, "output", []):
            for c in getattr(item, "content", []):
                t = getattr(c, "text", None)
                if t:
                    texts.append(t)
        if texts:
            return "\n".join(texts)
    except Exception:
        pass

    return "(LLM 응답 텍스트를 추출하지 못했습니다.)"


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(page_title="SDG 쿼리 키워드 탐색기", layout="wide")

BASE_DIR = Path(__file__).parent
sdg_data = load_all_sdgs(BASE_DIR)

st.title("SDG 쿼리 키워드 탐색기")
st.caption("SDG 선택 → 키워드 선택 → 검색 → 키워드 조합 확인")

if not sdg_data:
    st.error("SDG01.txt ~ SDG17.txt 파일을 app.py와 같은 폴더에 넣어주세요.")
    st.stop()

st.subheader("SDG 선택")

if "sdg_selected" not in st.session_state:
    st.session_state.sdg_selected = sorted(sdg_data.keys())[0]

available_sdgs = sorted(sdg_data.keys())
cols = st.columns(6)
for i, sdg in enumerate(available_sdgs):
    with cols[i % 6]:
        if st.button(
            f"SDG {sdg}",
            use_container_width=True,
            type="primary" if sdg == st.session_state.sdg_selected else "secondary",
        ):
            st.session_state.sdg_selected = sdg

sdg_selected = st.session_state.sdg_selected
data = sdg_data[sdg_selected]
terms_all: List[str] = data["terms_all"]
clauses: List[ClauseInfo] = data["clauses"]
query_text: str = data["query_text"]

st.markdown("---")
left, right = st.columns([0.45, 0.55], gap="large")

with left:
    st.subheader(f"키워드 목록 (SDG {sdg_selected})")
    q = st.text_input("키워드 검색(부분일치)", value="")
    show_list = terms_all if not q.strip() else [t for t in terms_all if q.lower() in t.lower()]
    st.write(f"표시 키워드: **{len(show_list)}개**")

    MAX_SHOW = 300
    if len(show_list) > MAX_SHOW:
        st.warning(f"키워드가 많아 상위 {MAX_SHOW}개만 표시합니다. 검색어를 더 구체화하세요.")
        show_list = show_list[:MAX_SHOW]

    col_btn1, col_btn2 = st.columns([0.5, 0.5])
    with col_btn1:
        do_search = st.button("검색", use_container_width=True, type="primary", disabled=(len(show_list) == 0))
    with col_btn2:
        do_reset = st.button("초기화", use_container_width=True)

    if do_reset:
        st.session_state.pop("last_keyword", None)
        st.session_state.pop("results", None)
        st.session_state.pop("llm_text", None)

    selected_term = st.radio(
        "키워드를 클릭해서 선택하세요",
        options=show_list if show_list else ["(없음)"],
        index=0,
        label_visibility="visible",
    )

with right:
    st.subheader("키워드 조합 확인")

    if do_search and selected_term != "(없음)":
        combos = build_combinations_for_keyword(clauses, selected_term, max_results=10, max_terms_per_sentence=10)
        st.session_state["last_keyword"] = selected_term
        st.session_state["results"] = combos

        # LLM 설명 생성
        try:
            prompt = build_llm_prompt(sdg_selected, selected_term, combos, query_text)
            st.session_state["llm_text"] = call_openai_explainer(prompt)
        except Exception as e:
            st.session_state["llm_text"] = (
                "설명 생성 기능을 사용할 수 없습니다. "
                f"(원인: {e})"
            )

    keyword = st.session_state.get("last_keyword")
    combos = st.session_state.get("results")
    llm_text = st.session_state.get("llm_text")

    if keyword and llm_text:
        st.markdown(f"### 선택 키워드: **{keyword}**")
        st.write(llm_text)

    # (원하시면 아래 조합/근거 출력은 숨겨도 됩니다)
    if keyword and combos:
        with st.expander("근거(쿼리 일부) 보기", expanded=False):
            for idx, r in enumerate(combos, start=1):
                if r.get("title_sentence"):
                    st.write(f"**{idx}.** {r['title_sentence']}")
                if r.get("auth_sentence"):
                    st.write(f"- {r['auth_sentence']}")
                st.code(r["clause_excerpt"], language="text")
                st.markdown("---")