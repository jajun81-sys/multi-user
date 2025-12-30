import os
import streamlit as st
import tempfile
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
import re
from uuid import uuid4
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import PGVector, FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

# LLM imports (선택적)
try:
    from langchain_anthropic import ChatAnthropic
    HAS_LANGCHAIN_ANTHROPIC = True
except ImportError:
    HAS_LANGCHAIN_ANTHROPIC = False
    ChatAnthropic = None

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    HAS_LANGCHAIN_GOOGLE = True
except ImportError:
    HAS_LANGCHAIN_GOOGLE = False
    ChatGoogleGenerativeAI = None

# Supabase imports
try:
    from supabase import create_client, Client as SupabaseClient
    HAS_SUPABASE = True
except ImportError:
    HAS_SUPABASE = False
    create_client = None
    SupabaseClient = None

# 로깅 설정
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_filename = os.path.join(log_dir, f"multi_users_{datetime.now().strftime('%Y%m%d')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# 환경 변수 로드 (.env 파일)
env_loaded = load_dotenv(override=True)
if env_loaded:
    logger.info(".env 파일이 성공적으로 로드되었습니다.")
else:
    env_file_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_file_path):
        logger.warning(f".env 파일이 존재하지만 로드에 실패했습니다: {env_file_path}")
    else:
        logger.info(".env 파일이 없습니다. 환경변수를 직접 설정하거나 .env 파일을 생성하세요.")

# HTTP 요청 로그 비활성화
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)

# 구분선 및 취소선 제거 함수
def remove_separators(text: str) -> str:
    """답변에서 구분선(---, ===, ___)과 취소선(~~텍스트~~)을 제거합니다."""
    if not text:
        return text
    text = re.sub(r'~~([^~]+)~~', r'\1', text)
    text = re.sub(r'\n\s*-{3,}\s*\n', '\n\n', text)
    text = re.sub(r'\n\s*={3,}\s*\n', '\n\n', text)
    text = re.sub(r'\n\s*_{3,}\s*\n', '\n\n', text)
    text = re.sub(r'^\s*-{3,}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*={3,}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*_{3,}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

# LLM 모델 선택 함수
def get_llm(model_name: str, temperature: float = 0.7, 
            openai_api_key: Optional[str] = None,
            anthropic_api_key: Optional[str] = None,
            google_api_key: Optional[str] = None) -> Any:
    """선택된 모델명에 따라 적절한 LLM 인스턴스를 반환합니다."""
    if model_name == "gpt-5.1":
        if not openai_api_key:
            st.error("OpenAI API 키가 설정되지 않았습니다. 사이드바에서 입력해주세요.")
            st.stop()
        return ChatOpenAI(model="gpt-5.1", temperature=temperature, openai_api_key=openai_api_key)
    elif model_name == "claude-sonnet-4-5":
        if not HAS_LANGCHAIN_ANTHROPIC or ChatAnthropic is None:
            st.error("langchain-anthropic 패키지가 설치되지 않았습니다.")
            st.stop()
        if not anthropic_api_key:
            st.error("Anthropic API 키가 설정되지 않았습니다. 사이드바에서 입력해주세요.")
            st.stop()
        return ChatAnthropic(model="claude-sonnet-4-5", temperature=temperature, anthropic_api_key=anthropic_api_key)
    elif model_name == "gemini-3-pro-preview":
        if not HAS_LANGCHAIN_GOOGLE or ChatGoogleGenerativeAI is None:
            st.error("langchain-google-genai 패키지가 설치되지 않았습니다.")
            st.stop()
        if not google_api_key:
            st.error("Google API 키가 설정되지 않았습니다. 사이드바에서 입력해주세요.")
            st.stop()
        return ChatGoogleGenerativeAI(model="gemini-3-pro-preview", google_api_key=google_api_key, temperature=temperature)
    else:
        # 기본값으로 OpenAI 사용
        if not openai_api_key:
            st.error("OpenAI API 키가 설정되지 않았습니다. 사이드바에서 입력해주세요.")
            st.stop()
        return ChatOpenAI(model="gpt-5.1", temperature=temperature, openai_api_key=openai_api_key)

# Supabase 클라이언트 초기화 (Streamlit Secrets 지원)
def init_supabase() -> Optional[Any]:
    """Supabase 클라이언트를 초기화합니다. Streamlit Secrets 또는 환경변수에서 로드합니다."""
    if not HAS_SUPABASE:
        logger.warning("supabase 패키지가 설치되지 않았습니다.")
        return None
    
    # Streamlit Secrets에서 먼저 읽기 (Streamlit Cloud 배포 시)
    supabase_url = None
    supabase_key = None
    
    try:
        # Streamlit Secrets 시도
        if hasattr(st, 'secrets') and st.secrets:
            try:
                supabase_url = st.secrets.get("SUPABASE_URL")
                supabase_key = st.secrets.get("SUPABASE_ANON_KEY") or st.secrets.get("SUPABASE_SERVICE_ROLE_KEY")
            except Exception as secrets_error:
                logger.debug(f"Streamlit Secrets 읽기 실패 (로컬 환경일 수 있음): {secrets_error}")
    except Exception:
        pass
    
    # Secrets에서 못 읽었으면 환경변수에서 읽기
    if not supabase_url:
        supabase_url = os.getenv("SUPABASE_URL")
    if not supabase_key:
        supabase_key = os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        logger.warning("SUPABASE_URL 또는 SUPABASE_KEY가 설정되지 않았습니다.")
        logger.info("설정 방법:")
        logger.info("1. 로컬 환경: .env 파일에 SUPABASE_URL과 SUPABASE_ANON_KEY 추가")
        logger.info("2. Streamlit Cloud: Settings > Secrets에 SUPABASE_URL과 SUPABASE_ANON_KEY 추가")
        return None
    
    # URL 형식 검증
    if not supabase_url.startswith("http://") and not supabase_url.startswith("https://"):
        logger.error(f"잘못된 SUPABASE_URL 형식: {supabase_url}")
        return None
    
    try:
        client = create_client(supabase_url, supabase_key)
        # 연결 테스트
        try:
            result = client.table("users").select("id").limit(1).execute()
            logger.info("Supabase 연결 성공")
            return client
        except Exception as table_error:
            error_str = str(table_error)
            if "401" in error_str or "Invalid API key" in error_str or "Unauthorized" in error_str:
                logger.error("API 키가 유효하지 않습니다.")
                return None
            logger.info("Supabase 클라이언트 생성 성공 (테이블은 아직 생성되지 않음)")
            return client
    except Exception as e:
        logger.error(f"Supabase 클라이언트 초기화 오류: {e}")
        return None

# 사용자 인증 함수
def authenticate_user(supabase: Any, email: str, password: str) -> Optional[Dict]:
    """사용자 로그인을 처리합니다."""
    try:
        response = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        if response.user:
            logger.info(f"사용자 로그인 성공: {email}")
            return {
                "user_id": response.user.id,
                "email": response.user.email,
                "session": response.session
            }
        return None
    except Exception as e:
        logger.error(f"로그인 오류: {e}")
        return None

# 사용자 회원가입 함수
def signup_user(supabase: Any, email: str, password: str) -> Optional[Dict]:
    """새 사용자 회원가입을 처리합니다."""
    try:
        response = supabase.auth.sign_up({
            "email": email,
            "password": password
        })
        if response.user:
            logger.info(f"사용자 회원가입 성공: {email}")
            return {
                "user_id": response.user.id,
                "email": response.user.email
            }
        return None
    except Exception as e:
        logger.error(f"회원가입 오류: {e}")
        return None

# 사용자 로그아웃 함수
def logout_user(supabase: Any):
    """사용자 로그아웃을 처리합니다."""
    try:
        supabase.auth.sign_out()
        logger.info("사용자 로그아웃")
    except Exception as e:
        logger.error(f"로그아웃 오류: {e}")

# 세션 제목 자동 생성
def generate_session_title(first_question: str, first_answer: str, llm: Any) -> str:
    """첫 번째 질문과 답변을 기반으로 세션 제목을 생성합니다."""
    try:
        prompt = f"""다음 질문과 답변을 요약하여 간결한 세션 제목을 만들어주세요.

질문: {first_question}

답변: {first_answer[:500]}...

요구사항:
- 제목은 최대 30자 이내로 작성
- 질문의 핵심 주제를 반영
- 한글로 작성
- 설명이나 추가 텍스트 없이 제목만 반환

제목:"""
        response = llm.invoke(prompt)
        title = response.content.strip() if hasattr(response, 'content') else str(response).strip()
        if len(title) > 30:
            title = title[:30]
        return title
    except Exception as e:
        logger.error(f"세션 제목 생성 오류: {e}")
        return f"세션 {datetime.now().strftime('%Y-%m-%d %H:%M')}"

# 세션 저장 (사용자별)
def save_session_to_supabase(
    supabase: Any,
    user_id: str,
    title: str,
    chat_history: List[Dict],
    processed_files: List[str],
    session_id: Optional[str] = None
) -> Optional[str]:
    """Supabase에 세션을 저장합니다 (사용자별)."""
    try:
        session_data = {
            "user_id": user_id,
            "title": title,
            "chat_history": chat_history,
            "processed_files": processed_files,
            "updated_at": datetime.now().isoformat()
        }
        
        if session_id:
            # 기존 세션 업데이트
            result = supabase.table("sessions").update(session_data).eq("id", session_id).eq("user_id", user_id).execute()
            return session_id
        else:
            # 새 세션 생성
            session_data["created_at"] = datetime.now().isoformat()
            result = supabase.table("sessions").insert(session_data).execute()
            if result.data and len(result.data) > 0:
                return result.data[0]["id"]
            return None
    except Exception as e:
        logger.error(f"세션 저장 오류: {e}")
        st.error(f"세션 저장 중 오류가 발생했습니다: {str(e)}")
        return None

# 세션 로드 (사용자별)
def load_session_from_supabase(supabase: Any, user_id: str, session_id: str) -> Optional[Dict]:
    """Supabase에서 세션을 로드합니다 (사용자별)."""
    try:
        result = supabase.table("sessions").select("*").eq("id", session_id).eq("user_id", user_id).execute()
        if result.data and len(result.data) > 0:
            return result.data[0]
        return None
    except Exception as e:
        logger.error(f"세션 로드 오류: {e}")
        st.error(f"세션 로드 중 오류가 발생했습니다: {str(e)}")
        return None

# 모든 세션 목록 조회 (사용자별)
def get_all_sessions(supabase: Any, user_id: str) -> List[Dict]:
    """사용자의 모든 세션 목록을 조회합니다."""
    try:
        result = supabase.table("sessions").select("id, title, created_at, updated_at").eq("user_id", user_id).order("updated_at", desc=True).execute()
        return result.data if result.data else []
    except Exception as e:
        logger.error(f"세션 목록 조회 오류: {e}")
        return []

# 세션 삭제 (사용자별)
def delete_session_from_supabase(supabase: Any, user_id: str, session_id: str) -> bool:
    """Supabase에서 세션을 삭제합니다 (사용자별)."""
    try:
        result = supabase.table("sessions").delete().eq("id", session_id).eq("user_id", user_id).execute()
        return True
    except Exception as e:
        logger.error(f"세션 삭제 오류: {e}")
        st.error(f"세션 삭제 중 오류가 발생했습니다: {str(e)}")
        return False

# PGVector 연결 문자열 생성
def get_connection_string() -> str:
    """Supabase PostgreSQL 연결 문자열을 생성합니다."""
    supabase_url = None
    supabase_db_password = None
    supabase_db_user = "postgres"
    supabase_db_host = None
    supabase_db_name = "postgres"
    supabase_db_port = "5432"
    
    # Streamlit Secrets에서 읽기 시도
    try:
        if hasattr(st, 'secrets') and st.secrets:
            try:
                supabase_url = st.secrets.get("SUPABASE_URL")
                supabase_db_password = st.secrets.get("SUPABASE_DB_PASSWORD")
                supabase_db_user = st.secrets.get("SUPABASE_DB_USER", "postgres")
                supabase_db_host = st.secrets.get("SUPABASE_DB_HOST")
                supabase_db_name = st.secrets.get("SUPABASE_DB_NAME", "postgres")
                supabase_db_port = st.secrets.get("SUPABASE_DB_PORT", "5432")
            except Exception:
                pass
    except Exception:
        pass
    
    # 환경변수에서 읽기
    if not supabase_url:
        supabase_url = os.getenv("SUPABASE_URL")
    if not supabase_db_password:
        supabase_db_password = os.getenv("SUPABASE_DB_PASSWORD")
    if not supabase_db_user or supabase_db_user == "postgres":
        supabase_db_user = os.getenv("SUPABASE_DB_USER", "postgres")
    if not supabase_db_host:
        supabase_db_host = os.getenv("SUPABASE_DB_HOST")
    if not supabase_db_name or supabase_db_name == "postgres":
        supabase_db_name = os.getenv("SUPABASE_DB_NAME", "postgres")
    if not supabase_db_port or supabase_db_port == "5432":
        supabase_db_port = os.getenv("SUPABASE_DB_PORT", "5432")
    
    # URL에서 호스트 추출
    if not supabase_db_host and supabase_url:
        import re
        match = re.search(r'https://([^.]+)', supabase_url)
        if match:
            supabase_db_host = f"{match.group(1)}.supabase.co"
    
    # 필수 환경변수 확인
    missing_vars = []
    if not supabase_db_password:
        missing_vars.append("SUPABASE_DB_PASSWORD")
    if not supabase_db_host:
        missing_vars.append("SUPABASE_DB_HOST (또는 SUPABASE_URL)")
    
    if missing_vars:
        logger.warning(f"필수 환경변수가 설정되지 않았습니다: {', '.join(missing_vars)}")
        return ""
    
    connection_string = f"postgresql://{supabase_db_user}:{supabase_db_password}@{supabase_db_host}:{supabase_db_port}/{supabase_db_name}"
    return connection_string

# Vector Store 초기화 (세션별)
def init_vectorstore(session_id: str, openai_api_key: Optional[str] = None) -> Optional[Any]:
    """세션별 Vector Store를 초기화합니다."""
    try:
        connection_string = get_connection_string()
        if not openai_api_key:
            return None
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        
        if connection_string:
            # Supabase PGVector 사용
            collection_name = f"session_{session_id}"
            try:
                vectorstore = PGVector(
                    connection_string=connection_string,
                    embedding_function=embeddings,
                    collection_name=collection_name
                )
                return vectorstore
            except Exception:
                # 새로 생성
                vectorstore = PGVector(
                    connection_string=connection_string,
                    embedding_function=embeddings,
                    collection_name=collection_name
                )
                return vectorstore
        else:
            logger.info("데이터베이스 연결 정보가 없어 로컬 Vector Store를 사용합니다.")
            return None
    except Exception as e:
        logger.error(f"Vector Store 초기화 오류: {e}")
        return None

# 페이지 설정
st.set_page_config(
    page_title="PDF 기반 멀티유저 멀티세션 RAG 챗봇",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 초기 상태 설정
if "supabase_client" not in st.session_state:
    st.session_state.supabase_client = init_supabase()

if "authenticated_user" not in st.session_state:
    st.session_state.authenticated_user = None

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "llm_model" not in st.session_state:
    st.session_state.llm_model = "gpt-5.1"

if "session_list" not in st.session_state:
    st.session_state.session_list = []

if "auto_save_enabled" not in st.session_state:
    st.session_state.auto_save_enabled = True

# API 키 저장소
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""

if "anthropic_api_key" not in st.session_state:
    st.session_state.anthropic_api_key = ""

if "google_api_key" not in st.session_state:
    st.session_state.google_api_key = ""

# CSS 스타일
st.markdown("""
<style>
/* 헤딩 스타일 */
h1 {
    font-size: 1.4rem !important;
    font-weight: 600 !important;
    color: #ff69b4 !important;
}
h2 {
    font-size: 1.2rem !important;
    font-weight: 600 !important;
    color: #ffd700 !important;
}
h3 {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    color: #1f77b4 !important;
}

/* 채팅 메시지 스타일 */
.stChatMessage {
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
}

.stChatMessage p {
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
    margin: 0.5rem 0 !important;
}

.stChatMessage ul, .stChatMessage ol {
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
    margin: 0.5rem 0 !important;
}

.stChatMessage li {
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
    margin: 0.3rem 0 !important;
}

.stChatMessage strong, .stChatMessage b {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
}

.stChatMessage blockquote {
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
    margin: 0.5rem 0 !important;
    padding-left: 1rem !important;
    border-left: 3px solid #e0e0e0 !important;
}

.stChatMessage code {
    font-size: 0.9rem !important;
    background-color: #f5f5f5 !important;
    padding: 0.2rem 0.4rem !important;
    border-radius: 3px !important;
}

.stChatMessage * {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}

.stButton > button {
    background-color: #ff69b4 !important;
    color: white !important;
    border: none !important;
    border-radius: 5px !important;
    padding: 0.5rem 1rem !important;
    font-weight: bold !important;
}

.stButton > button:hover {
    background-color: #ff1493 !important;
}
</style>
""", unsafe_allow_html=True)

# 제목 영역
st.markdown("""
<div style="margin-top: -3rem; margin-bottom: 1rem;">
""", unsafe_allow_html=True)

col_title, col_empty = st.columns([4, 1])

with col_title:
    st.markdown("""
    <div style="text-align: center; margin-top: 0.5rem; margin-bottom: 0.5rem;">
        <h1 style="font-size: 7rem; font-weight: bold; margin: 0; line-height: 1.2;">
            <span style="color: #1f77b4;">PDF 기반</span> 
            <span style="color: #ffd700;">멀티유저</span>
            <span style="color: #ff69b4;">멀티세션</span>
            <span style="color: #1f77b4;">RAG 챗봇</span>
        </h1>
    </div>
    """, unsafe_allow_html=True)

with col_empty:
    st.empty()

st.markdown("</div>", unsafe_allow_html=True)

# 사이드바
with st.sidebar:
    # API 키 입력 (상단)
    st.markdown('<h2 style="color: #1f77b4;">🔑 API 키 설정</h2>', unsafe_allow_html=True)
    
    openai_key = st.text_input("OpenAI API Key", value=st.session_state.openai_api_key, type="password", help="OpenAI API 키를 입력하세요")
    if openai_key != st.session_state.openai_api_key:
        st.session_state.openai_api_key = openai_key
    
    anthropic_key = st.text_input("Anthropic API Key", value=st.session_state.anthropic_api_key, type="password", help="Anthropic API 키를 입력하세요")
    if anthropic_key != st.session_state.anthropic_api_key:
        st.session_state.anthropic_api_key = anthropic_key
    
    google_key = st.text_input("Google API Key", value=st.session_state.google_api_key, type="password", help="Google Gemini API 키를 입력하세요")
    if google_key != st.session_state.google_api_key:
        st.session_state.google_api_key = google_key
    
    st.markdown("---")
    
    # 사용자 인증
    if not st.session_state.authenticated_user:
        st.markdown('<h2 style="color: #ffd700;">👤 로그인 / 회원가입</h2>', unsafe_allow_html=True)
        
        login_email = st.text_input("이메일", key="login_email")
        login_password = st.text_input("비밀번호", type="password", key="login_password")
        
        col_login, col_signup = st.columns(2)
        
        with col_login:
            if st.button("로그인", use_container_width=True):
                if not st.session_state.supabase_client:
                    st.error("Supabase 연결이 설정되지 않았습니다.")
                elif not login_email or not login_password:
                    st.error("이메일과 비밀번호를 입력해주세요.")
                else:
                    user = authenticate_user(st.session_state.supabase_client, login_email, login_password)
                    if user:
                        st.session_state.authenticated_user = user
                        st.success(f"로그인 성공: {user['email']}")
                        st.rerun()
                    else:
                        st.error("로그인 실패. 이메일과 비밀번호를 확인해주세요.")
        
        with col_signup:
            if st.button("회원가입", use_container_width=True):
                if not st.session_state.supabase_client:
                    st.error("Supabase 연결이 설정되지 않았습니다.")
                elif not login_email or not login_password:
                    st.error("이메일과 비밀번호를 입력해주세요.")
                else:
                    user = signup_user(st.session_state.supabase_client, login_email, login_password)
                    if user:
                        st.success(f"회원가입 성공: {user['email']}")
                        # 회원가입 후 자동 로그인
                        login_user = authenticate_user(st.session_state.supabase_client, login_email, login_password)
                        if login_user:
                            st.session_state.authenticated_user = login_user
                            st.rerun()
                    else:
                        st.error("회원가입 실패. 이미 존재하는 이메일일 수 있습니다.")
    else:
        st.markdown('<h2 style="color: #ffd700;">👤 사용자 정보</h2>', unsafe_allow_html=True)
        st.success(f"로그인: {st.session_state.authenticated_user['email']}")
        
        if st.button("로그아웃", use_container_width=True):
            if st.session_state.supabase_client:
                logout_user(st.session_state.supabase_client)
            st.session_state.authenticated_user = None
            st.session_state.current_session_id = None
            st.session_state.chat_history = []
            st.session_state.processed_files = []
            st.session_state.vectorstore = None
            st.session_state.session_list = []
            st.rerun()
    
    st.markdown("---")
    
    # 로그인한 사용자만 기능 사용 가능
    if st.session_state.authenticated_user:
        user_id = st.session_state.authenticated_user["user_id"]
        
        st.markdown('<h2 style="color: #1f77b4;">1. LLM 모델 선택</h2>', unsafe_allow_html=True)
        all_models = ["gpt-5.1", "claude-sonnet-4-5", "gemini-3-pro-preview"]
        
        selected_model = st.radio(
            "사용할 언어모델을 선택하세요",
            options=all_models,
            index=all_models.index(st.session_state.llm_model) if st.session_state.llm_model in all_models else 0,
            key='llm_model_radio'
        )
        st.session_state.llm_model = selected_model
        
        st.markdown("---")
        
        # 세션 관리
        st.markdown('<h2 style="color: #ffd700;">2. 세션 관리</h2>', unsafe_allow_html=True)
        
        # 세션 목록 새로고침
        if st.session_state.supabase_client:
            if st.button("🔄 세션 목록 새로고침", use_container_width=True):
                st.session_state.session_list = get_all_sessions(st.session_state.supabase_client, user_id)
                st.rerun()
            
            # 세션 목록 로드
            if not st.session_state.session_list:
                st.session_state.session_list = get_all_sessions(st.session_state.supabase_client, user_id)
            
            # 세션 선택
            if st.session_state.session_list:
                session_titles = [f"{s['title']} ({s['updated_at'][:10]})" for s in st.session_state.session_list]
                
                # 현재 선택된 세션 인덱스 찾기
                current_idx = None
                if st.session_state.current_session_id:
                    for idx, s in enumerate(st.session_state.session_list):
                        if s["id"] == st.session_state.current_session_id:
                            current_idx = idx
                            break
                
                # 이전에 선택한 세션 ID 저장
                if "previous_selected_session_id" not in st.session_state:
                    st.session_state.previous_selected_session_id = None
                
                selected_session_idx = st.selectbox(
                    "세션 선택",
                    options=range(len(session_titles)),
                    index=current_idx if current_idx is not None else 0,
                    format_func=lambda x: session_titles[x] if x < len(session_titles) else "",
                    key="session_selectbox"
                )
                
                if selected_session_idx is not None and selected_session_idx < len(st.session_state.session_list):
                    selected_session = st.session_state.session_list[selected_session_idx]
                    
                    # 세션 선택 시 자동 로드
                    if st.session_state.previous_selected_session_id != selected_session["id"]:
                        session_data = load_session_from_supabase(st.session_state.supabase_client, user_id, selected_session["id"])
                        if session_data:
                            st.session_state.current_session_id = selected_session["id"]
                            st.session_state.previous_selected_session_id = selected_session["id"]
                            st.session_state.chat_history = session_data.get("chat_history", [])
                            st.session_state.processed_files = session_data.get("processed_files", [])
                            
                            # Vector Store 로드
                            if st.session_state.current_session_id and st.session_state.openai_api_key:
                                st.session_state.vectorstore = init_vectorstore(st.session_state.current_session_id, st.session_state.openai_api_key)
                            
                            st.success(f"세션 '{selected_session['title']}'이(가) 자동으로 로드되었습니다.")
                            st.rerun()
                    
                    # 세션 로드 버튼 (수동 로드용)
                    if st.button("📂 세션로드", use_container_width=True, key="load_session_btn"):
                        session_data = load_session_from_supabase(st.session_state.supabase_client, user_id, selected_session["id"])
                        if session_data:
                            st.session_state.current_session_id = selected_session["id"]
                            st.session_state.previous_selected_session_id = selected_session["id"]
                            st.session_state.chat_history = session_data.get("chat_history", [])
                            st.session_state.processed_files = session_data.get("processed_files", [])
                            
                            # Vector Store 로드
                            if st.session_state.current_session_id and st.session_state.openai_api_key:
                                st.session_state.vectorstore = init_vectorstore(st.session_state.current_session_id, st.session_state.openai_api_key)
                            
                            st.success(f"세션 '{selected_session['title']}'이(가) 로드되었습니다.")
                            st.rerun()
            else:
                st.info("저장된 세션이 없습니다.")
        
        st.markdown("---")
        
        # Supabase 연결 상태 표시
        st.markdown('<h3 style="color: #1f77b4;">Supabase 연결 상태</h3>', unsafe_allow_html=True)
        if st.session_state.supabase_client:
            st.success("✅ Supabase 연결됨")
            if st.button("🔄 연결 재시도", use_container_width=True, key="retry_supabase_btn"):
                st.session_state.supabase_client = init_supabase()
                st.rerun()
        else:
            st.warning("⚠️ Supabase 연결 안 됨")
            
            if st.button("🔄 연결 재시도", use_container_width=True, key="retry_supabase_btn_2"):
                # .env 파일 다시 로드
                load_dotenv(override=True)
                st.session_state.supabase_client = init_supabase()
                if st.session_state.supabase_client:
                    st.success("✅ 연결 성공!")
                    st.rerun()
                else:
                    st.error("❌ 연결 실패. Streamlit Secrets 또는 환경변수를 확인해주세요.")
            
            with st.expander("📖 Supabase 연결 설정 가이드"):
                st.markdown("""
                **로컬 환경 (.env 파일 사용)**
                
                프로젝트 루트에 `.env` 파일을 생성하고 다음을 추가하세요:
                
                ```
                SUPABASE_URL=https://your-project-id.supabase.co
                SUPABASE_ANON_KEY=your_supabase_anon_key_here
                ```
                
                **Streamlit Cloud 배포 시**
                
                앱 대시보드 → Settings → Secrets에서 다음을 추가:
                
                ```
                SUPABASE_URL = "https://your-project-id.supabase.co"
                SUPABASE_ANON_KEY = "your_supabase_anon_key_here"
                ```
                
                **Supabase 키 찾는 방법:**
                1. [Supabase](https://supabase.com) 프로젝트 대시보드 접속
                2. Settings > API 메뉴로 이동
                3. Project URL을 `SUPABASE_URL`에 복사
                4. **'anon public' 키**를 `SUPABASE_ANON_KEY`에 복사 (⚠️ service_role 키 아님!)
                
                **설정 후:**
                - 로컬: 앱을 재시작하거나 위의 "🔄 연결 재시도" 버튼을 클릭
                - Streamlit Cloud: 앱이 자동으로 재배포됩니다
                
                **401 오류 해결 방법:**
                - 올바른 'anon public' 키를 사용하고 있는지 확인
                - RLS(Row Level Security) 정책이 올바르게 설정되어 있는지 확인
                """)
        
        st.markdown("---")
        
        # 세션 저장 버튼
        st.markdown('<h2 style="color: #ff69b4;">3. 세션 저장</h2>', unsafe_allow_html=True)
        if st.button("💾 세션저장", use_container_width=True, key="save_session_btn"):
            if not st.session_state.supabase_client:
                st.warning("Supabase 연결이 설정되지 않아 세션을 저장할 수 없습니다.")
            elif not st.session_state.chat_history:
                st.warning("저장할 대화 내용이 없습니다.")
            else:
                # 첫 번째 질문과 답변으로 제목 생성
                if len(st.session_state.chat_history) >= 2:
                    first_question = st.session_state.chat_history[0].get("content", "")
                    first_answer = st.session_state.chat_history[1].get("content", "")
                    llm = get_llm(
                        st.session_state.llm_model,
                        openai_api_key=st.session_state.openai_api_key,
                        anthropic_api_key=st.session_state.anthropic_api_key,
                        google_api_key=st.session_state.google_api_key
                    )
                    title = generate_session_title(first_question, first_answer, llm)
                else:
                    title = f"세션 {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                
                # 세션 저장
                session_id = save_session_to_supabase(
                    st.session_state.supabase_client,
                    user_id,
                    title,
                    st.session_state.chat_history,
                    st.session_state.processed_files,
                    st.session_state.current_session_id
                )
                
                if session_id:
                    st.session_state.current_session_id = session_id
                    st.session_state.session_list = get_all_sessions(st.session_state.supabase_client, user_id)
                    st.success(f"세션 '{title}'이(가) 저장되었습니다.")
                    st.rerun()
        
        st.markdown("---")
        
        # 세션 삭제 버튼
        st.markdown('<h2 style="color: #d62728;">4. 세션 삭제</h2>', unsafe_allow_html=True)
        if st.button("🗑️ 세션삭제", use_container_width=True, key="delete_session_btn"):
            if not st.session_state.current_session_id:
                st.warning("삭제할 세션이 선택되지 않았습니다.")
            elif st.session_state.supabase_client:
                if delete_session_from_supabase(st.session_state.supabase_client, user_id, st.session_state.current_session_id):
                    st.session_state.current_session_id = None
                    st.session_state.chat_history = []
                    st.session_state.processed_files = []
                    st.session_state.vectorstore = None
                    st.session_state.session_list = get_all_sessions(st.session_state.supabase_client, user_id)
                    st.success("세션이 삭제되었습니다.")
                    st.rerun()
        
        st.markdown("---")
        
        # 화면 초기화 버튼
        st.markdown('<h2 style="color: #9467bd;">5. 화면 초기화</h2>', unsafe_allow_html=True)
        if st.button("🔄 화면초기화", use_container_width=True, key="clear_screen_btn"):
            st.session_state.chat_history = []
            st.session_state.current_session_id = None
            st.session_state.processed_files = []
            st.session_state.vectorstore = None
            st.rerun()
        
        st.markdown("---")
        
        # Vector DB 파일 목록 버튼
        st.markdown('<h2 style="color: #8c564b;">6. Vector DB</h2>', unsafe_allow_html=True)
        if st.button("📊 vectordb", use_container_width=True, key="show_vectordb_btn"):
            if st.session_state.processed_files:
                st.markdown("### 처리된 파일 목록")
                for file in st.session_state.processed_files:
                    st.write(f"- {file}")
            else:
                st.info("처리된 파일이 없습니다.")
        
        st.markdown("---")
        
        # PDF 파일 업로드
        st.markdown('<h2 style="color: #2ca02c;">7. PDF 파일 업로드</h2>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader("PDF 파일을 선택하세요", type="pdf", accept_multiple_files=True)
        
        if uploaded_files:
            process_button = st.button("파일 처리하기", key="process_files_btn")
            
            if process_button:
                if not st.session_state.openai_api_key:
                    st.error("OpenAI API 키를 입력해주세요.")
                else:
                    with st.spinner("PDF 파일을 처리 중입니다..."):
                        try:
                            temp_dir = tempfile.TemporaryDirectory()
                            all_docs = []
                            new_files = []
                            
                            for uploaded_file in uploaded_files:
                                if uploaded_file.name in st.session_state.processed_files:
                                    continue
                                
                                temp_file_path = os.path.join(temp_dir.name, uploaded_file.name)
                                with open(temp_file_path, "wb") as f:
                                    f.write(uploaded_file.getbuffer())
                                
                                loader = PyPDFLoader(temp_file_path)
                                documents = loader.load()
                                
                                for doc in documents:
                                    doc.metadata["source"] = uploaded_file.name
                                    doc.metadata["session_id"] = st.session_state.current_session_id or str(uuid4())
                                
                                all_docs.extend(documents)
                                new_files.append(uploaded_file.name)
                            
                            if not all_docs:
                                st.success("모든 파일이 이미 처리되었습니다.")
                            else:
                                text_splitter = RecursiveCharacterTextSplitter(
                                    chunk_size=500,
                                    chunk_overlap=100,
                                    length_function=len
                                )
                                chunks = text_splitter.split_documents(all_docs)
                                
                                embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.openai_api_key)
                                connection_string = get_connection_string()
                                
                                try:
                                    # Supabase Vector Store 사용 가능 여부 확인
                                    if connection_string:
                                        # Supabase PGVector 사용
                                        if not st.session_state.current_session_id:
                                            session_id = str(uuid4())
                                            st.session_state.current_session_id = session_id
                                        else:
                                            session_id = st.session_state.current_session_id
                                        
                                        collection_name = f"session_{session_id}"
                                        
                                        # 기존 벡터 스토어가 있으면 추가, 없으면 생성
                                        if st.session_state.vectorstore:
                                            st.session_state.vectorstore.add_documents(chunks)
                                        else:
                                            vectorstore = PGVector.from_documents(
                                                documents=chunks,
                                                embedding=embeddings,
                                                connection_string=connection_string,
                                                collection_name=collection_name
                                            )
                                            st.session_state.vectorstore = vectorstore
                                    else:
                                        # 로컬 FAISS 사용
                                        if st.session_state.vectorstore:
                                            st.session_state.vectorstore.add_documents(chunks)
                                        else:
                                            vectorstore = FAISS.from_documents(chunks, embeddings)
                                            st.session_state.vectorstore = vectorstore
                                    
                                    st.session_state.processed_files.extend(new_files)
                                    
                                    # 자동 저장
                                    if st.session_state.auto_save_enabled and st.session_state.supabase_client:
                                        if st.session_state.chat_history:
                                            # 기존 세션이 있으면 업데이트, 없으면 새로 생성
                                            if not st.session_state.current_session_id:
                                                title = f"세션 {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                                            else:
                                                # 기존 세션 제목 가져오기
                                                session_data = load_session_from_supabase(
                                                    st.session_state.supabase_client,
                                                    user_id,
                                                    st.session_state.current_session_id
                                                )
                                                title = session_data.get("title", f"세션 {datetime.now().strftime('%Y-%m-%d %H:%M')}") if session_data else f"세션 {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                                            
                                            save_session_to_supabase(
                                                st.session_state.supabase_client,
                                                user_id,
                                                title,
                                                st.session_state.chat_history,
                                                st.session_state.processed_files,
                                                st.session_state.current_session_id
                                            )
                                    
                                    st.success(f"{len(new_files)}개 파일이 처리되었습니다!")
                                except Exception as e:
                                    st.error(f"파일 처리 중 오류가 발생했습니다: {str(e)}")
                                    logger.error(f"Vector Store 저장 오류: {e}")
                        
                        except Exception as e:
                            st.error(f"파일 처리 중 오류가 발생했습니다: {str(e)}")
                            logger.error(f"PDF 파일 처리 오류: {e}")
        
        # 처리된 파일 목록 표시
        if st.session_state.processed_files:
            st.markdown('<h3 style="color: #ffd700;">처리된 파일 목록</h3>', unsafe_allow_html=True)
            for file in st.session_state.processed_files:
                st.write(f"- {file}")
    else:
        st.info("로그인 후 기능을 사용할 수 있습니다.")

# 대화 내용 표시 (로그인한 사용자만)
if st.session_state.authenticated_user:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            if isinstance(message["content"], str):
                st.markdown(message["content"])
            else:
                st.write(message["content"])
    
    # 사용자 입력 영역
    if prompt := st.chat_input("질문을 입력하세요"):
        # 사용자 메시지 추가
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        # RAG 답변 생성
        if st.session_state.vectorstore:
            with st.spinner("PDF 기반 RAG 답변을 생성 중입니다..."):
                try:
                    # RAG 검색
                    retriever = st.session_state.vectorstore.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 10}
                    )
                    retrieved_docs = retriever.invoke(prompt)
                    
                    if not retrieved_docs:
                        response = f"죄송합니다. '{prompt}'에 대한 관련 문서를 찾을 수 없습니다."
                    else:
                        top_docs = retrieved_docs[:3]
                        context_text = ""
                        max_context_length = 8000
                        current_length = 0
                        
                        for i, doc in enumerate(top_docs):
                            doc_text = f"[문서 {i+1}]\n{doc.page_content}\n\n"
                            if current_length + len(doc_text) > max_context_length:
                                break
                            context_text += doc_text
                            current_length += len(doc_text)
                        
                        # 시스템 프롬프트 구성
                        system_prompt = f"""
                        질문: {prompt}
                        
                        관련 문서:
                        {context_text}
                        
                        위 문서 내용을 고려하여 질문에 답변해주세요.
                        
                        답변 형식:
                        - 답변은 반드시 제목과 본문으로 구분하여 작성하세요
                        - 제목(# H1)은 질문의 핵심을 짧고 명확하게 요약한 한 문장으로 작성하세요 (최대 20자 이내 권장)
                        - 제목 다음에 빈 줄을 하나 두고 본문을 작성하세요
                        - 본문은 ## (H2)와 ### (H3) 헤딩을 사용하여 구조화하세요
                        - 본문은 서술형으로 작성하되 존대말을 사용하세요
                        - 개조식이나 불완전한 문장을 사용하지 말고, 완전한 문장으로 서술하세요
                        
                        주의사항:
                        - 답변 중간에 (문서1), (문서2) 같은 참조 표시를 하지 마세요
                        - "참조 문서:", "제공된 문서", "문서 1, 문서 2" 같은 문구를 사용하지 마세요
                        - 답변은 순수한 내용만 포함하고, 참조 관련 문구는 전혀 포함하지 마세요
                        - 답변 끝에 참조 정보나 출처 관련 문구를 추가하지 마세요
                        - 답변 중간에 구분선(---, ===, ___)을 사용하지 마세요
                        - 마크다운 구분선이나 선을 그리는 기호를 절대 사용하지 마세요
                        - 취소선(~~텍스트~~)을 사용하지 마세요
                        """
                        
                        # LLM으로 답변 생성 (스트리밍 모드)
                        llm = get_llm(
                            st.session_state.llm_model,
                            temperature=1,
                            openai_api_key=st.session_state.openai_api_key,
                            anthropic_api_key=st.session_state.anthropic_api_key,
                            google_api_key=st.session_state.google_api_key
                        )
                        
                        response = ""
                        with st.chat_message("assistant"):
                            stream_placeholder = st.empty()
                            # 스트리밍으로 답변 생성
                            for chunk in llm.stream(system_prompt):
                                if hasattr(chunk, 'content'):
                                    chunk_text = chunk.content
                                else:
                                    chunk_text = str(chunk)
                                response += chunk_text
                                cleaned_response = remove_separators(response)
                                stream_placeholder.markdown(cleaned_response)
                        
                        response = remove_separators(response)
                        
                        # 다음 질문 3개 생성
                        try:
                            next_questions_prompt = f"""
                            질문자가 한 질문: {prompt}
                            
                            생성된 답변:
                            {response}
                            
                            위 질문과 답변 내용을 검토하여, 질문자가 다음에 할 수 있는 중요한 3가지 질문을 생성해주세요.
                            
                            요구사항:
                            - 답변 내용을 더 깊이 이해하기 위한 후속 질문
                            - 답변에서 언급된 내용을 구체화하거나 확장하는 질문
                            - 관련된 다른 주제나 관점을 탐색할 수 있는 질문
                            - 각 질문은 완전한 문장으로 작성하되, 간결하고 명확하게 작성
                            - 질문은 번호 없이 순서대로 나열하되, 각 질문은 별도의 줄에 작성
                            
                            형식:
                            질문1
                            질문2
                            질문3
                            
                            참고: 질문만 작성하고, 설명이나 추가 텍스트는 포함하지 마세요.
                            """
                            next_questions_response = llm.invoke(next_questions_prompt)
                            next_questions_text = next_questions_response.content if hasattr(next_questions_response, 'content') else str(next_questions_response)
                            next_questions = [q.strip() for q in next_questions_text.strip().split('\n') if q.strip() and not q.strip().startswith('#')]
                            next_questions = next_questions[:3]
                            
                            if next_questions:
                                response += "\n\n"
                                response += "### 💡 다음에 물어볼 수 있는 질문들\n\n"
                                for i, question in enumerate(next_questions, 1):
                                    response += f"{i}. {question}\n\n"
                                # 다음 질문 추가 후 다시 표시
                                with st.chat_message("assistant"):
                                    st.markdown(response)
                        except Exception as e:
                            logger.warning(f"다음 질문 생성 실패: {e}")
                        
                        # 대화 기록에 추가
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                        
                        # 자동 저장
                        if st.session_state.auto_save_enabled and st.session_state.supabase_client:
                            user_id = st.session_state.authenticated_user["user_id"]
                            # 첫 번째 질문과 답변이면 세션 제목 생성
                            if len(st.session_state.chat_history) == 2:
                                first_question = st.session_state.chat_history[0].get("content", "")
                                first_answer = st.session_state.chat_history[1].get("content", "")
                                llm = get_llm(
                                    st.session_state.llm_model,
                                    openai_api_key=st.session_state.openai_api_key,
                                    anthropic_api_key=st.session_state.anthropic_api_key,
                                    google_api_key=st.session_state.google_api_key
                                )
                                title = generate_session_title(first_question, first_answer, llm)
                            else:
                                # 기존 세션이 있으면 제목 가져오기
                                if st.session_state.current_session_id:
                                    session_data = load_session_from_supabase(
                                        st.session_state.supabase_client,
                                        user_id,
                                        st.session_state.current_session_id
                                    )
                                    title = session_data.get("title", f"세션 {datetime.now().strftime('%Y-%m-%d %H:%M')}") if session_data else f"세션 {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                                else:
                                    title = f"세션 {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                            
                            session_id = save_session_to_supabase(
                                st.session_state.supabase_client,
                                user_id,
                                title,
                                st.session_state.chat_history,
                                st.session_state.processed_files,
                                st.session_state.current_session_id
                            )
                            
                            if session_id:
                                st.session_state.current_session_id = session_id
                                st.session_state.session_list = get_all_sessions(st.session_state.supabase_client, user_id)
                
                except Exception as e:
                    with st.chat_message("assistant"):
                        st.write(f"오류가 발생했습니다: {str(e)}")
                    st.session_state.chat_history.append({"role": "assistant", "content": f"오류가 발생했습니다: {str(e)}"})
                    logger.error(f"RAG 답변 생성 오류: {e}")
        else:
            with st.chat_message("assistant"):
                st.warning("RAG를 사용하려면 먼저 PDF 파일을 업로드하고 처리해주세요.")
            st.session_state.chat_history.append({"role": "assistant", "content": "RAG를 사용하려면 먼저 PDF 파일을 업로드하고 처리해주세요."})
else:
    st.info("로그인 후 챗봇을 사용할 수 있습니다.")

if __name__ == "__main__":
    pass
