"""애플리케이션 설정. 모든 비밀값은 환경변수로만 받는다 (하드코딩 금지)."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM (Anthropic Claude) ---
    # API 키는 반드시 환경변수 ANTHROPIC_API_KEY 로만 주입한다.
    anthropic_api_key: str = ""
    # 프로토타입은 Claude Sonnet 5 단일 모델만 사용한다.
    claude_model: str = "claude-sonnet-5"

    # --- Embedding (후보 필터 전용, app/llm 추상화 뒤) ---
    # Anthropic은 임베딩 API가 없어 Voyage AI를 사용한다.
    voyage_api_key: str = ""
    embedding_model: str = "voyage-3"
    embedding_dim: int = 1024

    # --- Goal drift ---
    # 미분류(약한관련+무관) 자료 비율이 이 값 이상이면 드리프트 신호.
    drift_unclassified_threshold: float = 0.5

    # --- Database (PostgreSQL + pgvector) ---
    database_url: str = "postgresql+asyncpg://copilot:copilot@db:5432/copilot"

    # 시작 시 DB가 아직 안 떠 있으면 이 시간까지 재시도하며 기다린다(바로 죽어 재시작 루프가 되지 않게).
    db_init_max_wait_seconds: float = 60.0
    # readiness가 DB 응답을 기다리는 최대 시간(초). 멈춘 DB에 매달리지 않게 한다.
    readiness_db_timeout_seconds: float = 2.0

    # --- MCP 전송 ---
    # 무상태 HTTP: 요청마다 독립 처리해 세션을 파드 메모리에 두지 않는다.
    # 파드가 여러 개일 때 다른 파드로 간 요청이 세션을 못 찾아 404가 나는 문제를 막는다.
    mcp_stateless_http: bool = True

    # --- CORS ---
    # 웹은 같은 출처(/api 프록시·Ingress)로 API를 부르므로 기본은 비워 둔다(미들웨어 없음).
    # 다른 출처에서 직접 부를 일이 있을 때만 CORS_ORIGINS='["https://..."]'로 연다.
    cors_origins: list[str] = []

    # --- LLM 호출 보호 (비용 급증 방지) ---
    llm_timeout_seconds: float = 120.0
    # 429(요청 한도): 서버가 준 retry-after만큼 기다렸다가 재시도. 상한을 넘으면 즉시 실패.
    llm_rate_limit_retries: int = 2
    llm_retry_after_cap_seconds: float = 30.0
    # 529(과부하)·5xx·연결 오류: 지수 백오프로 짧게 재시도.
    llm_transient_retries: int = 2
    llm_backoff_base_seconds: float = 1.0
    # 서킷브레이커: 연속 일시 장애가 이만큼 쌓이면 cooldown 동안 호출 자체를 막는다.
    llm_breaker_failure_threshold: int = 5
    llm_breaker_cooldown_seconds: float = 60.0
    # 일일 토큰 예산(입력+출력). 0이면 무제한. 날짜 경계는 UTC+offset(기본 KST).
    daily_token_budget: int = 2_000_000
    budget_utc_offset_hours: int = 9
    # 이 프로세스가 동시에 보내는 LLM 요청 수 상한(자료 1건이 노드 수만큼 병렬 호출하므로).
    llm_max_concurrency: int = 4

    # --- 입력 제한 ---
    # 자료 길이 상한(글자). 넘으면 분석 전에 거절한다(비용·지연·예산 보호).
    max_content_chars: int = 60_000
    min_content_chars: int = 30
    max_goal_chars: int = 4_000
    # 이보다 긴 자료는 노드마다 전문 대신 관련 청크와 앞뒤만 발췌해 보낸다.
    full_document_max_chars: int = 8_000
    # 자료 1건당 판정할 후보 노드 수 상한. 노드 하나당 LLM 호출 1번이라 비용과 비례한다.
    max_candidates: int = 7
    excerpt_max_chars: int = 6_000

    # --- 분석 작업 ---
    # 이 시간 넘게 '처리 중'인 작업은 파드 재시작 등으로 끊긴 것으로 보고 다시 돌릴 수 있게 한다.
    analysis_stale_after_seconds: float = 900.0
    # 종료 신호를 받은 뒤 진행 중인 분석을 마저 끝내도록 기다리는 최대 시간.
    shutdown_drain_seconds: float = 280.0

    # --- 임베딩 ---
    embedding_batch_max_items: int = 128
    embedding_batch_max_chars: int = 80_000
    embedding_rate_limit_wait_seconds: float = 20.0
    embedding_max_wait_seconds: float = 45.0


settings = Settings()
