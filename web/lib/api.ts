// 같은 출처의 /api를 부른다. k8s에서는 Ingress가, 로컬에서는 Next rewrite가 api로 넘긴다.
const BASE = process.env.NEXT_PUBLIC_API_BASE ?? "";

// 서버 오류 본문: { detail: 사람이 읽을 문장, code: 원인 코드, ...추가 정보 }
export class ApiError extends Error {
  status: number;
  code: string;
  info: Record<string, any>;
  constructor(status: number, code: string, message: string, info: Record<string, any> = {}) {
    super(message);
    this.status = status;
    this.code = code;
    this.info = info;
  }
}

async function j(path: string, init?: RequestInit) {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
    });
  } catch {
    throw new ApiError(0, "network", "서버에 연결하지 못했습니다.");
  }
  if (!res.ok) {
    let body: any = null;
    try {
      body = await res.json();
    } catch {
      /* 프록시가 만든 HTML 오류 페이지 등 */
    }
    const code = body?.code ?? (res.status === 504 ? "gateway_timeout" : `http_${res.status}`);
    const detail = typeof body?.detail === "string" ? body.detail : res.statusText;
    throw new ApiError(res.status, code, detail, body ?? {});
  }
  return res.json();
}

export const api = {
  createBrief: (goal: string) =>
    j("/api/projects", { method: "POST", body: JSON.stringify({ goal }) }),
  getProject: (id: string) => j(`/api/projects/${id}`),
  listProjects: () => j("/api/projects"),
  usage: () => j("/api/usage"),
  // 접수만 하고 바로 돌아온다. 같은 자료면 저장된 결과가 바로 온다.
  startAnalysis: (project_id: string, content: string, source_url?: string) =>
    j("/api/analyze/jobs", {
      method: "POST",
      body: JSON.stringify({ project_id, content, source_url }),
    }),
  getContent: (contentId: string) => j(`/api/contents/${contentId}`),
  listContents: (projectId: string) => j(`/api/projects/${projectId}/contents`),
};

function fmtTime(iso?: string) {
  if (!iso) return "";
  const d = new Date(iso);
  return `${d.getMonth() + 1}월 ${d.getDate()}일 ${String(d.getHours()).padStart(2, "0")}:${String(
    d.getMinutes(),
  ).padStart(2, "0")}`;
}

// 원인별 안내. 사용자가 '내 탓인지 / 기다리면 되는지 / 관리자에게 알려야 하는지' 알 수 있게 한다.
export function explainError(code: string, detail: string, info: Record<string, any> = {}) {
  const wait = info.retry_after ? `${info.retry_after}초쯤 뒤에` : "잠시 뒤에";
  const map: Record<string, string> = {
    content_too_long: `자료가 너무 깁니다(최대 ${(info.max_chars ?? 60000).toLocaleString()}자). 관련된 장이나 부분만 나눠서 넣어 주세요.`,
    content_too_short: "자료가 너무 짧아 판단할 수 없습니다. 본문을 조금 더 넣어 주세요.",
    content_url_only:
      "URL만 들어왔습니다. 이 서비스는 주소의 내용을 가져오지 않습니다. 기사·리포트 본문을 복사해 붙여넣어 주세요.",
    content_empty: "자료 본문이 비어 있습니다.",
    budget_exceeded: info.needed
      ? `오늘 남은 분석 한도로는 이 자료를 끝까지 분석할 수 없습니다. 자료를 줄이거나 ${fmtTime(info.resets_at)} 이후 다시 시도해 주세요.`
      : `오늘 분석 한도를 모두 썼습니다. ${fmtTime(info.resets_at)}에 다시 채워집니다.`,
    llm_overloaded: `AI 서비스 쪽이 일시적으로 혼잡합니다(사용자 쪽 문제 아님). ${wait} 다시 시도해 주세요.`,
    llm_rate_limited: `AI 서비스 요청이 몰렸습니다. ${wait} 다시 시도해 주세요.`,
    llm_circuit_open: `AI 서비스 장애가 이어져 잠시 호출을 멈췄습니다. ${wait} 다시 시도해 주세요.`,
    embedding_unavailable: `자료 검색 서비스가 혼잡합니다. ${wait} 다시 시도해 주세요.`,
    llm_auth: "서버의 AI 서비스 설정(키·권한)에 문제가 있습니다. 관리자에게 알려 주세요.",
    llm_billing: "AI 서비스 결제·크레딧 문제로 분석할 수 없습니다. 관리자에게 알려 주세요.",
    llm_config: "서버 설정이 끝나지 않았습니다. 관리자에게 알려 주세요.",
    llm_model_unavailable: "설정된 AI 모델을 쓸 수 없습니다. 관리자에게 알려 주세요.",
    llm_input_too_long: "자료가 AI가 한 번에 읽을 수 있는 길이를 넘었습니다. 나눠서 넣어 주세요.",
    llm_output_truncated: "AI 응답이 길이 제한에서 잘렸습니다. 자료를 나눠서 다시 시도해 주세요.",
    llm_schema_invalid: "AI 응답 형식이 맞지 않아 결과를 만들지 못했습니다. 다시 시도해 주세요.",
    analysis_in_progress: "같은 자료를 이미 분석하고 있습니다. 끝나면 아래 목록에 나타납니다.",
    analysis_interrupted: "서버 재시작 등으로 분석이 중간에 끊겼습니다. 같은 자료로 다시 시도해 주세요.",
    invalid_project_id: "프로젝트 주소가 올바르지 않습니다.",
    project_not_found: "프로젝트를 찾을 수 없습니다.",
    gateway_timeout:
      "응답을 기다리다 연결이 끊겼습니다. 서버는 계속 처리 중일 수 있으니 잠시 뒤 새로고침해 보세요.",
    network: "서버에 연결하지 못했습니다. 네트워크 연결을 확인해 주세요.",
  };
  return map[code] ?? detail ?? "알 수 없는 오류가 발생했습니다.";
}
