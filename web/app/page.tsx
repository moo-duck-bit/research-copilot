"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { api, ApiError, explainError } from "../lib/api";

export default function Home() {
  const router = useRouter();
  const [goal, setGoal] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [projects, setProjects] = useState<any[]>([]);

  useEffect(() => {
    api.listProjects().then(setProjects).catch(() => {});
  }, []);

  async function submit() {
    setBusy(true);
    setError(null);
    try {
      const res = await api.createBrief(goal);
      if (res.status === "needs_clarification") {
        setError("한 가지만 더 알려주세요: " + res.clarification_question);
        setBusy(false);
        return;
      }
      router.push(`/project/${res.project_id}`);
    } catch (e: any) {
      setError(
        e instanceof ApiError ? explainError(e.code, e.message, e.info) : String(e?.message || e),
      );
      setBusy(false);
    }
  }

  return (
    <div className="stack">
      <div>
        <p className="eyebrow">새 프로젝트</p>
        <h1 className="display">무엇을 리서치하나요?</h1>
      </div>

      <div className="card raise">
        <div className="field">
          <span className="label">리서치 목표</span>
          <textarea
            className="textarea"
            rows={3}
            placeholder="예: 국내 중견 리테일 체인의 향후 3년 매출 성장 전략을 도출하고 싶다."
            value={goal}
            onChange={(e) => setGoal(e.target.value)}
          />
        </div>
        <p className="muted" style={{ fontSize: 13, marginTop: "var(--sp-3)" }}>
          목표를 문장으로 알려주세요. 이후 자료를 붙이면 각 자료가 어느 목표에 맞는지 정렬해
          드립니다. 판단과 결론은 언제나 컨설턴트가 합니다.
        </p>
        <div style={{ marginTop: "var(--sp-4)" }}>
          <button className="btn" onClick={submit} disabled={busy || !goal.trim()}>
            {busy ? "목표 정리 중…" : "목표 등록"}
          </button>
        </div>
        {error && <p className="errline" style={{ marginTop: "var(--sp-3)" }}>{error}</p>}
      </div>

      {projects.length > 0 && (
        <div className="card">
          <span className="label">이어서 작업</span>
          <div style={{ marginTop: "var(--sp-3)", display: "flex", flexDirection: "column", gap: "var(--sp-2)" }}>
            {projects.map((p) => (
              <a key={p.project_id} href={`/project/${p.project_id}`}>
                {p.primary_question}
              </a>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
