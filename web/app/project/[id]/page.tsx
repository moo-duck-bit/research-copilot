"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useParams } from "next/navigation";
import { api, ApiError, explainError } from "../../../lib/api";
import { GoalSummary } from "../../../components/GoalSummary";
import { HypothesisTree } from "../../../components/HypothesisTree";
import { AnalysisResult } from "../../../components/AnalysisResult";

const URL_ONLY = /^\s*(https?:\/\/\S+|www\.\S+)\s*$/i;
const POLL_MS = 2500;

type Item = {
  content_id: string;
  status: "processing" | "done" | "failed";
  source_url?: string | null;
  preview?: string;
  created_at?: string;
  result?: any;
  error?: { code: string; message: string };
  reused?: boolean;
};

function Elapsed({ since }: { since?: string }) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, []);
  if (!since) return null;
  const s = Math.max(0, Math.round((now - new Date(since).getTime()) / 1000));
  return <span>{s < 60 ? `${s}초` : `${Math.floor(s / 60)}분 ${s % 60}초`} 경과</span>;
}

export default function ProjectPage() {
  const params = useParams();
  const id = String(params.id);

  const [project, setProject] = useState<any>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [content, setContent] = useState("");
  const [sourceUrl, setSourceUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [items, setItems] = useState<Item[]>([]);
  const [usage, setUsage] = useState<any>(null);
  const maxChars: number = usage?.max_content_chars ?? 60000;
  const itemsRef = useRef<Item[]>([]);
  itemsRef.current = items;

  const refreshUsage = useCallback(() => {
    api.usage().then(setUsage).catch(() => {});
  }, []);

  useEffect(() => {
    api
      .getProject(id)
      .then(setProject)
      .catch((e: any) => setLoadError(e instanceof ApiError ? explainError(e.code, e.message) : String(e)));
    // 새로고침해도 이전 분석 결과와 진행 중인 작업이 그대로 보이게 서버에서 다시 읽는다.
    api.listContents(id).then(setItems).catch(() => {});
    refreshUsage();
  }, [id, refreshUsage]);

  // 처리 중인 작업이 있으면 주기적으로 상태를 확인한다.
  useEffect(() => {
    const pending = items.filter((x) => x.status === "processing");
    if (pending.length === 0) return;
    const t = setTimeout(async () => {
      const updates = await Promise.all(
        pending.map((p) => api.getContent(p.content_id).catch(() => null)),
      );
      setItems((cur) =>
        cur.map((x) => {
          const u = updates.find((v: any) => v && v.content_id === x.content_id);
          return u ? { ...x, ...u } : x;
        }),
      );
      if (updates.some((u: any) => u && u.status !== "processing")) refreshUsage();
    }, POLL_MS);
    return () => clearTimeout(t);
  }, [items, refreshUsage]);

  const tooLong = content.length > maxChars;
  const urlOnly = URL_ONLY.test(content);

  async function analyze() {
    if (busy) return; // 연속 클릭으로 같은 자료가 두 번 접수되지 않게
    setBusy(true);
    setError(null);
    try {
      const res = await api.startAnalysis(id, content, sourceUrl || undefined);
      const item: Item = {
        content_id: res.content_id,
        status: res.status,
        source_url: sourceUrl || null,
        preview: content.slice(0, 160),
        created_at: new Date().toISOString(),
        result: res.result,
        reused: res.reused,
      };
      setItems((cur) => [item, ...cur.filter((x) => x.content_id !== item.content_id)]);
      // 접수에 성공했을 때만 입력을 비운다. 실패하면 붙여넣은 내용이 그대로 남는다.
      setContent("");
      setSourceUrl("");
    } catch (e: any) {
      if (e instanceof ApiError && e.code === "analysis_in_progress" && e.info.content_id) {
        api.getContent(e.info.content_id).then((u) =>
          setItems((cur) => [u, ...cur.filter((x) => x.content_id !== u.content_id)]),
        );
      }
      setError(
        (e instanceof ApiError ? explainError(e.code, e.message, e.info) : String(e?.message || e)) +
          " 입력한 내용은 그대로 남아 있습니다.",
      );
    } finally {
      setBusy(false);
    }
  }

  if (loadError && !project) return <p className="errline">{loadError}</p>;
  if (!project) return <p className="muted">불러오는 중…</p>;

  const remainingPct =
    usage && usage.budget > 0 ? Math.round((100 * (usage.remaining ?? 0)) / usage.budget) : null;

  return (
    <div className="stack">
      <p className="crumb">
        <a href="/">← 새 프로젝트</a>
      </p>

      {/* 준비: 시스템이 이해한 목표 (사람 말, 내부구조는 접힘) */}
      <div className="card raise">
        <GoalSummary primaryQuestion={project.primary_question} nodes={project.hypothesis_tree} />
        <details className="struct">
          <summary>구조 자세히 보기 (관련성 판정에 쓰는 내부 구조)</summary>
          <div style={{ marginTop: "var(--sp-3)" }}>
            <HypothesisTree nodes={project.hypothesis_tree} />
          </div>
        </details>
      </div>

      {/* 주인공: 자료 → 목표 정렬 */}
      <div className="card">
        <p className="eyebrow">자료 붙여넣기</p>
        <p className="muted" style={{ fontSize: 13, margin: "var(--sp-1) 0 var(--sp-4)" }}>
          기사·리포트·메모 본문을 붙여넣으면 각 목표에 어떻게 정렬되는지 보여드립니다.
        </p>
        <div className="field" style={{ marginBottom: "var(--sp-3)" }}>
          <input
            className="input"
            type="text"
            placeholder="출처 URL (선택, 기록용)"
            value={sourceUrl}
            onChange={(e) => setSourceUrl(e.target.value)}
          />
          <p className="hint">출처를 기록해 두는 칸입니다. 이 주소의 내용을 가져오지는 않으니 본문은 아래에 붙여넣어 주세요.</p>
        </div>
        <div className="field">
          <textarea
            className="textarea"
            rows={6}
            placeholder="자료 본문을 붙여넣으세요."
            value={content}
            onChange={(e) => setContent(e.target.value)}
          />
          <div className="counter-row">
            {urlOnly ? (
              <span className="warnline">URL만 있습니다. 주소의 내용은 가져오지 않으니 본문을 붙여넣어 주세요.</span>
            ) : tooLong ? (
              <span className="warnline">너무 깁니다. 관련된 부분만 나눠서 넣어 주세요.</span>
            ) : (
              <span />
            )}
            <span className={`counter ${tooLong ? "over" : ""}`}>
              {content.length.toLocaleString()} / {maxChars.toLocaleString()}자
            </span>
          </div>
        </div>
        <div style={{ marginTop: "var(--sp-4)" }} className="actions-row">
          <button className="btn" onClick={analyze} disabled={busy || !content.trim() || tooLong || urlOnly}>
            {busy ? "접수 중…" : "이 자료가 어떤 목표와 맞는지 확인"}
          </button>
          {remainingPct !== null && (
            <span className="muted" style={{ fontSize: 12.5 }}>
              오늘 남은 분석 한도 {remainingPct}% · 매일 0시에 다시 채워집니다
            </span>
          )}
        </div>
        {error && <p className="errline" style={{ marginTop: "var(--sp-3)" }}>{error}</p>}
      </div>

      {items.map((it) => (
        <div className="card raise" key={it.content_id}>
          <p className="eyebrow">이 자료가 어느 목표에 정렬되나</p>
          <p className="muted" style={{ fontSize: 12.5, margin: "var(--sp-1) 0 var(--sp-4)" }}>
            {it.source_url || (it.preview ? `${it.preview.slice(0, 60)}…` : "붙여넣은 자료")}
          </p>
          {it.status === "processing" && (
            <div className="progress">
              <div className="spinner" />
              <div>
                <b>분석 중입니다</b> · <Elapsed since={it.created_at} />
                <div className="muted" style={{ fontSize: 13 }}>
                  관련 목표를 고르고, 목표마다 근거 문장을 원문과 대조합니다. 보통 30초~2분 걸립니다.
                  창을 닫거나 새로고침해도 결과는 저장되어 이 화면에 다시 나타납니다.
                </div>
              </div>
            </div>
          )}
          {it.status === "failed" && (
            <p className="errline">
              {explainError(it.error?.code ?? "", it.error?.message ?? "")} 같은 자료를 다시 넣으면 처음부터 다시 분석합니다.
            </p>
          )}
          {it.status === "done" && it.result && (
            <>
              {it.reused && (
                <p className="note" style={{ marginTop: 0 }}>
                  이미 분석한 자료라 저장된 결과를 보여드립니다(추가 사용량 없음).
                </p>
              )}
              <AnalysisResult result={it.result} />
            </>
          )}
        </div>
      ))}
    </div>
  );
}
