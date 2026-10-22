// k8s probe용. 페이지 렌더링 없이 바로 200을 준다.
export const dynamic = "force-dynamic";

export function GET() {
  return new Response("ok", { status: 200, headers: { "content-type": "text/plain" } });
}
