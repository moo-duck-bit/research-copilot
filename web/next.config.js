/** @type {import('next').NextConfig} */
// API 내부 주소. compose와 k8s 모두 서비스 이름이 "api"라서 기본값으로 둘 다 동작한다.
const API_INTERNAL_URL = process.env.API_INTERNAL_URL || "http://api:8000";

const nextConfig = {
  reactStrictMode: true,
  // 런타임에 필요한 파일만 모은 server.js를 만든다(작은 이미지, node_modules 없이 실행).
  output: "standalone",
  // 자료 판정은 LLM을 여러 번 불러 30초(기본 프록시 제한)를 넘길 수 있다.
  experimental: { proxyTimeout: 300_000 },
  async rewrites() {
    // 브라우저는 같은 출처의 /api만 부르고, Next 서버가 api 서비스로 넘긴다.
    // k8s에서는 Ingress가 /api를 먼저 api로 보내므로 이 규칙은 로컬용 보조 경로다.
    return [{ source: "/api/:path*", destination: `${API_INTERNAL_URL}/api/:path*` }];
  },
};

module.exports = nextConfig;
