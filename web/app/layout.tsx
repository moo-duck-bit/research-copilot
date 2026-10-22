import "./globals.css";

export const metadata = {
  title: "Research Copilot",
  description: "전략 컨설턴트용 목표 정렬(goal-alignment) 도구",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="ko">
      <body>
        <header className="appbar">
          <span className="logo" aria-hidden />
          <span className="brand">Research Copilot</span>
          <span className="tagline">목표 정렬 도구</span>
        </header>
        <main className="container">{children}</main>
      </body>
    </html>
  );
}
