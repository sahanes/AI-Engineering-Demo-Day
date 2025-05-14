import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Matrix Health Terminal",
  description: "A Matrix-style terminal interface for mental health counseling",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet" />
      </head>
      <body>
        <div className="matrix-container">
          <header className="matrix-header">
            <h1 className="matrix-title">NEURAL HEALTH INTERFACE v1.0</h1>
          </header>
        {children}
          <footer className="text-center py-4 text-xs opacity-50">
            SECURE CONNECTION ESTABLISHED • ALL CONVERSATIONS ENCRYPTED
          </footer>
        </div>
      </body>
    </html>
  );
}
