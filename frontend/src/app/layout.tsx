import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "GoldMind AI — AI Forex & Gold Trading Analysis Assistant",
  description: "Quantitative decision-support and capital protection assistant for XAUUSD & Forex trading. Real-time market structure, non-repainting Pine Script v6 alerts, and small-account risk sizing.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark h-full">
      <body className="min-h-full bg-[#07090e] text-slate-100 antialiased selection:bg-amber-500/30 selection:text-amber-200">
        {children}
      </body>
    </html>
  );
}
