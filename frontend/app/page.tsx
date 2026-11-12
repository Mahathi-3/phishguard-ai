"use client";
import { useState } from "react";
import Dashboard from "./components/Dashboard";
import Analyzer from "./components/Analyzer";
import Evaluation from "./components/Evaluation";
import RobustnessLab from "./components/RobustnessLab";

const TABS = ["Dashboard", "Analyzer", "Evaluation", "Robustness Lab"] as const;

export default function Home() {
  const [tab, setTab] = useState<string>("Dashboard");

  return (
    <div className="min-h-screen flex flex-col bg-slate-950 text-gray-100">
      <header className="bg-slate-900 border-b border-slate-700 px-6 py-4">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-cyan-500 rounded-lg flex items-center justify-center text-xl font-bold">🛡️</div>
            <div>
              <h1 className="text-xl font-bold text-white">PhishGuard AI</h1>
              <p className="text-xs text-gray-400">Explainable Phishing Detection</p>
            </div>
          </div>
          <nav className="flex gap-1">
            {TABS.map((t) => (
              <button key={t} onClick={() => setTab(t)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  tab === t ? "bg-cyan-600 text-white" : "text-gray-400 hover:text-white hover:bg-slate-800"
                }`}>{t}</button>
            ))}
          </nav>
        </div>
      </header>
      <main className="flex-1 max-w-7xl mx-auto w-full p-6">
        {tab === "Dashboard" && <Dashboard />}
        {tab === "Analyzer" && <Analyzer />}
        {tab === "Evaluation" && <Evaluation />}
        {tab === "Robustness Lab" && <RobustnessLab />}
      </main>
      <footer className="bg-slate-900 border-t border-slate-700 px-6 py-3 text-center text-xs text-gray-500">
        PhishGuard AI — Automated assessment, not a guarantee of safety
      </footer>
    </div>
  );
}
