"use client";
import { useEffect, useState } from "react";
import { apiFetch } from "@/lib/api";

interface Metrics {
  metadata: { dataset: string; total_rows: number; phishing_count: number; legitimate_count: number; test_size: number; date: string };
  logistic_regression: { accuracy: number; f1: number; precision: number; recall: number };
}

export default function Dashboard() {
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [health, setHealth] = useState<{ status: string; models_loaded: boolean } | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [sessionCount, setSessionCount] = useState(0);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      apiFetch<Metrics>("/metrics").catch(() => null),
      apiFetch<{ status: string; models_loaded: boolean }>("/health").catch(() => null),
    ]).then(([m, h]) => { setMetrics(m); setHealth(h); setLoading(false); })
      .catch(() => { setError("Failed to connect to backend"); setLoading(false); });
  }, []);

  useEffect(() => { setSessionCount(parseInt(sessionStorage.getItem("phishguard_count") || "0")); }, []);

  if (loading) return <Spinner />;
  if (error) return <ErrorBox msg={error} />;
  const lr = metrics?.logistic_regression;

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">Dashboard</h2>
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card title="System Status" value={health?.status === "ok" ? "Online" : "Offline"} color={health?.status === "ok" ? "text-green-400" : "text-red-400"} />
        <Card title="Models Loaded" value={health?.models_loaded ? "Yes" : "No"} color={health?.models_loaded ? "text-green-400" : "text-yellow-400"} />
        <Card title="Session Analyses" value={String(sessionCount)} color="text-cyan-400" />
        <Card title="Test F1 Score" value={lr ? (lr.f1 * 100).toFixed(1) + "%" : "N/A"} color="text-cyan-400" />
      </div>
      {metrics && (
        <>
          <div className="bg-slate-900 rounded-lg p-6 border border-slate-700">
            <h3 className="text-lg font-semibold mb-3">Dataset Information</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <Stat label="Dataset" value={metrics.metadata.dataset} />
              <Stat label="Total Rows" value={metrics.metadata.total_rows.toLocaleString()} />
              <Stat label="Phishing" value={metrics.metadata.phishing_count.toLocaleString()} />
              <Stat label="Legitimate" value={metrics.metadata.legitimate_count.toLocaleString()} />
            </div>
          </div>
          <div className="bg-slate-900 rounded-lg p-6 border border-slate-700">
            <h3 className="text-lg font-semibold mb-3">Model Performance (Test Set)</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <Stat label="Accuracy" value={(lr!.accuracy * 100).toFixed(2) + "%"} />
              <Stat label="Precision" value={(lr!.precision * 100).toFixed(2) + "%"} />
              <Stat label="Recall" value={(lr!.recall * 100).toFixed(2) + "%"} />
              <Stat label="F1 Score" value={(lr!.f1 * 100).toFixed(2) + "%"} />
            </div>
          </div>
        </>
      )}
      {!metrics && <Empty msg="No metrics available yet." sub="Run python backend/scripts/train_model.py to train the model." />}
    </div>
  );
}

function Card({ title, value, color }: { title: string; value: string; color: string }) {
  return (<div className="bg-slate-900 rounded-lg p-4 border border-slate-700"><p className="text-xs text-gray-400 uppercase tracking-wide">{title}</p><p className={`text-2xl font-bold mt-1 ${color}`}>{value}</p></div>);
}
function Stat({ label, value }: { label: string; value: string }) {
  return (<div><p className="text-gray-400">{label}</p><p className="font-semibold text-white">{value}</p></div>);
}
function Spinner() { return <div className="flex items-center justify-center h-64"><div className="animate-spin rounded-full h-12 w-12 border-4 border-cyan-500 border-t-transparent"></div></div>; }
function ErrorBox({ msg }: { msg: string }) { return <div className="bg-red-900/30 border border-red-700 rounded-lg p-4 text-red-300">{msg}</div>; }
function Empty({ msg, sub }: { msg: string; sub: string }) { return <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-8 text-center text-gray-400"><p className="text-lg">{msg}</p><p className="text-sm mt-2">{sub}</p></div>; }

export { Spinner, ErrorBox, Empty, Stat };
