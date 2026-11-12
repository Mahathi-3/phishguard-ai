"use client";
import { useEffect, useState } from "react";
import { apiFetch } from "@/lib/api";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts";
import { Spinner, ErrorBox } from "./Dashboard";

interface TransformResult {
  no_defense: Record<string, number>; with_defense: Record<string, number>;
  delta_f1_no_defense: number; delta_f1_with_defense: number;
  delta_fpr_no_defense: number; delta_fpr_with_defense: number;
  flipped_predictions: number; flipped_after_defense: number;
  examples: { original: string; perturbed: string; original_pred: number; perturbed_pred: number }[];
}
interface RobustnessData { original: Record<string, number>; transformations: Record<string, TransformResult> }

export default function RobustnessLab() {
  const [data, setData] = useState<RobustnessData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [sel, setSel] = useState<string | null>(null);

  useEffect(() => { apiFetch<RobustnessData>("/robustness").then(setData).catch(e => setError(e.message)).finally(() => setLoading(false)); }, []);

  if (loading) return <Spinner />;
  if (error) return <ErrorBox msg={error} />;
  if (!data) return <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-8 text-center text-gray-400">No robustness data. Run run_experiments.py first.</div>;

  const chartData = Object.entries(data.transformations).map(([name, t]) => ({
    name: name.replace(/_/g, " "), "No Defense": +(t.no_defense.f1*100).toFixed(1),
    "With Defense": +(t.with_defense.f1*100).toFixed(1), Original: +(data.original.f1*100).toFixed(1),
  }));

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">Robustness Lab</h2>
      <div className="bg-slate-900 rounded-lg p-6 border border-slate-700">
        <h3 className="text-lg font-semibold mb-3">Original Test Metrics (Baseline)</h3>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4 text-sm">
          {Object.entries(data.original).map(([k, v]) => (
            <div key={k}><p className="text-gray-400 capitalize">{k}</p><p className="text-lg font-bold text-cyan-400">{(v*100).toFixed(2)}%</p></div>
          ))}
        </div>
      </div>

      <div className="bg-slate-900 rounded-lg p-6 border border-slate-700">
        <h3 className="text-lg font-semibold mb-3">F1 Score Impact by Transformation</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="name" stroke="#94a3b8" fontSize={10} angle={-20} textAnchor="end" height={60} />
            <YAxis domain={[90, 100]} stroke="#94a3b8" fontSize={12} />
            <Tooltip contentStyle={{ backgroundColor: "#1e293b", border: "1px solid #475569", borderRadius: 8 }} />
            <Legend />
            <Bar dataKey="Original" fill="#6b7280" name="Original" />
            <Bar dataKey="No Defense" fill="#ef4444" name="No Defense" />
            <Bar dataKey="With Defense" fill="#06b6d4" name="With Defense" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="bg-slate-900 rounded-lg p-6 border border-slate-700">
        <h3 className="text-lg font-semibold mb-3">Detailed Results</h3>
        <div className="overflow-x-auto"><table className="w-full text-sm"><thead><tr className="border-b border-slate-600">
          <th className="text-left py-2 px-3 text-gray-400">Transformation</th>
          <th className="text-left py-2 px-3 text-gray-400">F1 (No Def)</th>
          <th className="text-left py-2 px-3 text-gray-400">F1 (With Def)</th>
          <th className="text-left py-2 px-3 text-gray-400">ΔF1</th>
          <th className="text-left py-2 px-3 text-gray-400">ΔFPR</th>
          <th className="text-left py-2 px-3 text-gray-400">Flipped</th>
          <th className="text-left py-2 px-3 text-gray-400">Flipped (Def)</th>
          <th className="text-left py-2 px-3 text-gray-400">Details</th>
        </tr></thead><tbody>
          {Object.entries(data.transformations).map(([name, t]) => (
            <tr key={name} className="border-b border-slate-700">
              <td className="py-2 px-3 font-semibold capitalize">{name.replace(/_/g, " ")}</td>
              <td className="py-2 px-3">{(t.no_defense.f1*100).toFixed(1)}%</td>
              <td className="py-2 px-3">{(t.with_defense.f1*100).toFixed(1)}%</td>
              <td className={`py-2 px-3 ${t.delta_f1_no_defense < 0 ? "text-red-400" : "text-green-400"}`}>{(t.delta_f1_no_defense*100).toFixed(2)}%</td>
              <td className={`py-2 px-3 ${t.delta_fpr_no_defense > 0 ? "text-red-400" : "text-green-400"}`}>{(t.delta_fpr_no_defense*100).toFixed(2)}%</td>
              <td className="py-2 px-3">{t.flipped_predictions}</td>
              <td className="py-2 px-3">{t.flipped_after_defense}</td>
              <td className="py-2 px-3"><button onClick={() => setSel(sel === name ? null : name)} className="text-cyan-400 hover:text-cyan-300 text-xs">{sel === name ? "Hide" : "Show"}</button></td>
            </tr>
          ))}
        </tbody></table></div>
      </div>

      {sel && data.transformations[sel]?.examples.length > 0 && (
        <div className="bg-slate-900 rounded-lg p-6 border border-slate-700">
          <h3 className="text-lg font-semibold mb-3 capitalize">{sel.replace(/_/g, " ")} — Flipped Examples</h3>
          <div className="space-y-4">{data.transformations[sel].examples.map((ex, i) => (
            <div key={i} className="grid grid-cols-1 md:grid-cols-2 gap-4 bg-slate-800 rounded-lg p-4 border border-slate-600">
              <div><p className="text-xs text-gray-400 mb-1">Original (pred: {ex.original_pred === 1 ? "Phishing" : "Legit"})</p><p className="text-sm text-gray-300 font-mono break-all">{ex.original}</p></div>
              <div><p className="text-xs text-gray-400 mb-1">Perturbed (pred: {ex.perturbed_pred === 1 ? "Phishing" : "Legit"})</p><p className="text-sm text-gray-300 font-mono break-all">{ex.perturbed}</p></div>
            </div>
          ))}</div>
        </div>
      )}
    </div>
  );
}
