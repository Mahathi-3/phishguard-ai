"use client";
import { useEffect, useState } from "react";
import { apiFetch } from "@/lib/api";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, LineChart, Line, ReferenceLine } from "recharts";
import { Spinner, ErrorBox } from "./Dashboard";

interface Metrics {
  metadata: { dataset: string; is_demo: boolean; total_rows: number; phishing_count: number; legitimate_count: number; train_size: number; val_size: number; test_size: number; split_method: string; positive_class: string; date: string };
  logistic_regression: MetricSet; multinomial_nb: MetricSet;
}
interface MetricSet {
  model: string; accuracy: number; precision: number; recall: number; f1: number; fpr: number;
  confusion_matrix: number[][]; brier_uncalibrated: number; brier_calibrated: number;
  reliability: { bin_start: number; bin_end: number; mean_predicted: number; mean_actual: number; count: number }[];
  classification_report: Record<string, any>;
}
interface CalibrationData { model: string; brier_before: number; brier_after: number; reliability: { bin_start: number; bin_end: number; mean_predicted: number; mean_actual: number; count: number }[]; note: string }

export default function Evaluation() {
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [calibration, setCalibration] = useState<CalibrationData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    setLoading(true);
    Promise.all([apiFetch<Metrics>("/metrics").catch(() => null), apiFetch<CalibrationData>("/calibration").catch(() => null)])
      .then(([m, c]) => { setMetrics(m); setCalibration(c); setLoading(false); })
      .catch(() => { setError("Failed to load metrics"); setLoading(false); });
  }, []);

  if (loading) return <Spinner />;
  if (error) return <ErrorBox msg={error} />;
  if (!metrics) return <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-8 text-center text-gray-400"><p>No metrics available. Run train_model.py first.</p></div>;

  const lr = metrics.logistic_regression, nb = metrics.multinomial_nb;
  const comparisonData = [
    { name: "Accuracy", LR: +(lr.accuracy*100).toFixed(1), NB: +(nb.accuracy*100).toFixed(1) },
    { name: "Precision", LR: +(lr.precision*100).toFixed(1), NB: +(nb.precision*100).toFixed(1) },
    { name: "Recall", LR: +(lr.recall*100).toFixed(1), NB: +(nb.recall*100).toFixed(1) },
    { name: "F1", LR: +(lr.f1*100).toFixed(1), NB: +(nb.f1*100).toFixed(1) },
  ];
  const reliabilityData = (calibration?.reliability || lr.reliability || []).map(r => ({
    predicted: +(r.mean_predicted*100).toFixed(1), actual: +(r.mean_actual*100).toFixed(1), count: r.count,
  }));

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">Model Evaluation</h2>
      <div className="bg-slate-900 rounded-lg p-6 border border-slate-700">
        <h3 className="text-lg font-semibold mb-3">Experiment Details</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div><p className="text-gray-400">Dataset</p><p className="font-semibold">{metrics.metadata.dataset}</p></div>
          <div><p className="text-gray-400">Split</p><p className="font-semibold">{metrics.metadata.split_method}</p></div>
          <div><p className="text-gray-400">Test Size</p><p className="font-semibold">{metrics.metadata.test_size.toLocaleString()}</p></div>
          <div><p className="text-gray-400">Date</p><p className="font-semibold">{new Date(metrics.metadata.date).toLocaleDateString()}</p></div>
          <div><p className="text-gray-400">Positive Class</p><p className="font-semibold">{metrics.metadata.positive_class}</p></div>
          <div><p className="text-gray-400">Class Balance</p><p className="font-semibold">{metrics.metadata.phishing_count} / {metrics.metadata.legitimate_count}</p></div>
          <div><p className="text-gray-400">Total Rows</p><p className="font-semibold">{metrics.metadata.total_rows.toLocaleString()}</p></div>
          <div><p className="text-gray-400">Demo Dataset</p><p className="font-semibold">{metrics.metadata.is_demo ? "Yes" : "No"}</p></div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-slate-900 rounded-lg p-6 border border-slate-700">
          <h3 className="text-lg font-semibold mb-3">Confusion Matrix (LR)</h3>
          <div className="grid grid-cols-2 gap-2 max-w-xs">
            <div className="bg-green-900/30 border border-green-800 rounded p-3 text-center"><p className="text-xs text-gray-400">TN</p><p className="text-xl font-bold text-green-400">{lr.confusion_matrix[0][0]}</p></div>
            <div className="bg-red-900/30 border border-red-800 rounded p-3 text-center"><p className="text-xs text-gray-400">FP</p><p className="text-xl font-bold text-red-400">{lr.confusion_matrix[0][1]}</p></div>
            <div className="bg-orange-900/30 border border-orange-800 rounded p-3 text-center"><p className="text-xs text-gray-400">FN</p><p className="text-xl font-bold text-orange-400">{lr.confusion_matrix[1][0]}</p></div>
            <div className="bg-green-900/30 border border-green-800 rounded p-3 text-center"><p className="text-xs text-gray-400">TP</p><p className="text-xl font-bold text-green-400">{lr.confusion_matrix[1][1]}</p></div>
          </div>
          <p className="text-xs text-gray-500 mt-2">FPR: {(lr.fpr*100).toFixed(2)}%</p>
        </div>
        <div className="bg-slate-900 rounded-lg p-6 border border-slate-700">
          <h3 className="text-lg font-semibold mb-3">Model Comparison</h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={comparisonData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="name" stroke="#94a3b8" fontSize={12} />
              <YAxis domain={[80, 100]} stroke="#94a3b8" fontSize={12} />
              <Tooltip contentStyle={{ backgroundColor: "#1e293b", border: "1px solid #475569", borderRadius: 8 }} />
              <Legend />
              <Bar dataKey="LR" fill="#06b6d4" name="Logistic Regression" />
              <Bar dataKey="NB" fill="#8b5cf6" name="Multinomial NB" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="bg-slate-900 rounded-lg p-6 border border-slate-700">
        <h3 className="text-lg font-semibold mb-3">Calibration &amp; Reliability</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <div className="grid grid-cols-2 gap-4 mb-4">
              <div className="bg-slate-800 rounded p-3"><p className="text-xs text-gray-400">Brier Score (Before)</p><p className="text-lg font-bold text-yellow-400">{(calibration?.brier_before ?? lr.brier_uncalibrated).toFixed(4)}</p></div>
              <div className="bg-slate-800 rounded p-3"><p className="text-xs text-gray-400">Brier Score (After)</p><p className="text-lg font-bold text-green-400">{(calibration?.brier_after ?? lr.brier_calibrated).toFixed(4)}</p></div>
            </div>
            <p className="text-xs text-gray-500">{calibration?.note || "Calibration fitted on validation set."}</p>
          </div>
          <div>
            <p className="text-sm text-gray-400 mb-2">Reliability Diagram</p>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={reliabilityData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="predicted" stroke="#94a3b8" fontSize={12} />
                <YAxis stroke="#94a3b8" fontSize={12} />
                <Tooltip contentStyle={{ backgroundColor: "#1e293b", border: "1px solid #475569", borderRadius: 8 }} />
                <Line type="monotone" dataKey="actual" stroke="#06b6d4" strokeWidth={2} dot={{ fill: "#06b6d4" }} name="Actual %" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="bg-slate-900 rounded-lg p-6 border border-slate-700">
        <h3 className="text-lg font-semibold mb-3">Classification Report (LR)</h3>
        <div className="overflow-x-auto"><table className="w-full text-sm"><thead><tr className="border-b border-slate-600">
          <th className="text-left py-2 px-3 text-gray-400">Class</th><th className="text-left py-2 px-3 text-gray-400">Precision</th><th className="text-left py-2 px-3 text-gray-400">Recall</th><th className="text-left py-2 px-3 text-gray-400">F1-Score</th><th className="text-left py-2 px-3 text-gray-400">Support</th>
        </tr></thead><tbody>{Object.entries(lr.classification_report).map(([cls, v]: [string, any]) => (
          typeof v === "object" && v.precision !== undefined && (
            <tr key={cls} className="border-b border-slate-700"><td className="py-2 px-3 font-semibold">{cls}</td>
              <td className="py-2 px-3">{(v.precision*100).toFixed(1)}%</td><td className="py-2 px-3">{(v.recall*100).toFixed(1)}%</td>
              <td className="py-2 px-3">{(v["f1-score"]*100).toFixed(1)}%</td><td className="py-2 px-3">{v.support}</td></tr>
          )))}</tbody></table></div>
      </div>
    </div>
  );
}
