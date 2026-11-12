"use client";
import { useState } from "react";
import { apiFetch } from "@/lib/api";

interface AnalyzeResult {
  prediction: number; label: string; probability: number;
  calibrated_probability: number; risk_category: string;
  indicators: { name: string; category: string; severity: string; evidence_snippet: string; explanation: string }[];
  warning: string;
}
interface ExplainResult {
  prediction: number; probability: number; indicators: any[];
  explanation: {
    summary: string; risk_interpretation: string;
    evidence_based_findings: string[]; uncertainty: string;
    recommended_actions: string[]; explanation_mode: string;
    verification: { claim: string; status: string; reason: string }[];
    verification_limits: string; disagreement?: string;
  };
}

const SAMPLE_PHISHING = `URGENT: Your PayPal account has been limited due to suspicious activity. You must verify your identity within 24 hours or your account will be permanently suspended.\n\nClick here to verify: http://paypal.security-check-2024.com/verify\n\nYou will need to provide:\n- Your login credentials\n- Credit card number and CVV\n- Social Security Number\n\nFailure to comply will result in immediate account closure.\n\nPayPal Security Team`;

const SAMPLE_LEGIT = `Hi Team,\n\nJust a reminder that our quarterly project review meeting is scheduled for this Friday at 2 PM in Conference Room B.\n\nPlease come prepared with your status updates and any blockers you'd like to discuss. I've attached the agenda for reference.\n\nIf you can't make it, please let me know and we'll arrange a catch-up.\n\nThanks,\nSarah Johnson\nProject Manager`;

export default function Analyzer() {
  const [text, setText] = useState("");
  const [analysis, setAnalysis] = useState<AnalyzeResult | null>(null);
  const [explanation, setExplanation] = useState<ExplainResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [explainLoading, setExplainLoading] = useState(false);
  const [error, setError] = useState("");

  const analyze = async () => {
    if (!text.trim()) { setError("Please enter text to analyze."); return; }
    if (text.trim().length < 3) { setError("Text too short for meaningful analysis."); return; }
    setLoading(true); setError(""); setAnalysis(null); setExplanation(null);
    try {
      const res = await apiFetch<AnalyzeResult>("/analyze", { method: "POST", body: JSON.stringify({ text: text.trim() }) });
      setAnalysis(res);
      const c = parseInt(sessionStorage.getItem("phishguard_count") || "0") + 1;
      sessionStorage.setItem("phishguard_count", String(c));
      setExplainLoading(true);
      try {
        const exp = await apiFetch<ExplainResult>("/explain", { method: "POST", body: JSON.stringify({ text: text.trim() }) });
        setExplanation(exp);
      } catch { /* explanation is optional */ }
      setExplainLoading(false);
    } catch (e: any) { setError(e.message || "Analysis failed"); }
    setLoading(false);
  };

  const clear = () => { setText(""); setAnalysis(null); setExplanation(null); setError(""); };
  const riskColor = (cat: string) => {
    switch (cat) {
      case "high": return "text-red-400 bg-red-900/30 border-red-700";
      case "medium": return "text-yellow-400 bg-yellow-900/30 border-yellow-700";
      case "uncertain": return "text-orange-400 bg-orange-900/30 border-orange-700";
      default: return "text-green-400 bg-green-900/30 border-green-700";
    }
  };
  const sevColor = (s: string) => s === "high" ? "text-red-400" : s === "medium" ? "text-yellow-400" : "text-blue-400";

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">Email/Message Analyzer</h2>
      <div className="bg-slate-900 rounded-lg p-6 border border-slate-700">
        <textarea value={text} onChange={(e) => setText(e.target.value)}
          placeholder="Paste an email or message to analyze..."
          className="w-full h-40 bg-slate-800 text-gray-100 rounded-lg p-4 border border-slate-600 focus:border-cyan-500 focus:outline-none resize-y placeholder-gray-500"
          maxLength={50000} />
        <div className="flex flex-wrap gap-3 mt-3">
          <button onClick={analyze} disabled={loading}
            className="px-6 py-2 bg-cyan-600 hover:bg-cyan-700 text-white rounded-lg font-medium disabled:opacity-50 transition-colors">
            {loading ? "Analyzing..." : "Analyze"}</button>
          <button onClick={clear} className="px-6 py-2 bg-slate-700 hover:bg-slate-600 text-gray-300 rounded-lg font-medium transition-colors">Clear</button>
          <button onClick={() => setText(SAMPLE_PHISHING)} className="px-4 py-2 bg-red-900/30 hover:bg-red-900/50 text-red-300 rounded-lg text-sm border border-red-800 transition-colors">Sample Phishing</button>
          <button onClick={() => setText(SAMPLE_LEGIT)} className="px-4 py-2 bg-green-900/30 hover:bg-green-900/50 text-green-300 rounded-lg text-sm border border-green-800 transition-colors">Sample Legitimate</button>
        </div>
        <p className="text-xs text-gray-500 mt-2">{text.length}/50000 characters</p>
      </div>

      {error && <div className="bg-red-900/30 border border-red-700 rounded-lg p-4 text-red-300">{error}</div>}
      {loading && <div className="flex items-center justify-center h-32"><div className="animate-spin rounded-full h-10 w-10 border-4 border-cyan-500 border-t-transparent"></div></div>}

      {analysis && (
        <div className="space-y-4">
          <div className={`rounded-lg p-6 border ${riskColor(analysis.risk_category)}`}>
            <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
              <div><p className="text-sm uppercase tracking-wide opacity-70">Prediction</p><p className="text-3xl font-bold">{analysis.label}</p></div>
              <div className="text-right"><p className="text-sm opacity-70">Calibrated Probability</p><p className="text-3xl font-bold">{(analysis.calibrated_probability * 100).toFixed(1)}%</p></div>
              <div className="text-right"><p className="text-sm opacity-70">Risk Category</p><p className="text-2xl font-bold capitalize">{analysis.risk_category}</p></div>
            </div>
          </div>

          <div className="bg-slate-900 rounded-lg p-6 border border-slate-700">
            <h3 className="text-lg font-semibold mb-3">Security Indicators</h3>
            {analysis.indicators.length === 0 ? <p className="text-gray-400">No specific security indicators detected.</p> : (
              <div className="space-y-3">{analysis.indicators.map((ind, i) => (
                <div key={i} className="bg-slate-800 rounded-lg p-4 border border-slate-600">
                  <div className="flex items-center gap-2 mb-1">
                    <span className={`text-sm font-bold ${sevColor(ind.severity)}`}>[{ind.severity.toUpperCase()}]</span>
                    <span className="font-semibold">{ind.name}</span>
                    <span className="text-xs text-gray-500 ml-auto">{ind.category}</span>
                  </div>
                  <p className="text-sm text-gray-300">{ind.explanation}</p>
                  <p className="text-xs text-gray-500 mt-1 font-mono">Evidence: &quot;{ind.evidence_snippet}&quot;</p>
                </div>
              ))}</div>
            )}
          </div>

          {explainLoading && <div className="bg-slate-900 rounded-lg p-6 border border-slate-700"><div className="flex items-center gap-3"><div className="animate-spin rounded-full h-5 w-5 border-2 border-cyan-500 border-t-transparent"></div><span className="text-gray-400">Generating explanation...</span></div></div>}

          {explanation && (
            <div className="bg-slate-900 rounded-lg p-6 border border-slate-700 space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold">AI Explanation</h3>
                <span className={`text-xs px-2 py-1 rounded ${explanation.explanation.explanation_mode === "gemini" ? "bg-purple-900/50 text-purple-300 border border-purple-700" : "bg-gray-800 text-gray-400 border border-gray-700"}`}>
                  {explanation.explanation.explanation_mode === "gemini" ? "Gemini LLM" : "Local explanation mode"}</span>
              </div>
              {explanation.explanation.disagreement && <div className="bg-yellow-900/30 border border-yellow-700 rounded p-3 text-yellow-300 text-sm">⚠️ {explanation.explanation.disagreement}</div>}
              <Section title="Summary" content={explanation.explanation.summary} />
              <Section title="Risk Interpretation" content={explanation.explanation.risk_interpretation} />
              <div><p className="text-sm text-gray-400 uppercase tracking-wide mb-1">Evidence-Based Findings</p>
                <ul className="list-disc list-inside space-y-1 text-gray-300 text-sm">{explanation.explanation.evidence_based_findings.map((f, i) => <li key={i}>{f}</li>)}</ul></div>
              <Section title="Uncertainty" content={explanation.explanation.uncertainty} />
              <div><p className="text-sm text-gray-400 uppercase tracking-wide mb-1">Recommended Actions</p>
                <ul className="list-disc list-inside space-y-1 text-gray-300 text-sm">{explanation.explanation.recommended_actions.map((a, i) => <li key={i}>{a}</li>)}</ul></div>
              <div><p className="text-sm text-gray-400 uppercase tracking-wide mb-2">Verification</p>
                <div className="overflow-x-auto"><table className="w-full text-sm"><thead><tr className="border-b border-slate-600">
                  <th className="text-left py-2 px-3 text-gray-400">Claim</th><th className="text-left py-2 px-3 text-gray-400">Status</th><th className="text-left py-2 px-3 text-gray-400">Reason</th>
                </tr></thead><tbody>{explanation.explanation.verification.map((v, i) => (
                  <tr key={i} className="border-b border-slate-700"><td className="py-2 px-3 text-gray-300">{v.claim}</td>
                    <td className="py-2 px-3"><span className={`px-2 py-0.5 rounded text-xs ${v.status === "supported" ? "bg-green-900/50 text-green-400" : v.status === "unsupported" ? "bg-red-900/50 text-red-400" : "bg-gray-800 text-gray-400"}`}>{v.status}</span></td>
                    <td className="py-2 px-3 text-gray-400 text-xs">{v.reason}</td></tr>
                ))}</tbody></table></div>
                <p className="text-xs text-gray-500 mt-2 italic">{explanation.explanation.verification_limits}</p>
              </div>
            </div>
          )}
          <div className="bg-amber-900/20 border border-amber-700 rounded-lg p-4 text-amber-300 text-sm">⚠️ {analysis.warning}</div>
        </div>
      )}
      {!analysis && !loading && !error && <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-12 text-center text-gray-400"><p className="text-lg">Paste an email or message above to analyze it for phishing indicators.</p><p className="text-sm mt-2">Or try the sample buttons to see how it works.</p></div>}
    </div>
  );
}

function Section({ title, content }: { title: string; content: string }) {
  return <div><p className="text-sm text-gray-400 uppercase tracking-wide mb-1">{title}</p><p className="text-gray-200">{content}</p></div>;
}
