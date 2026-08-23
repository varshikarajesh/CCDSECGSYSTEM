import React, { useState, useEffect, useRef } from "react";
import { 
  Activity, 
  Upload, 
  Settings, 
  MessageSquare, 
  BarChart2, 
  AlertCircle, 
  Database, 
  Calendar, 
  Clock, 
  CheckCircle, 
  ArrowRight,
  RefreshCw,
  Sliders,
  AlertTriangle,
  ZoomIn,
  ZoomOut,
  Maximize2,
  Bookmark,
  Send,
  User,
  Heart,
  Usb,
  Play,
  Square,
  Stethoscope
} from "lucide-react";
import { motion, AnimatePresence } from "motion/react";

import AnimatedECGBackground from "./components/AnimatedECGBackground";
import * as api from "./lib/api";

// 12 lead names array
const LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"];
const REPORT_SECTIONS = [["waveform", "Waveform & Metrics"], ["decision", "Decision"], ["retrieval", "Retrieval"], ["chat", "Clinical Q&A"], ["feedback", "Feedback"]] as const;

// Emergency fallback only. The complete registry is loaded from the API.
const LABEL_EXPANSIONS: Record<string, string> = {
  "NORM": "Normal ECG baseline rhythm",
  "1AVB": "First-degree atrioventricular block",
  "2AVB": "Second-degree atrioventricular block",
  "3AVB": "Third-degree atrioventricular block",
  "IRBBB": "Incomplete right bundle branch block",
  "AFIB": "Atrial fibrillation rhythm activity",
  "ISCAL": "Ischemic changes in anterolateral leads",
};

interface Message {
  role: "user" | "assistant";
  content: string;
  intent?: string;
  citations?: any[];
  evidence?: any;
}

export default function App() {
  const getMetric = (result: any, key: string) => {
    if (!result) return undefined;
    
    // Resolve overall stats object from different potential locations
    let overall = undefined;
    if (result.ecg_measurements?.overall) {
      overall = result.ecg_measurements.overall;
    } else if (result.recording_statistics?.overall) {
      overall = result.recording_statistics.overall;
    } else if (result.recording_statistics?.scope === "overall") {
      overall = result.recording_statistics;
    } else if (result.statistics?.whole_recording?.overall) {
      overall = result.statistics.whole_recording.overall;
    }
    
    if (overall) {
      if ((key === "heart_rate_bpm" || key === "hr") && overall.beat_detection?.mean_heart_rate_bpm !== undefined) {
        return overall.beat_detection.mean_heart_rate_bpm;
      }
      if ((key === "qrs_ms" || key === "qrs") && overall.morphology?.qrs_duration_median_ms_estimate !== undefined) {
        return overall.morphology.qrs_duration_median_ms_estimate;
      }
      if ((key === "qtc_ms" || key === "qtc") && overall.morphology?.qtc_bazett_ms !== undefined) {
        return overall.morphology.qtc_bazett_ms;
      }
      if ((key === "pr_ms" || key === "pr") && overall.morphology?.pr_interval_ms !== undefined) {
        return overall.morphology.pr_interval_ms;
      }
      if (key === "mean_rr_ms") return overall.time_domain_hrv?.mean_rr_ms;
      if (key === "sdnn_ms") return overall.time_domain_hrv?.sdnn_ms;
      if (key === "rmssd_ms") return overall.time_domain_hrv?.rmssd_ms;
      if (key === "r_peak_count") return overall.beat_detection?.r_peak_count;
      if (key === "duration_seconds") return overall.duration_seconds;
      if (key === "sampling_rate_hz") return overall.sampling_rate_hz;
    }
    
    // Fallback: Check flat keys in result.ecg_measurements / recording_statistics / statistics
    const flatTargets = [
      result.ecg_measurements,
      result.recording_statistics,
      result.statistics?.whole_recording
    ];
    for (const target of flatTargets) {
      if (!target) continue;
      if (target[key] !== undefined) {
        const val = target[key];
        return typeof val === "object" && val !== null ? val.value : val;
      }
    }
    
    return undefined;
  };

  const formatContradiction = (c: any): string => {
    if (typeof c === "string") return c;
    if (typeof c === "object" && c !== null) {
      if (c.type === "family_mismatch" || c.type === "family_head_disagreement") {
        const classifierFamily = c.classifier_family || "unknown";
        const independentFamily = c.independent_family || "unknown";
        return `Evidence-family disagreement: the multi-label classifier primarily supports the ${classifierFamily} family, while the independent family head favors ${independentFamily}. Both can coexist in a multi-label ECG, so this does not automatically invalidate either finding; it lowers confidence and requires waveform review.`;
      }
      return c.message || JSON.stringify(c);
    }
    return String(c);
  };

  // App view state: "landing" | "workspace"
  const [view, setView] = useState<"landing" | "workspace">("landing");
  const [activeReportSection, setActiveReportSection] = useState("waveform");
  
  // File inputs
  const [file, setFile] = useState<File | null>(null);
  const [mode, setMode] = useState<"10s" | "2min" | "5min">("10s");
  const [samplingRate, setSamplingRate] = useState<number>(100);
  const [inputMode, setInputMode] = useState<"upload" | "live">("upload");
  const [serialPorts, setSerialPorts] = useState<any[]>([]);
  const [selectedSerialPort, setSelectedSerialPort] = useState<any>(null);
  const [serialStatus, setSerialStatus] = useState("No ECG device selected");
  const [baudRate, setBaudRate] = useState(115200);
  const [isRecording, setIsRecording] = useState(false);
  const [recordedFrames, setRecordedFrames] = useState(0);
  const [recordingElapsed, setRecordingElapsed] = useState(0);
  const serialReaderRef = useRef<any>(null);
  const serialPortRef = useRef<any>(null);
  const liveFramesRef = useRef<number[][]>([]);
  const recordingStartedRef = useRef(0);
  const recordingTimerRef = useRef<any>(null);
  const analysisSequenceRef = useRef(0);
  
  // Loading status
  const [loading, setLoading] = useState(false);
  const [loadingStage, setLoadingStage] = useState(0);
  const loadingStages = [
    "Validating ECG signal parameters...",
    "Preprocessing recording waveform...",
    "Running window screening metrics...",
    "Computing diagnostic probabilities...",
    "Retrieving similar FAISS matches...",
    "Synthesizing system consensus...",
    "Generating advisory clinical explanation..."
  ];

  // System services health state
  const [systemHealth, setSystemHealth] = useState<Record<string, any> | null>(null);
  const [healthStatus, setHealthStatus] = useState<"connecting" | "healthy" | "degraded">("connecting");
  const [showHealth, setShowHealth] = useState(false);

  // Active analysis results
  const [recordingId, setRecordingId] = useState<string | null>(null);
  const [analysisResult, setAnalysisResult] = useState<any>(null);
  
  // Real ECG Viewer interaction states
  const [activeWindowIndex, setActiveWindowIndex] = useState<number>(0);
  const [activeWaveform, setActiveWaveform] = useState<any>(null);
  const [leadGain, setLeadGain] = useState<number>(1.0);
  const [timeZoom, setTimeZoom] = useState<number>(1.0);
  const [loadingWaveform, setLoadingWaveform] = useState(false);
  const [analyzingSelectedInterval, setAnalyzingSelectedInterval] = useState(false);
  const [selectedLead, setSelectedLead] = useState<string>("All");
  const [isEcgExpanded, setIsEcgExpanded] = useState<boolean>(false);
  const [viewerStart, setViewerStart] = useState<number>(0);
  const [viewerDuration, setViewerDuration] = useState<number>(10);
  const [labelRegistry, setLabelRegistry] = useState<Record<string, any>>({});
  const [neighborWaveforms, setNeighborWaveforms] = useState<Record<string, any>>({});
  const [neighborLead, setNeighborLead] = useState<string>("II");
  const [retrievalDisplayCount, setRetrievalDisplayCount] = useState<number>(5);
  
  // Interactive Chat State
  const [chatMessages, setChatMessages] = useState<Message[]>([]);
  const [userQuery, setUserQuery] = useState("");
  const [sendingChat, setSendingChat] = useState(false);
  const [floatingChatOpen, setFloatingChatOpen] = useState(false);
  const [floatingQuery, setFloatingQuery] = useState("");
  const [floatingAnswer, setFloatingAnswer] = useState("");
  const [floatingBusy, setFloatingBusy] = useState(false);

  // Feedback State
  const [reviewerId, setReviewerId] = useState("dr_smith");
  const [verdict, setVerdict] = useState("Cannot Determine");
  const [feedbackNotes, setFeedbackNotes] = useState("");
  const [correctedPrimary, setCorrectedPrimary] = useState("");
  const [correctedSecondary, setCorrectedSecondary] = useState("");
  const [correctedFamily, setCorrectedFamily] = useState("Other");
  const [confidenceRating, setConfidenceRating] = useState("Appropriate");
  const [explanationRating, setExplanationRating] = useState("Neutral");
  const [retrievalRatings, setRetrievalRatings] = useState<Record<string, string>>({});
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);

  // PWA installation state
  const [isInstallable, setIsInstallable] = useState(false);
  const [deferredPrompt, setDeferredPrompt] = useState<any>(null);

  // Periodic health checks on startup
  useEffect(() => {
    const checkHealth = async () => {
      const status = await api.checkSystemStatus();
      setSystemHealth(status.details);
      setHealthStatus(status.ready ? "healthy" : "degraded");
    };
    checkHealth();
    api.getLabelRegistry().then(setLabelRegistry).catch(() => setLabelRegistry({}));
    const interval = setInterval(checkHealth, 10000);

    // Listen for PWA installation prompt
    window.addEventListener("beforeinstallprompt", (e) => {
      e.preventDefault();
      setDeferredPrompt(e);
      setIsInstallable(true);
    });

    return () => {
      clearInterval(interval);
    };
  }, []);

  // Update loading stage intervals
  useEffect(() => {
    let interval: any;
    if (loading) {
      interval = setInterval(() => {
        setLoadingStage((prev) => (prev < loadingStages.length - 1 ? prev + 1 : prev));
      }, 1200);
    } else {
      setLoadingStage(0);
    }
    return () => clearInterval(interval);
  }, [loading]);

  // Load the active window waveform automatically when active index changes
  useEffect(() => {
    if (view === "workspace" && recordingId && analysisResult) {
      loadWindowWaveform();
    }
  }, [view, activeWindowIndex, recordingId]);

  useEffect(() => {
    if (view !== "workspace") return;
    const observer = new IntersectionObserver(entries => {
      const visible = entries.filter(entry => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
      if (visible?.target?.id) setActiveReportSection(visible.target.id);
    }, { rootMargin: "-20% 0px -55% 0px", threshold: [0.05, 0.2, 0.5] });
    REPORT_SECTIONS.forEach(([id]) => {
      const element = document.getElementById(id);
      if (element) observer.observe(element);
    });
    return () => observer.disconnect();
  }, [view, analysisResult]);

  const loadWindowWaveform = async () => {
    if (!recordingId || !analysisResult) return;
    
    setLoadingWaveform(true);
    try {
      let start = 0;
      let end = 10;
      
      // If we have sliced windows, extract active window boundaries
      if (analysisResult.windows && analysisResult.windows[activeWindowIndex]) {
        const activeWin = analysisResult.windows[activeWindowIndex];
        start = activeWin.start_seconds;
        end = activeWin.end_seconds;
      }
      
      const data = await api.getRecordingWindow(recordingId, start, end);
      setActiveWaveform(data);
    } catch (err: any) {
      console.error(err);
    } finally {
      setLoadingWaveform(false);
    }
  };

  const loadCustomWaveform = async () => {
    if (!recordingId) return;
    const start = Math.max(0, viewerStart);
    const duration = Math.max(1, Math.min(30, viewerDuration));
    setLoadingWaveform(true);
    try {
      setActiveWaveform(await api.getRecordingWindow(recordingId, start, start + duration));
      const matchingIndex = (analysisResult?.windows || []).findIndex((win: any, idx: number) => {
        const winStart = Number(win.start_seconds ?? idx * 5);
        const winEnd = Number(win.end_seconds ?? winStart + 10);
        return winEnd > start && winStart < start + duration;
      });
      if (matchingIndex >= 0) setActiveWindowIndex(matchingIndex);
    } finally {
      setLoadingWaveform(false);
    }
  };

  const displayLabel = (code: string) => {
    const key = String(code || "").toUpperCase();
    return labelRegistry[key]?.display_name || (LABEL_EXPANSIONS[key] ? `${LABEL_EXPANSIONS[key]} (${key})` : key);
  };

  const navigateToSection = (sectionId: string) => {
    setActiveReportSection(sectionId);
    document.getElementById(sectionId)?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  const cleanAnswerText = (text: string, hasStructuredCitations = false) => {
    let cleaned = String(text || "").replace(/\[\/?ANSWER\]/gi, "").trim();
    if (hasStructuredCitations) cleaned = cleaned.replace(/\n\s*Sources\s*:\s*[\s\S]*$/i, "").trim();
    return cleaned;
  };

  const uniqueCitations = (citations: any[] = []) => citations.filter((citation, index, all) => {
    const key = citation.id || citation.citation_id || `${citation.title}|${citation.section}|${citation.page_or_locator || citation.page || citation.url}`;
    return all.findIndex(other => (other.id || other.citation_id || `${other.title}|${other.section}|${other.page_or_locator || other.page || other.url}`) === key) === index;
  });

  const loadNeighborWaveform = async (match: any) => {
    const id = String(match.ecg_id ?? match.retrieved_ecg_id ?? match.faiss_row ?? "");
    if (!id || neighborWaveforms[id]) return;
    try {
      const waveform = await api.getRetrievalNeighborWaveform(Number(id));
      setNeighborWaveforms(prev => ({ ...prev, [id]: waveform }));
    } catch (error: any) {
      setNeighborWaveforms(prev => ({ ...prev, [id]: { error: error.message } }));
    }
  };

  useEffect(() => {
    const stats = analysisResult?.recording_statistics || analysisResult?.statistics?.whole_recording;
    if (stats) console.info("[Clinical ECG Support] Whole-recording ECG statistics", stats);
  }, [analysisResult]);

  const resetCaseScopedState = () => {
    setRecordingId(null);
    setAnalysisResult(null);
    setActiveWindowIndex(0);
    setActiveWaveform(null);
    setViewerStart(0);
    setNeighborWaveforms({});
    setRetrievalDisplayCount(5);
    setChatMessages([]);
    setUserQuery("");
    setSendingChat(false);
    setFloatingChatOpen(false);
    setFloatingQuery("");
    setFloatingAnswer("");
    setFloatingBusy(false);
    setActiveReportSection("waveform");
    setFeedbackNotes("");
    setCorrectedPrimary("");
    setCorrectedSecondary("");
    setCorrectedFamily("Other");
    setRetrievalRatings({});
    setFeedbackSubmitted(false);
  };

  const analyzeRecordingFile = async (recordingFile: File) => {
    const analysisSequence = ++analysisSequenceRef.current;
    resetCaseScopedState();
    setLoading(true);
    try {
      // All durations use the stateful recording endpoint. This creates the
      // case-scoped cache required by follow-up chat and manual window review.
      const result = await api.runRecordingInference({
        file: recordingFile,
        recordingMode: mode,
        samplingRateHz: samplingRate,
        topK: 20,
        question: "What is the primary finding and diagnostic conclusion?",
        includeExplanation: true,
        leadNames: LEAD_NAMES
      });

      // A newer subject may have been submitted while this request was running.
      // Never allow an older response to repopulate patient-scoped UI state.
      if (analysisSequence !== analysisSequenceRef.current) return;

      setAnalysisResult(result);
      const recId = result.recording_id || result.acquisition?.recording_id;
      if (!recId) throw new Error("Recording inference completed without a recording ID; chat state cannot be created safely.");
      setRecordingId(recId);
      setFile(recordingFile);
      
      // Initialize first chat assistant message
      const initialLabels = [result?.final_diagnostic_decision?.primary_label, ...(result?.final_diagnostic_decision?.supported_labels || []), ...(result?.final_diagnostic_decision?.partially_supported_labels || [])].filter(Boolean).map((v:string) => String(v).toUpperCase());
      const initialCitations = uniqueCitations(result.clinical_references || result.explanation?.citations || []).filter((citation:any) => {
        const id = String(citation.id || citation.citation_id || "").toUpperCase();
        return initialLabels.some((label:string) => id.startsWith(`SCP-${label}-`));
      });
      setChatMessages([
        {
          role: "assistant",
          content: cleanAnswerText(result.explanation?.text || "Analysis complete. Ask follow-up questions about the decision, retrieved neighbors, measurements, or validated knowledge.", initialCitations.length > 0),
          citations: initialCitations
        }
      ]);
      
      setActiveWindowIndex(0);
      setView("workspace");
    } catch (err: any) {
      if (analysisSequence !== analysisSequenceRef.current) return;
      alert(`Clinical ECG analysis error: ${err.message}`);
    } finally {
      if (analysisSequence === analysisSequenceRef.current) setLoading(false);
    }
  };

  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;
    await analyzeRecordingFile(file);
  };

  const targetDurationSeconds = () => mode === "10s" ? 10 : mode === "2min" ? 120 : 300;

  const discoverSerialDevices = async () => {
    const serial = (navigator as any).serial;
    if (!serial) {
      setSerialStatus("Web Serial is unavailable. Use Chrome/Edge on localhost or HTTPS, or upload a file.");
      return;
    }
    try {
      let ports = await serial.getPorts();
      if (!ports.length) {
        const chosen = await serial.requestPort();
        ports = [chosen];
      }
      setSerialPorts(ports);
      setSelectedSerialPort(ports[0]);
      setSerialStatus(`${ports.length} authorized ECG device${ports.length === 1 ? "" : "s"} found`);
    } catch (error: any) {
      setSerialStatus(error?.name === "NotFoundError" ? "No device was selected" : `Device discovery failed: ${error.message}`);
    }
  };

  const finishLiveRecording = async (analyze = true) => {
    if (recordingTimerRef.current) clearInterval(recordingTimerRef.current);
    recordingTimerRef.current = null;
    setIsRecording(false);
    try { await serialReaderRef.current?.cancel(); } catch (_) {}
    serialReaderRef.current = null;
    try { await serialPortRef.current?.close(); } catch (_) {}
    serialPortRef.current = null;

    const frames = liveFramesRef.current.slice();
    if (!analyze) {
      liveFramesRef.current = [];
      setRecordedFrames(0);
      setRecordingElapsed(0);
      setSerialStatus("Capture discarded; no patient waveform was retained in the browser");
      return;
    }
    const minimumFrames = Math.max(12, Math.floor(targetDurationSeconds() * samplingRate * 0.9));
    if (frames.length < minimumFrames) {
      setSerialStatus(`Capture stopped with ${frames.length} valid frames; ${minimumFrames} are required for this mode`);
      return;
    }
    const csv = `${LEAD_NAMES.join(",")}\n${frames.map(row => row.join(",")).join("\n")}`;
    const captured = new File([csv], `live_ecg_${new Date().toISOString().replace(/[:.]/g, "-")}.csv`, { type: "text/csv" });
    liveFramesRef.current = [];
    setSerialStatus("Capture complete; submitting through the validated recording pipeline");
    await analyzeRecordingFile(captured);
  };

  const startLiveRecording = async () => {
    if (!selectedSerialPort || isRecording) return;
    liveFramesRef.current = [];
    setRecordedFrames(0);
    setRecordingElapsed(0);
    try {
      await selectedSerialPort.open({ baudRate });
      serialPortRef.current = selectedSerialPort;
      const reader = selectedSerialPort.readable.getReader();
      serialReaderRef.current = reader;
      recordingStartedRef.current = Date.now();
      setIsRecording(true);
      setSerialStatus("Recording 12-lead frames… expected protocol: 12 comma-separated numeric values per line");
      recordingTimerRef.current = setInterval(() => {
        const elapsed = (Date.now() - recordingStartedRef.current) / 1000;
        setRecordingElapsed(elapsed);
        if (elapsed >= targetDurationSeconds()) void finishLiveRecording(true);
      }, 200);

      const decoder = new TextDecoder();
      let pending = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        pending += decoder.decode(value, { stream: true });
        const lines = pending.split(/\r?\n/);
        pending = lines.pop() || "";
        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) continue;
          let values: number[] = [];
          try {
            if (trimmed.startsWith("{")) {
              const parsed = JSON.parse(trimmed);
              values = parsed.leads || parsed.values || [];
            } else {
              values = trimmed.split(/[,;\t ]+/).map(Number);
            }
          } catch (_) { continue; }
          if (values.length === 12 && values.every(Number.isFinite)) {
            liveFramesRef.current.push(values);
            setRecordedFrames(liveFramesRef.current.length);
          }
        }
      }
    } catch (error: any) {
      if (isRecording || error?.name !== "NetworkError") setSerialStatus(`Recording error: ${error.message}`);
      await finishLiveRecording(false);
    }
  };

  const triggerChat = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!userQuery.trim() || !recordingId || sendingChat) return;

    const query = userQuery.trim();
    setUserQuery("");
    setChatMessages((prev) => [...prev, { role: "user", content: query }]);
    setSendingChat(true);

    try {
      const response = await api.runRecordingChat(recordingId, query);
      const citations = uniqueCitations(response.citations || []);
      setChatMessages((prev) => [
        ...prev, 
        { 
          role: "assistant", 
          content: cleanAnswerText(response.answer || response.text || "No advisory text received.", citations.length > 0),
          intent: response.intent,
          citations,
          evidence: response.evidence
        }
      ]);
    } catch (err: any) {
      setChatMessages((prev) => [
        ...prev,
        { role: "assistant", content: `Clinical Q&A request failed: ${err.message}` }
      ]);
    } finally {
      setSendingChat(false);
    }
  };

  const askSectionAssistant = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!recordingId || !floatingQuery.trim() || floatingBusy) return;
    const sectionName = REPORT_SECTIONS.find(([id]) => id === activeReportSection)?.[1] || "current ECG report";
    setFloatingBusy(true);
    setFloatingAnswer("");
    try {
      const response = await api.runRecordingChat(
        recordingId,
        `The clinician is currently reviewing the ${sectionName} section. Answer this section-specific question using the active ECG case, system consensus, measurements, retrieval and validated knowledge when relevant: ${floatingQuery.trim()}`
      );
      setFloatingAnswer(cleanAnswerText(response.answer || response.text || "No advisory answer was returned.", (response.citations || []).length > 0));
    } catch (error: any) {
      setFloatingAnswer(`Clinical assistant request failed: ${error.message}`);
    } finally {
      setFloatingBusy(false);
    }
  };

  const handleWindowSelect = async (winIndex: number) => {
    if (!recordingId || !analysisResult) return;
    
    // Toggle the selected state of the clicked window
    const updatedIndices = analysisResult.windows
      ?.map((w: any, idx: number) => {
        if (idx === winIndex) return { ...w, selected: !w.selected };
        return w;
      })
      .filter((w: any) => w.selected)
      .map((w: any) => w.window_index) || [];

    try {
      const result = await api.overrideRecordingWindows(recordingId, updatedIndices);
      setAnalysisResult(result);
      // Reset active window index to keep viewer valid
      if (updatedIndices.length > 0 && !updatedIndices.includes(activeWindowIndex)) {
        setActiveWindowIndex(updatedIndices[0]);
      }
    } catch (err: any) {
      alert(`Manual override selection failed: ${err.message}`);
    }
  };

  const analyzeSelectedInterval = async () => {
    if (!recordingId || !analysisResult) return;
    const start = Math.max(0, viewerStart);
    const end = Math.min(recordingDuration, start + Math.max(1, viewerDuration));
    const windows = analysisResult.windows || [];
    let selectedIndices = windows
      .map((win: any, idx: number) => ({
        idx: Number(win.window_index ?? idx),
        start: Number(win.start_seconds ?? idx * 5),
        end: Number(win.end_seconds ?? (idx * 5 + 10)),
      }))
      .filter((win: any) => win.end > start && win.start < end)
      .map((win: any) => win.idx);
    if (!selectedIndices.length) selectedIndices = [Math.max(0, Math.round(start / 5))];

    setAnalyzingSelectedInterval(true);
    try {
      const result = await api.overrideRecordingWindows(recordingId, selectedIndices);
      setAnalysisResult(result);
      setActiveWindowIndex(selectedIndices[0]);
      setChatMessages(prev => [...prev, {
        role: "assistant",
        content: `Clinician-selected interval ${start.toFixed(0)}–${end.toFixed(0)} seconds was reanalyzed using diagnostic windows ${selectedIndices.join(", ")}. The updated classifier, retrieval, statistics and bridge results are now displayed.`
      }]);
    } catch (err: any) {
      alert(`Selected-interval analysis failed: ${err.message}`);
    } finally {
      setAnalyzingSelectedInterval(false);
    }
  };

  const handleFeedbackSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!recordingId) return;

    try {
      await api.submitFeedback({
        case_id: recordingId,
        ecg_id: String(analysisResult?.ecg_id || recordingId),
        patient_id: String(analysisResult?.patient_id || recordingId),
        clinician_id: reviewerId,
        diagnosis_correctness: verdict,
        clinician_primary_scp: correctedPrimary.trim() || null,
        clinician_secondary_scps: correctedSecondary.split(",").map(v => v.trim().toUpperCase()).filter(Boolean),
        clinician_family: correctedFamily,
        confidence_rating: confidenceRating,
        bridge_explanation_rating: explanationRating,
        retrieval_evaluations: Object.entries(retrievalRatings).map(([ecg_id, relevance]) => ({ ecg_id, relevance })),
        general_comments: feedbackNotes,
        classifier_version: "PTB-XL selected",
        family_head_version: "hierarchical",
        retrieval_version: "V7",
        bridge_version: "V4",
        faiss_version: "V7",
        signal_quality: Number(analysisResult?.signal_quality?.score ?? 1),
        confidence_score: Math.round(Number(analysisResult?.final_diagnostic_decision?.confidence || analysisResult?.bridge?.confidence?.final_fused_confidence || 0) * 100),
        deployment_version: "final_version_optimized"
      });
      setFeedbackSubmitted(true);
      setFeedbackNotes("");
    } catch (err: any) {
      alert(`Feedback submission failed: ${err.message}`);
    }
  };

  const installPWA = () => {
    if (deferredPrompt) {
      deferredPrompt.prompt();
      deferredPrompt.userChoice.then((choice: any) => {
        if (choice.outcome === "accepted") {
          setIsInstallable(false);
        }
      });
    }
  };

  const recordingDuration = Number(getMetric(analysisResult, "duration_seconds") || (mode === "5min" ? 300 : mode === "2min" ? 120 : 10));
  const activeWindow = analysisResult?.windows?.[activeWindowIndex];
  const activeMetricSource = mode !== "10s" && activeWindow?.statistics
    ? { recording_statistics: { overall: activeWindow.statistics?.overall || activeWindow.statistics } }
    : analysisResult;
  const metricValue = (key: string) => getMetric(activeMetricSource, key) ?? getMetric(analysisResult, key);
  const waveformMetricStrip = (expanded = false) => {
    const metrics = [
      ["Heart rate", metricValue("heart_rate_bpm"), "bpm", true],
      ["QRS", metricValue("qrs_ms"), "ms", false],
      ["Mean RR", metricValue("mean_rr_ms"), "ms", false],
      ["SDNN", metricValue("sdnn_ms"), "ms", false],
      ["RMSSD", metricValue("rmssd_ms"), "ms", false],
      ["R peaks", metricValue("r_peak_count"), "", false],
    ] as const;
    return <div className={`grid grid-cols-2 md:grid-cols-6 gap-2 ${expanded ? "mb-3" : "mb-3"}`}>
      {metrics.map(([label, value, unit, pulse]) => <div key={label} className={`rounded-lg border px-3 py-2 ${pulse ? "border-red-200 bg-red-50" : "border-neutral-200 bg-neutral-50"}`}>
        <div className="flex items-center gap-1.5 text-[9px] font-bold uppercase tracking-wider text-neutral-500">
          {pulse && <span className="size-2 rounded-full bg-red-600 animate-pulse" />}{label}
        </div>
        <div className={`mt-0.5 text-sm font-black ${pulse ? "text-red-700" : "text-neutral-900"}`}>
          {value === undefined || value === null ? "—" : Number(value).toFixed(label === "R peaks" ? 0 : 1)} <span className="text-[9px] font-semibold text-neutral-500">{unit}</span>
        </div>
      </div>)}
    </div>;
  };

  return (
    <div className="relative w-full min-h-screen bg-neutral-50 text-neutral-900 font-sans flex flex-col antialiased overflow-x-clip">
      <div className="fixed inset-0 pointer-events-none z-0"><AnimatedECGBackground opacity={0.07} paused={false} /></div>
      {/* Maximized ECG Full Screen Overlay */}
      {isEcgExpanded && activeWaveform?.values && (
        <div className="fixed inset-0 bg-white z-[999] p-6 flex flex-col justify-between">
          <div className="flex items-center justify-between border-b border-neutral-200 pb-3 mb-4">
            <div className="flex items-center gap-3">
              <span className="text-sm font-extrabold text-neutral-900 tracking-wider">MAXIMIZED 12-LEAD WORKSTATION VIEW</span>
              <span className="text-[10px] px-2 py-0.5 rounded bg-red-50 text-red-600 border border-red-100 font-bold uppercase">
                {mode === "10s" ? "Whole Waveform" : `Window #${activeWindowIndex}`}
              </span>
            </div>
            
            {/* Expanded Controls */}
            <div className="flex items-center gap-4">
              {/* Lead Selector Select */}
              <div className="flex items-center gap-1.5">
                <span className="text-xs text-neutral-500 font-semibold uppercase">Vis:</span>
                <select 
                  value={selectedLead} 
                  onChange={(e) => setSelectedLead(e.target.value)}
                  className="bg-neutral-100 border border-neutral-200 rounded px-2.5 py-1 text-xs font-bold text-neutral-800 focus:outline-none"
                >
                  <option value="All">All 12 Leads</option>
                  {LEAD_NAMES.map(name => (
                    <option key={name} value={name}>{name}</option>
                  ))}
                </select>
              </div>

              <div className="flex items-center gap-1 bg-neutral-100 p-0.5 rounded border border-neutral-200 text-xs">
                <button onClick={() => setLeadGain(g => Math.max(0.2, g - 0.2))} className="px-2.5 py-0.5 hover:bg-white rounded transition shadow-sm font-bold">-</button>
                <span className="px-1 text-neutral-600">Gain: {leadGain.toFixed(1)}x</span>
                <button onClick={() => setLeadGain(g => Math.min(2.5, g + 0.2))} className="px-2.5 py-0.5 hover:bg-white rounded transition shadow-sm font-bold">+</button>
              </div>

              <div className="flex items-center gap-1 bg-neutral-100 p-0.5 rounded border border-neutral-200 text-xs">
                <button onClick={() => setTimeZoom(z => Math.max(0.5, z - 0.2))} className="px-2.5 py-0.5 hover:bg-white rounded transition shadow-sm font-bold">-</button>
                <span className="px-1 text-neutral-600">Time: {timeZoom.toFixed(1)}x</span>
                <button onClick={() => setTimeZoom(z => Math.min(2.0, z + 0.2))} className="px-2.5 py-0.5 hover:bg-white rounded transition shadow-sm font-bold">+</button>
              </div>

              <button 
                onClick={() => setIsEcgExpanded(false)}
                className="text-xs px-3 py-1 bg-neutral-950 text-white rounded hover:bg-neutral-800 transition font-bold"
              >
                Close Fullscreen
              </button>
            </div>
          </div>
          {waveformMetricStrip(true)}
          <div className="flex-1 bg-white border border-neutral-200 rounded-lg overflow-hidden relative">
            <ECGWaveformCanvas 
              waveform={activeWaveform.values}
              gain={leadGain}
              zoom={timeZoom}
              leadNames={activeWaveform.lead_order || LEAD_NAMES}
              selectedLead={selectedLead}
            />
          </div>
        </div>
      )}

      {/* Header Bar */}
      <header className={`sticky top-0 z-40 border-b border-neutral-200/80 bg-white/95 backdrop-blur-md px-6 py-3 flex items-center ${view === "landing" ? "justify-end" : "justify-between"}`}>
        {view !== "landing" && <div className="flex items-center gap-3">
          <div className="bg-red-600 text-white rounded p-1.5 flex items-center justify-center shadow-md">
            <Activity className="size-5" />
          </div>
          <div>
            <span className="font-bold tracking-wide text-base text-neutral-950">Clinical Decision Support System</span>
            <span className="text-xs text-neutral-500 ml-2 border-l border-neutral-300 pl-2">ECG Diagnosis</span>
          </div>
        </div>}

        <div className="flex items-center gap-4">
          {view !== "landing" && isInstallable && (
            <button 
              onClick={installPWA}
              className="text-xs px-3 py-1 bg-red-50 text-red-600 rounded border border-red-200 hover:bg-red-100 transition"
            >
              Install Workstation PWA
            </button>
          )}

          {/* Health Indicator Badge */}
          <button 
            onClick={() => setShowHealth(!showHealth)}
            className="flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full bg-neutral-100 border border-neutral-200 hover:bg-neutral-200 transition"
          >
            <span className={`size-2 rounded-full ${healthStatus === "healthy" ? "bg-green-600 animate-pulse" : (healthStatus === "connecting" ? "bg-amber-500" : "bg-red-600")}`} />
            <span className="text-neutral-600 capitalize font-medium">{healthStatus} runtime</span>
          </button>
        </div>
      </header>

      {/* Health status dropdown overlay */}
      {showHealth && systemHealth && (
        <div className="absolute right-6 top-16 z-50 w-80 bg-white rounded-lg border border-neutral-200 shadow-xl p-4 text-xs">
          <h4 className="font-bold border-b border-neutral-100 pb-2 mb-2 flex items-center justify-between text-neutral-900">
            <span>Clinical ECG Pipeline Status</span>
            <span className="text-neutral-400">Bridge / FAISS</span>
          </h4>
          <div className="space-y-1.5 text-neutral-600">
            <div className="flex justify-between"><span>FastAPI Status:</span><span className="font-bold text-green-600">Active</span></div>
            <div className="flex justify-between"><span>Diagnosis Model:</span><span className="font-bold text-green-600">{systemHealth.diagnosis_model_loaded ? "Ready" : "Unavailable"}</span></div>
            <div className="flex justify-between"><span>Inference Device:</span><span className="font-mono text-neutral-800 uppercase">{systemHealth.configured_device}</span></div>
            <div className="flex justify-between"><span>Clinical KB:</span><span className="font-bold text-green-600">{systemHealth.feedback_service_available ? "Ready" : "Unavailable"}</span></div>
            <div className="flex justify-between"><span>Local LLM:</span><span className="font-mono text-neutral-800">{systemHealth.configured_llm_mode} mode</span></div>
          </div>
        </div>
      )}

      {/* Main View Area */}
      <main className="flex-1 w-full flex relative z-10 overflow-visible">
        <AnimatePresence mode="wait">
          {view === "landing" ? (
            <motion.div 
              key="landing"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              className="flex-1 max-w-4xl mx-auto flex flex-col justify-center px-6 py-12"
            >
              <h1 className="mb-3 text-center text-2xl font-extrabold tracking-tight text-neutral-950 sm:text-3xl">
                Clinical Decision Support System - ECG Decision Support
              </h1>
              <p className="mx-auto mb-7 max-w-3xl text-center text-[11px] leading-relaxed text-neutral-500 sm:text-xs">
                The system combines ECG classification, FAISS retrieval, rhythm and measurements into a traceable consensus with citations, while the clinical assistant provides contextual explanations.
              </p>

              {/* Acquisition Card */}
              <div className="bg-white border border-neutral-200 shadow-xl rounded-xl p-8 max-w-xl mx-auto w-full relative overflow-hidden">
                {loading && (
                  <div className="absolute inset-0 bg-white/90 backdrop-blur-sm z-50 flex flex-col items-center justify-center p-6 text-center">
                    <RefreshCw className="size-10 text-red-600 animate-spin mb-4" />
                    <h3 className="font-bold text-neutral-950 text-base mb-1">Analyzing ECG Waveform</h3>
                    <p className="text-xs text-neutral-500 max-w-xs">{loadingStages[loadingStage]}</p>
                    <div className="w-full bg-neutral-100 rounded-full h-1.5 mt-4 max-w-xs overflow-hidden">
                      <div className="bg-red-600 h-1.5 rounded-full transition-all duration-1200" style={{ width: `${((loadingStage + 1) / loadingStages.length) * 100}%` }} />
                    </div>
                  </div>
                )}

                <div className="grid grid-cols-2 gap-1 rounded-lg bg-neutral-100 border border-neutral-200 p-1 mb-6">
                  <button type="button" onClick={() => setInputMode("upload")} className={`rounded-md py-2 text-xs font-semibold ${inputMode === "upload" ? "bg-white shadow-sm text-neutral-950" : "text-neutral-500"}`}>Upload recording</button>
                  <button type="button" onClick={() => setInputMode("live")} className={`rounded-md py-2 text-xs font-semibold ${inputMode === "live" ? "bg-white shadow-sm text-neutral-950" : "text-neutral-500"}`}>Live ECG device</button>
                </div>

                <form onSubmit={handleUpload} className="space-y-6">
                  {inputMode === "upload" ? <div className="border-2 border-dashed border-neutral-300 rounded-lg p-6 text-center hover:border-red-500 transition cursor-pointer relative bg-neutral-50/50">
                    <input 
                      type="file" 
                      onChange={(e) => setFile(e.target.files?.[0] || null)}
                      className="absolute inset-0 size-full opacity-0 cursor-pointer"
                    />
                    <Upload className="size-8 mx-auto text-neutral-400 mb-2" />
                    {file ? (
                      <div>
                        <p className="text-sm font-bold text-neutral-900 truncate max-w-xs mx-auto">{file.name}</p>
                        <p className="text-xs text-neutral-400 mt-1">{(file.size / 1024).toFixed(1)} KB · Double click to change</p>
                      </div>
                    ) : (
                      <div>
                        <p className="text-sm text-neutral-700 font-medium">Upload raw ECG signal file</p>
                        <p className="text-xs text-neutral-400 mt-1">Supported formats: Any dataset or raw binary files (12 leads, 100-1000 Hz)</p>
                      </div>
                    )}
                  </div> : <div className="space-y-4 rounded-lg border border-neutral-200 bg-neutral-50 p-4">
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <p className="text-sm font-bold text-neutral-900">12-lead serial ECG acquisition</p>
                        <p className="text-xs text-neutral-500 mt-1">One frame per line: 12 finite values in canonical lead order I, II, III, aVR, aVL, aVF, V1–V6.</p>
                      </div>
                      <Usb className="size-6 text-red-600 shrink-0" />
                    </div>
                    <div className="grid grid-cols-[1fr_auto] gap-2">
                      <button type="button" disabled={isRecording} onClick={discoverSerialDevices} className="rounded-lg border border-neutral-300 bg-white px-3 py-2 text-xs font-semibold hover:border-red-500 disabled:opacity-50">Find / pair ECG device</button>
                      <select value={baudRate} disabled={isRecording} onChange={e => setBaudRate(Number(e.target.value))} className="rounded-lg border border-neutral-300 bg-white px-2 py-2 text-xs">
                        {[9600, 19200, 38400, 57600, 115200, 230400, 460800, 921600].map(rate => <option key={rate} value={rate}>{rate} baud</option>)}
                      </select>
                    </div>
                    {serialPorts.length > 1 && <select value={serialPorts.indexOf(selectedSerialPort)} onChange={e => setSelectedSerialPort(serialPorts[Number(e.target.value)])} className="w-full rounded-lg border border-neutral-300 bg-white px-3 py-2 text-xs">
                      {serialPorts.map((_, index) => <option key={index} value={index}>Authorized ECG device {index + 1}</option>)}
                    </select>}
                    <div className="rounded-lg border border-neutral-200 bg-white p-3">
                      <div className="flex justify-between text-xs"><span className="font-semibold">Capture timer</span><span className="font-mono">{Math.min(recordingElapsed, targetDurationSeconds()).toFixed(1)} / {targetDurationSeconds()} s</span></div>
                      <div className="h-2 bg-neutral-100 rounded-full overflow-hidden mt-2"><div className="h-full bg-red-600 transition-all" style={{width: `${Math.min(100, recordingElapsed / targetDurationSeconds() * 100)}%`}} /></div>
                      <div className="flex justify-between text-[11px] text-neutral-500 mt-2"><span>{recordedFrames.toLocaleString()} valid frames</span><span>Target: {(targetDurationSeconds() * samplingRate).toLocaleString()}</span></div>
                    </div>
                    <p className="text-xs text-neutral-600 min-h-8">{serialStatus}</p>
                    <div className="grid grid-cols-2 gap-2">
                      <button type="button" disabled={!selectedSerialPort || isRecording} onClick={startLiveRecording} className="rounded-lg bg-red-600 text-white px-3 py-2 text-xs font-bold flex items-center justify-center gap-2 disabled:bg-neutral-300"><Play className="size-4" />Start recording</button>
                      <button type="button" disabled={!isRecording} onClick={() => void finishLiveRecording(true)} className="rounded-lg bg-neutral-900 text-white px-3 py-2 text-xs font-bold flex items-center justify-center gap-2 disabled:bg-neutral-300"><Square className="size-4" />Stop and analyze</button>
                    </div>
                    <button type="button" disabled={!isRecording && recordedFrames === 0} onClick={() => void finishLiveRecording(false)} className="w-full text-xs text-neutral-500 hover:text-red-700 disabled:opacity-40">Discard temporary capture</button>
                  </div>}

                  {/* Mode select segmented control */}
                  <div className="space-y-2">
                    <label className="text-xs font-semibold text-neutral-500 uppercase tracking-wider block">ECG Recording Mode</label>
                    <div className="grid grid-cols-3 gap-2 bg-neutral-100 p-1.5 rounded-lg border border-neutral-200">
                      {(["10s", "2min", "5min"] as const).map((m) => (
                        <button
                          key={m}
                          type="button"
                          onClick={() => setMode(m)}
                          className={`py-1.5 text-xs font-medium rounded-md transition ${mode === m ? "bg-white text-neutral-950 shadow-sm border border-neutral-200/50" : "text-neutral-500 hover:text-neutral-950"}`}
                        >
                          {m === "10s" ? "10 seconds" : (m === "2min" ? "2 min" : "5 min")}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Sampling rate input */}
                  <div className="space-y-1.5">
                    <label className="text-xs font-semibold text-neutral-500 uppercase tracking-wider block">ECG Sampling Rate (Hz)</label>
                    <input 
                      type="number"
                      value={samplingRate}
                      onChange={(e) => setSamplingRate(Math.max(10, parseInt(e.target.value) || 100))}
                      className="w-full border border-neutral-200 rounded-lg px-3 py-2 text-sm bg-neutral-50 focus:bg-white focus:outline-none focus:ring-1 focus:ring-red-500"
                    />
                  </div>

                  <button
                    type="submit"
                    disabled={!file || inputMode !== "upload"}
                    className={`${inputMode === "upload" ? "flex" : "hidden"} w-full py-3 px-4 rounded-lg font-bold text-white text-sm shadow-md transition items-center justify-center gap-2 ${file ? "bg-red-600 hover:bg-red-700" : "bg-neutral-300 cursor-not-allowed"}`}
                  >
                    <span>ANALYZE ECG</span>
                    <ArrowRight className="size-4" />
                  </button>
                </form>
              </div>
            </motion.div>
          ) : (
            // Continuous clinician report with anchored navigation.
            <motion.div 
              key="workspace"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex-1 w-full max-w-[1500px] mx-auto px-5 py-6 space-y-6 bg-neutral-50/80"
            >
              
              {/* Sticky anchored report navigation */}
              <aside className="sticky top-[57px] z-30 select-none bg-white/95 backdrop-blur border border-neutral-200 rounded-xl shadow-sm px-4 py-3 flex flex-wrap items-center justify-between gap-3">
                  {/* Branding */}
                  <div>
                    <h3 className="text-sm font-bold tracking-widest text-neutral-900 flex items-center gap-1.5">
                      <Heart className="size-4 text-red-600 fill-red-600" />
                      <span>ECG CONSOLE</span>
                    </h3>
                  </div>

                  {/* Loaded file indicator */}
                  <div className="bg-neutral-50 rounded-lg border border-neutral-200 px-3 py-2">
                    <div className="text-[10px] font-semibold text-neutral-400 uppercase tracking-wider">Active Dataset</div>
                    <div className="text-xs font-bold text-neutral-900 truncate">{file?.name || "sample_ecg.npy"}</div>
                    <div className="text-[10px] text-neutral-500">{(file?.size ? (file.size/1024).toFixed(1) + " KB" : "")} · Mode: {mode}</div>
                  </div>

                  {/* Actions */}
                  <nav className="flex flex-wrap gap-1 text-xs font-semibold">
                    {REPORT_SECTIONS.map(([id,label]) => (
                      <a key={id} href={`#${id}`} onClick={(event) => { event.preventDefault(); navigateToSection(id); }} className={`px-3 py-2 rounded-lg outline-none ${activeReportSection === id ? "text-red-700 bg-red-50" : "text-neutral-600 hover:text-red-700 hover:bg-red-50 focus:text-red-700 focus:bg-red-50"}`}>{label}</a>
                    ))}
                  </nav>
                  <div>
                    <button
                      onClick={() => setView("landing")}
                      className="w-full flex items-center gap-2 px-3 py-2 text-xs font-bold text-neutral-600 rounded-lg hover:bg-neutral-100 hover:text-neutral-950 transition"
                    >
                      <Upload className="size-4" />
                      <span>Ingest New ECG</span>
                    </button>
                  </div>
                  <span className="text-[10px] text-neutral-500 font-mono">{systemHealth?.configured_device || "CPU"} · Bridge / FAISS</span>
              </aside>

              {/* Col 2, Row 1: ECG INPUT / LIVE ECG REGION */}
              <section id="waveform" className="scroll-mt-32 bg-white border border-neutral-200 rounded-xl shadow-sm p-5 relative flex flex-col min-h-[560px]">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-bold text-neutral-500 uppercase tracking-wider">ECG INPUT VIEWER</span>
                    <span className="text-[10px] px-2 py-0.5 rounded bg-red-50 text-red-600 border border-red-100 font-bold uppercase">
                      {mode === "10s" ? "Whole Waveform" : `Window #${activeWindowIndex} (${(activeWindowIndex * 5)}-${(activeWindowIndex * 5 + 10)}s)`}
                    </span>
                  </div>
                  
                  {/* Waveform Controls */}
                  <div className="flex items-center gap-3">
                    <div className="flex items-center gap-1.5 bg-neutral-50 border border-neutral-200 rounded-lg px-2 py-1">
                      <label className="text-[10px] font-bold text-neutral-500">Start (s)</label>
                      <input type="number" min="0" step="1" value={viewerStart} onChange={e => setViewerStart(Number(e.target.value))} className="w-16 bg-white border rounded px-1 py-0.5 text-xs" />
                      <label className="text-[10px] font-bold text-neutral-500">Duration</label>
                      <select value={viewerDuration} onChange={e => setViewerDuration(Number(e.target.value))} className="bg-white border rounded px-1 py-0.5 text-xs">
                        {[5,10,20,30].map(v => <option key={v} value={v}>{v}s</option>)}
                      </select>
                      <button onClick={loadCustomWaveform} className="bg-neutral-900 text-white rounded px-2 py-1 text-[10px] font-bold">View</button>
                      <button onClick={analyzeSelectedInterval} disabled={analyzingSelectedInterval} className="bg-red-600 disabled:opacity-50 text-white rounded px-2 py-1 text-[10px] font-bold whitespace-nowrap">
                        {analyzingSelectedInterval ? "Analyzing…" : "Analyze interval"}
                      </button>
                    </div>
                    {/* Lead filter drop-down */}
                    <div className="flex items-center gap-1.5">
                      <select 
                        value={selectedLead} 
                        onChange={(e) => setSelectedLead(e.target.value)}
                        className="bg-neutral-100 border border-neutral-200 rounded px-1.5 py-0.5 text-[10px] font-bold text-neutral-800 focus:outline-none"
                      >
                        <option value="All">All Leads</option>
                        {LEAD_NAMES.map(name => (
                          <option key={name} value={name}>{name}</option>
                        ))}
                      </select>
                    </div>

                    <div className="flex items-center gap-1 bg-neutral-100 p-0.5 rounded border border-neutral-200 text-[10px]">
                      <button onClick={() => setLeadGain(g => Math.max(0.2, g - 0.2))} className="p-1 hover:bg-white rounded transition shadow-sm hover:text-neutral-950 font-bold">-</button>
                      <span className="px-1 text-neutral-600">Gain: {leadGain.toFixed(1)}x</span>
                      <button onClick={() => setLeadGain(g => Math.min(2.5, g + 0.2))} className="p-1 hover:bg-white rounded transition shadow-sm hover:text-neutral-950 font-bold">+</button>
                    </div>

                    <div className="flex items-center gap-1 bg-neutral-100 p-0.5 rounded border border-neutral-200 text-[10px]">
                      <button onClick={() => setTimeZoom(z => Math.max(0.5, z - 0.2))} className="p-1 hover:bg-white rounded transition shadow-sm hover:text-neutral-950 font-bold">-</button>
                      <span className="px-1 text-neutral-600">Time: {timeZoom.toFixed(1)}x</span>
                      <button onClick={() => setTimeZoom(z => Math.min(2.0, z + 0.2))} className="p-1 hover:bg-white rounded transition shadow-sm hover:text-neutral-950 font-bold">+</button>
                    </div>

                    {/* Expand Modal Overlay button */}
                    <button 
                      onClick={() => setIsEcgExpanded(true)}
                      title="Expand to Fullscreen View"
                      className="p-1 bg-neutral-100 hover:bg-neutral-200 rounded border border-neutral-200 text-neutral-700 hover:text-neutral-950 transition shadow-sm"
                    >
                      <Maximize2 className="size-3" />
                    </button>
                  </div>
                </div>

                {waveformMetricStrip(false)}

                {/* HTML Canvas Real ECG trace plotting */}
                <div className="h-[410px] relative bg-white border border-neutral-100 rounded-lg overflow-hidden flex items-center justify-center">
                  {loadingWaveform ? (
                    <div className="flex items-center gap-2 text-neutral-400 text-xs">
                      <RefreshCw className="size-4 animate-spin text-red-600" />
                      <span>Retrieving ECG lead waveforms...</span>
                    </div>
                  ) : activeWaveform?.values ? (
                    <div className="size-full">
                      {/* Grid representation */}
                      <ECGWaveformCanvas 
                        waveform={activeWaveform.values}
                        gain={leadGain}
                        zoom={timeZoom}
                        leadNames={activeWaveform.lead_order || LEAD_NAMES}
                        selectedLead={selectedLead}
                      />
                    </div>
                  ) : (
                    <div className="text-xs text-neutral-400 font-medium">Waveform data format invalid or missing.</div>
                  )}
                </div>
                {mode !== "10s" && <div className="mt-3 border-t border-neutral-100 pt-3">
                  <div className="flex justify-between text-[10px] text-neutral-500 mb-1"><span>Drag horizontally to choose the visible interval</span><span>{viewerStart.toFixed(0)}–{Math.min(recordingDuration, viewerStart + viewerDuration).toFixed(0)} s of {recordingDuration.toFixed(0)} s</span></div>
                  <input aria-label="Recording horizontal window navigator" type="range" min="0" max={Math.max(0, recordingDuration - viewerDuration)} step="1" value={Math.min(viewerStart, Math.max(0, recordingDuration-viewerDuration))} onChange={e => setViewerStart(Number(e.target.value))} onMouseUp={loadCustomWaveform} onTouchEnd={loadCustomWaveform} className="w-full accent-red-600 cursor-ew-resize" />
                </div>}
              </section>

              {/* Col 2, Row 2: SECONDARY STRIP (BRIDGE OUTPUT, MODE SELECTOR, RETRIEVAL OUTPUT) */}
              <section className="bg-white border border-neutral-200 rounded-xl shadow-sm grid md:grid-cols-3 divide-x divide-neutral-100 select-none">
                
                {/* Bridge output */}
                <div className="p-4 flex flex-col justify-between overflow-hidden">
                  <div>
                    <h4 className="text-[10px] font-bold text-neutral-400 uppercase tracking-wider mb-1.5">System Consensus</h4>
                    <div className="text-sm font-black text-neutral-900 truncate">
                      {displayLabel(analysisResult?.final_diagnostic_decision?.primary_label || analysisResult?.bridge?.primary_label || analysisResult?.primary_candidate?.label || "") || "Indeterminate"}
                    </div>
                    <p className="mt-1 text-[10px] leading-relaxed text-neutral-500 line-clamp-2">
                      {analysisResult?.final_diagnostic_decision?.summary || analysisResult?.final_diagnostic_decision?.decision_summary || analysisResult?.bridge?.summary || "Most probable finding after deterministic fusion of classifier, retrieval, rhythm and measurement evidence; review the detailed decision below."}
                    </p>
                  </div>
                  <div className="flex items-center justify-between text-[10px]">
                    <span className="text-neutral-500">Confidence:</span>
                    <span className="font-bold text-red-600">
                      {((analysisResult?.final_diagnostic_decision?.confidence || analysisResult?.bridge?.confidence?.final_fused_confidence || (typeof analysisResult?.bridge?.confidence === "number" ? analysisResult?.bridge?.confidence : 0) || 0) * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>

                {/* Mode Selector */}
                <div className="p-4 flex flex-col justify-between">
                  <h4 className="text-[10px] font-bold text-neutral-400 uppercase tracking-wider mb-1.5">Record Length Mode</h4>
                  <div className="text-sm font-extrabold text-neutral-800 capitalize flex items-center gap-1.5">
                    <Clock className="size-4 text-neutral-400" />
                    <span>{mode === "10s" ? "10s Acquisition" : (mode === "2min" ? "2 Minute Strip" : "5 Minute Record")}</span>
                  </div>
                </div>

                {/* Retrieval output */}
                <div className="p-4 flex flex-col justify-between overflow-hidden">
                  <div>
                    <h4 className="text-[10px] font-bold text-neutral-400 uppercase tracking-wider mb-1.5">FAISS ECG Retrieval</h4>
                    <div className="text-xs text-neutral-700 truncate">
                      Matched: <span className="font-bold text-neutral-900">{analysisResult?.retrieval?.raw_neighbors?.length || 0} similar cases</span>
                    </div>
                  </div>
                  <div className="text-[10px] text-neutral-500">
                    Max Sim: <span className="font-bold text-neutral-800">
                      {((analysisResult?.retrieval?.raw_neighbors?.[0]?.raw_similarity || 0) * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              </section>

              {/* Col 3, Row 2-4: WINDOW SELECTION TIMELINE & OVERRIDES */}
              <section id="retrieval" className="scroll-mt-32 bg-white border border-neutral-200 rounded-xl shadow-sm p-5 select-none">
                <div className="space-y-4">
                  <div>
                    <h3 className="text-xs font-bold text-neutral-500 uppercase tracking-wider mb-2">WINDOW TIMELINE</h3>
                    {/* Render timeline for long record modes */}
                    {mode !== "10s" ? (
                      <div className="space-y-3">
                        <div className="flex flex-wrap gap-1 bg-neutral-50 p-2 rounded-lg border border-neutral-200/50">
                          {analysisResult?.windows?.map((win: any, idx: number) => {
                            let color = "bg-neutral-100 text-neutral-400 hover:bg-neutral-200 border-transparent";
                            if (win.status === "stable") color = "bg-green-100 text-green-700 border-green-200";
                            else if (win.status === "candidate") color = "bg-amber-100 text-amber-700 border-amber-200";
                            else if (win.status === "persistent_abnormal_baseline_candidate") color = "bg-red-50 text-red-600 border-red-200";
                            
                            const isActive = activeWindowIndex === idx;

                            return (
                              <button
                                key={idx}
                                onClick={() => setActiveWindowIndex(idx)}
                                onDoubleClick={() => handleWindowSelect(idx)}
                                title={`Window #${idx} (${win.status}). Double-click to toggle manual selection.`}
                                className={`size-7 text-[10px] font-bold rounded border flex items-center justify-center transition cursor-pointer ${color} ${isActive ? "ring-2 ring-red-500 ring-offset-1 border-transparent" : ""} ${win.selected ? "border-red-500 font-extrabold border-2" : ""}`}
                              >
                                {idx}
                              </button>
                            );
                          })}
                        </div>
                        <p className="text-[10px] text-neutral-400 leading-relaxed">
                          Click index to view lead waveform. Double-click to toggle clinician manual selection overrides.
                        </p>
                      </div>
                    ) : (
                      <div className="bg-neutral-50 text-neutral-400 border border-neutral-200/50 rounded-lg p-3 text-center text-xs">
                        Single window mode. Slicing disabled.
                      </div>
                    )}
                  </div>

                  {/* Merged Episodes list */}
                  {analysisResult?.episodes && analysisResult.episodes.length > 0 && (
                    <div className="border-t border-neutral-100 pt-3">
                      <h4 className="text-[10px] font-bold text-neutral-400 uppercase tracking-wider mb-2">Detected Episodes</h4>
                      <div className="space-y-3">
                        {analysisResult.episodes.map((ep: any, index: number) => (
                          <div 
                            key={index}
                            onClick={() => setActiveWindowIndex(ep.representative_window_index)}
                            className="bg-neutral-50 rounded border border-neutral-200/60 p-2 text-xs flex justify-between items-center hover:bg-neutral-100/80 transition cursor-pointer"
                          >
                            <div>
                              <span className="font-bold text-neutral-900">Episode #{index}</span>
                              <div className="text-[10px] text-neutral-500">Times: {ep.start_seconds}-{ep.end_seconds}s</div>
                            </div>
                            <span className="px-1.5 py-0.5 rounded bg-red-50 text-red-600 text-[10px] font-bold">
                              Win {ep.representative_window_index}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* FAISS Nearest Neighbors comparison matches */}
                  {analysisResult?.retrieval?.raw_neighbors && (
                    <div className="border-t border-neutral-100 pt-3">
                      <div className="flex items-center justify-between mb-2">
                        <h4 className="text-[10px] font-bold text-neutral-400 uppercase tracking-wider">FAISS Retrieved Neighbors</h4>
                        <label className="text-[10px] text-neutral-500">Show top <select value={retrievalDisplayCount} onChange={e => setRetrievalDisplayCount(Number(e.target.value))} className="ml-1 border rounded px-2 py-1 bg-white text-neutral-800"><option value={5}>5</option><option value={10}>10</option><option value={20}>20</option></select></label>
                      </div>
                      <div className="space-y-3">
                        {analysisResult.retrieval.raw_neighbors.slice(0, retrievalDisplayCount).map((match: any, index: number) => (
                          <div 
                            key={index}
                            className="bg-neutral-50 rounded-xl border border-neutral-200 p-4 text-xs space-y-3"
                          >
                            <div className="flex justify-between items-center">
                              <span className="font-bold text-neutral-900">Rank #{index+1} · ID {match.ecg_id || match.faiss_row || "Unknown"}</span>
                              <span className="text-neutral-500">{(match.raw_similarity * 100).toFixed(1)}% Sim</span>
                            </div>
                            <div className="text-neutral-500 truncate">
                              Labels: {(match.scp_codes || match.labels || []).map((code: string) => displayLabel(code)).join("; ") || "None"}
                            </div>
                            <button onClick={() => loadNeighborWaveform(match)} className="text-[10px] font-bold px-3 py-1.5 rounded bg-white border border-neutral-300 hover:border-red-400">Show retrieved waveform</button>
                            {(() => {
                              const id = String(match.ecg_id ?? match.retrieved_ecg_id ?? match.faiss_row ?? "");
                              const wave = neighborWaveforms[id];
                              if (!wave) return null;
                              if (wave.error) return <p className="text-amber-700 text-[10px]">Waveform unavailable: {wave.error}</p>;
                              return <div className="space-y-2">
                                <select value={neighborLead} onChange={e => setNeighborLead(e.target.value)} className="border rounded px-2 py-1 text-xs">
                                  {LEAD_NAMES.map(lead => <option key={lead}>{lead}</option>)}
                                </select>
                                <div className="h-52 bg-white border rounded-lg overflow-hidden"><ECGWaveformCanvas waveform={wave.values} gain={1} zoom={1} leadNames={wave.lead_order || LEAD_NAMES} selectedLead={neighborLead} /></div>
                              </div>;
                            })()}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

              </section>

              {/* Col 2, Row 3: FINAL DIAGNOSIS - DOMINANT CENTRAL REGION */}
              <section id="decision" className="scroll-mt-32 bg-white border border-neutral-200 rounded-xl shadow-sm p-6 select-none flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between mb-3 border-b border-neutral-100 pb-2">
                    <span className="text-xs font-bold text-neutral-500 uppercase tracking-wider">System Consensus Diagnostic Decision</span>
                    <span className="text-xs text-neutral-400">ECG Decision Support</span>
                  </div>

                  <div className="flex items-start gap-4">
                    {/* Confidence score block */}
                    <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex flex-col items-center justify-center min-w-28 shadow-sm">
                      <span className="text-2xl font-black text-red-600">
                        {((analysisResult?.final_diagnostic_decision?.confidence || analysisResult?.bridge?.confidence?.final_fused_confidence || (typeof analysisResult?.bridge?.confidence === "number" ? analysisResult?.bridge?.confidence : 0) || 0) * 100).toFixed(0)}%
                      </span>
                      <span className="text-[10px] text-red-600 font-bold uppercase tracking-wider mt-0.5">
                        {analysisResult?.final_diagnostic_decision?.confidence_level || 
                         (analysisResult?.bridge?.confidence?.final_fused_confidence !== undefined ? 
                          (analysisResult.bridge.confidence.final_fused_confidence >= 0.8 ? "HIGH" : "MODERATE") : "MODERATE")}
                      </span>
                    </div>

                    <div className="space-y-3 flex-1">
                      <div>
                        <div className="text-[10px] font-bold text-neutral-400 uppercase tracking-wider">Primary Finding</div>
                        <h2 className="text-lg font-black text-neutral-900">
                          {displayLabel(analysisResult?.final_diagnostic_decision?.primary_label || analysisResult?.bridge?.primary_label || analysisResult?.primary_candidate?.label || "") || "Indeterminate Decision"}
                        </h2>
                        <p className="mt-2 text-sm text-neutral-600 leading-relaxed">
                          {analysisResult?.final_diagnostic_decision?.summary || analysisResult?.final_diagnostic_decision?.decision_summary || analysisResult?.bridge?.summary || analysisResult?.explanation?.text || `The bridge identifies ${displayLabel(analysisResult?.final_diagnostic_decision?.primary_label || analysisResult?.bridge?.primary_label || "")} as the most probable ECG finding after combining classifier, retrieval, rhythm and whole-recording measurement evidence. This remains decision support and should be confirmed against the waveform and clinical context.`}
                        </p>
                      </div>

                      {/* Supported labels badges */}
                      <div className="flex flex-wrap gap-2">
                        {(analysisResult?.final_diagnostic_decision?.supported_labels || analysisResult?.bridge?.supported_labels || []).map((label: string, index: number) => (
                          <span key={index} className="px-2 py-0.5 rounded bg-green-50 text-green-700 border border-green-200 text-xs font-medium">
                            Supported: {displayLabel(label)}
                          </span>
                        ))}
                        {(analysisResult?.final_diagnostic_decision?.partially_supported_labels || analysisResult?.bridge?.partially_supported_labels || []).map((label: string, index: number) => (
                          <span key={index} className="px-2 py-0.5 rounded bg-amber-50 text-amber-700 border border-amber-200 text-xs font-medium">
                            Uncertain: {displayLabel(label)}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Warnings / Contradictions if present */}
                {((analysisResult?.final_diagnostic_decision?.contradictions || analysisResult?.bridge?.contradictions) && 
                  (analysisResult?.final_diagnostic_decision?.contradictions || analysisResult?.bridge?.contradictions || []).length > 0) && (
                  <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 flex gap-2.5 items-start mt-4">
                    <AlertTriangle className="size-5 text-amber-600 shrink-0" />
                    <div>
                      <span className="text-xs font-bold text-amber-800">Contradictions Detected</span>
                      <ul className="list-disc pl-4 text-[10px] text-amber-700 space-y-0.5 mt-1">
                        {(analysisResult?.final_diagnostic_decision?.contradictions || analysisResult?.bridge?.contradictions || []).map((c: any, i: number) => (
                          <li key={i}>{formatContradiction(c)}</li>
                        ))}
                      </ul>
                    </div>
                  </div>
                )}
              </section>

              {/* Col 2, Row 4: CLINICAL Q&A - LARGE LOWER PANEL */}
              <section id="chat" className="scroll-mt-32 bg-white border border-neutral-200 rounded-xl shadow-sm p-6 flex flex-col min-h-[650px]">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-xs font-bold text-neutral-500 uppercase tracking-wider flex items-center gap-1.5">
                    <MessageSquare className="size-4 text-neutral-400" />
                    <span>Gemma Clinical Q&A</span>
                  </h3>
                  <span className="text-[10px] text-neutral-400">Advisory-only responses grounded on clinical guideline sources</span>
                </div>

                {/* Conversation area */}
                <div className="min-h-[340px] mb-4 border border-neutral-100 rounded-lg p-4 bg-neutral-50/50 space-y-4">
                  {chatMessages.map((msg, index) => (
                    <div key={index} className={`flex gap-3 items-start ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                      {msg.role === "assistant" && (
                        <div className="bg-red-100 text-red-600 rounded p-1.5 shadow-sm">
                          <Activity className="size-4" />
                        </div>
                      )}
                      <div className={`rounded-xl px-4 py-2.5 text-xs max-w-xl leading-relaxed shadow-sm ${msg.role === "user" ? "bg-neutral-900 text-white rounded-tr-none" : "bg-white border border-neutral-200/80 rounded-tl-none text-neutral-800"}`}>
                        <p className="whitespace-pre-line">{msg.content}</p>
                        
                        {/* Render structured citations if any exist */}
                        {msg.citations && msg.citations.length > 0 && (
                          <div className="border-t border-neutral-100 pt-2 mt-2 space-y-1">
                            <span className="text-[9px] font-bold text-neutral-400 uppercase tracking-wider block">Grounded Citations</span>
                            {msg.citations.map((cit, ci) => (
                              <div key={ci} className="text-[10px] text-neutral-500">
                                📚 {cit.url ? <a className="underline" href={cit.url} target="_blank" rel="noreferrer">{cit.title}</a> : cit.title} {cit.section ? `· ${cit.section}` : ""} {(cit.page_or_locator && !String(cit.page_or_locator).startsWith("http")) ? `(${cit.page_or_locator})` : ""}
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                      {msg.role === "user" && (
                        <div className="bg-neutral-200 text-neutral-700 rounded p-1.5 shadow-sm">
                          <User className="size-4" />
                        </div>
                      )}
                    </div>
                  ))}
                  {sendingChat && (
                    <div className="flex gap-2 items-center text-xs text-neutral-400">
                      <RefreshCw className="size-4 animate-spin text-red-600" />
                      <span>Generating clinician advisory...</span>
                    </div>
                  )}
                </div>

                {/* Chat composer with suggestion chips */}
                <div className="space-y-3">
                  {/* Suggestion Chips */}
                  <div className="flex gap-2 overflow-x-auto pb-1 scrollbar-none select-none">
                    {[
                      "What supports this diagnosis?",
                      "Is this condition serious?",
                      "What are the primary symptoms?",
                      "What is the recommended next step?"
                    ].map((sug, si) => (
                      <button
                        key={si}
                        type="button"
                        onClick={() => setUserQuery(sug)}
                        className="shrink-0 text-[10px] bg-neutral-100 hover:bg-neutral-200 border border-neutral-200/60 rounded-full px-3 py-1 font-medium text-neutral-600 hover:text-neutral-900 transition"
                      >
                        {sug}
                      </button>
                    ))}
                  </div>

                  <form onSubmit={triggerChat} className="flex gap-3">
                    <input
                      type="text"
                      placeholder="Ask about these ECG findings, evidence, symptoms, or risk indicators..."
                      value={userQuery}
                      onChange={(e) => setUserQuery(e.target.value)}
                      className="flex-1 border border-neutral-200 rounded-lg px-4 py-2 text-xs focus:outline-none focus:ring-1 focus:ring-red-500 bg-neutral-50/50 focus:bg-white transition"
                    />
                    <button
                      type="submit"
                      disabled={sendingChat || !userQuery.trim()}
                      className="bg-neutral-900 text-white rounded-lg px-4 py-2 text-xs font-bold hover:bg-neutral-800 disabled:opacity-50 disabled:cursor-not-allowed transition flex items-center gap-1.5"
                    >
                      <span>Send</span>
                      <Send className="size-3" />
                    </button>
                  </form>
                </div>
              </section>

              {/* Final section: detailed append-only clinician feedback */}
              <section id="feedback" className="scroll-mt-32 bg-white border border-neutral-200 rounded-xl shadow-sm p-6">
                <div className="border-b border-neutral-100 pb-3 mb-5">
                  <h3 className="text-xs font-bold text-neutral-600 uppercase tracking-wider">Clinician Audit and Learning Feedback</h3>
                  <p className="text-xs text-neutral-500 mt-1">Stored append-only for adjudication and future model evaluation. It never changes the active system consensus or FAISS index.</p>
                </div>
                {feedbackSubmitted ? <div className="bg-green-50 border border-green-200 rounded-lg p-4 text-sm text-green-700">Feedback recorded and queued for independent review.</div> : (
                  <form onSubmit={handleFeedbackSubmit} className="space-y-5">
                    <div className="grid md:grid-cols-3 gap-4">
                      <label className="text-xs font-semibold">Diagnostic correctness<select value={verdict} onChange={e => setVerdict(e.target.value)} className="mt-1 w-full border rounded-lg p-2 font-normal"><option>Correct</option><option>Partially Correct</option><option>Incorrect</option><option>Cannot Determine</option><option>Poor ECG Quality</option></select></label>
                      <label className="text-xs font-semibold">Confidence calibration<select value={confidenceRating} onChange={e => setConfidenceRating(e.target.value)} className="mt-1 w-full border rounded-lg p-2 font-normal"><option>Too High</option><option>Appropriate</option><option>Too Low</option></select></label>
                      <label className="text-xs font-semibold">Bridge explanation quality<select value={explanationRating} onChange={e => setExplanationRating(e.target.value)} className="mt-1 w-full border rounded-lg p-2 font-normal"><option>Very Useful</option><option>Useful</option><option>Neutral</option><option>Misleading</option><option>Incorrect</option></select></label>
                      <label className="text-xs font-semibold">Corrected primary SCP label<input value={correctedPrimary} onChange={e => setCorrectedPrimary(e.target.value.toUpperCase())} placeholder="e.g. IRBBB" className="mt-1 w-full border rounded-lg p-2 font-normal" /></label>
                      <label className="text-xs font-semibold">Other valid SCP labels<input value={correctedSecondary} onChange={e => setCorrectedSecondary(e.target.value)} placeholder="Comma-separated labels" className="mt-1 w-full border rounded-lg p-2 font-normal" /></label>
                      <label className="text-xs font-semibold">Corrected clinical family<select value={correctedFamily} onChange={e => setCorrectedFamily(e.target.value)} className="mt-1 w-full border rounded-lg p-2 font-normal">{["Normal","Rhythm","Conduction","Infarction","Hypertrophy","Repolarization","Ischemia","Pacing","Other"].map(v => <option key={v}>{v}</option>)}</select></label>
                    </div>
                    <div>
                      <h4 className="text-xs font-bold mb-2">Retrieved-neighbor relevance</h4>
                      <div className="grid md:grid-cols-2 gap-2">{(analysisResult?.retrieval?.matches || analysisResult?.retrieval_matches || []).slice(0,5).map((match:any, index:number) => { const id=String(match.ecg_id ?? match.faiss_row ?? index); return <label key={id} className="flex items-center justify-between border rounded-lg p-2 text-xs"><span>Rank {index+1} · ECG {id}</span><select value={retrievalRatings[id] || "Relevant"} onChange={e => setRetrievalRatings(p => ({...p,[id]:e.target.value}))} className="border rounded p-1"><option>Highly Relevant</option><option>Relevant</option><option>Somewhat Relevant</option><option>Not Relevant</option><option>Misleading</option></select></label>; })}</div>
                    </div>
                    <div className="grid md:grid-cols-[220px_1fr] gap-4"><label className="text-xs font-semibold">Clinician identifier<input value={reviewerId} onChange={e => setReviewerId(e.target.value)} className="mt-1 w-full border rounded-lg p-2 font-normal" /></label><label className="text-xs font-semibold">Corrections, reasoning, citation, language, window or measurement notes<textarea value={feedbackNotes} onChange={e => setFeedbackNotes(e.target.value)} className="mt-1 w-full border rounded-lg p-2 min-h-24 font-normal" placeholder="Describe missed labels/windows, incorrect measurements, retrieval relevance, citation issues, or explanation quality." /></label></div>
                    <button type="submit" className="w-full bg-neutral-900 text-white rounded-lg py-3 text-sm font-bold hover:bg-neutral-800">Submit Feedback for Adjudication</button>
                  </form>
                )}
              </section>

              {/* Section-aware assistant: shares the active case chat state. */}
              <div className="fixed bottom-6 right-6 z-[120] flex flex-col items-end gap-3">
                {floatingChatOpen && <div className="w-[min(420px,calc(100vw-3rem))] rounded-2xl border border-neutral-200 bg-white shadow-2xl overflow-hidden">
                  <div className="flex items-center justify-between bg-neutral-950 px-4 py-3 text-white">
                    <div>
                      <div className="text-xs font-bold">Clinical ECG Assistant</div>
                      <div className="text-[10px] text-neutral-300">Context: {REPORT_SECTIONS.find(([id]) => id === activeReportSection)?.[1] || "Current report"}</div>
                    </div>
                    <button type="button" onClick={() => setFloatingChatOpen(false)} className="text-neutral-300 hover:text-white text-lg leading-none">×</button>
                  </div>
                  <div className="max-h-64 overflow-y-auto p-4 text-xs leading-relaxed text-neutral-700 bg-neutral-50">
                    {floatingBusy ? <div className="flex items-center gap-2"><RefreshCw className="size-4 animate-spin text-red-600" />Reviewing this section and active case…</div> : floatingAnswer || "Ask about the waveform, current metrics, consensus, retrieval, evidence or feedback section without leaving this page."}
                  </div>
                  <form onSubmit={askSectionAssistant} className="flex gap-2 border-t border-neutral-200 p-3">
                    <input value={floatingQuery} onChange={e => setFloatingQuery(e.target.value)} placeholder="Ask about this section…" className="min-w-0 flex-1 rounded-lg border border-neutral-300 px-3 py-2 text-xs focus:outline-none focus:ring-1 focus:ring-red-500" />
                    <button disabled={!floatingQuery.trim() || floatingBusy} className="rounded-lg bg-red-600 px-3 py-2 text-white disabled:opacity-40"><Send className="size-4" /></button>
                  </form>
                </div>}
                <button type="button" onClick={() => setFloatingChatOpen(open => !open)} aria-label="Open section-aware clinical ECG assistant" className="relative size-16 rounded-full bg-red-600 text-white shadow-xl border-4 border-white hover:bg-red-700 transition animate-pulse">
                  <Heart className="absolute left-3 top-3 size-8 fill-white" />
                  <Stethoscope className="absolute right-1 bottom-1 size-6 rounded-full bg-neutral-950 p-1" />
                </button>
              </div>

            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
}

// Reusable micro-metric cards
interface MetricCardProps {
  label: string;
  value: any;
  unit?: string;
  status?: "normal" | "abnormal" | "warning";
}

function MetricCard({ label, value, unit, status = "normal" }: MetricCardProps) {
  const isAvailable = value !== undefined && value !== null;
  return (
    <div className="bg-neutral-50/50 border border-neutral-200/70 rounded-lg p-2.5 space-y-0.5 select-none">
      <div className="text-[9px] font-bold text-neutral-400 uppercase tracking-wider">{label}</div>
      <div className="flex items-baseline gap-0.5">
        <span className={`text-sm font-black ${isAvailable ? "text-neutral-900" : "text-neutral-400"}`}>
          {isAvailable ? value : "—"}
        </span>
        {isAvailable && unit && <span className="text-[9px] text-neutral-400 font-semibold">{unit}</span>}
      </div>
      {!isAvailable && <div className="text-[8px] text-neutral-400 font-mono italic">Not measured</div>}
    </div>
  );
}

// HTML Canvas ECG leads trace plotting component
interface CanvasProps {
  waveform: number[][]; // [12, samples]
  gain: number;
  zoom: number;
  leadNames: string[];
  selectedLead?: string;
}

function ECGWaveformCanvas({ waveform, gain, zoom, leadNames, selectedLead = "All" }: CanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    
    const resize = () => {
      canvas.width = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
    };
    resize();

    // Render loop
    const h = canvas.height;
    const w = canvas.width;
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, w, h);

    // Minor / Major grid sizes
    const sz = 20;
    ctx.lineWidth = 0.5;
    ctx.strokeStyle = "rgba(229,57,53,0.06)";
    for (let x = 0; x <= w; x += sz) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    }
    for (let y = 0; y <= h; y += sz) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }
    ctx.strokeStyle = "rgba(229,57,53,0.12)";
    for (let x = 0; x <= w; x += sz * 5) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    }
    for (let y = 0; y <= h; y += sz * 5) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }

    // Filter leads if single lead is chosen
    const leadsToRender: Array<{ values: number[]; label: string; index: number }> = [];
    if (selectedLead === "All") {
      waveform.forEach((vals, i) => {
        leadsToRender.push({ values: vals, label: leadNames[i] || `Lead ${i + 1}`, index: i });
      });
    } else {
      const idx = leadNames.indexOf(selectedLead);
      if (idx !== -1 && waveform[idx]) {
        leadsToRender.push({ values: waveform[idx], label: selectedLead, index: 0 });
      }
    }

    const numRows = leadsToRender.length;
    const rowH = h / numRows;
    const paddingLeft = 45;

    leadsToRender.forEach((lead, renderIdx) => {
      const midY = rowH * renderIdx + rowH * 0.5;
      const label = lead.label;
      const leadValues = lead.values;

      // Row separator
      ctx.strokeStyle = "rgba(229,57,53,0.10)";
      ctx.lineWidth = 0.5;
      ctx.beginPath();
      ctx.moveTo(0, rowH * renderIdx);
      ctx.lineTo(w, renderIdx === numRows - 1 ? midY : rowH * (renderIdx + 1)); // clean separator line
      ctx.stroke();

      // Label text
      ctx.font = "bold 10px 'JetBrains Mono', 'Courier New', monospace";
      ctx.fillStyle = "rgba(229,57,53,0.55)";
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillText(label, 8, midY);

      // Baseline
      ctx.strokeStyle = "rgba(229,57,53,0.05)";
      ctx.lineWidth = 0.5;
      ctx.beginPath();
      ctx.moveTo(paddingLeft, midY);
      ctx.lineTo(w, midY);
      ctx.stroke();

      // Plot trace
      if (leadValues && leadValues.length > 0) {
        ctx.save();
        ctx.beginPath();
        ctx.rect(paddingLeft, rowH * renderIdx + 1, w - paddingLeft, rowH - 2);
        ctx.clip();

        ctx.strokeStyle = "#E53935";
        ctx.lineWidth = selectedLead === "All" ? 1.2 : 1.7; // slightly thicker trace in single view
        ctx.lineJoin = "round";
        ctx.beginPath();

        const samples = leadValues.length;
        const widthPerSample = (w - paddingLeft) / samples * zoom;

        // Robust per-lead display normalization. It preserves morphology while
        // preventing a few extreme samples from clipping every visible peak.
        const finite = leadValues.filter(Number.isFinite).sort((a, b) => a - b);
        const median = finite.length ? finite[Math.floor(finite.length * 0.5)] : 0;
        const deviations = finite.map(v => Math.abs(v - median)).sort((a, b) => a - b);
        const robustAmplitude = deviations.length ? Math.max(deviations[Math.floor(deviations.length * 0.995)], 1e-6) : 1;

        for (let s = 0; s < samples; s++) {
          const x = paddingLeft + s * widthPerSample;
          // Scale so lead amplitude values comfort inside row heights
          const normalized = Math.max(-1.0, Math.min(1.0, (leadValues[s] - median) / robustAmplitude));
          const y = midY - normalized * (rowH * 0.40) * gain;
          
          if (s === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.restore();
      }
    });

  }, [waveform, gain, zoom, leadNames, selectedLead]);

  return <canvas ref={canvasRef} className="size-full block" />;
}
