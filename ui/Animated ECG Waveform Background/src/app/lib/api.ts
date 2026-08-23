// Development uses Vite's same-origin /api proxy, avoiding browser CORS and
// allowing any local frontend port. Jetson deployments may set an explicit
// VITE_TRACE_API_URL when the API is hosted separately.
const API_BASE = (import.meta.env.VITE_TRACE_API_URL || "").replace(/\/$/, "");

async function apiError(res: Response, fallback: string) {
  const body = await res.json().catch(() => ({}));
  return body.message || body.detail || `${fallback} (HTTP ${res.status})`;
}

export interface ChatResponse {
  answer: string;
  text: string;
  intent: string;
  active_condition: {
    label: string;
    name: string;
  };
  evidence: {
    source_type: string;
    chunk_ids: string[];
  };
  citations: Array<{
    title: string;
    organization_or_authors: string;
    year: string;
    section: string;
    page_or_locator: string;
    doi: string;
    url: string;
  }>;
  status: string;
}

export interface WindowSelectionItem {
  window_index: number;
  screening: {
    status: string;
    combined_score: number;
    selected: boolean;
  };
  diagnostic: {
    primary_label: string;
    supported_labels: Array<{ label: string; probability: number }>;
    probabilities: Record<string, number>;
    confidence: number;
  };
  statistics: Record<string, any>;
  retrieval: Record<string, any>;
}

export async function runInference(params: {
  file: File;
  samplingRateHz?: number;
  topK?: number;
  question?: string;
  includeRetrieval?: boolean;
  includeKnowledge?: boolean;
  includeExplanation?: boolean;
  leadNames?: string[];
  conversationId?: string;
}) {
  const formData = new FormData();
  formData.append("file", params.file);
  formData.append("sampling_rate_hz", String(params.samplingRateHz ?? 100));
  formData.append("top_k", String(params.topK ?? 5));
  formData.append("question", params.question ?? "What is the primary finding and diagnostic conclusion?");
  formData.append("include_retrieval", String(params.includeRetrieval ?? true));
  formData.append("include_knowledge", String(params.includeKnowledge ?? true));
  formData.append("include_explanation", String(params.includeExplanation ?? true));
  if (params.leadNames) {
    formData.append("lead_names_json", JSON.stringify(params.leadNames));
  }
  if (params.conversationId) {
    formData.append("conversation_id", params.conversationId);
  }

  const res = await fetch(`${API_BASE}/api/inference`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const errorJson = await res.json().catch(() => ({}));
    throw new Error(errorJson.message || errorJson.detail || `Inference failed with status ${res.status}`);
  }
  return res.json();
}

export async function runRecordingInference(params: {
  file: File;
  recordingMode: "10s" | "2min" | "5min";
  samplingRateHz?: number;
  topK?: number;
  manualWindowIndices?: number[];
  leadNames?: string[];
  question?: string;
  includeExplanation?: boolean;
}) {
  const formData = new FormData();
  formData.append("file", params.file);
  formData.append("recording_mode", params.recordingMode);
  formData.append("sampling_rate_hz", String(params.samplingRateHz ?? 100));
  formData.append("top_k", String(params.topK ?? 5));
  formData.append("manual_window_indices_json", JSON.stringify(params.manualWindowIndices ?? []));
  formData.append("question", params.question ?? "What is the primary finding and diagnostic conclusion?");
  formData.append("include_explanation", String(params.includeExplanation ?? true));
  if (params.leadNames) {
    formData.append("lead_names_json", JSON.stringify(params.leadNames));
  }

  const res = await fetch(`${API_BASE}/api/recording-inference`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const errorJson = await res.json().catch(() => ({}));
    throw new Error(errorJson.message || errorJson.detail || `Recording inference failed with status ${res.status}`);
  }
  return res.json();
}

export async function getRecordingWindow(recordingId: string, startSeconds: number, endSeconds: number) {
  const params = new URLSearchParams({
    start_seconds: String(startSeconds),
    end_seconds: String(endSeconds),
  });
  const res = await fetch(`${API_BASE}/api/recordings/${recordingId}/window?${params.toString()}`);
  if (!res.ok) {
    const errorJson = await res.json().catch(() => ({}));
    throw new Error(errorJson.message || errorJson.detail || `Failed to retrieve window waveform`);
  }
  return res.json();
}

export async function runRecordingChat(recordingId: string, question: string): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE}/api/recordings/${recordingId}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!res.ok) {
    const errorJson = await res.json().catch(() => ({}));
    throw new Error(errorJson.message || errorJson.detail || `Chat request failed`);
  }
  return res.json();
}

export async function overrideRecordingWindows(recordingId: string, windowIndices: number[]) {
  const res = await fetch(`${API_BASE}/api/recordings/${recordingId}/windows/select`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ window_indices: windowIndices }),
  });
  if (!res.ok) {
    const errorJson = await res.json().catch(() => ({}));
    throw new Error(errorJson.message || errorJson.detail || `Window selection override failed`);
  }
  return res.json();
}

export async function submitFeedback(payload: {
  recording_id: string;
  reviewer_id: string;
  new_status?: string;
  notes?: string;
  verdict?: string;
  [key: string]: any;
}) {
  const res = await fetch(`${API_BASE}/api/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const errorJson = await res.json().catch(() => ({}));
    throw new Error(errorJson.message || errorJson.detail || `Feedback submission failed`);
  }
  return res.json();
}

export async function getRetrievalNeighborWaveform(ecgId: number) {
  const res = await fetch(`${API_BASE}/api/retrieval/neighbors/${ecgId}/waveform`);
  if (!res.ok) {
    const errorJson = await res.json().catch(() => ({}));
    throw new Error(errorJson.message || errorJson.detail || `Failed to fetch neighbor waveform`);
  }
  return res.json();
}

export async function getLabelRegistry() {
  const res = await fetch(`${API_BASE}/api/labels`);
  if (!res.ok) throw new Error("Failed to load diagnostic label registry");
  return res.json();
}

export async function checkSystemStatus() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    if (!res.ok) return { ready: false, details: `Health status returned ${res.status}` };
    const data = await res.json();
    return {
      ready: data.status === "healthy",
      details: data,
    };
  } catch (err: any) {
    return { ready: false, details: err.message || "Failed to contact backend services" };
  }
}
