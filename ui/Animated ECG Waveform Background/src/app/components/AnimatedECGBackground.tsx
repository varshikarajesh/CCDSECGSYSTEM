import { useEffect, useRef } from "react";

const LEADS = [
  { name: "I",   p:  0.14, q: -0.08, r:  0.72, s: -0.14, t:  0.26 },
  { name: "II",  p:  0.20, q: -0.10, r:  1.00, s: -0.18, t:  0.36 },
  { name: "III", p:  0.07, q: -0.14, r:  0.40, s: -0.28, t:  0.12 },
  { name: "aVR", p: -0.17, q: -0.05, r: -0.76, s:  0.10, t: -0.26 },
  { name: "aVL", p:  0.05, q: -0.10, r:  0.33, s: -0.18, t:  0.09 },
  { name: "aVF", p:  0.17, q: -0.09, r:  0.68, s: -0.20, t:  0.26 },
  { name: "V1",  p:  0.07, q:  0.00, r:  0.20, s: -0.74, t: -0.14 },
  { name: "V2",  p:  0.09, q:  0.00, r:  0.38, s: -0.58, t:  0.18 },
  { name: "V3",  p:  0.11, q: -0.04, r:  0.60, s: -0.42, t:  0.24 },
  { name: "V4",  p:  0.14, q: -0.07, r:  0.84, s: -0.26, t:  0.30 },
  { name: "V5",  p:  0.15, q: -0.09, r:  0.92, s: -0.18, t:  0.33 },
  { name: "V6",  p:  0.15, q: -0.08, r:  0.80, s: -0.12, t:  0.29 },
] as const;

type Lead = (typeof LEADS)[number];

function ecgSample(t: number, lead: Lead): number {
  const g = (mu: number, sigma: number, amp: number) =>
    amp * Math.exp(-0.5 * ((t - mu) / sigma) ** 2);
  return (
    g(0.130, 0.026, lead.p) +
    g(0.265, 0.011, lead.q) +
    g(0.305, 0.014, lead.r) +
    g(0.365, 0.011, lead.s) +
    g(0.580, 0.050, lead.t)
  );
}

const BG         = "#ffffff";
const ECG_COLOR  = "#E53935";
const GRID_MINOR = "rgba(229,57,53,0.10)";
const GRID_MAJOR = "rgba(229,57,53,0.22)";
const LABEL_CLR  = "rgba(229,57,53,0.45)";
const BEAT_PX    = 340;
const SPEED      = 1;

interface BackgroundProps {
  opacity?: number;
  paused?: boolean;
}

export default function AnimatedECGBackground({ opacity = 0.25, paused = false }: BackgroundProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const bufsRef = useRef<number[][]>(LEADS.map(() => []));
  const offsetRef = useRef(0);
  const rafRef = useRef(0);
  const isPausedRef = useRef(paused);

  useEffect(() => {
    isPausedRef.current = paused;
  }, [paused]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;

    // Check media queries for reduced motion
    const mediaQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
    let prefersReducedMotion = mediaQuery.matches;

    const handleMotionChange = (e: MediaQueryListEvent) => {
      prefersReducedMotion = e.matches;
    };
    mediaQuery.addEventListener("change", handleMotionChange);

    const resize = () => {
      canvas.width  = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
      bufsRef.current = LEADS.map(() => new Array(canvas.width).fill(0));
    };
    resize();

    const ro = new ResizeObserver(resize);
    ro.observe(canvas);

    const drawGrid = (w: number, h: number) => {
      const sz = 20;
      ctx.lineWidth = 0.5;
      ctx.strokeStyle = GRID_MINOR;
      for (let x = 0; x <= w; x += sz) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
      }
      for (let y = 0; y <= h; y += sz) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
      }

      ctx.strokeStyle = GRID_MAJOR;
      for (let x = 0; x <= w; x += sz * 5) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
      }
      for (let y = 0; y <= h; y += sz * 5) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
      }
    };

    const drawTrace = (buf: number[], midY: number, amp: number, w: number) => {
      ctx.save();
      ctx.shadowColor = ECG_COLOR;
      ctx.shadowBlur  = 10;
      ctx.strokeStyle = "rgba(229,57,53,0.18)";
      ctx.lineWidth   = 5;
      ctx.lineJoin    = "round";
      ctx.beginPath();
      for (let x = 0; x < w; x++) {
        const y = midY - buf[x] * amp;
        x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.restore();

      ctx.save();
      ctx.shadowColor = ECG_COLOR;
      ctx.shadowBlur  = 3;
      ctx.strokeStyle = ECG_COLOR;
      ctx.lineWidth   = 1.4;
      ctx.lineJoin    = "round";
      ctx.beginPath();
      for (let x = 0; x < w; x++) {
        const y = midY - buf[x] * amp;
        x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.restore();
    };

    let documentHidden = false;
    const handleVisibilityChange = () => {
      documentHidden = document.hidden;
    };
    document.addEventListener("visibilitychange", handleVisibilityChange);

    const frame = () => {
      const w = canvas.width;
      const h = canvas.height;
      if (w === 0 || h === 0) {
        rafRef.current = requestAnimationFrame(frame);
        return;
      }

      // Hidden tabs do not need redraw work. Static/reduced-motion modes still
      // render a complete first frame instead of leaving a blank canvas.
      if (documentHidden) {
        rafRef.current = requestAnimationFrame(frame);
        return;
      }

      const bufs  = bufsRef.current;
      const rowH  = h / LEADS.length;
      const amp   = rowH * 0.38;

      const staticMode = prefersReducedMotion || isPausedRef.current;
      if (staticMode) {
        LEADS.forEach((lead, li) => {
          bufs[li] = Array.from({ length: w }, (_, x) => ecgSample((x % BEAT_PX) / BEAT_PX, lead));
        });
      } else {
        for (let i = 0; i < SPEED; i++) {
          const t = (offsetRef.current % BEAT_PX) / BEAT_PX;
          LEADS.forEach((lead, li) => {
            bufs[li].shift();
            bufs[li].push(ecgSample(t, lead));
          });
          offsetRef.current++;
        }
        }

      LEADS.forEach((_, li) => {
        while (bufs[li].length < w) bufs[li].unshift(0);
        while (bufs[li].length > w) bufs[li].shift();
      });

      ctx.fillStyle = BG;
      ctx.fillRect(0, 0, w, h);
      drawGrid(w, h);

      const TRACE_START = 52;
      const LABEL_X     = 10;

      LEADS.forEach((lead, li) => {
        const midY = rowH * li + rowH * 0.5;

        ctx.strokeStyle = "rgba(229,57,53,0.18)";
        ctx.lineWidth   = 0.5;
        ctx.beginPath();
        ctx.moveTo(0, rowH * li);
        ctx.lineTo(w, rowH * li);
        ctx.stroke();

        ctx.strokeStyle = "rgba(229,57,53,0.12)";
        ctx.lineWidth   = 0.5;
        ctx.beginPath();
        ctx.moveTo(TRACE_START, midY);
        ctx.lineTo(w, midY);
        ctx.stroke();

        ctx.font      = "500 11px 'JetBrains Mono', 'Courier New', monospace";
        ctx.fillStyle = LABEL_CLR;
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(lead.name, LABEL_X, midY);

        ctx.save();
        ctx.beginPath();
        ctx.rect(TRACE_START, rowH * li + 2, w - TRACE_START, rowH - 4);
        ctx.clip();

        const slicedBuf = bufs[li].slice(0, w - TRACE_START);
        drawTrace(slicedBuf, midY, amp, w - TRACE_START);

        ctx.restore();
      });

      const fade = ctx.createLinearGradient(TRACE_START, 0, TRACE_START + (w - TRACE_START) * 0.22, 0);
      fade.addColorStop(0, "rgba(255,255,255,0.60)");
      fade.addColorStop(1, "rgba(255,255,255,0)");
      ctx.fillStyle = fade;
      ctx.fillRect(TRACE_START, 0, (w - TRACE_START) * 0.22, h);

      rafRef.current = requestAnimationFrame(frame);
    };

    rafRef.current = requestAnimationFrame(frame);

    return () => {
      cancelAnimationFrame(rafRef.current);
      ro.disconnect();
      mediaQuery.removeEventListener("change", handleMotionChange);
      document.removeEventListener("visibilitychange", handleVisibilityChange);
    };
  }, []);

  return (
    <div className="absolute inset-0 size-full select-none pointer-events-none transition-opacity duration-700" style={{ opacity }}>
      <canvas ref={canvasRef} className="size-full block" />
    </div>
  );
}
