require('dotenv').config();

const express    = require('express');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const helmet     = require('helmet');
const rateLimit  = require('express-rate-limit');
const cors       = require('cors');
const NodeCache  = require('node-cache');
const { v4: uuidv4 } = require('uuid');

// ── Environment Configuration ──────────────────────────────────────────────
const requiredEnvVars = ['GEMINI_API_KEY'];
const optionalEnvVars = {
  PORT:                3000,
  MAX_QUESTION_LENGTH: 500,
  RATE_LIMIT_WINDOW:   60,
  RATE_LIMIT_MAX:      30,
  REQUEST_TIMEOUT_MS:  15000,
  STT_TIMEOUT_MS:      20000,
  NODE_ENV:            'production',
  GEMINI_CHAT_MODEL:   'gemini-2.5-flash',
  GEMINI_TTS_MODEL:    'gemini-2.5-flash-preview-tts',
  TTS_VOICE_NAME:      'Leda',
  SESSION_TTL_SEC:     1800,   // 30 min — session memory expires after inactivity
  MAX_HISTORY_TURNS:   10      // keep last 10 exchanges to avoid huge context windows
};

requiredEnvVars.forEach(varName => {
  if (!process.env[varName]) {
    console.error(`FATAL: ${varName} is not set in environment variables.`);
    process.exit(1);
  }
});

Object.entries(optionalEnvVars).forEach(([key, defaultValue]) => {
  if (!process.env[key]) {
    process.env[key] = defaultValue.toString();
    console.log(`Using default for ${key}: ${defaultValue}`);
  }
});

// ── Initialize App ─────────────────────────────────────────────────────────
const app   = express();
const port  = process.env.PORT;
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

// Cache text answers only — never audio blobs (memory safety)
const textCache = new NodeCache({ stdTTL: 300 }); // 5 min TTL for repeated questions

// ── Session Memory Store ───────────────────────────────────────────────────
// Stores conversation history per session so Carrot remembers past exchanges.
// Key: sessionId (string)
// Value: array of { role: 'user'|'model', parts: [{ text }] }
//
// Sessions expire after SESSION_TTL_SEC seconds of inactivity.
// Only the last MAX_HISTORY_TURNS exchanges are kept to control context size.
const sessionStore = new NodeCache({
  stdTTL:      parseInt(process.env.SESSION_TTL_SEC),
  checkperiod: 120   // clean up expired sessions every 2 minutes
});

// ── Custom Error Class ─────────────────────────────────────────────────────
class CarrotError extends Error {
  constructor(message, type = 'GENERAL_ERROR', statusCode = 500) {
    super(message);
    this.name       = 'CarrotError';
    this.type       = type;
    this.statusCode = statusCode;
    this.timestamp  = new Date().toISOString();
  }
}

// ── Metrics ────────────────────────────────────────────────────────────────
const metrics = {
  requests:          0,
  errors:            0,
  cacheHits:         0,
  totalResponseTime: 0,
  avgResponseTime:   0,
  startTime:         Date.now()
};

// ── Security Middleware ────────────────────────────────────────────────────
app.use(cors({
  origin: process.env.NODE_ENV === 'production'
    ? process.env.ALLOWED_ORIGINS?.split(',') || true
    : true,
  methods:        ['GET', 'POST'],
  allowedHeaders: ['Content-Type', 'Authorization']
}));

app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      scriptSrc:  ["'self'"],
      styleSrc:   ["'self'"],
      imgSrc:     ["'self'", "data:", "https:"],
    },
  },
  hsts: { maxAge: 31536000, includeSubDomains: true, preload: true }
}));

// 2mb limit — enough for ~5s INMP441 audio as Base64
app.use(express.json({ limit: '2mb' }));

// ── Request ID Middleware ──────────────────────────────────────────────────
app.use((req, res, next) => {
  req.id = uuidv4();
  res.setHeader('X-Request-ID', req.id);
  next();
});

// ── Request Logger ─────────────────────────────────────────────────────────
app.use((req, res, next) => {
  const startTime = Date.now();
  res.on('finish', () => {
    const duration = Date.now() - startTime;
    console.log(JSON.stringify({
      requestId: req.id,
      timestamp: new Date().toISOString(),
      method:    req.method,
      url:       req.url,
      status:    res.statusCode,
      duration:  `${duration}ms`,
      ip:        req.ip,
      userAgent: req.get('user-agent')
    }));
    metrics.requests++;
    metrics.totalResponseTime += duration;
    metrics.avgResponseTime = metrics.totalResponseTime / metrics.requests;
  });
  next();
});

// ── Rate Limiting ──────────────────────────────────────────────────────────
const limiter = rateLimit({
  windowMs:        parseInt(process.env.RATE_LIMIT_WINDOW) * 1000,
  max:             parseInt(process.env.RATE_LIMIT_MAX),
  standardHeaders: true,
  legacyHeaders:   false,
  message:         { error: "Too many requests, slow down!" },
  keyGenerator:    (req) => req.ip || req.id,
  skip:            (req) => req.path === '/health' || req.path === '/status'
});
app.use('/ask', limiter);

// ── Request Timeout Helper ─────────────────────────────────────────────────
const withTimeout = (promise, ms = parseInt(process.env.REQUEST_TIMEOUT_MS)) =>
  Promise.race([
    promise,
    new Promise((_, reject) =>
      setTimeout(
        () => reject(new CarrotError('Request timed out', 'TIMEOUT_ERROR', 504)),
        ms
      )
    )
  ]);

// ── Session History Helpers ────────────────────────────────────────────────

// Get existing history for a session, or start fresh
function getHistory(sessionId) {
  return sessionStore.get(sessionId) || [];
}

// Append a user turn and a model turn, then trim to MAX_HISTORY_TURNS
// Each "turn" = one user message + one model reply = 2 entries
function saveHistory(sessionId, userText, modelText) {
  const maxTurns   = parseInt(process.env.MAX_HISTORY_TURNS); // exchanges
  let   history    = getHistory(sessionId);

  // Append new exchange
  history.push({ role: 'user',  parts: [{ text: userText  }] });
  history.push({ role: 'model', parts: [{ text: modelText }] });

  // Trim oldest turns if over limit (each turn = 2 entries)
  const maxEntries = maxTurns * 2;
  if (history.length > maxEntries) {
    history = history.slice(history.length - maxEntries);
  }

  // Reset TTL on every update so active sessions stay alive
  sessionStore.set(sessionId, history);
}

// ── Robot Commands (Offline / Local ESP32 Mode) ────────────────────────────
const ROBOT_COMMANDS = [
  {
    keywords: ['move forward', 'go forward', 'forward', 'move ahead', 'go ahead'],
    command:  { command: 'FORWARD',   speed: 'normal' }
  },
  {
    keywords: ['move back', 'go back', 'backward', 'move backward', 'reverse'],
    command:  { command: 'BACKWARD',  speed: 'normal' }
  },
  {
    keywords: ['turn left', 'go left', 'left'],
    command:  { command: 'LEFT',      angle: 90 }
  },
  {
    keywords: ['turn right', 'go right', 'right'],
    command:  { command: 'RIGHT',     angle: 90 }
  },
  {
    keywords: ['stop', 'halt', 'freeze', 'stay'],
    command:  { command: 'STOP',      emergency: false }
  },
  {
    keywords: ['go faster', 'speed up', 'faster'],
    command:  { command: 'SPEED_UP',  increment: 10 }
  },
  {
    keywords: ['go slower', 'slow down', 'slower'],
    command:  { command: 'SLOW_DOWN', decrement: 10 }
  }
];

// ── POST /ask ──────────────────────────────────────────────────────────────
// Accepts either:
//   { audioData: "<base64>", mimeType: "audio/wav", sessionId: "abc" }
//   { question:  "hey carrot what is AI",           sessionId: "abc" }
//
// sessionId is optional — if omitted, a new session is created each request
// (no memory). ESP32 should generate one UUID on boot and reuse it.
app.post('/ask', async (req, res) => {
  try {
    const {
      question,
      audioData: incomingAudio,  // Base64 audio from ESP32 INMP441
      mimeType:  incomingMime,   // e.g. "audio/wav"
      sessionId: clientSessionId // ESP32 sends this to maintain memory
    } = req.body;

    // Use provided sessionId or generate a new one for this request
    const sessionId = clientSessionId || uuidv4();

    // ── STEP 0: "The Ear" — STT via Gemini multimodal ─────────────────────
    let transcribedText = '';

    if (incomingAudio) {
      console.log(`[${req.id}] Raw audio received from INMP441 — transcribing...`);

      const sttModel = genAI.getGenerativeModel({ model: process.env.GEMINI_CHAT_MODEL });

      const sttResult = await withTimeout(
        sttModel.generateContent([
          "Transcribe exactly what the user is saying in this audio. Reply ONLY with the transcribed text, nothing else.",
          {
            inlineData: {
              data:     incomingAudio,
              mimeType: incomingMime || 'audio/wav'
            }
          }
        ]),
        parseInt(process.env.STT_TIMEOUT_MS)
      );

      transcribedText = sttResult.response.text()?.trim() || '';
      console.log(`[${req.id}] STT result: "${transcribedText}"`);

      if (!transcribedText) {
        return res.json({
          answer:    "Sorry, I didn't catch that. Please try again!",
          audio:     null,
          mimeType:  null,
          type:      'stt_failure',
          sessionId,
          requestId: req.id
        });
      }

    } else if (question && typeof question === 'string' && question.trim().length > 0) {
      transcribedText = question.trim();

    } else {
      throw new CarrotError(
        "Payload must contain 'audioData' (Base64) or 'question' (string)",
        'VALIDATION_ERROR',
        400
      );
    }

    // Normalise — lowercase, strip punctuation, trim
    const cleanQ = transcribedText.toLowerCase().replace(/[.,!?]/g, '').trim();
    console.log(`[${req.id}] [session:${sessionId}] Cleaned input: "${cleanQ}"`);

    // ── ROUTE 1: OFFLINE / LOCAL COMMAND MODE ─────────────────────────────
    // Robot commands don't need memory — they are instant physical actions
    if (cleanQ.startsWith('carrot') && !cleanQ.startsWith('hey carrot')) {
      console.log(`[${req.id}] Route: LOCAL COMMAND MODE`);

      const isolatedCommand = cleanQ.replace(/^carrot\s*/, '').trim();
      console.log(`[${req.id}] Isolated command: "${isolatedCommand}"`);

      const matched = ROBOT_COMMANDS.find(entry =>
        entry.keywords.some(k => isolatedCommand.includes(k))
      );

      if (matched) {
        console.log(`[${req.id}] Matched command: ${matched.command.command}`);
        return res.json({
          answer:    `Executing: ${matched.command.command}`,
          command:   matched.command,
          audio:     null,
          mimeType:  null,
          type:      'robot_command',
          sessionId,
          requestId: req.id
        });
      }

      console.log(`[${req.id}] No command matched — falling back to AI mode`);
    }

    // ── ROUTE 2: AI BRAIN MODE (with conversation memory) ─────────────────
    console.log(`[${req.id}] Route: AI BRAIN MODE`);

    // Strip wake words so AI only sees the actual question
    const actualQuestion = cleanQ
      .replace(/^hey carrot\s*/, '')
      .replace(/^carrot\s*/, '')
      .trim() || cleanQ;

    // ── STEP 1: "The Brain" — generate answer with full conversation history
    // Load this session's conversation history
    const history = getHistory(sessionId);
    console.log(`[${req.id}] [session:${sessionId}] History length: ${history.length} entries`);

    // System instruction — defines Carrot's personality and memory behaviour
// NEW
const systemInstruction = `You are Carrot, a cute and friendly robot companion.
You have a memory — you remember everything said earlier in this conversation.
If the user refers to something previously mentioned (like a name, fact, or topic), use that context in your reply.
Reply in 2 to 3 sentences maximum. Keep it cheerful, warm, and conversational.
Avoid bullet points or lists — speak naturally like a friendly robot.`;

    // Build the full conversation to send to Gemini:
    // [system context as first user turn] + [history] + [new question]
    const contents = [
      // Inject system personality as the very first exchange so it's always in context
      {
        role:  'user',
        parts: [{ text: systemInstruction }]
      },
      {
        role:  'model',
        parts: [{ text: "Got it! I'm Carrot, your friendly robot companion. I'll remember our conversation!" }]
      },
      // All previous turns for this session
      ...history,
      // The new question from the user
      {
        role:  'user',
        parts: [{ text: actualQuestion }]
      }
    ];

    // Note: skip text cache for sessions with history — context changes the answer
    const shouldUseCache = history.length === 0;
    const cacheKey       = `q:${actualQuestion}`;
    let   answerText;

    if (shouldUseCache) {
      const cachedText = textCache.get(cacheKey);
      if (cachedText) {
        console.log(`[${req.id}] Cache hit — reusing answer (no history)`);
        metrics.cacheHits++;
        answerText = cachedText;
      }
    }

    if (!answerText) {
      const chatModel  = genAI.getGenerativeModel({ model: process.env.GEMINI_CHAT_MODEL });
      const chatResult = await withTimeout(
        chatModel.generateContent({ contents })
      );

      answerText = chatResult.response.text()?.trim();
      if (!answerText) {
        answerText = "Sorry, I couldn't think of an answer!";
      }

      // Only cache if this is a fresh session (no prior context)
      if (shouldUseCache) {
        textCache.set(cacheKey, answerText);
      }
    }

    // Save this exchange to session memory for future requests
    saveHistory(sessionId, actualQuestion, answerText);
    console.log(`[${req.id}] [session:${sessionId}] Brain answer: "${answerText}"`);

    // ── STEP 2: "The Voice Box" — convert answer to audio (non-fatal) ──────
    let outAudio    = null;
    let outMimeType = null;

    try {
      const ttsModel  = genAI.getGenerativeModel({ model: process.env.GEMINI_TTS_MODEL });
      const ttsResult = await withTimeout(
        ttsModel.generateContent({
          contents: [{ role: 'user', parts: [{ text: answerText }] }],
          generationConfig: {
            responseModalities: ['AUDIO'],
            speechConfig: {
              voiceConfig: {
                prebuiltVoiceConfig: { voiceName: process.env.TTS_VOICE_NAME }
              }
            }
          }
        })
      );

      const parts     = ttsResult.response.candidates?.[0]?.content?.parts ?? [];
      const audioPart = parts.find(p => p.inlineData?.data);

      if (audioPart) {
        outAudio    = audioPart.inlineData.data;
        outMimeType = audioPart.inlineData.mimeType;

        // Safety cap — drop audio if too large for ESP32 to buffer
        if (outAudio.length > 500000) {
          console.warn(`[${req.id}] Audio too large (${outAudio.length} chars) — dropping to protect ESP32`);
          outAudio    = null;
          outMimeType = null;
        } else {
          console.log(`[${req.id}] TTS audio generated (${outMimeType}, ${outAudio.length} chars)`);
        }
      } else {
        console.warn(`[${req.id}] TTS returned no audio — text-only response`);
      }

    } catch (ttsError) {
      console.error(`[${req.id}] TTS failed (non-fatal):`, ttsError.message);
      metrics.errors++;
    }

    // Always return answer text — OLED animation never breaks
    return res.json({
      answer:    answerText,   // Always present — ESP32 OLED safe
      audio:     outAudio,     // Base64 PCM — null if TTS failed or too large
      mimeType:  outMimeType,  // e.g. "audio/pcm"
      type:      'ai_response',
      sessionId,               // Return sessionId so ESP32 can reuse it next request
      requestId: req.id
    });

  } catch (error) {
    metrics.errors++;

    if (error instanceof CarrotError) {
      console.error(`[${req.id}] ${error.type}:`, error.message);
      return res.status(error.statusCode).json({
        error:     error.message,
        type:      error.type,
        requestId: req.id,
        timestamp: error.timestamp
      });
    }

    console.error(`[${req.id}] Unhandled Error:`, error.message ?? error);
    return res.status(500).json({
      error:     "Carrot's brain is tired!",
      type:      'INTERNAL_ERROR',
      requestId: req.id,
      timestamp: new Date().toISOString()
    });
  }
});

// ── POST /session/clear — reset a session's memory ────────────────────────
// Call this from ESP32 on reboot or when user wants Carrot to forget
app.post('/session/clear', (req, res) => {
  const { sessionId } = req.body;
  if (!sessionId) {
    return res.status(400).json({ error: "Missing 'sessionId' in request body." });
  }
  sessionStore.del(sessionId);
  console.log(`[session:${sessionId}] Memory cleared`);
  res.json({
    message:   'Session memory cleared. Carrot has forgotten everything!',
    sessionId,
    timestamp: new Date().toISOString()
  });
});

// ── GET / ──────────────────────────────────────────────────────────────────
app.get('/', (_req, res) => {
  res.send('🥕 Carrot AI Backend is Running');
});

// ── GET /robot ─────────────────────────────────────────────────────────────
app.get('/robot', (_req, res) => {
  res.json({
    name:      'Carrot',
    version:   '1.0',
    status:    'online',
    modes:     ['ai_mode', 'local_command_mode'],
    wake_words: {
      ai_mode:    'hey carrot',
      local_mode: 'carrot'
    },
    memory: {
      session_ttl_minutes: parseInt(process.env.SESSION_TTL_SEC) / 60,
      max_history_turns:   parseInt(process.env.MAX_HISTORY_TURNS),
      active_sessions:     sessionStore.keys().length
    },
    commands: ROBOT_COMMANDS.map(entry => ({
      command:  entry.command.command,
      keywords: entry.keywords
    })),
    timestamp: new Date().toISOString()
  });
});

// ── GET /status ────────────────────────────────────────────────────────────
app.get('/status', (_req, res) => {
  const uptime    = process.uptime();
  const uptimeStr = `${Math.floor(uptime / 3600)}h ${Math.floor((uptime % 3600) / 60)}m ${Math.floor(uptime % 60)}s`;
  res.json({
    status:      'ok',
    message:     'Carrot is Awake! 🥕',
    timestamp:   new Date().toISOString(),
    uptime:      uptimeStr,
    environment: process.env.NODE_ENV
  });
});

// ── GET /health ────────────────────────────────────────────────────────────
app.get('/health', (req, res) => {
  const healthcheck = {
    requestId:    req.id,
    uptime:       process.uptime(),
    timestamp:    new Date().toISOString(),
    status:       'OK',
    dependencies: {
      gemini: process.env.GEMINI_API_KEY ? 'configured' : 'missing',
      cache:  'healthy'
    },
    metrics: {
      totalRequests:   metrics.requests,
      totalErrors:     metrics.errors,
      cacheHits:       metrics.cacheHits,
      cachedAnswers:   textCache.keys().length,
      activeSessions:  sessionStore.keys().length,
      avgResponseTime: `${metrics.avgResponseTime.toFixed(2)}ms`,
      uptime:          `${Math.floor(process.uptime() / 3600)}h ${Math.floor((process.uptime() % 3600) / 60)}m`
    }
  };

  try {
    textCache.set('health-test', 'ok', 10);
    if (textCache.get('health-test') !== 'ok') {
      healthcheck.dependencies.cache = 'unhealthy';
      healthcheck.status = 'DEGRADED';
    }
  } catch {
    healthcheck.dependencies.cache = 'unhealthy';
    healthcheck.status = 'DEGRADED';
  }

  res.status(healthcheck.status === 'OK' ? 200 : 503).json(healthcheck);
});

// ── GET /metrics ───────────────────────────────────────────────────────────
app.get('/metrics', (_req, res) => {
  const runtime = Date.now() - metrics.startTime;
  res.json({
    ...metrics,
    runtime:        `${Math.floor(runtime / 3600000)}h ${Math.floor((runtime % 3600000) / 60000)}m`,
    cachedAnswers:  textCache.keys().length,
    activeSessions: sessionStore.keys().length,
    memoryUsage:    process.memoryUsage(),
    timestamp:      new Date().toISOString()
  });
});

// ── Web Control Command Store ──────────────────────────────────────────────
// ESP32 polls GET /control every 1-2 seconds
// Dashboard POSTs to POST /control to set a command
// Command is consumed (cleared) after ESP32 reads it — one-shot execution
let pendingWebCommand = null;

// POST /control — dashboard sends a command
app.post('/control', (req, res) => {
  const { command } = req.body;
  const validCommands = ['FORWARD', 'BACKWARD', 'LEFT', 'RIGHT', 'STOP',
                         'SPEED_UP', 'SLOW_DOWN'];
  if (!command || !validCommands.includes(command.toUpperCase())) {
    return res.status(400).json({ error: 'Invalid or missing command.' });
  }
  pendingWebCommand = command.toUpperCase();
  console.log(`[WEB CONTROL] Command queued: ${pendingWebCommand}`);
  res.json({
    message:   `Command '${pendingWebCommand}' queued for ESP32`,
    command:   pendingWebCommand,
    timestamp: new Date().toISOString()
  });
});

// GET /control — ESP32 polls this endpoint
// Returns command and immediately clears it (one-shot)
app.get('/control', (req, res) => {
  const cmd = pendingWebCommand;
  pendingWebCommand = null; // clear after sending — prevents repeat execution
  res.json({
    command:   cmd,         // null if no pending command
    timestamp: new Date().toISOString()
  });
});

// ── 404 Handler ────────────────────────────────────────────────────────────
app.use((req, res) => {
  res.status(404).json({
    error:     'Route not found.',
    requestId: req.id,
    timestamp: new Date().toISOString()
  });
});

// ── Global Error Handler ───────────────────────────────────────────────────
app.use((err, req, res, _next) => {
  metrics.errors++;
  console.error(`[${req.id}] Unhandled error:`, err);
  res.status(500).json({
    error:     'Something went wrong.',
    requestId: req.id,
    timestamp: new Date().toISOString()
  });
});

// ── Start Server ───────────────────────────────────────────────────────────
const server = app.listen(port, () => {
  console.log(`🥕 Carrot Backend running on port ${port}`);
  console.log(`Environment: ${process.env.NODE_ENV}`);
  console.log(`Chat model:  ${process.env.GEMINI_CHAT_MODEL}`);
  console.log(`TTS model:   ${process.env.GEMINI_TTS_MODEL}`);
  console.log(`Timeout:     ${process.env.REQUEST_TIMEOUT_MS}ms`);
  console.log(`STT Timeout: ${process.env.STT_TIMEOUT_MS}ms`);
  console.log(`Session TTL: ${process.env.SESSION_TTL_SEC}s`);
  console.log(`Max history: ${process.env.MAX_HISTORY_TURNS} turns`);
});

// ── Graceful Shutdown ──────────────────────────────────────────────────────
function gracefulShutdown(signal) {
  console.log(`Received ${signal} — starting graceful shutdown...`);
  server.close(() => {
    console.log('HTTP server closed');
    textCache.close();
    sessionStore.close();
    console.log('Cache and session store closed');
    const runtime = Date.now() - metrics.startTime;
    console.log('Final metrics:', {
      ...metrics,
      runtime:   `${Math.floor(runtime / 3600000)}h ${Math.floor((runtime % 3600000) / 60000)}m`,
      timestamp: new Date().toISOString()
    });
    process.exit(0);
  });
  setTimeout(() => {
    console.error('Force shutdown — connections did not close in time');
    process.exit(1);
  }, 10000);
}

process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
process.on('SIGINT',  () => gracefulShutdown('SIGINT'));

process.on('uncaughtException', (error) => {
  console.error('Uncaught Exception:', error);
  metrics.errors++;
  gracefulShutdown('UNCAUGHT_EXCEPTION');
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
  metrics.errors++;
});

module.exports = app;
