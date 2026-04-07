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
  SESSION_TTL_SEC:     1800,
  MAX_HISTORY_TURNS:   10
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

// RAM cache — text answers only, 5 min TTL
const textCache = new NodeCache({ stdTTL: 300 });

// Session memory store
const sessionStore = new NodeCache({
  stdTTL:      parseInt(process.env.SESSION_TTL_SEC),
  checkperiod: 120
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
  methods:        ['GET', 'POST', 'DELETE'],
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

// 2mb — incoming audio from INMP441 can be up to ~200KB Base64
// Response is text-only so outgoing is tiny
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
      ip:        req.ip
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
function getHistory(sessionId) {
  return sessionStore.get(sessionId) || [];
}

function saveHistory(sessionId, userText, modelText) {
  const maxTurns = parseInt(process.env.MAX_HISTORY_TURNS);
  let   history  = getHistory(sessionId);

  history.push({ role: 'user',  parts: [{ text: userText  }] });
  history.push({ role: 'model', parts: [{ text: modelText }] });

  const maxEntries = maxTurns * 2;
  if (history.length > maxEntries) {
    history = history.slice(history.length - maxEntries);
  }
  sessionStore.set(sessionId, history);
}

// ── Robot Commands ─────────────────────────────────────────────────────────
const ROBOT_COMMANDS = [
  {
    keywords: ['move forward', 'go forward', 'forward', 'move ahead', 'go ahead'],
    command:  { command: 'FORWARD',  speed: 'normal' }
  },
  {
    keywords: ['move back', 'go back', 'backward', 'move backward', 'reverse'],
    command:  { command: 'BACKWARD', speed: 'normal' }
  },
  {
    keywords: ['turn left', 'go left', 'left'],
    command:  { command: 'LEFT',     angle: 90 }
  },
  {
    keywords: ['turn right', 'go right', 'right'],
    command:  { command: 'RIGHT',    angle: 90 }
  },
  {
    keywords: ['stop', 'halt', 'freeze', 'stay'],
    command:  { command: 'STOP',     emergency: false }
  },
  {
    keywords: ['go faster', 'speed up', 'faster'],
    command:  { command: 'SPEED_UP', increment: 10 }
  },
  {
    keywords: ['go slower', 'slow down', 'slower'],
    command:  { command: 'SLOW_DOWN', decrement: 10 }
  }
];

// ── System Prompt — optimized for ESP32 OLED display ──────────────────────
// Rules:
//  1. NO audio data returned ever
//  2. Text only — max 2-3 short sentences
//  3. Optimized for 128x64 OLED (21 chars per line, 4 lines visible)
//  4. Remember conversation context
const SYSTEM_PROMPT = `You are Carrot, a friendly robot assistant running on an ESP32 microcontroller.

STRICT RULES — you must follow these every single response:
1. Reply in plain text only. No markdown, no bullet points, no symbols.
2. Maximum 2 sentences. Never more.
3. Each sentence must be under 20 words.
4. If you remember something from earlier in this conversation, use it.
5. Be cheerful and warm but extremely concise.
6. Never ask follow-up questions.

Your responses will be shown on a tiny 128x64 OLED screen. Short = better.`;

// ── POST /ask ──────────────────────────────────────────────────────────────
// Accepts:
//   { audioData: "<base64>", mimeType: "audio/wav", sessionId: "abc" }
//   { question:  "hey carrot what is AI",           sessionId: "abc" }
//
// Returns TEXT ONLY — no audio in response (ESP32 RAM limitation)
app.post('/ask', async (req, res) => {
  try {
    const {
      question,
      audioData: incomingAudio,
      mimeType:  incomingMime,
      sessionId: clientSessionId
    } = req.body;

    const sessionId = clientSessionId || uuidv4();

    // ── STEP 0: "The Ear" — STT (audio input only) ────────────────────────
    let transcribedText = '';

    if (incomingAudio) {
      console.log(`[${req.id}] Audio received — running STT...`);

      const sttModel  = genAI.getGenerativeModel({ model: process.env.GEMINI_CHAT_MODEL });
      const sttResult = await withTimeout(
        sttModel.generateContent([
          "Transcribe exactly what the user says. Reply ONLY with the transcribed text. Nothing else.",
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
      console.log(`[${req.id}] STT: "${transcribedText}"`);

      // STT failed — return short OLED-safe message, no audio
      if (!transcribedText) {
        return res.json({
          answer:    "Didn't catch that. Try again!",
          type:      'stt_failure',
          sessionId,
          requestId: req.id
        });
      }

    } else if (question && typeof question === 'string' && question.trim().length > 0) {
      transcribedText = question.trim();

    } else {
      throw new CarrotError(
        "Send 'audioData' (Base64 WAV) or 'question' (string)",
        'VALIDATION_ERROR',
        400
      );
    }

    // Normalise
    const cleanQ = transcribedText.toLowerCase().replace(/[.,!?]/g, '').trim();
    console.log(`[${req.id}] [session:${sessionId}] Input: "${cleanQ}"`);

    // ── ROUTE 1: ROBOT COMMAND (no AI needed) ─────────────────────────────
    if (cleanQ.startsWith('carrot') && !cleanQ.startsWith('hey carrot')) {
      console.log(`[${req.id}] Route: ROBOT COMMAND`);

      const isolatedCommand = cleanQ.replace(/^carrot\s*/, '').trim();
      const matched = ROBOT_COMMANDS.find(entry =>
        entry.keywords.some(k => isolatedCommand.includes(k))
      );

      if (matched) {
        console.log(`[${req.id}] Command: ${matched.command.command}`);
        return res.json({
          answer:    `OK! ${matched.command.command}`,  // short OLED text
          command:   matched.command,
          type:      'robot_command',
          sessionId,
          requestId: req.id
        });
      }
      console.log(`[${req.id}] No command matched — routing to AI`);
    }

    // ── ROUTE 2: AI BRAIN (text answer only, no TTS) ──────────────────────
    console.log(`[${req.id}] Route: AI BRAIN`);

    const actualQuestion = cleanQ
      .replace(/^hey carrot\s*/, '')
      .replace(/^carrot\s*/, '')
      .trim() || cleanQ;

    // Load session history
    const history        = getHistory(sessionId);
    const shouldUseCache = history.length === 0;
    const cacheKey       = `q:${actualQuestion}`;
    let   answerText;

    // Check RAM cache first (no tokens, 5 min)
    if (shouldUseCache) {
      const cached = textCache.get(cacheKey);
      if (cached) {
        console.log(`[${req.id}] Cache hit`);
        metrics.cacheHits++;
        answerText = cached;
      }
    }

    // Call Gemini if not cached
    if (!answerText) {
      const chatModel = genAI.getGenerativeModel({ model: process.env.GEMINI_CHAT_MODEL });

      const contents = [
        // System prompt — first exchange sets personality + rules
        { role: 'user',  parts: [{ text: SYSTEM_PROMPT }] },
        { role: 'model', parts: [{ text: "Understood. I am Carrot. Short text replies only. Ready!" }] },
        // Session history (remembers past exchanges)
        ...history,
        // Current question
        { role: 'user',  parts: [{ text: actualQuestion }] }
      ];

      const chatResult = await withTimeout(
        chatModel.generateContent({ contents })
      );

      answerText = chatResult.response.text()?.trim();

      // Fallback if Gemini returns empty
      if (!answerText) {
        answerText = "Hmm, I'm not sure. Ask me again!";
      }

      // Hard truncate — safety net for OLED
      // 4 lines x 21 chars = 84 chars max comfortable display
      // Allow slightly more for scrolling displays
      if (answerText.length > 120) {
        answerText = answerText.substring(0, 117) + '...';
      }

      // Cache only fresh sessions (no personal context)
      if (shouldUseCache) {
        textCache.set(cacheKey, answerText);
      }
    }

    // Save to session memory
    saveHistory(sessionId, actualQuestion, answerText);
    console.log(`[${req.id}] Answer: "${answerText}"`);

    // ── RESPONSE — text only, no audio field ──────────────────────────────
    return res.json({
      answer:    answerText,   // Plain text — display on OLED
      type:      'ai_response',
      sessionId,               // ESP32 saves this and sends it next time
      requestId: req.id
      // NOTE: no 'audio' field — intentionally removed to save ESP32 RAM
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

// ── POST /session/clear ────────────────────────────────────────────────────
app.post('/session/clear', (req, res) => {
  const { sessionId } = req.body;
  if (!sessionId) {
    return res.status(400).json({ error: "Missing 'sessionId'." });
  }
  sessionStore.del(sessionId);
  console.log(`[session:${sessionId}] Cleared`);
  res.json({
    message:   'Carrot has forgotten everything!',
    sessionId,
    timestamp: new Date().toISOString()
  });
});

// ── Web Control Command Store ──────────────────────────────────────────────
let pendingWebCommand = null;

// POST /control — web dashboard sends a movement command
app.post('/control', (req, res) => {
  const { command } = req.body;
  const valid = ['FORWARD', 'BACKWARD', 'LEFT', 'RIGHT', 'STOP', 'SPEED_UP', 'SLOW_DOWN'];

  if (!command || !valid.includes(command.toUpperCase())) {
    return res.status(400).json({ error: 'Invalid command.' });
  }

  pendingWebCommand = command.toUpperCase();
  console.log(`[WEB CTRL] Queued: ${pendingWebCommand}`);
  res.json({
    message:   `Command '${pendingWebCommand}' queued for ESP32`,
    command:   pendingWebCommand,
    timestamp: new Date().toISOString()
  });
});

// GET /control — ESP32 polls this every 2 seconds
// One-shot: command is cleared after ESP32 reads it
app.get('/control', (req, res) => {
  const cmd = pendingWebCommand;
  pendingWebCommand = null;
  res.json({
    command:   cmd,   // null = no pending command
    timestamp: new Date().toISOString()
  });
});

// ── GET / ──────────────────────────────────────────────────────────────────
app.get('/', (_req, res) => {
  res.send('🥕 Carrot AI Backend is Running — Text Only Mode');
});

// ── GET /robot ─────────────────────────────────────────────────────────────
app.get('/robot', (_req, res) => {
  res.json({
    name:    'Carrot',
    version: '1.0',
    status:  'online',
    mode:    'text-only (audio-free for ESP32 RAM)',
    wake_words: {
      ai_mode:    'hey carrot',
      local_mode: 'carrot'
    },
    memory: {
      session_ttl_minutes: parseInt(process.env.SESSION_TTL_SEC) / 60,
      max_history_turns:   parseInt(process.env.MAX_HISTORY_TURNS),
      active_sessions:     sessionStore.keys().length
    },
    commands:  ROBOT_COMMANDS.map(e => ({ command: e.command.command, keywords: e.keywords })),
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
    mode:         'text-only',
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
  console.log(`Mode:        TEXT ONLY — no audio responses`);
  console.log(`Environment: ${process.env.NODE_ENV}`);
  console.log(`Chat model:  ${process.env.GEMINI_CHAT_MODEL}`);
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
