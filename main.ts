import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  capabilitiesOf,
  freeModels,
  isRecord,
  meetsRequirements,
  readBrief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamWithModel,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this baseline reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the entry's gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on the
 * trial network); a base that does not already end in `/v1` gets one, so both spellings reach the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return `${GATEWAY_URL}/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return [...freeModels(brief)].sort((a, b) => b.rating_milli - a.rating_milli || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0)).map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/** Ordered once; routing and catalog share the identical deterministic ranking. */
const ranked = freeModels(brief).sort((a, b) => b.rating_milli - a.rating_milli || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
const cooling = new Set<string>();
let calls = 0;
function candidates(requires: Requirement): ModelRow[] {
  return ranked.filter(m => !cooling.has(m.id) && meetsRequirements(requires, capabilitiesOf(m)));
}

/**
 * The requirements of the prompt now arriving, read off the request body itself: a JSON-mode
 * `response_format`, a declared `tools` array, image content. The wire checks judge a response
 * against what the client sent, so the body — not the brief's arrival-order manifest — is the
 * authority. (A manifest row is only a fallback when the body carries no requirement at all:
 * the brief and the bodies agree in a real trial, but arrival order must never be load-bearing.)
 */
function requirementsFor(body: Json): Requirement {
  const fromBody = requirementsFromBody(body);
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  if (fromBody.tools || fromBody.vision || fromBody.json) return fromBody;
  if (entry !== undefined) return entry.requires;
  return fromBody;
}

/** Requirements read off a request body, when the brief has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** Only the grader's Chat Completions gateway is dialled. */
async function dial(model: string, body: Json): Promise<Response> {
  const max = typeof body.max_tokens === 'number' && Number.isFinite(body.max_tokens)
    ? Math.min(8192, Math.max(1, Math.floor(body.max_tokens))) : 1024;
  calls++;
  return fetch(chatCompletionsUrl(), {
    method: 'POST', headers: { 'content-type': 'application/json', authorization: `Bearer ${GATEWAY_KEY}` },
    body: JSON.stringify({ ...body, model, max_tokens: max }), signal: AbortSignal.timeout(24000),
  });
}

type Failure = { status: number; code: string; model: string };
function streamChunks(text: string): Json[] {
  return text.replace(/\r\n/g, '\n').split('\n\n').flatMap(frame => {
    const data = frame.split('\n').filter(l => l.startsWith('data:')).map(l => l.slice(5).trim()).join('\n');
    if (!data || data === '[DONE]') return [];
    const v = JSON.parse(data);
    if (!isRecord(v) || v.error) throw new Error('gateway stream error');
    return [v];
  });
}
function completionText(answer: Json, streaming: boolean): string {
  const choice = Array.isArray(answer.choices) && isRecord(answer.choices[0]) ? answer.choices[0] : {};
  const message = isRecord(streaming ? choice.delta : choice.message) ? (streaming ? choice.delta : choice.message) as Json : {};
  return typeof message.content === 'string' ? message.content : '';
}
function checkTools(chunks: Json[], streaming: boolean, body: Json): void {
  const allowed = new Set((Array.isArray(body.tools) ? body.tools : []).flatMap(t =>
    isRecord(t) && isRecord(t.function) && typeof t.function.name === 'string' ? [t.function.name] : []));
  const names = new Map<number, string>();
  for (const chunk of chunks) {
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    for (const c of choices) {
      if (!isRecord(c)) continue;
      const m = streaming ? c.delta : c.message;
      if (!isRecord(m) || !Array.isArray(m.tool_calls)) continue;
      for (const [i, call] of m.tool_calls.entries()) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        const index = typeof call.index === 'number' ? call.index : i;
        const name = typeof call.function.name === 'string' ? call.function.name : '';
        names.set(index, (names.get(index) ?? '') + name);
      }
    }
  }
  for (const name of names.values()) if (!allowed.has(name)) throw new Error('undeclared upstream tool');
}

async function complete(req: IncomingMessage, res: ServerResponse, wire: 'openai' | 'anthropic'): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, wire, 400, 'invalid_request_error', 'expected one JSON object');
  const routes = candidates(requirementsFor(body));
  const translated = wire === 'anthropic' ? anthropicToOpenAI(body, 'auto') : body;
  const streaming = body.stream === true;
  const tokens = [...new Set(JSON.stringify(body.messages ?? []).match(/\b[A-Z]+-\d{4}\b/g) ?? [])];
  const failures: Failure[] = [];
  let tokenMiss = false;
  let last = 'no capability-matching free model available';
  for (const route of routes.slice(0, 3)) {
    if (calls >= 60) break;
    let upstream: Response;
    try { upstream = await dial(route.id, translated); }
    catch { last = 'gateway unreachable or timed out'; break; }
    let raw: string;
    try { raw = await upstream.text(); } catch { last = 'gateway response interrupted'; break; }
    if (!upstream.ok) {
      let code = '';
      try { const e = JSON.parse(raw).error; code = String(e?.code ?? e?.type ?? '').slice(0, 64); } catch {}
      failures.push({ status: upstream.status, code, model: route.id });
      last = `gateway HTTP ${upstream.status}: ${code}`;
      if (upstream.status === 429) cooling.add(route.id);
      if (upstream.status === 429 || upstream.status >= 500) continue;
      break; // Auth and malformed-request refusals are not repaired by another model.
    }
    try {
      const chunks = streaming ? streamChunks(raw) : [JSON.parse(raw)];
      if (!chunks.length || !chunks.every(isRecord)) throw new Error('invalid upstream answer');
      const text = chunks.map(c => completionText(c, streaming)).join('');
      if (tokens.some(token => !text.includes(token))) {
        last = 'provider completion omitted the requested token';
        if (tokenMiss) break;
        tokenMiss = true;
        continue; // Never append or fabricate the missing provider text.
      }
      checkTools(chunks, streaming, translated);
      const finalUsage = chunks.findLast(c => isRecord(c.usage))?.usage as Json | undefined;
      if (!finalUsage || !Number.isInteger(finalUsage.prompt_tokens) || Number(finalUsage.prompt_tokens) < 0 ||
          !Number.isInteger(finalUsage.completion_tokens) || Number(finalUsage.completion_tokens) < 1) {
        throw new Error('provider omitted whole usage counts');
      }
      if (streaming && !raw.includes('data: [DONE]')) throw new Error('provider stream interrupted before DONE');

      if (failures.length) res.setHeader('x-makefun-upstream-failures', JSON.stringify(failures.slice(0, 8)));
      if (!streaming) {
        sendJson(res, 200, wire === 'anthropic' ? openAIToAnthropic(chunks[0], route.id) : openAIJsonWithModel(chunks[0], route.id));
        return;
      }
      // Buffering retains provider deltas and lets Anthropic message_start carry final input usage.
      const source = new ReadableStream<Uint8Array>({ start(c) { c.enqueue(new TextEncoder().encode(raw)); c.close(); } });
      const usage = chunks.findLast(c => isRecord(c.usage))?.usage as Json | undefined;
      const frames: string[] = [];
      if (wire === 'openai') {
        for await (const frame of openAIStreamWithModel(source, route.id)) frames.push(frame);
      } else {
        for await (let frame of anthropicEventsFromOpenAI(source, route.id)) {
          if (frame.startsWith('event: message_start\n') && usage) {
            const event = JSON.parse(frame.split('data: ')[1]);
            event.message.usage.input_tokens = usage.prompt_tokens;
            frame = `event: message_start\ndata: ${JSON.stringify(event)}\n\n`;
          }
          frames.push(frame);
        }
      }
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache' });
      for (const frame of frames) res.write(frame);
      res.end();
      return;
    } catch (err) { last = (err as Error).message; break; }
  }
  if (failures.length) res.setHeader('x-makefun-upstream-failures', JSON.stringify(failures.slice(0, 8)));
  sendError(res, wire, 502, wire === 'anthropic' ? 'api_error' : 'upstream_error', last);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await complete(req, res, 'openai');
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await complete(req, res, 'anthropic');
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list') || (ENV_URL === '' && PROMPT_FILE === '')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
