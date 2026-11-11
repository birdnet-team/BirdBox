// SPDX-FileCopyrightText: 2026 BirdNET-Team
// SPDX-License-Identifier: MIT
/**
 * Minimal ONNX protobuf surgery for the demo. Weights and nodes stay untouched.
 *
 * addGraphOutput declares a tensor the graph already computes as an additional
 * graph output. Tests use it for raw_all, the per-window boxes that feed the
 * graph's window fusion (see onnx_export.py).
 *
 * streamGraphs builds the two streaming graphs a BirdBox model stores as ONNX
 * functions, so the demo can show boxes every few seconds while detection runs
 * (chunked-graph.js).
 */

export const FLOAT = 1;   // TensorProto.DataType

function readVarint(buf, pos) {
  let result = 0, scale = 1;
  for (;;) {
    const b = buf[pos++];
    result += (b & 0x7f) * scale;
    if (!(b & 0x80)) return [result, pos];
    scale *= 128;
  }
}

function varint(value) {
  const out = [];
  do {
    let b = value % 128;
    value = Math.floor(value / 128);
    if (value) b |= 0x80;
    out.push(b);
  } while (value);
  return Uint8Array.from(out);
}

function concat(parts) {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}

const utf8 = new TextEncoder(), text = new TextDecoder();
const lenField = (field, payload) => concat([varint((field << 3) | 2), varint(payload.length), payload]);
const varintField = (field, value) => concat([varint(field << 3), varint(value)]);
const str = (buf, f) => text.decode(buf.subarray(f.valueStart, f.valueEnd));

function* fields(buf, start = 0, end = buf.length) {
  let pos = start;
  while (pos < end) {
    const tagStart = pos;
    let tag;
    [tag, pos] = readVarint(buf, pos);
    const field = Math.floor(tag / 8), wire = tag & 7;
    let valueStart = pos;
    if (wire === 0) [, pos] = readVarint(buf, pos);
    else if (wire === 1) pos += 8;
    else if (wire === 5) pos += 4;
    else if (wire === 2) {
      let len;
      [len, pos] = readVarint(buf, pos);
      valueStart = pos;
      pos += len;
    } else throw new Error(`Unsupported protobuf wire type ${wire}`);
    yield { field, tagStart, valueStart, valueEnd: pos, end: pos };
  }
}

/**
 * Model bytes with `name` added to the graph outputs (unchanged if it is one
 * already). dims entries are symbolic names (string) or fixed sizes (number).
 * @param {Uint8Array} model  ModelProto bytes
 */
export function addGraphOutput(model, name, elemType, dims) {
  const graphField = [...fields(model)].find(f => f.field === 7);
  if (!graphField) throw new Error('No graph in the ONNX model');
  const graph = model.subarray(graphField.valueStart, graphField.valueEnd);
  const produced = new Set(), outputs = [];
  for (const f of fields(graph)) {
    if (f.field === 1) {   // NodeProto, field 2 = output
      for (const g of fields(graph, f.valueStart, f.valueEnd)) if (g.field === 2) produced.add(str(graph, g));
    } else if (f.field === 12) {   // ValueInfoProto, field 1 = name
      for (const g of fields(graph, f.valueStart, f.valueEnd)) if (g.field === 1) outputs.push(str(graph, g));
    }
  }
  if (outputs.includes(name)) return model;
  if (!produced.has(name)) throw new Error(`The ONNX graph has no tensor named ${name}`);

  // ValueInfoProto { name, type: TypeProto { tensor_type { elem_type, shape { dim... } } } }
  const shape = concat(dims.map(d => lenField(1, typeof d === 'string' ? lenField(2, utf8.encode(d)) : varintField(1, d))));
  const tensorType = concat([varintField(1, elemType), lenField(2, shape)]);
  const valueInfo = concat([lenField(1, utf8.encode(name)), lenField(2, lenField(1, tensorType))]);
  const tail = lenField(12, valueInfo);
  // The graph grows by `tail`, so only its length prefix changes
  return concat([
    model.subarray(0, graphField.tagStart),
    varint((7 << 3) | 2), varint(graph.length + tail.length), graph, tail,
    model.subarray(graphField.end),
  ]);
}

const ELEM_TYPES = { float32: 1, uint8: 2, int8: 3, int32: 6, int64: 7, bool: 9, float64: 11 };

function parseNode(buf, f) {
  const node = { inputs: [], outputs: [], op: '', record: buf.subarray(f.tagStart, f.end), start: f.valueStart, end: f.valueEnd, buf };
  for (const g of fields(buf, f.valueStart, f.valueEnd)) {
    if (g.field === 1) node.inputs.push(str(buf, g));
    else if (g.field === 2) node.outputs.push(str(buf, g));
    else if (g.field === 4) node.op = str(buf, g);
  }
  return node;
}

/** Nodes and initializers of a GraphProto. */
function parseGraph(buf) {
  const nodes = [], inits = new Map();
  for (const f of fields(buf)) {
    if (f.field === 1) nodes.push(parseNode(buf, f));
    else if (f.field === 5) {   // TensorProto, field 8 = name
      for (const g of fields(buf, f.valueStart, f.valueEnd)) if (g.field === 8) inits.set(str(buf, g), buf.subarray(f.tagStart, f.end));
    }
  }
  return { nodes, inits, producers: new Map(nodes.flatMap(n => n.outputs.map(o => [o, n]))) };
}

/** The nodes `roots` depend on, and the tensors they read but nothing here produces. */
function ancestors(graph, roots, stop = new Set()) {
  const nodes = new Set(), leaves = new Set(), seen = new Set(), todo = [...roots];
  while (todo.length) {
    const t = todo.pop();
    if (!t || seen.has(t)) continue;
    seen.add(t);
    const node = stop.has(t) ? null : graph.producers.get(t);
    if (!node) { leaves.add(t); continue; }
    nodes.add(node);
    todo.push(...node.inputs);
  }
  return { nodes, leaves };
}

/** ValueInfoProto; a null type leaves it to ONNX Runtime's inference, null dims leave the shape open. */
function valueInfo(name, type, dims = null) {
  if (!type) return lenField(1, utf8.encode(name));
  const shape = dims ? [lenField(2, concat(dims.map(d => lenField(1, varintField(1, d)))))] : [];
  const tensorType = concat([varintField(1, ELEM_TYPES[type]), ...shape]);
  return concat([lenField(1, utf8.encode(name)), lenField(2, lenField(1, tensorType))]);
}

/**
 * The model's metadata_props (ModelProto field 14) as an object.
 * @param {Uint8Array} model  ModelProto bytes
 * @returns {Object<string, string>}
 */
export function modelMetadata(model) {
  const out = {};
  for (const f of fields(model)) {
    if (f.field !== 14) continue;
    let key = null, value = '';
    for (const g of fields(model, f.valueStart, f.valueEnd)) {
      if (g.field === 1) key = str(model, g);
      else if (g.field === 2) value = str(model, g);
    }
    if (key != null) out[key] = value;
  }
  return out;
}

/** A ModelProto around new nodes, initializers, inputs and outputs. */
function buildModel(envelope, name, nodes, inits, inputs, outputs) {
  const graph = concat([
    ...nodes.map(n => n.record),
    lenField(2, utf8.encode(name)),
    ...inits,
    ...inputs.map(v => lenField(11, v)),
    ...outputs.map(v => lenField(12, v)),
  ]);
  return concat([envelope.before, lenField(7, graph), envelope.after]);
}

/**
 * The streaming graphs of a BirdBox model (onnx_export.py, StreamSpectrogram and
 * StreamDetect), as two runnable models. The export stores them as ONNX functions
 * (domain birdbox.stream) whose extra inputs are tensors of the model: weights of the
 * segment Loop body, and the DFT kernel the parent graph builds. They become
 * initializers or prelude nodes here, so the weights are stored once.
 *
 *   spectrogram(audio) -> db (n_mels, frames), sound (frames)
 *   detect(db, sound, first_frame, times, conf) -> raw
 *
 * @param {Uint8Array} model  ModelProto bytes
 * @returns {{ spectrogram: Uint8Array, detect: Uint8Array } | null}  null without the functions
 */
export function streamGraphs(model) {
  const functions = new Map();
  for (const f of fields(model)) {
    if (f.field !== 25) continue;   // FunctionProto: name 1, input 4, output 5, node 7, domain 10
    const fn = { name: '', domain: '', inputs: [], outputs: [], nodes: [] };
    for (const g of fields(model, f.valueStart, f.valueEnd)) {
      if (g.field === 1) fn.name = str(model, g);
      else if (g.field === 10) fn.domain = str(model, g);
      else if (g.field === 4) fn.inputs.push(str(model, g));
      else if (g.field === 5) fn.outputs.push(str(model, g));
      else if (g.field === 7) {
        const node = parseNode(model, g);
        node.record = lenField(1, model.subarray(g.valueStart, g.valueEnd));   // as a GraphProto node
        fn.nodes.push(node);
      }
    }
    if (fn.domain === 'birdbox.stream') functions.set(fn.name, fn);
  }
  const spec = functions.get('stream_spectrogram'), det = functions.get('stream_detect');
  if (!spec || !det) return null;

  const graphField = [...fields(model)].find(f => f.field === 7);
  const envelope = { before: model.subarray(0, graphField.tagStart), after: model.subarray(graphField.end) };
  const top = parseGraph(model.subarray(graphField.valueStart, graphField.valueEnd));
  const loop = top.nodes.find(n => n.op === 'Loop');
  let body = { inits: new Map() };
  for (const a of loop ? fields(loop.buf, loop.start, loop.end) : []) {
    if (a.field !== 5) continue;
    const g = [...fields(loop.buf, a.valueStart, a.valueEnd)].find(x => x.field === 6);
    if (g) body = parseGraph(loop.buf.subarray(g.valueStart, g.valueEnd));
  }
  const build = (fn, types) => {
    const weights = fn.inputs.filter(t => body.inits.has(t));
    const outer = fn.inputs.filter(t => !body.inits.has(t) && (top.producers.has(t) || top.inits.has(t)));
    const prelude = ancestors(top, outer);
    const inputs = fn.inputs.filter(t => !weights.includes(t) && !outer.includes(t));
    const missing = inputs.filter(t => !types[t]);
    if (missing.length) throw new Error(`${fn.name} reads ${missing.join(', ')}, which the model does not have`);
    return buildModel(
      envelope, fn.name,
      [...top.nodes.filter(n => prelude.nodes.has(n)), ...fn.nodes],
      [...weights.map(t => body.inits.get(t)), ...[...prelude.leaves].filter(t => top.inits.has(t)).map(t => top.inits.get(t))],
      inputs.map(t => valueInfo(t, types[t][0], types[t][1])),
      fn.outputs.map(t => valueInfo(t, null)),
    );
  };
  return {
    spectrogram: build(spec, { audio: ['float32', null] }),
    detect: build(det, { db: ['float32', null], sound: ['bool', null], first_frame: ['int64', []], times: ['float32', null], conf: ['float32', null] }),
  };
}

