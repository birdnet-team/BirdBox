// SPDX-FileCopyrightText: 2026 BirdNET-Team
// SPDX-License-Identifier: MIT
/**
 * Minimal ONNX protobuf surgery for the demo. Weights and nodes stay untouched.
 *
 * addGraphOutput declares a tensor the graph already computes as an additional
 * graph output. Tests use it for raw_all, the per-window boxes that feed the
 * graph's window fusion (see onnx_export.py).
 *
 * splitSegmentGraph cuts the graph's segment Loop in two models, so the demo
 * can show boxes every few seconds while detection runs (chunked-graph.js).
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
 * Cut the graph's segment Loop in two, so a segment's clips can go through the
 * network a few at a time (see chunked-graph.js):
 *
 *   features(audio, audio_in, iter) -> trip_count, clips and clip starts of
 *       segment `iter`, plus the tensors the second part reads. All of the
 *       spectrogram work, including the noise floor or PCEN of the segment.
 *   detect(clips, clip starts, conf, ...) -> raw, the per-window boxes of the
 *       given clips: rendering, network, NMS, pixels to seconds and Hz.
 *
 * The body picks its clips with two Gathers that share one Range index
 * (detect_segment_tensor in onnx_export.py), and that is where it is cut.
 * Weights and all other nodes stay as exported.
 *
 * @param {Uint8Array} model  ModelProto bytes
 * @returns {{ features: Uint8Array, detect: (types: Object<string, string>) => Uint8Array,
 *             takes: {from: string, to: string}[], shared: string[] }}
 *   `takes` holds each cut (`from` is an output of features, `to` an input of
 *   detect). `shared` holds the other tensors that detect needs from features.
 *   detect(types) builds the second model from the element type of each of them.
 */
export function splitSegmentGraph(model) {
  const graphField = [...fields(model)].find(f => f.field === 7);
  if (!graphField) throw new Error('No graph in the ONNX model');
  const envelope = { before: model.subarray(0, graphField.tagStart), after: model.subarray(graphField.end) };
  const top = parseGraph(model.subarray(graphField.valueStart, graphField.valueEnd));

  const loop = top.nodes.find(n => n.op === 'Loop');
  if (!loop) throw new Error('The ONNX graph has no segment Loop');
  let bodyBytes = null;
  for (const a of fields(loop.buf, loop.start, loop.end)) {
    if (a.field !== 5) continue;
    const g = [...fields(loop.buf, a.valueStart, a.valueEnd)].find(x => x.field === 6);
    if (g) bodyBytes = loop.buf.subarray(g.valueStart, g.valueEnd);
  }
  if (!bodyBytes) throw new Error('The segment Loop has no body');
  const body = parseGraph(bodyBytes);

  const index = body.nodes.find(n => n.op === 'Range')?.outputs[0];
  const gathers = body.nodes.filter(n => n.op === 'Gather' && n.inputs[1] === index);
  const rawNode = body.producers.get('raw');
  if (gathers.length !== 2 || !rawNode) throw new Error('The segment Loop does not have the expected clip selection');
  const takes = gathers.map(n => ({ from: n.inputs[0], to: n.outputs[0] }));

  // Everything up to the cut belongs to the features; constants may be copied to both sides
  const before = ancestors(body, takes.map(t => t.to));
  const computed = new Set([...before.nodes].filter(n => n.op !== 'Constant').flatMap(n => n.outputs));
  const after = ancestors(body, ['raw'], computed);
  const cut = new Set(takes.map(t => t.to));
  const shared = [...after.leaves].filter(t => computed.has(t) && !cut.has(t));
  const unknown = [...after.leaves].filter(t => !computed.has(t) && !body.inits.has(t) && t !== 'conf_in');
  if (unknown.length) throw new Error(`The network part of the segment Loop also reads ${unknown.join(', ')}`);

  const sources = [...takes.map(t => t.from), ...shared];
  const features = ancestors(body, sources);
  const outer = [...features.leaves].filter(t => !body.inits.has(t) && t !== 'audio_in' && t !== 'iter');
  const prelude = ancestors(top, [...outer, 'trip_count']);
  const initsOf = (graph, leaves) => [...leaves].filter(t => graph.inits.has(t)).map(t => graph.inits.get(t));
  const inOrder = (graph, set) => graph.nodes.filter(n => set.has(n));

  return {
    takes,
    shared,
    features: buildModel(
      envelope, 'birdbox_features',
      [...inOrder(top, prelude.nodes), ...inOrder(body, features.nodes)],
      [...initsOf(top, prelude.leaves), ...initsOf(body, features.leaves)],
      [valueInfo('audio', 'float32'), valueInfo('audio_in', 'float32'), valueInfo('iter', 'int64', [])],
      ['trip_count', ...sources].map(name => valueInfo(name, null)),
    ),
    detect: types => buildModel(
      envelope, 'birdbox_detect',
      inOrder(body, after.nodes),
      initsOf(body, after.leaves),
      [...[...takes.map(t => t.to), ...shared].map(name => valueInfo(name, types[name])), valueInfo('conf_in', 'float32')],
      [valueInfo('raw', 'float32')],
    ),
  };
}
