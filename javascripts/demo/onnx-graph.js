// SPDX-FileCopyrightText: 2026 BirdNET-Team
// SPDX-License-Identifier: MIT
/**
 * Minimal ONNX protobuf surgery for the demo: declare a tensor the graph
 * already computes as an additional graph output, so ONNX Runtime can return
 * it. Weights and nodes stay untouched.
 *
 * The demo uses it for raw_all, the per-window boxes that feed the graph's
 * window fusion (see onnx_export.py). Requesting only that output lets ONNX
 * Runtime skip the fusion, which fusion.js then repeats for any slider value.
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
