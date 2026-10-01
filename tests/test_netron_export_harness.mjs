// Netron vendor-execution harness (lane F14, netron memo D-08 / tier T3).
//
// Drives the INSTALLED netron wheel's real onnx.js ModelFactory over a
// TorchLens export -- no DOM, no transcription: what this prints is what
// netron would draw. Invoked by tests/test_netron_export_vendor.py as
//   node test_netron_export_harness.mjs <netron-js-dir> <artifact.json>
// and emits ONE JSON object on stdout for the python side to assert on.
import * as fs from 'node:fs';
import * as path from 'node:path';

const NETRON = path.resolve(process.argv[2]);
const artifact = process.argv[3];
const base = await import(path.join(NETRON, 'base.js'));
const { ModelFactory } = await import(path.join(NETRON, 'onnx.js'));

// Minimal netron view.Context stand-in: enough for ModelFactory.match/open.
class Ctx {
    constructor(obj, identifier) {
        this.obj = obj;
        this.identifier = identifier;
        this._value = null;
        this.type = null;
        this.stream = new base.BinaryStream(new TextEncoder().encode(JSON.stringify(obj)));
    }
    async tags() { return new Map(); }
    async peek(kind) { return kind === 'json' ? this.obj : null; }
    async read(kind) { if (kind === 'json') return this.obj; throw new Error('read ' + kind); }
    async require(id) { return await import(path.join(NETRON, id.replace('./', '') + '.js')); }
    async asset(name) { return fs.readFileSync(path.join(NETRON, name), 'utf-8'); }
    set(name, value) { this.type = name; this._value = value; return this; }
    get value() { return this._value; }
}

// Transcribes netron view.js edge labelling: dims joined with U+00D7, >16 chars suppressed.
function edgeLabel(type) {
    if (type && type.shape && type.shape.dimensions && type.shape.dimensions.length > 0 &&
        type.shape.dimensions.every((d) => !d || Number.isInteger(d) || typeof d === 'bigint' || typeof d === 'string')) {
        const joined = type.shape.dimensions
            .map((d) => (d !== null && d !== undefined && d !== -1 && d !== -1n) ? d : '?')
            .join('×');
        return joined.length > 16 ? '' : joined;
    }
    return '';
}

const obj = JSON.parse(fs.readFileSync(artifact, 'utf-8'));
const context = new Ctx(obj, path.basename(artifact));
const factory = new ModelFactory();
const matched = await factory.match(context);
const result = { matched: Boolean(matched), reader: context.type || null };
if (!matched) {
    console.log(JSON.stringify(result));
    process.exit(0);
}
const model = await factory.open(context);
const graph = model.modules[0];
result.format = model.format;
result.producer = model.producer;
result.rootNodeCount = graph.nodes.length;
result.rootInputCount = graph.inputs.length;
result.rootOutputCount = graph.outputs.length;
result.domains = [...new Set(graph.nodes.map((n) => (n.type && n.type.module) || ''))];
result.modelMetadata = Object.fromEntries(
    (model.metadata || []).map((m) => [m.name, String(m.value)]));

// Edge labels netron would paint (typed-value coverage, memo D-04 / #71).
let painted = 0, untyped = 0;
const samples = [];
for (const node of graph.nodes) {
    for (const output of node.outputs || []) {
        for (const value of output.value || []) {
            if (!value.type) { untyped++; continue; }
            const label = edgeLabel(value.type);
            if (label) { painted++; if (samples.length < 4) samples.push(label); }
        }
    }
}
result.paintedEdgeLabels = painted;
result.untypedValues = untyped;
result.edgeLabelSamples = samples;

// Function drill-down: list, emptiness, nested resolution, call-graph acyclicity.
const functions = model.functions || [];
result.functionNames = functions.map((f) => f.name);
result.emptyFunctions = functions.filter((f) => !(f.nodes || []).length).map((f) => f.name);
const bodyCalls = new Map(functions.map((f) => [
    f.name,
    (f.nodes || []).map((n) => n.type && n.type.name).filter((name) => result.functionNames.includes(name)),
]));
result.nestedFunctionCalls = [...bodyCalls.values()].flat();
const visiting = new Set(), done = new Set();
let cyclic = false;
const walk = (name) => {
    if (done.has(name)) return;
    if (visiting.has(name)) { cyclic = true; return; }
    visiting.add(name);
    for (const callee of bodyCalls.get(name) || []) walk(callee);
    visiting.delete(name);
    done.add(name);
};
for (const name of result.functionNames) walk(name);
result.functionCallGraphAcyclic = !cyclic;

// Sidebar probe: one lossy-domain node's exact attribute strings + port names.
const probe = graph.nodes.find((n) => (n.attributes || []).length > 0) || graph.nodes[0];
result.probeNode = probe ? {
    name: probe.name,
    type: probe.type ? probe.type.name : null,
    attributes: Object.fromEntries((probe.attributes || []).map((a) => [a.name, String(a.value)])),
    inputPortNames: (probe.inputs || []).map((a) => a.name),
} : null;
console.log(JSON.stringify(result));
