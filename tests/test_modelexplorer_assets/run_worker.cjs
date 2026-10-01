// Headless oracle: run Google Model Explorer's real pinned dist/worker.js
// graph processor under Node and report per-graph processing stats.
//
// Usage: node run_worker.cjs <worker.js path> <graph-or-collection JSON path>
//        [--keep-single-child] (sets keepLayersWithASingleChild=true)
//
// The worker is a browser Web Worker; this runner supplies the minimal
// worker-global surface it touches (message events, postMessage, an
// OffscreenCanvas 2D context stub for label measurement) inside a `vm`
// sandbox, then feeds each graph a PROCESS_GRAPH message exactly like the
// app does. Output: a JSON array with one row per graph:
//   { collection, graphId, ms, err, stats: { totalNodes, opNodes,
//     groupNodes, totalIncomingEdges, maxNamespaceDepth, layoutGraphError,
//     rootChildren, identicalGroupNodes, identicalGroupCount, groupNs } }
//
// NO SILENT NODE LOSS is asserted by callers as
//   stats.opNodes === declared op-node count and
//   stats.totalIncomingEdges === declared edge count.

'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const workerPath = process.argv[2];
const graphJsonPath = process.argv[3];
const keepSingleChild = process.argv.includes('--keep-single-child');

class Ctx2D {
  constructor() {
    this.font = '';
    this.textBaseline = '';
    this.fillStyle = '';
  }
  measureText(t) {
    return {
      width: String(t).length * 6.2,
      actualBoundingBoxAscent: 8,
      actualBoundingBoxDescent: 2,
    };
  }
  fillText() {}
  clearRect() {}
  fillRect() {}
  save() {}
  restore() {}
  scale() {}
  translate() {}
  getImageData() {
    return { data: new Uint8ClampedArray(4) };
  }
  drawImage() {}
  beginPath() {}
  closePath() {}
  stroke() {}
  fill() {}
}
class OffscreenCanvasShim {
  constructor(w, h) {
    this.width = w;
    this.height = h;
  }
  getContext() {
    return new Ctx2D();
  }
}

const handlers = {};
const outbox = [];
const sandbox = {
  console,
  addEventListener: (t, fn) => {
    handlers[t] = fn;
  },
  postMessage: (m) => {
    outbox.push(m);
  },
  setTimeout,
  clearTimeout,
  setInterval,
  clearInterval,
  TextDecoder,
  TextEncoder,
  URL,
  performance,
  navigator: { userAgent: 'node' },
  document: {
    createElement: () => ({ style: {}, getContext: () => new Ctx2D() }),
    body: {},
    documentElement: { matches: () => false },
    fonts: { load: async () => {}, ready: Promise.resolve() },
  },
  OffscreenCanvas: OffscreenCanvasShim,
  ImageData: class {
    constructor() {
      this.data = new Uint8ClampedArray(4);
    }
  },
  Image: class {},
  createImageBitmap: async () => ({}),
  require,
};
sandbox.self = sandbox;
sandbox.globalThis = sandbox;
sandbox.window = sandbox;
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(workerPath, 'utf8'), sandbox, {
  filename: path.basename(workerPath),
});

if (!handlers.message) {
  console.error('NO_MESSAGE_HANDLER');
  process.exit(2);
}

const payload = JSON.parse(fs.readFileSync(graphJsonPath, 'utf8'));
const collections = payload.graphCollections || [
  { label: payload.label, graphs: payload.graphs },
];
const results = [];
for (const coll of collections) {
  for (const graph of coll.graphs) {
    const t0 = Date.now();
    let err = null;
    outbox.length = 0;
    try {
      handlers.message({
        data: {
          eventType: 0,
          paneId: 'p0',
          graph,
          showOnNodeItemTypes: {},
          nodeDataProviderRuns: {},
          config: {},
          groupNodeChildrenCountThreshold: 1000,
          flattenLayers: false,
          keepLayersWithASingleChild: keepSingleChild,
          initialLayout: true,
        },
      });
    } catch (e) {
      err = String((e && e.stack) || e);
    }
    const ms = Date.now() - t0;
    const resp = outbox.find((m) => m && m.eventType === 1);
    let stats = null;
    if (resp && resp.modelGraph) {
      const mg = resp.modelGraph;
      const nodes = Object.values(mg.nodesById || {});
      const depth = (ns) => (ns ? ns.split('/').length : 0);
      let maxDepth = 0;
      for (const n of nodes) maxDepth = Math.max(maxDepth, depth(n.namespace));
      stats = {
        totalNodes: nodes.length,
        totalIncomingEdges: nodes.reduce(
          (a, n) => a + (n.incomingEdges || []).length,
          0,
        ),
        groupNs: nodes
          .filter((n) => n.nsChildrenIds !== undefined)
          .map((n) => (n.namespace ? n.namespace + '/' : '') + (n.label || '')),
        opNodes: nodes.filter((n) => n.nsChildrenIds === undefined).length,
        groupNodes: nodes.filter((n) => n.nsChildrenIds !== undefined).length,
        maxNamespaceDepth: maxDepth,
        layoutGraphError: mg.layoutGraphError || null,
        rootChildren: (mg.rootNodes || []).length,
        identicalGroupNodes: nodes.filter(
          (n) => n.identicalGroupIndex !== undefined,
        ).length,
        identicalGroupCount: new Set(
          nodes
            .filter((n) => n.identicalGroupIndex !== undefined)
            .map((n) => n.identicalGroupIndex),
        ).size,
      };
    }
    results.push({
      collection: coll.label,
      graphId: graph.id,
      ms,
      err,
      stats,
      respTypes: outbox.map((m) => m && m.eventType),
    });
  }
}
console.log(JSON.stringify(results));
