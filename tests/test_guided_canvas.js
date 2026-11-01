// 在无浏览器环境中验证画布控制点的缩放与旋转手势。
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname, '..', 'validation_templates', 'guided_validation.html'), 'utf8');
const descriptors = [
  {index: 0, name: 'reference.png', width: 200, height: 120, preview_url: '/preview/0'},
  {index: 1, name: 'moving.png', width: 200, height: 120, preview_url: '/preview/1'},
];
const source = html.match(/<script>([\s\S]*?)<\/script>/)[1]
  .replace('{{ images|tojson }}', JSON.stringify(descriptors));
const listeners = {};
const elements = new Map();
const context2d = new Proxy({}, {get: () => () => {}, set: () => true});

function element(id) {
  if (!elements.has(id)) {
    elements.set(id, {
      value: id === 'opacity' ? '72' : '',
      style: {},
      classList: {add() {}, remove() {}, toggle() {}},
      clientWidth: id === 'workspace' ? 800 : id === 'result-viewer' ? 900 : 0,
      clientHeight: id === 'workspace' ? 600 : id === 'result-viewer' ? 500 : 0,
      getContext: () => context2d,
      getBoundingClientRect: () => ({left: 0, top: 0}),
      setPointerCapture() {},
      addEventListener(name, callback) { if (id === 'canvas') listeners[name] = callback; },
      appendChild() {},
      append() {},
      replaceChildren() {},
      dispatchEvent(event) { if (this.oninput) this.oninput({target: this}); },
    });
  }
  return elements.get(id);
}

class FakeImage {
  set src(value) { if (this.onload) this.onload(); }
}

const sandbox = {
  document: {
    getElementById: element,
    createElement: () => element(`created-${elements.size}`),
    querySelectorAll: () => [],
    activeElement: {tagName: 'BODY'},
  },
  window: {devicePixelRatio: 1, addEventListener() {}},
  ResizeObserver: class {constructor(callback) {this.callback = callback;} observe() {this.callback();}},
  Image: FakeImage,
  Event: class {constructor(type) {this.type = type;}},
  console,
};
vm.runInNewContext(`${source}\nglobalThis.testState = state; globalThis.testRotationHandle = rotationHandle; globalThis.testResultView = resultView; globalThis.testFitResult = fitResult; globalThis.testZoomResult = zoomResult; globalThis.testFormatApiDebug = formatApiDebug; globalThis.testShowProgress = showProgress; globalThis.testUpdateProgress = updateProgress;`, sandbox);
const state = sandbox.testState;
const reference = state.images[0].corners.map(point => [...point]);
const moving = state.images[1];
const screen = point => [point[0] * state.zoom + state.panX, point[1] * state.zoom + state.panY];
const pointer = (type, point, extra = {}) => {
  const [clientX, clientY] = screen(point);
  listeners[type]({clientX, clientY, button: type === 'pointermove' ? -1 : 0,
                   pointerId: 1, shiftKey: false, ...extra});
};
const close = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-6, `${actual} != ${expected}`);

const start = moving.corners.map(point => [...point]);
const corner = start[1], anchor = start[3];
pointer('pointerdown', corner);
pointer('pointermove', [anchor[0] + 1.2 * (corner[0] - anchor[0]),
                        anchor[1] + 1.2 * (corner[1] - anchor[1])]);
pointer('pointerup', moving.corners[1]);
close(Math.hypot(moving.corners[1][0] - moving.corners[0][0],
                 moving.corners[1][1] - moving.corners[0][1]), 240);
close(Math.hypot(moving.corners[3][0] - moving.corners[0][0],
                 moving.corners[3][1] - moving.corners[0][1]), 144);

const center = [moving.corners.reduce((sum, point) => sum + point[0], 0) / 4,
                moving.corners.reduce((sum, point) => sum + point[1], 0) / 4];
const handle = sandbox.testRotationHandle(moving);
const angle = Math.PI / 6;
const dx = handle[0] - center[0], dy = handle[1] - center[1];
const rotatedHandle = [center[0] + Math.cos(angle) * dx - Math.sin(angle) * dy,
                       center[1] + Math.sin(angle) * dx + Math.cos(angle) * dy];
pointer('pointerdown', handle);
pointer('pointermove', rotatedHandle);
pointer('pointerup', rotatedHandle);
const edge = [moving.corners[1][0] - moving.corners[0][0],
              moving.corners[1][1] - moving.corners[0][1]];
close(Math.atan2(edge[1], edge[0]), angle);
close(Math.hypot(...edge), 240);
assert.equal(JSON.stringify(state.images[0].corners), JSON.stringify(reference));
assert.doesNotMatch(html, /id="(?:apply-position|rotate-left|rotate-right|scale-down|scale-up)"/);
const resultImage = element('result-preview');
resultImage.naturalWidth = 1800;
resultImage.naturalHeight = 1000;
sandbox.testFitResult();
close(sandbox.testResultView.scale, .5);
close(sandbox.testResultView.x, 0);
sandbox.testZoomResult(2, 450, 250);
close(sandbox.testResultView.scale, 1);
close(sandbox.testResultView.x, -450);
close(sandbox.testResultView.y, -250);
const debugText = sandbox.testFormatApiDebug({
  algorithm: 'guided_affine', output_width: 500, output_height: 900,
  debug_info: {image_count: 2, stages: [{index: 17, total: 17, percentage: 95,
    message: '最终全景图完成', elapsed_seconds: 2.3}], timings: {}, seam_mesh: {
      enabled: true, status: 'applied', matches_used: 18, seam_refined: true,
      images: [
        {image_index: 0, status: 'fixed_reference', max_displacement_pixels: 0,
          seam_support_pixels: 350},
        {image_index: 1, status: 'refined', max_displacement_pixels: 2.375,
          seam_support_pixels: 320, minimum_jacobian: 0.91},
      ],
    }},
  settings: {guided_mesh_grid_spacing: 128, guided_mesh_seam_band_ratio: .15,
    guided_mesh_max_displacement_ratio: .02, guided_mesh_min_matches: 12},
});
assert.match(debugText, /\[17\/17\] 最终全景图完成/);
assert.match(debugText, /输出：500 × 900/);
assert.match(debugText, /状态：已应用网格形变；启用：是；接缝匹配：18；实际应用：是/);
assert.match(debugText, /图 2：已变形；最大位移 2\.375 px/);
const homographyDebugText = sandbox.testFormatApiDebug({
  transform_model: 'homography', registration_status: 'full',
  image_diagnostics: [{image_index: 1, status: 'refined', attempted_correction: {
    max_corner_shift_pixels: 15.25, mean_corner_shift_pixels: 9.5,
  }}],
  debug_info: {image_count: 2, stages: [], timings: {}, pair_diagnostics: [], seam_mesh: {
    enabled: true, status: 'fallback_insufficient_seam_matches', matches_used: 3, images: [],
  }},
  settings: {guided_transform_model: 'homography', guided_mesh_seam_band_ratio: .25},
});
assert.match(homographyDebugText, /全局几何模型：homography/);
assert.match(homographyDebugText, /最大角点位移 15\.25 px；平均角点位移 9\.50 px/);
sandbox.testShowProgress();
assert.equal(element('progress-overlay').hidden, false);
sandbox.testUpdateProgress({exists: true, percentage: 65, status: '估计曝光补偿'});
assert.equal(element('progress-bar').value, 65);
assert.equal(element('progress-percent').textContent, '65.0%');
assert.equal(element('progress-stage').textContent, '估计曝光补偿');
console.log('Canvas gestures and result viewer zoom OK');

// Region editing uses world coordinates, and image placement does not move boxes.
element('edit-mode').onchange({target:{value:'regions'}});
pointer('pointerdown', [-40,20]);
pointer('pointermove', [80,90]);
pointer('pointerup', [80,90]);
assert.equal(state.regions.length,1);
state.regions[0].rect.forEach((v,i) => close(v,[-40,20,120,70][i]));
pointer('pointerdown', [20,50]);
pointer('pointermove', [30,60]);
pointer('pointerup', [30,60]);
state.regions[0].rect.forEach((v,i) => close(v,[-30,30,120,70][i]));
pointer('pointerdown', [90,100]);
pointer('pointermove', [110,130]);
pointer('pointerup', [110,130]);
state.regions[0].rect.forEach((v,i) => close(v,[-30,30,140,100][i]));
const fixedRect = [...state.regions[0].rect];
element('edit-mode').onchange({target:{value:'images'}});
assert.deepEqual(Array.from(state.regions[0].rect),fixedRect);
element('edit-mode').onchange({target:{value:'regions'}});
const oldPan = state.panX;
state.space = true;
pointer('pointerdown',[10,60]);
pointer('pointermove',[40,60]);
pointer('pointerup',[40,60]);
state.space = false;
assert.notEqual(state.panX,oldPan);
assert.deepEqual(Array.from(state.regions[0].rect),fixedRect);
element('delete-region').onclick();
assert.equal(state.regions.length,0);
pointer('pointerdown',[0,0]);
pointer('pointermove',[40,50]);
pointer('pointerup',[40,50]);
element('clear-regions').onclick();
assert.equal(state.regions.length,0);
console.log('Region create, move, resize, delete and canvas pan OK');
state.images[0].corners = [[0,0],[200,0],[200,120],[0,120]];
state.images[1].corners = [[160,0],[360,0],[360,120],[160,120]];
const related = vm.runInNewContext('regionPairs({rect:[170,10,20,90]})',sandbox);
assert.deepEqual(Array.from(related),['1↔2']);
assert.equal(vm.runInNewContext('regionPairs({rect:[-100,-100,10,10]}).length',sandbox),0);
pointer('pointerdown',[170,10]); pointer('pointermove',[190,100]); pointer('pointerup',[190,100]);
assert.equal(state.regions.length,1);
element('reset').onclick();
assert.equal(state.regions.length,0);
console.log('Region pair associations and reset OK');

// A smaller moving image uses its own canvas short side for the search margin.
state.images[1].corners = [[208,0],[308,0],[308,60],[208,60]];
assert.equal(vm.runInNewContext('regionPairs({rect:[195,0,30,60]}).length',sandbox),0);
const partialDebug = sandbox.testFormatApiDebug({
  algorithm: 'guided_affine', registration_status: 'partial',
  output_width: 500, output_height: 900,
  image_diagnostics: [{image_index: 2, status: 'fallback_initial', reason: '没有通过可靠匹配连接到第一张参考图'}],
  debug_info: {image_count: 3, stages: [], timings: {}},
});
assert.match(partialDebug, /部分配准/);
assert.match(partialDebug, /图 3：保留人工位置/);
const rejectedDebug = sandbox.testFormatApiDebug({
  registration_status: 'manual_only',
  image_diagnostics: [{image_index: 1, status: 'fallback_initial', reason: '仿射修正超过微调上限',
    attempted_correction: {translation_ratio: 0.08, rotation_degrees: 0, scale_change: 0.01, shear: 0.01}}],
  debug_info: {stages: [], timings: {}},
});
assert.match(rejectedDebug, /超过微调上限/);
assert.match(rejectedDebug, /中心平移 8.00% 短边/);

// File-generation success must not be presented as successful registration.
(async () => {
  element('output-name').value = 'result.png';
  element('final-megapix').value = '1';
  element('transform-model').value = 'homography';
  sandbox.setInterval = () => 1;
  sandbox.clearInterval = () => {};
  const submittedBodies = [];
  for (const [status, text] of [['full','完整配准'], ['partial','部分配准'], ['manual_only','仅按人工布局合成']]) {
    sandbox.fetch = async (url, options = {}) => {
      if (url.includes('baseline=1')) return {ok: true, json: async () => ({version: '0'})};
      submittedBodies.push(JSON.parse(options.body));
      return {ok: true, json: async () => ({
        registration_status: status, output_width: 500, output_height: 900,
        output_path: '/output/result.png', preview_url: '/result/result.png', download_url: '/download/result.png',
        debug_info: {stages: [], timings: {}, pair_diagnostics: []},
      })};
    };
    await element('run').onclick();
    assert.match(element('status').textContent, new RegExp(text));
    assert.equal(element('status').className, status === 'full' ? 'success' : 'muted');
    assert.equal(element('result-panel').style.display, 'block');
  }
  assert.equal(submittedBodies.length, 3);
  assert.ok(submittedBodies.every(body => body.guided_transform_model === 'homography'));
  console.log('Registration status, fallback display and pair margins OK');
})().catch(error => {console.error(error); process.exitCode = 1;});
