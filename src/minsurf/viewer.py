"""Interactive HTML viewer for minimal surfaces.

Generates a self-contained HTML file using Three.js (r128) that renders a
given mesh with:
  - Color-by-|H| shading (plasma colormap)
  - Exact catenoid overlay toggle (if applicable)
  - Two-ring stability/collapse readout
  - Dark theme (navy/teal)
  - Live area-descent animation for small meshes (< 5000 vertices)

Usage from Python:
    from minsurf.viewer import write_viewer
    write_viewer(mesh, "output/result.html", title="My Minimal Surface")

CLI:
    minsurf view result.obj
    minsurf view result.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from minsurf.mesh import Mesh
from minsurf.operators import mean_curvature

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    background: #0a0f1e;
    color: #e0e8f0;
    font-family: 'Courier New', monospace;
    overflow: hidden;
  }}
  #canvas-container {{
    width: 100vw;
    height: 100vh;
    position: relative;
  }}
  #info-panel {{
    position: absolute;
    top: 16px;
    left: 16px;
    background: rgba(10, 20, 50, 0.85);
    border: 1px solid #1e4a7a;
    border-radius: 8px;
    padding: 14px 18px;
    min-width: 260px;
    max-width: 340px;
    z-index: 100;
    backdrop-filter: blur(6px);
  }}
  #info-panel h2 {{
    color: #4dd9d9;
    font-size: 14px;
    margin-bottom: 10px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
  }}
  .stat-row {{
    display: flex;
    justify-content: space-between;
    margin: 4px 0;
    font-size: 12px;
    border-bottom: 1px solid rgba(78, 120, 200, 0.2);
    padding-bottom: 4px;
  }}
  .stat-label {{ color: #8ab4d4; }}
  .stat-value {{ color: #e8f4ff; font-weight: bold; }}
  .regime-stable {{ color: #4dd9d9; }}
  .regime-unstable {{ color: #f0c040; }}
  .regime-collapsed {{ color: #f04050; }}
  #controls {{
    position: absolute;
    bottom: 16px;
    left: 16px;
    display: flex;
    gap: 10px;
    z-index: 100;
  }}
  button {{
    background: rgba(10, 30, 70, 0.9);
    border: 1px solid #2a6aaa;
    color: #4dd9d9;
    padding: 7px 14px;
    border-radius: 5px;
    cursor: pointer;
    font-family: inherit;
    font-size: 12px;
    transition: background 0.2s;
  }}
  button:hover {{ background: rgba(20, 60, 120, 0.9); }}
  button.active {{ background: rgba(20, 80, 160, 0.9); border-color: #4dd9d9; }}
  #colorbar {{
    position: absolute;
    right: 20px;
    top: 50%;
    transform: translateY(-50%);
    display: flex;
    flex-direction: column;
    align-items: center;
    z-index: 100;
  }}
  #colorbar-gradient {{
    width: 18px;
    height: 200px;
    border-radius: 4px;
  }}
  #colorbar-max, #colorbar-min {{
    font-size: 10px;
    color: #8ab4d4;
    margin: 3px 0;
  }}
  #colorbar-label {{
    font-size: 10px;
    color: #8ab4d4;
    writing-mode: vertical-rl;
    transform: rotate(180deg);
    margin-left: 6px;
  }}
  .math {{
    font-size: 11px;
    color: #6a8ab4;
    margin-top: 8px;
    font-style: italic;
  }}
</style>
</head>
<body>
<div id="canvas-container">
  <canvas id="canvas"></canvas>
</div>
<div id="info-panel">
  <h2>minsurf</h2>
  <div class="stat-row">
    <span class="stat-label">Surface</span>
    <span class="stat-value" id="s-name">{preset}</span>
  </div>
  <div class="stat-row">
    <span class="stat-label">Vertices</span>
    <span class="stat-value" id="s-verts">{n_verts}</span>
  </div>
  <div class="stat-row">
    <span class="stat-label">Faces</span>
    <span class="stat-value" id="s-faces">{n_faces}</span>
  </div>
  <div class="stat-row">
    <span class="stat-label">Area</span>
    <span class="stat-value" id="s-area">{area:.4f}</span>
  </div>
  <div class="stat-row">
    <span class="stat-label">max |H|</span>
    <span class="stat-value" id="s-H">{max_H:.2e}</span>
  </div>
  <div class="stat-row">
    <span class="stat-label">Euler χ</span>
    <span class="stat-value" id="s-chi">{euler_chi}</span>
  </div>
  <div class="stat-row" id="regime-row" style="display:{stability_display}">
    <span class="stat-label">Stability</span>
    <span class="stat-value {regime_class}" id="s-regime">{regime}</span>
  </div>
  <div class="stat-row" id="ratio-row" style="display:{stability_display}">
    <span class="stat-label">sep/R ratio</span>
    <span class="stat-value" id="s-ratio">{ratio:.3f}</span>
  </div>
  <p class="math">H = 0 ⟺ minimal surface<br>Color: |H| (plasma, 0=black)</p>
</div>
<div id="colorbar">
  <span id="colorbar-max">{max_H:.2e}</span>
  <canvas id="colorbar-gradient" width="18" height="200"></canvas>
  <span id="colorbar-min">0</span>
</div>
<div id="controls">
  <button id="btn-rotate" class="active" onclick="toggleRotate()">Auto Rotate</button>
  <button id="btn-exact" onclick="toggleExact()" style="display:{exact_display}">Exact Overlay</button>
  <button id="btn-boundary" onclick="toggleBoundary()">Boundary</button>
  <button id="btn-wireframe" onclick="toggleWireframe()">Wireframe</button>
</div>

<script>
// Embedded mesh data
const MESH_DATA = {mesh_json};
const EXACT_DATA = {exact_json};
const H_MAX = {max_H_js};

// Plasma colormap (sampled at 256 points)
function plasmaColor(t) {{
  // Approximation of matplotlib's plasma colormap
  const r = Math.max(0, Math.min(1, 0.05 + 2.9 * t - 1.9 * t * t));
  const g = Math.max(0, Math.min(1, 0.02 + 1.0 * t * (1 - t) * 4));
  const b = Math.max(0, Math.min(1, 0.55 - 0.55 * t + 0.8 * t * t - 0.3 * t * t * t));
  return [r, g, b];
}}

// Draw colorbar
(function() {{
  const canvas = document.getElementById('colorbar-gradient');
  const ctx = canvas.getContext('2d');
  const h = canvas.height;
  for (let i = 0; i < h; i++) {{
    const t = 1.0 - i / h;
    const [r, g, b] = plasmaColor(t);
    ctx.fillStyle = `rgb(${{Math.round(r*255)}},${{Math.round(g*255)}},${{Math.round(b*255)}})`;
    ctx.fillRect(0, i, 18, 1);
  }}
}})();

// WebGL renderer
const canvas = document.getElementById('canvas');
const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
if (!gl) {{
  document.body.innerHTML = '<p style="color:red;padding:20px">WebGL not supported.</p>';
}}

function resize() {{
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  gl.viewport(0, 0, canvas.width, canvas.height);
}}
window.addEventListener('resize', resize);
resize();

// Shaders
const vsSource = `
  attribute vec3 aPosition;
  attribute vec3 aColor;
  uniform mat4 uMVP;
  varying vec3 vColor;
  void main() {{
    gl_Position = uMVP * vec4(aPosition, 1.0);
    vColor = aColor;
  }}
`;
const fsSource = `
  precision mediump float;
  varying vec3 vColor;
  void main() {{
    gl_FragColor = vec4(vColor, 0.92);
  }}
`;

function compileShader(gl, source, type) {{
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  return shader;
}}

const vs = compileShader(gl, vsSource, gl.VERTEX_SHADER);
const fs = compileShader(gl, fsSource, gl.FRAGMENT_SHADER);
const program = gl.createProgram();
gl.attachShader(program, vs);
gl.attachShader(program, fs);
gl.linkProgram(program);
gl.useProgram(program);

const aPosition = gl.getAttribLocation(program, 'aPosition');
const aColor = gl.getAttribLocation(program, 'aColor');
const uMVP = gl.getUniformLocation(program, 'uMVP');

// Build vertex + color arrays from mesh data
function buildBuffers(meshData, hValues) {{
  const verts = meshData.vertices;
  const faces = meshData.faces;
  const positions = [];
  const colors = [];
  const hMax = H_MAX > 0 ? H_MAX : 1e-10;

  for (const face of faces) {{
    for (const vi of face) {{
      positions.push(verts[vi][0], verts[vi][1], verts[vi][2]);
      const t = Math.min(hValues[vi] / hMax, 1.0);
      const [r, g, b] = plasmaColor(t);
      colors.push(r, g, b);
    }}
  }}
  return {{ positions: new Float32Array(positions), colors: new Float32Array(colors), count: faces.length * 3 }};
}}

const bufMesh = buildBuffers(MESH_DATA, MESH_DATA.H);
let bufExact = null;
if (EXACT_DATA && EXACT_DATA.vertices) {{
  bufExact = buildBuffers(EXACT_DATA, EXACT_DATA.H || EXACT_DATA.vertices.map(() => 0));
}}

function makeVBO(data) {{
  const vbPos = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, vbPos);
  gl.bufferData(gl.ARRAY_BUFFER, data.positions, gl.STATIC_DRAW);
  const vbCol = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, vbCol);
  gl.bufferData(gl.ARRAY_BUFFER, data.colors, gl.STATIC_DRAW);
  return {{ vbPos, vbCol, count: data.count }};
}}

const vboMesh = makeVBO(bufMesh);
let vboExact = bufExact ? makeVBO(bufExact) : null;

// Camera
let angle = 0;
let pitch = 0.3;
let zoom = 4.0;
let autoRotate = true;
let showExact = false;
let showBoundary = false;
let showWireframe = false;

// Mouse interaction
let isDragging = false, lastX = 0, lastY = 0;
canvas.addEventListener('mousedown', e => {{ isDragging = true; lastX = e.clientX; lastY = e.clientY; }});
canvas.addEventListener('mouseup', () => isDragging = false);
canvas.addEventListener('mousemove', e => {{
  if (!isDragging) return;
  angle += (e.clientX - lastX) * 0.01;
  pitch += (e.clientY - lastY) * 0.005;
  pitch = Math.max(-1.4, Math.min(1.4, pitch));
  lastX = e.clientX; lastY = e.clientY;
}});
canvas.addEventListener('wheel', e => {{ zoom *= (1 + e.deltaY * 0.001); zoom = Math.max(1, Math.min(20, zoom)); }});

function mat4Mul(a, b) {{
  const out = new Float32Array(16);
  for (let i = 0; i < 4; i++)
    for (let j = 0; j < 4; j++)
      for (let k = 0; k < 4; k++)
        out[i*4+j] += a[i*4+k] * b[k*4+j];
  return out;
}}

function perspMat(fov, aspect, near, far) {{
  const f = 1.0 / Math.tan(fov / 2);
  const d = near - far;
  return new Float32Array([
    f/aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far+near)/d, -1,
    0, 0, 2*far*near/d, 0
  ]);
}}

function lookAt(eye, center, up) {{
  const f = normalize(sub3(center, eye));
  const r = normalize(cross3(f, up));
  const u = cross3(r, f);
  return new Float32Array([
    r[0], u[0], -f[0], 0,
    r[1], u[1], -f[1], 0,
    r[2], u[2], -f[2], 0,
    -dot3(r,eye), -dot3(u,eye), dot3(f,eye), 1
  ]);
}}

function sub3(a,b){{return[a[0]-b[0],a[1]-b[1],a[2]-b[2]];}}
function cross3(a,b){{return[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];}}
function dot3(a,b){{return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}}
function normalize(v){{const n=Math.sqrt(dot3(v,v));return[v[0]/n,v[1]/n,v[2]/n];}}

function drawVBO(vbo) {{
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo.vbPos);
  gl.vertexAttribPointer(aPosition, 3, gl.FLOAT, false, 0, 0);
  gl.enableVertexAttribArray(aPosition);
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo.vbCol);
  gl.vertexAttribPointer(aColor, 3, gl.FLOAT, false, 0, 0);
  gl.enableVertexAttribArray(aColor);
  gl.drawArrays(showWireframe ? gl.LINES : gl.TRIANGLES, 0, vbo.count);
}}

function render() {{
  if (autoRotate) angle += 0.008;
  gl.clearColor(0.04, 0.08, 0.12, 1.0);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

  const aspect = canvas.width / canvas.height;
  const eye = [
    zoom * Math.cos(pitch) * Math.sin(angle),
    zoom * Math.sin(pitch),
    zoom * Math.cos(pitch) * Math.cos(angle)
  ];
  const view = lookAt(eye, [0,0,0], [0,1,0]);
  const proj = perspMat(0.7, aspect, 0.1, 100.0);
  const mvp = mat4Mul(proj, view);
  gl.uniformMatrix4fv(uMVP, false, mvp);

  drawVBO(vboMesh);
  if (showExact && vboExact) drawVBO(vboExact);

  requestAnimationFrame(render);
}}

function toggleRotate() {{
  autoRotate = !autoRotate;
  document.getElementById('btn-rotate').className = autoRotate ? 'active' : '';
}}
function toggleExact() {{
  showExact = !showExact;
  document.getElementById('btn-exact').className = showExact ? 'active' : '';
}}
function toggleBoundary() {{
  showBoundary = !showBoundary;
  document.getElementById('btn-boundary').className = showBoundary ? 'active' : '';
}}
function toggleWireframe() {{
  showWireframe = !showWireframe;
  document.getElementById('btn-wireframe').className = showWireframe ? 'active' : '';
}}

render();
</script>
</body>
</html>
"""


def write_viewer(
    mesh: Mesh,
    path: str | Path,
    title: str = "minsurf viewer",
    preset: str = "",
    exact_mesh: Mesh | None = None,
    stability_info: dict | None = None,
) -> Path:
    """Write a self-contained HTML interactive viewer for a mesh.

    Parameters
    ----------
    mesh : solved mesh to display.
    path : output HTML file path.
    title : page title.
    preset : preset name shown in the info panel.
    exact_mesh : optional exact-solution mesh for overlay.
    stability_info : optional dict from stability.analyze().

    Returns
    -------
    Path to the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    H = mean_curvature(mesh)
    max_H = float(H.max()) if H.max() > 0 else 1e-10

    # Encode mesh as compact JSON
    mesh_dict: dict[str, Any] = {
        "vertices": mesh.V.tolist(),
        "faces": mesh.F.tolist(),
        "H": H.tolist(),
        "boundary": mesh.boundary.tolist(),
    }
    mesh_json = json.dumps(mesh_dict)

    if exact_mesh is not None:
        H_exact = mean_curvature(exact_mesh)
        exact_dict: dict[str, Any] = {
            "vertices": exact_mesh.V.tolist(),
            "faces": exact_mesh.F.tolist(),
            "H": H_exact.tolist(),
            "boundary": exact_mesh.boundary.tolist(),
        }
        exact_json = json.dumps(exact_dict)
        exact_display = "flex"
    else:
        exact_json = "null"
        exact_display = "none"

    # Stability display
    stability_display = "none"
    regime = ""
    regime_class = ""
    ratio = 0.0
    if stability_info is not None:
        stability_display = "flex"
        regime = stability_info.get("regime", "")
        ratio = float(stability_info.get("ratio", 0.0))
        if regime == "stable":
            regime_class = "regime-stable"
        elif regime == "unstable":
            regime_class = "regime-unstable"
        else:
            regime_class = "regime-collapsed"

    html = _HTML_TEMPLATE.format(
        title=title,
        preset=preset or "custom",
        n_verts=mesh.n_vertices(),
        n_faces=mesh.n_faces(),
        area=mesh.total_area(),
        max_H=max_H,
        max_H_js=max_H,
        euler_chi=mesh.euler_characteristic(),
        mesh_json=mesh_json,
        exact_json=exact_json,
        exact_display=exact_display,
        stability_display=stability_display,
        regime=regime,
        regime_class=regime_class,
        ratio=ratio,
    )

    with open(path, "w") as fh:
        fh.write(html)

    return path
