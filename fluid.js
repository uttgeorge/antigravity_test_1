// ===== WebGL Fluid Simulation Engine =====
// Based on GPU-accelerated Navier-Stokes equations

class FluidSimulation {
    constructor(canvas) {
        this.canvas = canvas;
        this.gl = canvas.getContext('webgl2', {
            alpha: false,
            depth: false,
            stencil: false,
            antialias: false,
            preserveDrawingBuffer: false
        });

        if (!this.gl) {
            alert('WebGL 2 not supported!');
            return;
        }


        // Enable extensions if available
        this.gl.getExtension('EXT_color_buffer_float');
        this.gl.getExtension('OES_texture_float_linear');

        // Use universally supported RGBA8 format for maximum compatibility
        this.formatRGBA = {
            internalFormat: this.gl.RGBA,
            format: this.gl.RGBA,
            type: this.gl.UNSIGNED_BYTE
        };
        this.formatRG = this.formatRGBA;  // Fallback to RGBA for RG
        this.formatR = this.formatRGBA;   // Fallback to RGBA for R

        // Simulation parameters
        this.config = {
            simResolution: 256,
            dyeResolution: 1024,
            viscosity: 20,
            diffusion: 0.8,
            pressure: 20,
            curl: 30,
            splatRadius: 0.5,
            colorScheme: 'neon'
        };

        // Color schemes
        this.colorSchemes = {
            neon: [
                { r: 0.0, g: 0.95, b: 1.0 },  // Cyan
                { r: 1.0, g: 0.0, b: 1.0 },   // Magenta
                { r: 0.69, g: 0.15, b: 1.0 }  // Purple
            ],
            rainbow: [
                { r: 1.0, g: 0.0, b: 0.0 },   // Red
                { r: 1.0, g: 0.5, b: 0.0 },   // Orange
                { r: 1.0, g: 1.0, b: 0.0 },   // Yellow
                { r: 0.0, g: 1.0, b: 0.0 },   // Green
                { r: 0.0, g: 0.5, b: 1.0 },   // Blue
                { r: 0.5, g: 0.0, b: 1.0 }    // Purple
            ],
            fire: [
                { r: 1.0, g: 0.0, b: 0.0 },   // Red
                { r: 1.0, g: 0.5, b: 0.0 },   // Orange
                { r: 1.0, g: 1.0, b: 0.0 }    // Yellow
            ],
            ocean: [
                { r: 0.0, g: 0.4, b: 0.8 },   // Deep Blue
                { r: 0.0, g: 0.8, b: 1.0 },   // Cyan
                { r: 0.0, g: 1.0, b: 0.8 }    // Turquoise
            ]
        };

        this.pointers = [];
        this.splatStack = [];

        this.init();
    }

    init() {
        this.resizeCanvas();
        this.initPrograms();
        this.initFramebuffers();
        this.setupEventListeners();
        this.update();
    }

    resizeCanvas() {
        const width = window.innerWidth;
        const height = window.innerHeight;

        if (this.canvas.width !== width || this.canvas.height !== height) {
            this.canvas.width = width;
            this.canvas.height = height;
        }
    }

    // ===== Shader Programs =====
    initPrograms() {
        const gl = this.gl;

        // Base vertex shader (used by all programs)
        const baseVertexShader = `
            precision highp float;
            attribute vec2 aPosition;
            varying vec2 vUv;
            varying vec2 vL;
            varying vec2 vR;
            varying vec2 vT;
            varying vec2 vB;
            uniform vec2 texelSize;
            
            void main() {
                vUv = aPosition * 0.5 + 0.5;
                vL = vUv - vec2(texelSize.x, 0.0);
                vR = vUv + vec2(texelSize.x, 0.0);
                vT = vUv + vec2(0.0, texelSize.y);
                vB = vUv - vec2(0.0, texelSize.y);
                gl_Position = vec4(aPosition, 0.0, 1.0);
            }
        `;

        // Display shader
        const displayFragmentShader = `
            precision highp float;
            varying vec2 vUv;
            uniform sampler2D uTexture;
            
            void main() {
                vec3 color = texture2D(uTexture, vUv).rgb;
                gl_FragColor = vec4(color, 1.0);
            }
        `;

        // Splat shader (add velocity and color)
        const splatFragmentShader = `
            precision highp float;
            varying vec2 vUv;
            uniform sampler2D uTarget;
            uniform float aspectRatio;
            uniform vec3 color;
            uniform vec2 point;
            uniform float radius;
            
            void main() {
                vec2 p = vUv - point.xy;
                p.x *= aspectRatio;
                vec3 splat = exp(-dot(p, p) / radius) * color;
                vec3 base = texture2D(uTarget, vUv).xyz;
                gl_FragColor = vec4(base + splat, 1.0);
            }
        `;

        // Advection shader (move quantities through velocity field)
        const advectionFragmentShader = `
            precision highp float;
            varying vec2 vUv;
            uniform sampler2D uVelocity;
            uniform sampler2D uSource;
            uniform vec2 texelSize;
            uniform float dt;
            uniform float dissipation;
            
            void main() {
                vec2 coord = vUv - dt * texture2D(uVelocity, vUv).xy * texelSize;
                gl_FragColor = dissipation * texture2D(uSource, coord);
            }
        `;

        // Divergence shader
        const divergenceFragmentShader = `
            precision highp float;
            varying vec2 vUv;
            varying vec2 vL;
            varying vec2 vR;
            varying vec2 vT;
            varying vec2 vB;
            uniform sampler2D uVelocity;
            
            void main() {
                float L = texture2D(uVelocity, vL).x;
                float R = texture2D(uVelocity, vR).x;
                float T = texture2D(uVelocity, vT).y;
                float B = texture2D(uVelocity, vB).y;
                float div = 0.5 * (R - L + T - B);
                gl_FragColor = vec4(div, 0.0, 0.0, 1.0);
            }
        `;

        // Curl shader
        const curlFragmentShader = `
            precision highp float;
            varying vec2 vUv;
            varying vec2 vL;
            varying vec2 vR;
            varying vec2 vT;
            varying vec2 vB;
            uniform sampler2D uVelocity;
            
            void main() {
                float L = texture2D(uVelocity, vL).y;
                float R = texture2D(uVelocity, vR).y;
                float T = texture2D(uVelocity, vT).x;
                float B = texture2D(uVelocity, vB).x;
                float vorticity = R - L - T + B;
                gl_FragColor = vec4(0.5 * vorticity, 0.0, 0.0, 1.0);
            }
        `;

        // Vorticity shader
        const vorticityFragmentShader = `
            precision highp float;
            varying vec2 vUv;
            varying vec2 vL;
            varying vec2 vR;
            varying vec2 vT;
            varying vec2 vB;
            uniform sampler2D uVelocity;
            uniform sampler2D uCurl;
            uniform float curl;
            uniform float dt;
            
            void main() {
                float L = texture2D(uCurl, vL).x;
                float R = texture2D(uCurl, vR).x;
                float T = texture2D(uCurl, vT).x;
                float B = texture2D(uCurl, vB).x;
                float C = texture2D(uCurl, vUv).x;
                
                vec2 force = 0.5 * vec2(abs(T) - abs(B), abs(R) - abs(L));
                force /= length(force) + 0.0001;
                force *= curl * C;
                force.y *= -1.0;
                
                vec2 velocity = texture2D(uVelocity, vUv).xy;
                velocity += force * dt;
                velocity = min(max(velocity, -1000.0), 1000.0);
                gl_FragColor = vec4(velocity, 0.0, 1.0);
            }
        `;

        // Pressure shader (Jacobi iteration)
        const pressureFragmentShader = `
            precision highp float;
            varying vec2 vUv;
            varying vec2 vL;
            varying vec2 vR;
            varying vec2 vT;
            varying vec2 vB;
            uniform sampler2D uPressure;
            uniform sampler2D uDivergence;
            
            void main() {
                float L = texture2D(uPressure, vL).x;
                float R = texture2D(uPressure, vR).x;
                float T = texture2D(uPressure, vT).x;
                float B = texture2D(uPressure, vB).x;
                float C = texture2D(uDivergence, vUv).x;
                float pressure = (L + R + T + B - C) * 0.25;
                gl_FragColor = vec4(pressure, 0.0, 0.0, 1.0);
            }
        `;

        // Gradient subtraction shader
        const gradientSubtractFragmentShader = `
            precision highp float;
            varying vec2 vUv;
            varying vec2 vL;
            varying vec2 vR;
            varying vec2 vT;
            varying vec2 vB;
            uniform sampler2D uPressure;
            uniform sampler2D uVelocity;
            
            void main() {
                float L = texture2D(uPressure, vL).x;
                float R = texture2D(uPressure, vR).x;
                float T = texture2D(uPressure, vT).x;
                float B = texture2D(uPressure, vB).x;
                vec2 velocity = texture2D(uVelocity, vUv).xy;
                velocity.xy -= vec2(R - L, T - B);
                gl_FragColor = vec4(velocity, 0.0, 1.0);
            }
        `;

        // Clear shader
        const clearFragmentShader = `
            precision highp float;
            varying vec2 vUv;
            uniform sampler2D uTexture;
            uniform float value;
            
            void main() {
                gl_FragColor = value * texture2D(uTexture, vUv);
            }
        `;

        // Compile and link programs
        this.programs = {
            display: this.createProgram(baseVertexShader, displayFragmentShader),
            splat: this.createProgram(baseVertexShader, splatFragmentShader),
            advection: this.createProgram(baseVertexShader, advectionFragmentShader),
            divergence: this.createProgram(baseVertexShader, divergenceFragmentShader),
            curl: this.createProgram(baseVertexShader, curlFragmentShader),
            vorticity: this.createProgram(baseVertexShader, vorticityFragmentShader),
            pressure: this.createProgram(baseVertexShader, pressureFragmentShader),
            gradientSubtract: this.createProgram(baseVertexShader, gradientSubtractFragmentShader),
            clear: this.createProgram(baseVertexShader, clearFragmentShader)
        };

        // Create vertex buffer
        const vertices = new Float32Array([-1, -1, -1, 1, 1, 1, 1, -1]);
        this.vertexBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
    }

    createProgram(vertexSource, fragmentSource) {
        const gl = this.gl;

        const vertexShader = gl.createShader(gl.VERTEX_SHADER);
        gl.shaderSource(vertexShader, vertexSource);
        gl.compileShader(vertexShader);

        const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
        gl.shaderSource(fragmentShader, fragmentSource);
        gl.compileShader(fragmentShader);

        const program = gl.createProgram();
        gl.attachShader(program, vertexShader);
        gl.attachShader(program, fragmentShader);
        gl.linkProgram(program);

        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
            console.error('Program link error:', gl.getProgramInfoLog(program));
        }

        return program;
    }

    // ===== Framebuffers =====
    initFramebuffers() {
        const simRes = this.config.simResolution;
        const dyeRes = this.config.dyeResolution;

        this.velocity = this.createDoubleFBO(
            simRes, simRes,
            this.formatRG.internalFormat,
            this.formatRG.format,
            this.formatRG.type
        );
        this.density = this.createDoubleFBO(
            dyeRes, dyeRes,
            this.formatRGBA.internalFormat,
            this.formatRGBA.format,
            this.formatRGBA.type
        );
        this.divergence = this.createFBO(
            simRes, simRes,
            this.formatR.internalFormat,
            this.formatR.format,
            this.formatR.type
        );
        this.curl = this.createFBO(
            simRes, simRes,
            this.formatR.internalFormat,
            this.formatR.format,
            this.formatR.type
        );
        this.pressure = this.createDoubleFBO(
            simRes, simRes,
            this.formatR.internalFormat,
            this.formatR.format,
            this.formatR.type
        );
    }

    createFBO(w, h, internalFormat, format, type) {
        const gl = this.gl;

        const texture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, texture);

        // LINEAR filtering works well with UNSIGNED_BYTE
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, w, h, 0, format, type, null);

        const fbo = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
        gl.viewport(0, 0, w, h);

        // Unbind framebuffer
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);

        return { texture, fbo, width: w, height: h };
    }

    createDoubleFBO(w, h, internalFormat, format, type) {
        let fbo1 = this.createFBO(w, h, internalFormat, format, type);
        let fbo2 = this.createFBO(w, h, internalFormat, format, type);

        return {
            read: fbo1,
            write: fbo2,
            swap() {
                let temp = fbo1;
                fbo1 = fbo2;
                fbo2 = temp;
                this.read = fbo1;
                this.write = fbo2;
            }
        };
    }

    // ===== Event Listeners =====
    setupEventListeners() {
        this.canvas.addEventListener('mousedown', (e) => {
            const pointer = {
                id: -1,
                x: e.clientX,
                y: e.clientY,
                dx: 0,
                dy: 0,
                down: true,
                moved: false,
                color: this.getRandomColor()
            };
            this.pointers.push(pointer);
        });

        this.canvas.addEventListener('mousemove', (e) => {
            const pointer = this.pointers.find(p => p.id === -1);
            if (pointer) {
                pointer.moved = pointer.down;
                pointer.dx = (e.clientX - pointer.x) * 5.0;
                pointer.dy = (e.clientY - pointer.y) * 5.0;
                pointer.x = e.clientX;
                pointer.y = e.clientY;
            }
        });

        this.canvas.addEventListener('mouseup', () => {
            this.pointers = this.pointers.filter(p => p.id !== -1);
        });

        // Touch events
        this.canvas.addEventListener('touchstart', (e) => {
            e.preventDefault();
            const touches = e.targetTouches;
            for (let i = 0; i < touches.length; i++) {
                const touch = touches[i];
                const pointer = {
                    id: touch.identifier,
                    x: touch.clientX,
                    y: touch.clientY,
                    dx: 0,
                    dy: 0,
                    down: true,
                    moved: false,
                    color: this.getRandomColor()
                };
                this.pointers.push(pointer);
            }
        });

        this.canvas.addEventListener('touchmove', (e) => {
            e.preventDefault();
            const touches = e.targetTouches;
            for (let i = 0; i < touches.length; i++) {
                const touch = touches[i];
                const pointer = this.pointers.find(p => p.id === touch.identifier);
                if (pointer) {
                    pointer.moved = pointer.down;
                    pointer.dx = (touch.clientX - pointer.x) * 5.0;
                    pointer.dy = (touch.clientY - pointer.y) * 5.0;
                    pointer.x = touch.clientX;
                    pointer.y = touch.clientY;
                }
            }
        });

        this.canvas.addEventListener('touchend', (e) => {
            const touches = e.changedTouches;
            for (let i = 0; i < touches.length; i++) {
                const touch = touches[i];
                this.pointers = this.pointers.filter(p => p.id !== touch.identifier);
            }
        });

        window.addEventListener('resize', () => {
            this.resizeCanvas();
        });
    }

    getRandomColor() {
        const scheme = this.colorSchemes[this.config.colorScheme];
        return scheme[Math.floor(Math.random() * scheme.length)];
    }

    // ===== Simulation Update =====
    update() {
        const dt = 0.016; // ~60 FPS

        this.resizeCanvas();
        this.applyInputs();
        this.step(dt);
        this.render();

        requestAnimationFrame(() => this.update());
    }

    applyInputs() {
        if (this.splatStack.length > 0) {
            this.multipleSplats(this.splatStack.pop());
        }

        for (let i = 0; i < this.pointers.length; i++) {
            const pointer = this.pointers[i];
            if (pointer.moved) {
                this.splat(pointer.x, pointer.y, pointer.dx, pointer.dy, pointer.color);
                pointer.moved = false;
            }
        }
    }

    step(dt) {
        const gl = this.gl;

        // Curl
        this.runProgram(this.programs.curl, this.curl, {
            texelSize: [1.0 / this.velocity.read.width, 1.0 / this.velocity.read.height],
            uVelocity: this.velocity.read.texture
        });

        // Vorticity
        this.runProgram(this.programs.vorticity, this.velocity.write, {
            texelSize: [1.0 / this.velocity.read.width, 1.0 / this.velocity.read.height],
            uVelocity: this.velocity.read.texture,
            uCurl: this.curl.texture,
            curl: this.config.curl,
            dt: dt
        });
        this.velocity.swap();

        // Divergence
        this.runProgram(this.programs.divergence, this.divergence, {
            texelSize: [1.0 / this.velocity.read.width, 1.0 / this.velocity.read.height],
            uVelocity: this.velocity.read.texture
        });

        // Pressure
        this.runProgram(this.programs.clear, this.pressure.write, {
            uTexture: this.pressure.read.texture,
            value: 0.8
        });
        this.pressure.swap();

        for (let i = 0; i < this.config.pressure; i++) {
            this.runProgram(this.programs.pressure, this.pressure.write, {
                texelSize: [1.0 / this.velocity.read.width, 1.0 / this.velocity.read.height],
                uPressure: this.pressure.read.texture,
                uDivergence: this.divergence.texture
            });
            this.pressure.swap();
        }

        // Gradient subtract
        this.runProgram(this.programs.gradientSubtract, this.velocity.write, {
            texelSize: [1.0 / this.velocity.read.width, 1.0 / this.velocity.read.height],
            uPressure: this.pressure.read.texture,
            uVelocity: this.velocity.read.texture
        });
        this.velocity.swap();

        // Advection
        this.runProgram(this.programs.advection, this.velocity.write, {
            texelSize: [1.0 / this.velocity.read.width, 1.0 / this.velocity.read.height],
            uVelocity: this.velocity.read.texture,
            uSource: this.velocity.read.texture,
            dt: dt,
            dissipation: 1.0 - this.config.viscosity / 100.0
        });
        this.velocity.swap();

        this.runProgram(this.programs.advection, this.density.write, {
            texelSize: [1.0 / this.velocity.read.width, 1.0 / this.velocity.read.height],
            uVelocity: this.velocity.read.texture,
            uSource: this.density.read.texture,
            dt: dt,
            dissipation: 1.0 - this.config.diffusion / 10.0
        });
        this.density.swap();
    }

    render() {
        const gl = this.gl;
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.viewport(0, 0, this.canvas.width, this.canvas.height);

        this.runProgram(this.programs.display, null, {
            uTexture: this.density.read.texture
        });
    }

    runProgram(program, target, uniforms) {
        const gl = this.gl;

        gl.useProgram(program);

        // Bind vertex buffer
        const aPosition = gl.getAttribLocation(program, 'aPosition');
        gl.enableVertexAttribArray(aPosition);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer);
        gl.vertexAttribPointer(aPosition, 2, gl.FLOAT, false, 0, 0);

        // Set uniforms
        for (const name in uniforms) {
            const location = gl.getUniformLocation(program, name);
            const value = uniforms[name];

            if (value instanceof WebGLTexture) {
                gl.uniform1i(location, 0);
                gl.activeTexture(gl.TEXTURE0);
                gl.bindTexture(gl.TEXTURE_2D, value);
            } else if (Array.isArray(value)) {
                if (value.length === 2) gl.uniform2f(location, value[0], value[1]);
                else if (value.length === 3) gl.uniform3f(location, value[0], value[1], value[2]);
            } else {
                gl.uniform1f(location, value);
            }
        }

        // Bind target framebuffer
        if (target) {
            gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo);
            gl.viewport(0, 0, target.width, target.height);
        }

        // Draw
        gl.drawArrays(gl.TRIANGLE_FAN, 0, 4);
    }

    splat(x, y, dx, dy, color) {
        this.runProgram(this.programs.splat, this.velocity.write, {
            uTarget: this.velocity.read.texture,
            aspectRatio: this.canvas.width / this.canvas.height,
            point: [x / this.canvas.width, 1.0 - y / this.canvas.height],
            color: [dx, -dy, 1.0],
            radius: this.config.splatRadius / 100.0
        });
        this.velocity.swap();

        this.runProgram(this.programs.splat, this.density.write, {
            uTarget: this.density.read.texture,
            aspectRatio: this.canvas.width / this.canvas.height,
            point: [x / this.canvas.width, 1.0 - y / this.canvas.height],
            color: [color.r, color.g, color.b],
            radius: this.config.splatRadius / 100.0
        });
        this.density.swap();
    }

    multipleSplats(amount) {
        for (let i = 0; i < amount; i++) {
            const color = this.getRandomColor();
            const x = Math.random() * this.canvas.width;
            const y = Math.random() * this.canvas.height;
            const dx = 1000 * (Math.random() - 0.5);
            const dy = 1000 * (Math.random() - 0.5);
            this.splat(x, y, dx, dy, color);
        }
    }

    clear() {
        const gl = this.gl;
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.density.write.fbo);
        gl.clear(gl.COLOR_BUFFER_BIT);
        this.density.swap();
    }
}

// ===== UI Controls =====
class UIController {
    constructor(simulation) {
        this.sim = simulation;
        this.stats = {
            fps: 60,
            frameTime: 16,
            lastTime: performance.now(),
            frames: 0
        };

        this.initControls();
        this.updateStats();
    }

    initControls() {
        // Viscosity
        const viscositySlider = document.getElementById('viscosity');
        const viscosityValue = document.getElementById('viscosity-value');
        viscositySlider.addEventListener('input', (e) => {
            this.sim.config.viscosity = parseFloat(e.target.value);
            viscosityValue.textContent = e.target.value;
        });

        // Diffusion
        const diffusionSlider = document.getElementById('diffusion');
        const diffusionValue = document.getElementById('diffusion-value');
        diffusionSlider.addEventListener('input', (e) => {
            this.sim.config.diffusion = parseFloat(e.target.value);
            diffusionValue.textContent = e.target.value;
        });

        // Pressure
        const pressureSlider = document.getElementById('pressure');
        const pressureValue = document.getElementById('pressure-value');
        pressureSlider.addEventListener('input', (e) => {
            this.sim.config.pressure = parseInt(e.target.value);
            pressureValue.textContent = e.target.value;
        });

        // Curl
        const curlSlider = document.getElementById('curl');
        const curlValue = document.getElementById('curl-value');
        curlSlider.addEventListener('input', (e) => {
            this.sim.config.curl = parseInt(e.target.value);
            curlValue.textContent = e.target.value;
        });

        // Splat Radius
        const splatRadiusSlider = document.getElementById('splatRadius');
        const splatRadiusValue = document.getElementById('splatRadius-value');
        splatRadiusSlider.addEventListener('input', (e) => {
            this.sim.config.splatRadius = parseFloat(e.target.value) * 100;
            splatRadiusValue.textContent = e.target.value;
        });

        // Color schemes
        const schemeButtons = document.querySelectorAll('.scheme-btn');
        schemeButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                schemeButtons.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.sim.config.colorScheme = btn.dataset.scheme;
            });
        });

        // Clear button
        document.getElementById('clearBtn').addEventListener('click', () => {
            this.sim.clear();
        });

        // Stats toggle
        const statsDisplay = document.getElementById('stats');
        document.getElementById('statsToggle').addEventListener('click', () => {
            statsDisplay.classList.toggle('active');
        });
    }

    updateStats() {
        const now = performance.now();
        this.stats.frames++;

        if (now >= this.stats.lastTime + 1000) {
            this.stats.fps = Math.round((this.stats.frames * 1000) / (now - this.stats.lastTime));
            this.stats.frameTime = Math.round(1000 / this.stats.fps);
            this.stats.frames = 0;
            this.stats.lastTime = now;

            document.getElementById('fps').textContent = this.stats.fps;
            document.getElementById('frameTime').textContent = this.stats.frameTime + 'ms';
        }

        requestAnimationFrame(() => this.updateStats());
    }
}

// ===== Initialize Application =====
window.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('fluidCanvas');
    const simulation = new FluidSimulation(canvas);
    const ui = new UIController(simulation);

    // Add some initial splats for visual interest
    setTimeout(() => {
        simulation.multipleSplats(Math.random() * 10 + 5);
    }, 100);
});
