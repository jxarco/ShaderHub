// snippets collected from Xor's fragcoord.xyz website

const CONSTANTS_LIBRARY_SNIPPETS = [{
    name: "Common",
    description: "PI, TAU, HPI, square roots",
    code: `#define PI         3.14159265359
#define TAU        6.28318530718
#define HPI        1.57079632679
#define SQRT_HALF  0.70710678118
#define SQRT2      1.41421356237
#define SQRT_THIRD 0.57735026919`
}, {
    name: "Simplex",
    description: "Skew / unskew factors for simplex noise (2D–4D)",
    code: `#define SQRT_3_4  0.86602540378 // sqrt(0.75)
#define SQRT_HALF 0.70710678118 // sqrt(0.5)

// Simplex skew:   (sqrt(N+1)-1)/N
// Simplex unskew: (1-1/sqrt(N+1))/N
#define F2 0.36602540378
#define G2 0.21132486541
#define F3 0.33333333333
#define G3 0.16666666667
#define F4 0.30901699437
#define G4 0.13819660113`
}];

export const WGSL_CODE_LIBRARY = [
    {
        name: "Constants",
        icon: "Pi",
        snippets: [
            ...CONSTANTS_LIBRARY_SNIPPETS,
            {
                name: "Golden Ratio",
                description: "PHI, golden angle, 2D & 3D rotation matrices",
                code: `#define PHI          1.61803398875
#define GOLDEN_ANGLE 2.39996322972

const GOLDEN_ROT2: mat2x2f = mat2x2f(
    -0.73736887808,  0.67549029426,
    -0.67549029426, -0.73736887808);

const GOLDEN_ROT3: mat3x3f = mat3x3f(
    -0.571464913,  0.814921382, 0.096597072,
    -0.278044873, -0.303026659, 0.911518454,
     0.772087367,  0.494042493, 0.399753815);`
            }
        ]
    },
    {
        name: "Noise",
        icon: "Hash",
        snippets: [{
            name: "Hash (1D → 1D)",
            description: "Simple pseudorandom hash from a float",
            code: `fn hash_11(s: f32) -> f32
{
    var p: f32 = fract(s * 0.1031);
    p *= p + 33.33;
    p *= p + p;
    return fract(p);
}`
        }, {
            name: "Hash (2D → 1D)",
            description: "Pseudorandom float from a vec2 seed",
            code: `fn hash_21(p: vec2f) -> f32
{
    var p3: vec3f = fract(vec3f(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}`
        }, {
            name: "Hash (2D → 2D)",
            description: "Pseudorandom vec2 from a vec2 seed",
            code: `fn hash_22(p: vec2f) -> vec2f
{
    var p3: vec3f = fract(vec3f(p.xyx) * vec3f(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}`
        }, {
            name: "Value Noise 2D",
            description: "Smooth interpolated value noise",
            code: `fn value_noise(p: vec2f) -> f32
{
    let i: vec2f = floor(p);
    var f: vec2f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    let a: f32 = hash_21(i);
    let b: f32 = hash_21(i + vec2(1.0, 0.0));
    let c: f32 = hash_21(i + vec2(0.0, 1.0));
    let d: f32 = hash_21(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}`
        }, {
            name: "Simplex Noise 2D",
            description: "Classic 2D simplex noise",
            author: "Ashima Arts, Stefan Gustavson (webgl-noise)",
            code: `fn mod_289_3(x: vec3f) -> vec3f { return x - floor(x * (1.0 / 289.0)) * 289.0; }
fn mod_289_2(x: vec2f) -> vec2f { return x - floor(x * (1.0 / 289.0)) * 289.0; }
fn permute(x: vec3f) -> vec3f { return mod_289_3(((x * 34.0) + 1.0) * x); }

fn simplex_noise(v: vec2f) -> f32
{
    const C: vec4f = vec4f(0.211324865405187, 0.366025403784439,
                       -0.577350269189626, 0.024390243902439);
    var i: vec2f = floor(v + dot(v, C.yy));
    let x0: vec2f = v - i + dot(i, C.xx);
    let i1: vec2f = select(vec2f(0.0, 1.0), vec2f(1.0, 0.0), x0.x > x0.y);
    var x12: vec4f = x0.xyxy + C.xxzz;
    x12.x -= i1.x;
    x12.y -= i1.y;
    i = mod_289_2(i);
    let p: vec3f = permute(permute(i.y + vec3f(0.0, i1.y, 1.0))
                            + i.x + vec3f(0.0, i1.x, 1.0));
    var m: vec3f = max(0.5 - vec3f(dot(x0, x0), dot(x12.xy, x12.xy),
                            dot(x12.zw, x12.zw)), vec3f(0.0));
    m = m * m;
    m = m * m;
    let x: vec3f = 2.0 * fract(p * C.www) - 1.0;
    let h: vec3f = abs(x) - 0.5;
    let ox: vec3f = floor(x + 0.5);
    let a0: vec3f = x - ox;
    m *= 1.79284291400159 - 0.85373472095314 * (a0 * a0 + h * h);
    var g: vec3f;
    g.x = a0.x * x0.x + h.x * x0.y;
    let Gyz: vec2f = a0.yz * x12.xz + h.yz * x12.yw; 
    g.y = Gyz.x;
    g.z = Gyz.y;
    return 130.0 * dot(m, g);
}`
        }, {
            name: "FBM (Fractal Brownian Motion)",
            description: "Layered noise with configurable octaves",
            code: `fn fbm(p: vec2f) -> f32
{
    var value: f32 = 0.0;
    var amplitude: f32 = 0.5;
    var frequency: f32 = 1.0;
    for (var i: i32 = 0; i < 6; i++)
    {
        value += amplitude * value_noise(p * frequency);
        frequency *= 2.0;
        amplitude *= 0.5;
    }
    return value;
}`
        }, {
            name: "Voronoi / Worley Noise",
            description: "Cell noise with distance to nearest point",
            code: `fn voronoi(p: vec2f) -> f32
{
    let i: vec2f = floor(p);
    let f: vec2f = fract(p);
    var min_dist: f32 = 1.0;
    for (var y: i32 = -1; y <= 1; y++)
    {
        for (var x: i32 = -1; x <= 1; x++)
        {
            let neighbor: vec2f = vec2f(f32(x), f32(y));
            let point: vec2f = hash_22(i + neighbor);
            let diff: vec2f = neighbor + point - f;
            let dist: f32 = length(diff);
            min_dist = min(min_dist, dist);
        }
    }
    return min_dist;
}`
        }, {
            name: "Turbulence",
            description: "Fake fluid dynamics via layered rotated sine waves",
            author: "Xor (mini.gmshaders.com/p/turbulence)",
            code: `// Turbulence: approximate fluid motion by layering
// rotated sine‑wave displacements.
//
// Parameters
//   TURB_NUM   – number of octaves (more = finer detail)
//   TURB_AMP   – overall wave amplitude (0–1)
//   TURB_SPEED – scroll speed (set 0 for static)
//   TURB_FREQ  – starting frequency
//   TURB_EXP   – frequency multiplier per octave

#define TURB_NUM   10
#define TURB_AMP   0.7
#define TURB_SPEED 0.3
#define TURB_FREQ  2.0
#define TURB_EXP   1.4

fn turbulence(p: vec2f, time: f32) -> vec2f
{
    var freq: f32 = TURB_FREQ;
    var rot: mat2x2f = mat2x2f(0.6, -0.8, 0.8, 0.6);
    var pos: vec2f = p;

    for (var i: i32 = 0; i < TURB_NUM; i++)
    {
        let phase: f32 = freq * (pos * rot).y + TURB_SPEED * time + f32(i);
        pos += TURB_AMP * rot[0] * sin(phase) / freq;
        rot *= mat2x2f(0.6, -0.8, 0.8, 0.6);
        freq *= TURB_EXP;
    }
    return pos;
}`
        }, {
            name: "Dot Noise 3D",
            description: "Cheap aperiodic 3D noise using golden‑ratio gyroids",
            author: "Xor (mini.gmshaders.com/p/dot-noise)",
            code: `// Dot Noise: a fast alternative to 3D value / Perlin / simplex
// noise.  Uses a gyroid formula with golden‑ratio rotation so
// the pattern never repeats.  Returns a value in [‑3, +3].

fn dot_noise(p: vec3f) -> f32
{
    const PHI: f32 = 1.618033988;

    // Golden‑angle rotation on the vec3(1, φ, φ²) axis
    const GOLD: mat3x3f = mat3x3f(
        -0.571464913,  0.814921382,  0.096597072,
        -0.278044873, -0.303026659,  0.911518454,
         0.772087367,  0.494042493,  0.399753815);

    return dot(cos(GOLD * p), sin(PHI * p * GOLD));
}

// Fractal layering for richer detail (returns ~[‑1, +1])
fn dot_noise_fbm(p: vec3f) -> f32
{
    const PHI: f32 = 1.618033988;
    const GOLD: mat3x3f = mat3x3f(
        -0.571464913,  0.814921382,  0.096597072,
        -0.278044873, -0.303026659,  0.911518454,
         0.772087367,  0.494042493,  0.399753815);

    var value: f32 = 0.0;
    var amplitude: f32 = 0.5;
    var pos: vec3f = p;
    for (var i: i32 = 0; i < 5; i++)
    {
        value += amplitude * dot(cos(GOLD * pos), sin(PHI * pos * GOLD)) / 3.0;
        pos = GOLD * pos * 2.0;
        amplitude *= 0.5;
    }
    return value;
}`
        }]
    }, {
        name: "Color",
        icon: "Contrast",
        snippets: [{
            name: "HSV to RGB",
            description: "Convert hue/saturation/value to RGB",
            code: `vec3 hsv_to_rgb(vec3 c)
{
    vec3 p = abs(fract(c.xxx + vec3(1.0, 2.0/3.0, 1.0/3.0)) * 6.0 - 3.0);
    return c.z * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), c.y);
}`
        }, {
            name: "RGB to HSV",
            description: "Convert RGB to hue/saturation/value",
            author: "Sam Hocevar",
            code: `vec3 rgb_to_hsv(vec3 c)
{
    vec4 K = vec4(0.0, -1.0/3.0, 2.0/3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}`
        }, {
            name: "Cosine Palette",
            description: "Cosine gradient palette (4 vec3 parameters)",
            author: "Inigo Quilez (iquilezles.org/articles/palettes)",
            code: `vec3 palette(float t, vec3 a, vec3 b, vec3 c, vec3 d)
{
    return a + b * cos(TAU * (c * t + d));
}

// Example presets:
// Rainbow:   palette(t, vec3(0.5), vec3(0.5), vec3(1.0), vec3(0.0, 0.33, 0.67))
// Warm:      palette(t, vec3(0.5), vec3(0.5), vec3(1.0), vec3(0.0, 0.1, 0.2))
// Cool:      palette(t, vec3(0.5), vec3(0.5), vec3(1.0, 1.0, 0.5), vec3(0.8, 0.9, 0.3))`
        }, {
            name: "sRGB Conversion",
            description: "Accurate linear ↔ sRGB with the piecewise IEC 61966-2-1 transfer curve",
            code: `vec3 linear_to_srgb(vec3 c)
{
    vec3 lo = 12.92 * c;
    vec3 hi = 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055;
    return mix(lo, hi, step(vec3(0.0031308), c));
}

vec3 srgb_to_linear(vec3 c)
{
    vec3 lo = c / 12.92;
    vec3 hi = pow((c + 0.055) / 1.055, vec3(2.4));
    return mix(lo, hi, step(vec3(0.04045), c));
}`
        }, {
            name: "Tonemapping",
            description: "ACES filmic and Reinhard tonemapping operators for HDR → LDR",
            code: `// ACES filmic tone mapping (Krzysztof Narkowicz fit)
vec3 tonemap_aces(vec3 x)
{
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

// Reinhard tonemapping
vec3 tonemap_reinhard(vec3 x)
{
    return x / (1.0 + x);
}

// Extended Reinhard with white point control
vec3 tonemap_reinhard_ext(vec3 x, float white)
{
    vec3 num = x * (1.0 + x / (white * white));
    return num / (1.0 + x);
}`,
            author: "Krzysztof Narkowicz (ACES), Erik Reinhard et al."
        }, {
            name: "Hue Rotation",
            description: "Rotate the hue of an RGB color",
            author: "W3C CSS Filters (hue-rotate matrix)",
            code: `vec3 hue_rotate(vec3 col, float angle)
{
    float s = sin(angle);
    float c = cos(angle);
    mat3 m = mat3(
        0.299 + 0.701*c + 0.168*s,  0.587 - 0.587*c + 0.330*s,  0.114 - 0.114*c - 0.497*s,
        0.299 - 0.299*c - 0.328*s,  0.587 + 0.413*c + 0.035*s,  0.114 - 0.114*c + 0.292*s,
        0.299 - 0.300*c + 1.250*s,  0.587 - 0.588*c - 1.050*s,  0.114 + 0.886*c - 0.203*s
    );
    return m * col;
}`
        }, {
            name: "Luminance",
            description: "Perceived brightness of an RGB color",
            author: "ITU-R BT.709 coefficients",
            code: `float luminance(vec3 col)
{
    return dot(col, vec3(0.2126, 0.7152, 0.0722));
}`
        }, {
            name: "OKLab (Linear RGB ↔ L,a,b)",
            description: "Perceptually uniform color space; use linear RGB (srgbToLinear first)",
            author: "Björn Ottosson (bottosson.github.io/posts/oklab)",
            code: `float ok_cbrt(float x)
{
    return sign(x) * pow(abs(x), 1.0 / 3.0);
}

vec3 rgb_to_oklab(vec3 c)
{
    float l = 0.4122214708 * c.r + 0.5363325363 * c.g + 0.0514459929 * c.b;
    float m = 0.2119034982 * c.r + 0.6806995451 * c.g + 0.1073969566 * c.b;
    float s = 0.0883024619 * c.r + 0.2817188376 * c.g + 0.6299787005 * c.b;
    float l_ = ok_cbrt(l), m_ = ok_cbrt(m), s_ = ok_cbrt(s);
    return vec3(
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    );
}

vec3 oklab_to_rgb(vec3 c)
{
    float l_ = c.r + 0.3963377774 * c.g + 0.2158037573 * c.b;
    float m_ = c.r - 0.1055613458 * c.g - 0.0638541728 * c.b;
    float s_ = c.r - 0.0894841775 * c.g - 1.2914855480 * c.b;
    float l = l_ * l_ * l_, m = m_ * m_ * m_, s = s_ * s_ * s_;
    return vec3(
        4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    );
}`
        }, {
            name: "OKLch (Lightness, Chroma, Hue)",
            description: "Cylindrical OKLab; L=lightness, C=chroma, H=hue in radians",
            author: "Björn Ottosson (bottosson.github.io/posts/oklab)",
            code: `vec3 oklab_to_oklch(vec3 ok)
{
    float L = ok.r;
    float C = sqrt(ok.g * ok.g + ok.b * ok.b);
    float H = atan(ok.b, ok.g);
    return vec3(L, C, H);
}

vec3 oklch_to_oklab(vec3 lch)
{
    float L = lch.r, C = lch.g, H = lch.b;
    return vec3(L, C * cos(H), C * sin(H));
}

// Full pipeline: sRGB → linear → OKLab → OKLch
// vec3 lab = rgb_to_oklab(srgb_to_linear(col));
// vec3 lch = oklab_to_oklch(lab);`
        }]
    }, {
        name: "Math",
        icon: "Function",
        snippets: [{
            name: "2D Rotation",
            description: "Rotate a vec2 by an angle in radians",
            code: `mat2 rotate_2d(float angle)
{
    float s = sin(angle);
    float c = cos(angle);
    return mat2(c, -s, s, c);
}

// Usage: p.xy *= rotate_2d(angle);`
        }, {
            name: "3D Rotation (Euler Angles)",
            description: "Rotate a vec3 with roll, pitch, and yaw via successive 2D rotations",
            author: "Xor (mini.gmshaders.com/p/3d-rotation)",
            code: `mat2 rot2d(float a) { return mat2(cos(a), -sin(a), sin(a), cos(a)); }

vec3 euler_rotate(vec3 v, float roll, float pitch, float yaw)
{
    v.yz *= rot2d(roll);
    v.xz *= rot2d(pitch);
    v.xy *= rot2d(yaw);
    return v;
}`
        }, {
            name: "3D Rotation (Axis-Angle)",
            description: "Rotate a vec3 by an angle around an arbitrary unit axis",
            author: "Xor / Fabrice Neyret (mini.gmshaders.com/p/3d-rotation)",
            code: `vec3 axis_rotate(vec3 v, vec3 axis, float angle)
{
    return mix(dot(v, axis) * axis, v, cos(angle))
         + sin(angle) * cross(v, axis);
}`
        }, {
            name: "Remap",
            description: "Map a value from one range to another",
            code: `float remap(float value, float in_min, float in_max, float out_min, float out_max)
{
    return out_min + (out_max - out_min) * (value - in_min) / (in_max - in_min);
}`
        }, {
            name: "Smooth Min / Max",
            description: "Smooth minimum and maximum blending",
            author: "Inigo Quilez (iquilezles.org/articles/smin)",
            code: `float smin(float a, float b, float k)
{
    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

float smax(float a, float b, float k)
{
    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(a, b, h) + k * h * (1.0 - h);
}`
        }, {
            name: "Smootherstep",
            description: "Improved smoothstep (C2 continuous)",
            author: "Ken Perlin",
            code: `float smootherstep(float edge0, float edge1, float x)
{
    x = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0);
}`
        }, {
            name: "Polar Coordinates",
            description: "Convert between cartesian and polar coords",
            code: `vec2 to_polar(vec2 p)
{
    return vec2(length(p), atan(p.y, p.x));
}

vec2 to_cartesian(vec2 polar)
{
    return vec2(polar.x * cos(polar.y), polar.x * sin(polar.y));
}`
        }, {
            name: "Repeat / Tile",
            description: "Repeat space for infinite tiling",
            code: `vec2 repeat(vec2 p, vec2 size)
{
    return mod(p + size * 0.5, size) - size * 0.5;
}

// Mirror repeat (ping-pong)
vec2 mirror_repeat(vec2 p, vec2 size)
{
    vec2 half = size * 0.5;
    vec2 q = mod(p + half, size * 2.0) - size;
    return half - abs(q);
}`
        }, {
            name: "Anti-Aliasing (SDF)",
            description: "Smooth edges for SDFs by matching gradient to pixel scale",
            author: "Xor (mini.gmshaders.com/p/antialiasing)",
            code: `// For SDFs with a known pixel scale (texel size).
// Blends 0→1 over the width of one pixel at the edge.
float antialias_sdf(float dist, float texel)
{
    return clamp(dist / texel + 0.5, 0.0, 1.0);
}`
        }, {
            name: "Anti-Aliasing (Derivative)",
            description: "Smooth edges from any continuous function using fwidth / derivatives",
            author: "Xor (mini.gmshaders.com/p/antialiasing)",
            code: `// Automatic edge smoothing via fwidth (cheap, handles most cases)
float antialias(float d)
{
    float w = fwidth(d);
    float scale = w > 0.0 ? 1.0 / w : 1e7;
    return clamp(0.5 + 0.5 * scale * d, 0.0, 1.0);
}

// Higher-quality variant using gradient length (L2 norm)
float antialias_l2(float d)
{
    vec2 dxy = vec2(dFdx(d), dFdy(d));
    float w = length(dxy);
    float scale = w > 0.0 ? 1.0 / w : 1e7;
    return clamp(0.5 + 0.7 * scale * d, 0.0, 1.0);
}

// Manual derivatives version (for discontinuous gradients)
float antialias_l2_dxy(float d, vec2 dxy)
{
    float w = length(dxy);
    float scale = w > 0.0 ? 1.0 / w : 1e7;
    return clamp(0.5 + 0.7 * scale * d, 0.0, 1.0);
}`
        }]
    }, {
        name: "2D Shapes",
        icon: "Hexagon",
        snippets: [{
            name: "Circle",
            description: "Signed distance to a circle",
            author: "Inigo Quilez (iquilezles.org)",
            code: `float sd_circle(vec2 p, float r)
{
    return length(p) - r;
}`
        }, {
            name: "Box",
            description: "Signed distance to an axis-aligned box",
            author: "Inigo Quilez (iquilezles.org)",
            code: `float sd_box(vec2 p, vec2 b)
{
    vec2 d = abs(p) - b;
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}`
        }, {
            name: "Rounded Box",
            description: "Box with individually rounded corners",
            author: "Inigo Quilez (iquilezles.org)",
            code: `float sd_rounded_box(vec2 p, vec2 b, vec4 r)
{
    r.xy = (p.x > 0.0) ? r.xy : r.zw;
    r.x  = (p.y > 0.0) ? r.x  : r.y;
    vec2 q = abs(p) - b + r.x;
    return min(max(q.x, q.y), 0.0) + length(max(q, 0.0)) - r.x;
}`
        }, {
            name: "Line Segment",
            description: "Distance to a line segment between two points",
            author: "Inigo Quilez (iquilezles.org)",
            code: `float sd_segment(vec2 p, vec2 a, vec2 b)
{
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}`
        }, {
            name: "Equilateral Triangle",
            description: "SDF for an equilateral triangle",
            author: "Inigo Quilez (iquilezles.org)",
            code: `float sd_triangle(vec2 p, float r)
{
    const float k = sqrt(3.0);
    p.x = abs(p.x) - r;
    p.y = p.y + r / k;
    if (p.x + k * p.y > 0.0) p = vec2(p.x - k * p.y, -k * p.x - p.y) / 2.0;
    p.x -= clamp(p.x, -2.0 * r, 0.0);
    return -length(p) * sign(p.y);
}`
        }, {
            name: "Ring",
            description: "Signed distance to a ring (annulus)",
            author: "Inigo Quilez (iquilezles.org)",
            code: `float sd_ring(vec2 p, float r, float thickness)
{
    return abs(length(p) - r) - thickness;
}`
        }, {
            name: "Capsule",
            description: "Segment with radius (pill shape)",
            author: "Inigo Quilez (iquilezles.org)",
            code: `float sd_capsule(vec2 p, vec2 a, vec2 b, float r)
{
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}`
        }, {
            name: "Vesica",
            description: "Lens shape from two overlapping circles; w = width, h = height",
            author: "Inigo Quilez (iquilezles.org)",
            code: `float sd_vesica(vec2 p, float w, float h)
{
    vec2 p2 = abs(p);
    float d = 0.5 * (w * w - h * h) / h;
    vec3 c = (w * p2.y < d * (p2.x - w))
        ? vec3(0.0, w, 0.0)
        : vec3(-d, 0.0, d + h);
    return length(p2 - c.yx) - c.z;
}`
        }, {
            name: "Hexagon",
            description: "Regular hexagon (flat-top), r = half-width",
            author: "Inigo Quilez (iquilezles.org)",
            code: `float sd_hexagon(vec2 p, float r)
{
    const vec3 k = vec3(-0.866025404, 0.5, 0.577350269);
    p = abs(p);
    p -= 2.0 * min(dot(k.xy, p), 0.0) * k.xy;
    p -= vec2(clamp(p.x, -k.z * r, k.z * r), r);
    return length(p) * sign(p.y);
}`
        }, {
            name: "Pentagon",
            description: "Regular pentagon",
            author: "Inigo Quilez (iquilezles.org)",
            code: `float sd_pentagon(vec2 p, float r)
{
    const vec3 k = vec3(0.809016994, 0.587785252, 0.726542528);
    p.x = abs(p.x);
    p -= 2.0 * min(dot(vec2(-k.x, k.y), p), 0.0) * vec2(-k.x, k.y);
    p -= 2.0 * min(dot(vec2(k.x, k.y), p), 0.0) * vec2(k.x, k.y);
    p -= vec2(clamp(p.x, -r * k.z, r * k.z), r);
    return length(p) * sign(p.y);
}`
        }, {
            name: "Star (5-point)",
            description: "Regular 5-point star (pentagram)",
            author: "Inigo Quilez (iquilezles.org)",
            code: `float sd_pentagram(vec2 p, float r)
{
    const float k1 = 0.809016994;
    const float k2 = 0.309016994;
    const vec2 v1 = vec2(k1, -0.587785252);
    const vec2 v2 = vec2(-k1, -0.587785252);
    const vec2 v3 = vec2(k2, -0.951056516);
    p.x = abs(p.x);
    p -= 2.0 * max(dot(v1, p), 0.0) * v1;
    p -= 2.0 * max(dot(v2, p), 0.0) * v2;
    p.x = abs(p.x);
    p.y -= r;
    return length(p - v3 * clamp(dot(p, v3), 0.0, 0.726542528 * r)) * sign(p.y * v3.x - p.x * v3.y);
}`
        }, {
            name: "Arc",
            description: "Circular arc; sc = sin/cos of half aperture, ra = radius, rb = thickness",
            author: "Inigo Quilez (iquilezles.org)",
            code: `float sd_arc(vec2 p, vec2 sc, float ra, float rb)
{
    p.x = abs(p.x);
    return ((sc.y * p.x > sc.x * p.y)
        ? length(p - sc * ra)
        : abs(length(p) - ra)) - rb;
}`
        }, {
            name: "Ellipse",
            description: "Axis-aligned ellipse; ab = semi-axes",
            author: "Inigo Quilez (iquilezles.org)",
            code: `float sd_ellipse(vec2 p, vec2 ab)
{
    p = abs(p);
    if (p.x > p.y) { p = p.yx; ab = ab.yx; }
    float l = ab.y * ab.y - ab.x * ab.x;
    float m = ab.x * p.x / l;
    float m2 = m * m;
    float n = ab.y * p.y / l;
    float n2 = n * n;
    float c = (m2 + n2 - 1.0) / 3.0;
    float c3 = c * c * c;
    float q = c3 + m2 * n2 * 2.0;
    float d = c3 + m2 * n2;
    float g = m + m * n2;
    float co;
    if (d < 0.0) {
        float h = acos(q / c3) / 3.0;
        float s = cos(h), t = sin(h) * sqrt(3.0);
        float rx = sqrt(-c * (s + t + 2.0) + m2);
        float ry = sqrt(-c * (s - t + 2.0) + m2);
        co = (ry + sign(l) * rx + abs(g) / (rx * ry) - m) * 0.5;
    } else {
        float h = 2.0 * m * n * sqrt(d);
        float s = sign(q + h) * pow(abs(q + h), 1.0 / 3.0);
        float u = sign(q - h) * pow(abs(q - h), 1.0 / 3.0);
        float rx = -s - u - c * 4.0 + 2.0 * m2;
        float ry = (s - u) * sqrt(3.0);
        float rm = sqrt(rx * rx + ry * ry);
        co = (ry / sqrt(rm - rx) + 2.0 * g / rm - m) * 0.5;
    }
    vec2 r = ab * vec2(co, sqrt(1.0 - co * co));
    return length(r - p) * sign(p.y - r.y);
}`
        }, {
            name: "SDF Operations",
            description: "Union, intersection, subtraction, and smooth variants",
            author: "Inigo Quilez (iquilezles.org)",
            code: `float op_union(float d1, float d2) { return min(d1, d2); }
float op_intersect(float d1, float d2) { return max(d1, d2); }
float op_subtract(float d1, float d2) { return max(-d1, d2); }

float op_smooth_union(float d1, float d2, float k)
{
    float h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return mix(d2, d1, h) - k * h * (1.0 - h);
}

float op_smooth_intersect(float d1, float d2, float k)
{
    float h = clamp(0.5 - 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return mix(d2, d1, h) + k * h * (1.0 - h);
}

float op_smooth_subtract(float d1, float d2, float k)
{
    float h = clamp(0.5 - 0.5 * (d2 + d1) / k, 0.0, 1.0);
    return mix(d2, -d1, h) + k * h * (1.0 - h);
}`
        }]
    }, {
        name: "3D Shapes",
        icon: "Pyramid",
        snippets: [{
            name: "3D SDF Sphere",
            description: "Signed distance to a sphere in 3D",
            author: "Inigo Quilez (iquilezles.org)",
            code: `float sd_sphere(vec3 p, float r)
{
    return length(p) - r;
}`
        }, {
            name: "3D SDF Box",
            description: "Signed distance to an axis-aligned 3D box",
            author: "Inigo Quilez (iquilezles.org)",
            code: `float sd_box_3d(vec3 p, vec3 b)
{
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}`
        }, {
            name: "3D SDF Torus",
            description: "Signed distance to a torus",
            author: "Inigo Quilez (iquilezles.org)",
            code: `float sd_torus(vec3 p, vec2 t)
{
    vec2 q = vec2(length(p.xz) - t.x, p.y);
    return length(q) - t.y;
}`
        }, {
            name: "3D Capsule",
            description: "Line segment with radius (pill)",
            author: "Inigo Quilez (iquilezles.org)",
            code: `float sd_capsule(vec3 p, vec3 a, vec3 b, float r)
{
    vec3 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}`
        }, {
            name: "3D Capped Cylinder",
            description: "Cylinder with flat ends (vertical axis)",
            author: "Inigo Quilez (iquilezles.org)",
            code: `float sd_capped_cylinder(vec3 p, float r, float h)
{
    vec2 d = abs(vec2(length(p.xz), p.y)) - vec2(r, h);
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}`
        }, {
            name: "3D Cone",
            description: "Cone; c = sin/cos of angle, h = height",
            author: "Inigo Quilez (iquilezles.org)",
            code: `float sd_cone(vec3 p, vec2 c, float h)
{
    vec2 q = h * vec2(c.x / c.y, -1.0);
    vec2 w = vec2(length(p.xz), p.y);
    vec2 a = w - q * clamp(dot(w, q) / dot(q, q), 0.0, 1.0);
    vec2 b = w - q * vec2(clamp(w.x / q.x, 0.0, 1.0), 1.0);
    float k = sign(q.y);
    float d = min(dot(a, a), dot(b, b));
    float s = max(k * (w.x * q.y - w.y * q.x), k * (w.y - q.y));
    return sqrt(d) * sign(s);
}`
        }, {
            name: "3D Plane",
            description: "Infinite plane; n = normal (normalized), h = distance from origin",
            author: "Inigo Quilez (iquilezles.org)",
            code: `float sd_plane(vec3 p, vec3 n, float h)
{
    return dot(p, n) + h;
}`
        }, {
            name: "3D Round Cone",
            description: "Cone with rounded ends (r1 at base, r2 at tip)",
            author: "Inigo Quilez (iquilezles.org)",
            code: `float sd_round_cone(vec3 p, float r1, float r2, float h)
{
    float b = (r1 - r2) / h;
    float a = sqrt(1.0 - b * b);
    vec2 q = vec2(length(p.xz), p.y);
    float k = dot(q, vec2(-b, a));
    if (k < 0.0) return length(q) - r1;
    if (k > a * h) return length(q - vec2(0.0, h)) - r2;
    return dot(q, vec2(a, b)) - r1;
}`
        }, {
            name: "3D Hexagonal Prism",
            description: "Hexagonal prism; h.x = half-width, h.y = half-height",
            author: "Inigo Quilez (iquilezles.org)",
            code: `float sd_hex_prism(vec3 p, vec2 h)
{
    const vec3 k = vec3(-0.8660254, 0.5, 0.57735);
    p = abs(p);
    p.xy -= 2.0 * min(dot(k.xy, p.xy), 0.0) * k.xy;
    vec2 d = vec2(
        length(p.xy - vec2(clamp(p.x, -k.z * h.x, k.z * h.x), h.x)) * sign(p.y - h.x),
        p.z - h.y
    );
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}`
        }, {
            name: "3D Round Box",
            description: "Box with rounded edges",
            author: "Inigo Quilez (iquilezles.org)",
            code: `float sd_round_box(vec3 p, vec3 b, float r)
{
    vec3 q = abs(p) - b + r;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}`
        }]
    }, {
        name: "Ray Marching",
        icon: "Globe",
        snippets: [{
            name: "Ray March Loop",
            description: "Basic sphere-tracing ray march loop",
            code: `#define MAX_STEPS 128
#define MAX_DIST  100.0
#define SURF_DIST 0.001

float ray_march(vec3 ro, vec3 rd)
{
    float t = 0.0;
    for (int i = 0; i < MAX_STEPS; i++)
    {
        vec3 p = ro + rd * t;
        float d = scene(p); // Replace with your scene SDF
        if (d < SURF_DIST) break;
        t += d;
        if (t > MAX_DIST) break;
    }
    return t;
}`
        }, {
            name: "Normal Estimation",
            description: "Compute surface normal from SDF gradient",
            code: `#define EPSILON 0.0001

vec3 get_normal(vec3 p)
{
    vec2 e = vec2(EPSILON, 0.0);
    return normalize(vec3(
        scene(p + e.xyy) - scene(p - e.xyy),
        scene(p + e.yxy) - scene(p - e.yxy),
        scene(p + e.yyx) - scene(p - e.yyx)
    ));
}`
        }, {
            name: "Basic Lighting",
            description: "Diffuse + specular Phong-style lighting",
            code: `vec3 lighting(vec3 p, vec3 normal, vec3 rd, vec3 light_dir, vec3 col)
{
    float diff = max(dot(normal, light_dir), 0.0);
    vec3 half_dir = normalize(light_dir - rd);
    float spec = pow(max(dot(normal, half_dir), 0.0), 32.0);
    vec3 ambient = 0.05 * col;
    return ambient + col * diff + vec3(1.0) * spec * 0.5;
}`
        }, {
            name: "Camera Setup",
            description: "Look-at camera matrix for ray marching",
            code: `mat3 camera(vec3 eye, vec3 target, float roll)
{
    vec3 cw = normalize(target - eye);
    vec3 cp = vec3(sin(roll), cos(roll), 0.0);
    vec3 cu = normalize(cross(cw, cp));
    vec3 cv = cross(cu, cw);
    return mat3(cu, cv, cw);
}`
        }]
    }];
export const WGSL_SHADER_CONSTANTS = [{
    name: "PI",
    value: "3.14159265359",
    detail: "π — ratio of circumference to diameter"
}, {
    name: "TAU",
    value: "6.28318530718",
    detail: "2π — full circle in radians"
}, {
    name: "HPI",
    value: "1.57079632679",
    detail: "π/2 — quarter turn"
}, {
    name: "SQRT_HALF",
    value: "0.70710678118",
    detail: "√(1/2)"
}, {
    name: "SQRT2",
    value: "1.41421356237",
    detail: "√2"
}, {
    name: "SQRT_THIRD",
    value: "0.57735026919",
    detail: "√(1/3)"
}, {
    name: "PHI",
    value: "1.61803398875",
    detail: "Golden ratio (φ)"
}, {
    name: "GOLDEN_ANGLE",
    value: "2.39996322972",
    detail: "Golden angle in radians ≈ 137.508°"
}, {
    name: "SQRT_3_4",
    value: "0.86602540378",
    detail: "√(3/4) — simplex height factor"
}, {
    name: "F2",
    value: "0.36602540378",
    detail: "2D simplex skew: (√3−1)/2"
}, {
    name: "G2",
    value: "0.21132486541",
    detail: "2D simplex unskew: (3−√3)/6"
}, {
    name: "F3",
    value: "0.33333333333",
    detail: "3D simplex skew: 1/3"
}, {
    name: "G3",
    value: "0.16666666667",
    detail: "3D simplex unskew: 1/6"
}, {
    name: "F4",
    value: "0.30901699437",
    detail: "4D simplex skew: (√5−1)/4"
}, {
    name: "G4",
    value: "0.13819660113",
    detail: "4D simplex unskew: (1−1/√5)/4"
}]
export const WGSL_SHADER_FUNCTIONS = [{
    name: "sin",
    signature: "sin(x)",
    description: "Sine of angle in radians"
}, {
    name: "cos",
    signature: "cos(x)",
    description: "Cosine of angle in radians"
}, {
    name: "tan",
    signature: "tan(x)",
    description: "Tangent of angle in radians"
}, {
    name: "asin",
    signature: "asin(x)",
    description: "Arc sine, returns radians"
}, {
    name: "acos",
    signature: "acos(x)",
    description: "Arc cosine, returns radians"
}, {
    name: "atan",
    signature: "atan(y, x)",
    description: "Arc tangent of y/x"
}, {
    name: "radians",
    signature: "radians(degrees)",
    description: "Convert degrees to radians"
}, {
    name: "degrees",
    signature: "degrees(radians)",
    description: "Convert radians to degrees"
}, {
    name: "pow",
    signature: "pow(x, y)",
    description: "x raised to the power y"
}, {
    name: "exp",
    signature: "exp(x)",
    description: "e raised to the power x"
}, {
    name: "exp2",
    signature: "exp2(x)",
    description: "2 raised to the power x"
}, {
    name: "log",
    signature: "log(x)",
    description: "Natural logarithm"
}, {
    name: "log2",
    signature: "log2(x)",
    description: "Base-2 logarithm"
}, {
    name: "sqrt",
    signature: "sqrt(x)",
    description: "Square root"
}, {
    name: "inversesqrt",
    signature: "inversesqrt(x)",
    description: "1 / sqrt(x)"
}, {
    name: "abs",
    signature: "abs(x)",
    description: "Absolute value"
}, {
    name: "sign",
    signature: "sign(x)",
    description: "Returns -1, 0, or 1"
}, {
    name: "floor",
    signature: "floor(x)",
    description: "Round down to integer"
}, {
    name: "ceil",
    signature: "ceil(x)",
    description: "Round up to integer"
}, {
    name: "fract",
    signature: "fract(x)",
    description: "Fractional part (x - floor(x))"
}, {
    name: "mod",
    signature: "mod(x, y)",
    description: "Modulo (x - y * floor(x/y))"
}, {
    name: "min",
    signature: "min(x, y)",
    description: "Minimum of x and y"
}, {
    name: "max",
    signature: "max(x, y)",
    description: "Maximum of x and y"
}, {
    name: "clamp",
    signature: "clamp(x, min, max)",
    description: "Clamp x between min and max"
}, {
    name: "mix",
    signature: "mix(a, b, t)",
    description: "Linear interpolation: a*(1-t) + b*t"
}, {
    name: "step",
    signature: "step(edge, x)",
    description: "0.0 if x < edge, else 1.0"
}, {
    name: "smoothstep",
    signature: "smoothstep(a, b, x)",
    description: "Hermite interpolation between a and b"
}, {
    name: "length",
    signature: "length(v)",
    description: "Length of vector"
}, {
    name: "distance",
    signature: "distance(a, b)",
    description: "Distance between two points"
}, {
    name: "dot",
    signature: "dot(a, b)",
    description: "Dot product"
}, {
    name: "cross",
    signature: "cross(a, b)",
    description: "Cross product (vec3)"
}, {
    name: "normalize",
    signature: "normalize(v)",
    description: "Unit vector in same direction"
}, {
    name: "reflect",
    signature: "reflect(I, N)",
    description: "Reflection of incident vector"
}, {
    name: "refract",
    signature: "refract(I, N, eta)",
    description: "Refraction of incident vector"
}, {
    name: "faceforward",
    signature: "faceforward(N, I, Nref)",
    description: "Flip N if facing away from I"
}, {
    name: "texture",
    signature: "texture(sampler, uv)",
    description: "Sample a texture at UV coordinates"
}, {
    name: "dFdx",
    signature: "dFdx(p)",
    description: "Partial derivative in x"
}, {
    name: "dFdy",
    signature: "dFdy(p)",
    description: "Partial derivative in y"
}, {
    name: "fwidth",
    signature: "fwidth(p)",
    description: "abs(dFdx(p)) + abs(dFdy(p))"
}];

export const GLSL_CODE_LIBRARY = [
    {
        name: "Constants",
        icon: "Pi",
        snippets: [
            ...CONSTANTS_LIBRARY_SNIPPETS,
            {
                name: "Golden Ratio",
                description: "PHI, golden angle, 2D & 3D rotation matrices",
                code: `#define PHI          1.61803398875
#define GOLDEN_ANGLE 2.39996322972

const mat2 GOLDEN_ROT2 = mat2(
    -0.73736887808,  0.67549029426,
    -0.67549029426, -0.73736887808);

const mat3 GOLDEN_ROT3 = mat3(
    -0.571464913, +0.814921382, +0.096597072,
    -0.278044873, -0.303026659, +0.911518454,
    +0.772087367, +0.494042493, +0.399753815);`
            }
        ]
    },
    {
        name: "Noise",
        icon: "Hash",
        snippets: [{
            name: "Hash (1D → 1D)",
            description: "Simple pseudorandom hash from a float",
            code: `float hash_11(float p)
{
    p = fract(p * 0.1031);
    p *= p + 33.33;
    p *= p + p;
    return fract(p);
}`
        }, {
            name: "Hash (2D → 1D)",
            description: "Pseudorandom float from a vec2 seed",
            code: `float hash_21(vec2 p)
{
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}`
        }, {
            name: "Hash (2D → 2D)",
            description: "Pseudorandom vec2 from a vec2 seed",
            code: `vec2 hash_22(vec2 p)
{
    vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}`
        }, {
            name: "Value Noise 2D",
            description: "Smooth interpolated value noise",
            code: `float value_noise(vec2 p)
{
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash_21(i);
    float b = hash_21(i + vec2(1.0, 0.0));
    float c = hash_21(i + vec2(0.0, 1.0));
    float d = hash_21(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}`
        }, {
            name: "Simplex Noise 2D",
            description: "Classic 2D simplex noise",
            author: "Ashima Arts, Stefan Gustavson (webgl-noise)",
            code: `vec3 mod_289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec2 mod_289(vec2 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec3 permute(vec3 x) { return mod_289(((x * 34.0) + 1.0) * x); }

float simplex_noise(vec2 v)
{
    const vec4 C = vec4(0.211324865405187, 0.366025403784439,
                       -0.577350269189626, 0.024390243902439);
    vec2 i = floor(v + dot(v, C.yy));
    vec2 x0 = v - i + dot(i, C.xx);
    vec2 i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec4 x12 = x0.xyxy + C.xxzz;
    x12.xy -= i1;
    i = mod_289(i);
    vec3 p = permute(permute(i.y + vec3(0.0, i1.y, 1.0))
                            + i.x + vec3(0.0, i1.x, 1.0));
    vec3 m = max(0.5 - vec3(dot(x0, x0), dot(x12.xy, x12.xy),
                            dot(x12.zw, x12.zw)), 0.0);
    m = m * m;
    m = m * m;
    vec3 x = 2.0 * fract(p * C.www) - 1.0;
    vec3 h = abs(x) - 0.5;
    vec3 ox = floor(x + 0.5);
    vec3 a0 = x - ox;
    m *= 1.79284291400159 - 0.85373472095314 * (a0 * a0 + h * h);
    vec3 g;
    g.x = a0.x * x0.x + h.x * x0.y;
    g.yz = a0.yz * x12.xz + h.yz * x12.yw;
    return 130.0 * dot(m, g);
}`
        }, {
            name: "FBM (Fractal Brownian Motion)",
            description: "Layered noise with configurable octaves",
            code: `float fbm(vec2 p)
{
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;
    for (int i = 0; i < 6; i++)
    {
        value += amplitude * value_noise(p * frequency);
        frequency *= 2.0;
        amplitude *= 0.5;
    }
    return value;
}`
        }, {
            name: "Voronoi / Worley Noise",
            description: "Cell noise with distance to nearest point",
            code: `float voronoi(vec2 p)
{
    vec2 i = floor(p);
    vec2 f = fract(p);
    float min_dist = 1.0;
    for (int y = -1; y <= 1; y++)
    {
        for (int x = -1; x <= 1; x++)
        {
            vec2 neighbor = vec2(float(x), float(y));
            vec2 point = hash_22(i + neighbor);
            vec2 diff = neighbor + point - f;
            float dist = length(diff);
            min_dist = min(min_dist, dist);
        }
    }
    return min_dist;
}`
        }, {
            name: "Turbulence",
            description: "Fake fluid dynamics via layered rotated sine waves",
            author: "Xor (mini.gmshaders.com/p/turbulence)",
            code: `// Turbulence: approximate fluid motion by layering
// rotated sine‑wave displacements.
//
// Parameters
//   TURB_NUM   – number of octaves (more = finer detail)
//   TURB_AMP   – overall wave amplitude (0–1)
//   TURB_SPEED – scroll speed (set 0 for static)
//   TURB_FREQ  – starting frequency
//   TURB_EXP   – frequency multiplier per octave

#define TURB_NUM   10.0
#define TURB_AMP   0.7
#define TURB_SPEED 0.3
#define TURB_FREQ  2.0
#define TURB_EXP   1.4

vec2 turbulence(vec2 pos, float time)
{
    float freq = TURB_FREQ;
    mat2 rot = mat2(0.6, -0.8, 0.8, 0.6);

    for (float i = 0.0; i < TURB_NUM; i++)
    {
        float phase = freq * (pos * rot).y + TURB_SPEED * time + i;
        pos += TURB_AMP * rot[0] * sin(phase) / freq;
        rot *= mat2(0.6, -0.8, 0.8, 0.6);
        freq *= TURB_EXP;
    }
    return pos;
}`
        }, {
            name: "Dot Noise 3D",
            description: "Cheap aperiodic 3D noise using golden‑ratio gyroids",
            author: "Xor (mini.gmshaders.com/p/dot-noise)",
            code: `// Dot Noise: a fast alternative to 3D value / Perlin / simplex
// noise.  Uses a gyroid formula with golden‑ratio rotation so
// the pattern never repeats.  Returns a value in [‑3, +3].

float dot_noise(vec3 p)
{
    const float PHI = 1.618033988;

    // Golden‑angle rotation on the vec3(1, φ, φ²) axis
    const mat3 GOLD = mat3(
        -0.571464913, +0.814921382, +0.096597072,
        -0.278044873, -0.303026659, +0.911518454,
        +0.772087367, +0.494042493, +0.399753815);

    return dot(cos(GOLD * p), sin(PHI * p * GOLD));
}

// Fractal layering for richer detail (returns ~[‑1, +1])
float dot_noise_fbm(vec3 p)
{
    const float PHI = 1.618033988;
    const mat3 GOLD = mat3(
        -0.571464913, +0.814921382, +0.096597072,
        -0.278044873, -0.303026659, +0.911518454,
        +0.772087367, +0.494042493, +0.399753815);

    float value = 0.0;
    float amplitude = 0.5;
    for (int i = 0; i < 5; i++)
    {
        value += amplitude * dot(cos(GOLD * p), sin(PHI * p * GOLD)) / 3.0;
        p = GOLD * p * 2.0;
        amplitude *= 0.5;
    }
    return value;
}`
        }]
    }, {
        name: "Color",
        icon: "Contrast",
        snippets: [{
            name: "HSV to RGB",
            description: "Convert hue/saturation/value to RGB",
            code: `vec3 hsv_to_rgb(vec3 c)
{
    vec3 p = abs(fract(c.xxx + vec3(1.0, 2.0/3.0, 1.0/3.0)) * 6.0 - 3.0);
    return c.z * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), c.y);
}`
        }, {
            name: "RGB to HSV",
            description: "Convert RGB to hue/saturation/value",
            author: "Sam Hocevar",
            code: `vec3 rgb_to_hsv(vec3 c)
{
    vec4 K = vec4(0.0, -1.0/3.0, 2.0/3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}`
        }, {
            name: "Cosine Palette",
            description: "Cosine gradient palette (4 vec3 parameters)",
            author: "Inigo Quilez (iquilezles.org/articles/palettes)",
            code: `vec3 palette(float t, vec3 a, vec3 b, vec3 c, vec3 d)
{
    return a + b * cos(TAU * (c * t + d));
}

// Example presets:
// Rainbow:   palette(t, vec3(0.5), vec3(0.5), vec3(1.0), vec3(0.0, 0.33, 0.67))
// Warm:      palette(t, vec3(0.5), vec3(0.5), vec3(1.0), vec3(0.0, 0.1, 0.2))
// Cool:      palette(t, vec3(0.5), vec3(0.5), vec3(1.0, 1.0, 0.5), vec3(0.8, 0.9, 0.3))`
        }, {
            name: "sRGB Conversion",
            description: "Accurate linear ↔ sRGB with the piecewise IEC 61966-2-1 transfer curve",
            code: `vec3 linear_to_srgb(vec3 c)
{
    vec3 lo = 12.92 * c;
    vec3 hi = 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055;
    return mix(lo, hi, step(vec3(0.0031308), c));
}

vec3 srgb_to_linear(vec3 c)
{
    vec3 lo = c / 12.92;
    vec3 hi = pow((c + 0.055) / 1.055, vec3(2.4));
    return mix(lo, hi, step(vec3(0.04045), c));
}`
        }, {
            name: "Tonemapping",
            description: "ACES filmic and Reinhard tonemapping operators for HDR → LDR",
            code: `// ACES filmic tone mapping (Krzysztof Narkowicz fit)
vec3 tonemap_aces(vec3 x)
{
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

// Reinhard tonemapping
vec3 tonemap_reinhard(vec3 x)
{
    return x / (1.0 + x);
}

// Extended Reinhard with white point control
vec3 tonemap_reinhard_ext(vec3 x, float white)
{
    vec3 num = x * (1.0 + x / (white * white));
    return num / (1.0 + x);
}`,
            author: "Krzysztof Narkowicz (ACES), Erik Reinhard et al."
        }, {
            name: "Hue Rotation",
            description: "Rotate the hue of an RGB color",
            author: "W3C CSS Filters (hue-rotate matrix)",
            code: `vec3 hue_rotate(vec3 col, float angle)
{
    float s = sin(angle);
    float c = cos(angle);
    mat3 m = mat3(
        0.299 + 0.701*c + 0.168*s,  0.587 - 0.587*c + 0.330*s,  0.114 - 0.114*c - 0.497*s,
        0.299 - 0.299*c - 0.328*s,  0.587 + 0.413*c + 0.035*s,  0.114 - 0.114*c + 0.292*s,
        0.299 - 0.300*c + 1.250*s,  0.587 - 0.588*c - 1.050*s,  0.114 + 0.886*c - 0.203*s
    );
    return m * col;
}`
        }, {
            name: "Luminance",
            description: "Perceived brightness of an RGB color",
            author: "ITU-R BT.709 coefficients",
            code: `float luminance(vec3 col)
{
    return dot(col, vec3(0.2126, 0.7152, 0.0722));
}`
        }, {
            name: "OKLab (Linear RGB ↔ L,a,b)",
            description: "Perceptually uniform color space; use linear RGB (srgbToLinear first)",
            author: "Björn Ottosson (bottosson.github.io/posts/oklab)",
            code: `float ok_cbrt(float x)
{
    return sign(x) * pow(abs(x), 1.0 / 3.0);
}

vec3 rgb_to_oklab(vec3 c)
{
    float l = 0.4122214708 * c.r + 0.5363325363 * c.g + 0.0514459929 * c.b;
    float m = 0.2119034982 * c.r + 0.6806995451 * c.g + 0.1073969566 * c.b;
    float s = 0.0883024619 * c.r + 0.2817188376 * c.g + 0.6299787005 * c.b;
    float l_ = ok_cbrt(l), m_ = ok_cbrt(m), s_ = ok_cbrt(s);
    return vec3(
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    );
}

vec3 oklab_to_rgb(vec3 c)
{
    float l_ = c.r + 0.3963377774 * c.g + 0.2158037573 * c.b;
    float m_ = c.r - 0.1055613458 * c.g - 0.0638541728 * c.b;
    float s_ = c.r - 0.0894841775 * c.g - 1.2914855480 * c.b;
    float l = l_ * l_ * l_, m = m_ * m_ * m_, s = s_ * s_ * s_;
    return vec3(
        4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    );
}`
        }, {
            name: "OKLch (Lightness, Chroma, Hue)",
            description: "Cylindrical OKLab; L=lightness, C=chroma, H=hue in radians",
            author: "Björn Ottosson (bottosson.github.io/posts/oklab)",
            code: `vec3 oklab_to_oklch(vec3 ok)
{
    float L = ok.r;
    float C = sqrt(ok.g * ok.g + ok.b * ok.b);
    float H = atan(ok.b, ok.g);
    return vec3(L, C, H);
}

vec3 oklch_to_oklab(vec3 lch)
{
    float L = lch.r, C = lch.g, H = lch.b;
    return vec3(L, C * cos(H), C * sin(H));
}

// Full pipeline: sRGB → linear → OKLab → OKLch
// vec3 lab = rgb_to_oklab(srgb_to_linear(col));
// vec3 lch = oklab_to_oklch(lab);`
        }]
    }, {
        name: "Math",
        icon: "Function",
        snippets: [{
            name: "2D Rotation",
            description: "Rotate a vec2 by an angle in radians",
            code: `mat2 rotate_2d(float angle)
{
    float s = sin(angle);
    float c = cos(angle);
    return mat2(c, -s, s, c);
}

// Usage: p.xy *= rotate_2d(angle);`
        }, {
            name: "3D Rotation (Euler Angles)",
            description: "Rotate a vec3 with roll, pitch, and yaw via successive 2D rotations",
            author: "Xor (mini.gmshaders.com/p/3d-rotation)",
            code: `mat2 rot2d(float a) { return mat2(cos(a), -sin(a), sin(a), cos(a)); }

vec3 euler_rotate(vec3 v, float roll, float pitch, float yaw)
{
    v.yz *= rot2d(roll);
    v.xz *= rot2d(pitch);
    v.xy *= rot2d(yaw);
    return v;
}`
        }, {
            name: "3D Rotation (Axis-Angle)",
            description: "Rotate a vec3 by an angle around an arbitrary unit axis",
            author: "Xor / Fabrice Neyret (mini.gmshaders.com/p/3d-rotation)",
            code: `vec3 axis_rotate(vec3 v, vec3 axis, float angle)
{
    return mix(dot(v, axis) * axis, v, cos(angle))
         + sin(angle) * cross(v, axis);
}`
        }, {
            name: "Remap",
            description: "Map a value from one range to another",
            code: `float remap(float value, float in_min, float in_max, float out_min, float out_max)
{
    return out_min + (out_max - out_min) * (value - in_min) / (in_max - in_min);
}`
        }, {
            name: "Smooth Min / Max",
            description: "Smooth minimum and maximum blending",
            author: "Inigo Quilez (iquilezles.org/articles/smin)",
            code: `float smin(float a, float b, float k)
{
    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

float smax(float a, float b, float k)
{
    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(a, b, h) + k * h * (1.0 - h);
}`
        }, {
            name: "Smootherstep",
            description: "Improved smoothstep (C2 continuous)",
            author: "Ken Perlin",
            code: `float smootherstep(float edge0, float edge1, float x)
{
    x = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0);
}`
        }, {
            name: "Polar Coordinates",
            description: "Convert between cartesian and polar coords",
            code: `vec2 to_polar(vec2 p)
{
    return vec2(length(p), atan(p.y, p.x));
}

vec2 to_cartesian(vec2 polar)
{
    return vec2(polar.x * cos(polar.y), polar.x * sin(polar.y));
}`
        }, {
            name: "Repeat / Tile",
            description: "Repeat space for infinite tiling",
            code: `vec2 repeat(vec2 p, vec2 size)
{
    return mod(p + size * 0.5, size) - size * 0.5;
}

// Mirror repeat (ping-pong)
vec2 mirror_repeat(vec2 p, vec2 size)
{
    vec2 half = size * 0.5;
    vec2 q = mod(p + half, size * 2.0) - size;
    return half - abs(q);
}`
        }, {
            name: "Anti-Aliasing (SDF)",
            description: "Smooth edges for SDFs by matching gradient to pixel scale",
            author: "Xor (mini.gmshaders.com/p/antialiasing)",
            code: `// For SDFs with a known pixel scale (texel size).
// Blends 0→1 over the width of one pixel at the edge.
float antialias_sdf(float dist, float texel)
{
    return clamp(dist / texel + 0.5, 0.0, 1.0);
}`
        }, {
            name: "Anti-Aliasing (Derivative)",
            description: "Smooth edges from any continuous function using fwidth / derivatives",
            author: "Xor (mini.gmshaders.com/p/antialiasing)",
            code: `// Automatic edge smoothing via fwidth (cheap, handles most cases)
float antialias(float d)
{
    float w = fwidth(d);
    float scale = w > 0.0 ? 1.0 / w : 1e7;
    return clamp(0.5 + 0.5 * scale * d, 0.0, 1.0);
}

// Higher-quality variant using gradient length (L2 norm)
float antialias_l2(float d)
{
    vec2 dxy = vec2(dFdx(d), dFdy(d));
    float w = length(dxy);
    float scale = w > 0.0 ? 1.0 / w : 1e7;
    return clamp(0.5 + 0.7 * scale * d, 0.0, 1.0);
}

// Manual derivatives version (for discontinuous gradients)
float antialias_l2_dxy(float d, vec2 dxy)
{
    float w = length(dxy);
    float scale = w > 0.0 ? 1.0 / w : 1e7;
    return clamp(0.5 + 0.7 * scale * d, 0.0, 1.0);
}`
        }]
    }, {
        name: "2D Shapes",
        icon: "Hexagon",
        snippets: [{
            name: "Circle",
            description: "Signed distance to a circle",
            author: "Inigo Quilez (iquilezles.org)",
            code: `float sd_circle(vec2 p, float r)
{
    return length(p) - r;
}`
        }, {
            name: "Box",
            description: "Signed distance to an axis-aligned box",
            author: "Inigo Quilez (iquilezles.org)",
            code: `float sd_box(vec2 p, vec2 b)
{
    vec2 d = abs(p) - b;
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}`
        }, {
            name: "Rounded Box",
            description: "Box with individually rounded corners",
            author: "Inigo Quilez (iquilezles.org)",
            code: `float sd_rounded_box(vec2 p, vec2 b, vec4 r)
{
    r.xy = (p.x > 0.0) ? r.xy : r.zw;
    r.x  = (p.y > 0.0) ? r.x  : r.y;
    vec2 q = abs(p) - b + r.x;
    return min(max(q.x, q.y), 0.0) + length(max(q, 0.0)) - r.x;
}`
        }, {
            name: "Line Segment",
            description: "Distance to a line segment between two points",
            author: "Inigo Quilez (iquilezles.org)",
            code: `float sd_segment(vec2 p, vec2 a, vec2 b)
{
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}`
        }, {
            name: "Equilateral Triangle",
            description: "SDF for an equilateral triangle",
            author: "Inigo Quilez (iquilezles.org)",
            code: `float sd_triangle(vec2 p, float r)
{
    const float k = sqrt(3.0);
    p.x = abs(p.x) - r;
    p.y = p.y + r / k;
    if (p.x + k * p.y > 0.0) p = vec2(p.x - k * p.y, -k * p.x - p.y) / 2.0;
    p.x -= clamp(p.x, -2.0 * r, 0.0);
    return -length(p) * sign(p.y);
}`
        }, {
            name: "Ring",
            description: "Signed distance to a ring (annulus)",
            author: "Inigo Quilez (iquilezles.org)",
            code: `float sd_ring(vec2 p, float r, float thickness)
{
    return abs(length(p) - r) - thickness;
}`
        }, {
            name: "Capsule",
            description: "Segment with radius (pill shape)",
            author: "Inigo Quilez (iquilezles.org)",
            code: `float sd_capsule(vec2 p, vec2 a, vec2 b, float r)
{
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}`
        }, {
            name: "Vesica",
            description: "Lens shape from two overlapping circles; w = width, h = height",
            author: "Inigo Quilez (iquilezles.org)",
            code: `float sd_vesica(vec2 p, float w, float h)
{
    vec2 p2 = abs(p);
    float d = 0.5 * (w * w - h * h) / h;
    vec3 c = (w * p2.y < d * (p2.x - w))
        ? vec3(0.0, w, 0.0)
        : vec3(-d, 0.0, d + h);
    return length(p2 - c.yx) - c.z;
}`
        }, {
            name: "Hexagon",
            description: "Regular hexagon (flat-top), r = half-width",
            author: "Inigo Quilez (iquilezles.org)",
            code: `float sd_hexagon(vec2 p, float r)
{
    const vec3 k = vec3(-0.866025404, 0.5, 0.577350269);
    p = abs(p);
    p -= 2.0 * min(dot(k.xy, p), 0.0) * k.xy;
    p -= vec2(clamp(p.x, -k.z * r, k.z * r), r);
    return length(p) * sign(p.y);
}`
        }, {
            name: "Pentagon",
            description: "Regular pentagon",
            author: "Inigo Quilez (iquilezles.org)",
            code: `float sd_pentagon(vec2 p, float r)
{
    const vec3 k = vec3(0.809016994, 0.587785252, 0.726542528);
    p.x = abs(p.x);
    p -= 2.0 * min(dot(vec2(-k.x, k.y), p), 0.0) * vec2(-k.x, k.y);
    p -= 2.0 * min(dot(vec2(k.x, k.y), p), 0.0) * vec2(k.x, k.y);
    p -= vec2(clamp(p.x, -r * k.z, r * k.z), r);
    return length(p) * sign(p.y);
}`
        }, {
            name: "Star (5-point)",
            description: "Regular 5-point star (pentagram)",
            author: "Inigo Quilez (iquilezles.org)",
            code: `float sd_pentagram(vec2 p, float r)
{
    const float k1 = 0.809016994;
    const float k2 = 0.309016994;
    const vec2 v1 = vec2(k1, -0.587785252);
    const vec2 v2 = vec2(-k1, -0.587785252);
    const vec2 v3 = vec2(k2, -0.951056516);
    p.x = abs(p.x);
    p -= 2.0 * max(dot(v1, p), 0.0) * v1;
    p -= 2.0 * max(dot(v2, p), 0.0) * v2;
    p.x = abs(p.x);
    p.y -= r;
    return length(p - v3 * clamp(dot(p, v3), 0.0, 0.726542528 * r)) * sign(p.y * v3.x - p.x * v3.y);
}`
        }, {
            name: "Arc",
            description: "Circular arc; sc = sin/cos of half aperture, ra = radius, rb = thickness",
            author: "Inigo Quilez (iquilezles.org)",
            code: `float sd_arc(vec2 p, vec2 sc, float ra, float rb)
{
    p.x = abs(p.x);
    return ((sc.y * p.x > sc.x * p.y)
        ? length(p - sc * ra)
        : abs(length(p) - ra)) - rb;
}`
        }, {
            name: "Ellipse",
            description: "Axis-aligned ellipse; ab = semi-axes",
            author: "Inigo Quilez (iquilezles.org)",
            code: `float sd_ellipse(vec2 p, vec2 ab)
{
    p = abs(p);
    if (p.x > p.y) { p = p.yx; ab = ab.yx; }
    float l = ab.y * ab.y - ab.x * ab.x;
    float m = ab.x * p.x / l;
    float m2 = m * m;
    float n = ab.y * p.y / l;
    float n2 = n * n;
    float c = (m2 + n2 - 1.0) / 3.0;
    float c3 = c * c * c;
    float q = c3 + m2 * n2 * 2.0;
    float d = c3 + m2 * n2;
    float g = m + m * n2;
    float co;
    if (d < 0.0) {
        float h = acos(q / c3) / 3.0;
        float s = cos(h), t = sin(h) * sqrt(3.0);
        float rx = sqrt(-c * (s + t + 2.0) + m2);
        float ry = sqrt(-c * (s - t + 2.0) + m2);
        co = (ry + sign(l) * rx + abs(g) / (rx * ry) - m) * 0.5;
    } else {
        float h = 2.0 * m * n * sqrt(d);
        float s = sign(q + h) * pow(abs(q + h), 1.0 / 3.0);
        float u = sign(q - h) * pow(abs(q - h), 1.0 / 3.0);
        float rx = -s - u - c * 4.0 + 2.0 * m2;
        float ry = (s - u) * sqrt(3.0);
        float rm = sqrt(rx * rx + ry * ry);
        co = (ry / sqrt(rm - rx) + 2.0 * g / rm - m) * 0.5;
    }
    vec2 r = ab * vec2(co, sqrt(1.0 - co * co));
    return length(r - p) * sign(p.y - r.y);
}`
        }, {
            name: "SDF Operations",
            description: "Union, intersection, subtraction, and smooth variants",
            author: "Inigo Quilez (iquilezles.org)",
            code: `float op_union(float d1, float d2) { return min(d1, d2); }
float op_intersect(float d1, float d2) { return max(d1, d2); }
float op_subtract(float d1, float d2) { return max(-d1, d2); }

float op_smooth_union(float d1, float d2, float k)
{
    float h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return mix(d2, d1, h) - k * h * (1.0 - h);
}

float op_smooth_intersect(float d1, float d2, float k)
{
    float h = clamp(0.5 - 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return mix(d2, d1, h) + k * h * (1.0 - h);
}

float op_smooth_subtract(float d1, float d2, float k)
{
    float h = clamp(0.5 - 0.5 * (d2 + d1) / k, 0.0, 1.0);
    return mix(d2, -d1, h) + k * h * (1.0 - h);
}`
        }]
    }, {
        name: "3D Shapes",
        icon: "Pyramid",
        snippets: [{
            name: "3D SDF Sphere",
            description: "Signed distance to a sphere in 3D",
            author: "Inigo Quilez (iquilezles.org)",
            code: `float sd_sphere(vec3 p, float r)
{
    return length(p) - r;
}`
        }, {
            name: "3D SDF Box",
            description: "Signed distance to an axis-aligned 3D box",
            author: "Inigo Quilez (iquilezles.org)",
            code: `float sd_box_3d(vec3 p, vec3 b)
{
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}`
        }, {
            name: "3D SDF Torus",
            description: "Signed distance to a torus",
            author: "Inigo Quilez (iquilezles.org)",
            code: `float sd_torus(vec3 p, vec2 t)
{
    vec2 q = vec2(length(p.xz) - t.x, p.y);
    return length(q) - t.y;
}`
        }, {
            name: "3D Capsule",
            description: "Line segment with radius (pill)",
            author: "Inigo Quilez (iquilezles.org)",
            code: `float sd_capsule(vec3 p, vec3 a, vec3 b, float r)
{
    vec3 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}`
        }, {
            name: "3D Capped Cylinder",
            description: "Cylinder with flat ends (vertical axis)",
            author: "Inigo Quilez (iquilezles.org)",
            code: `float sd_capped_cylinder(vec3 p, float r, float h)
{
    vec2 d = abs(vec2(length(p.xz), p.y)) - vec2(r, h);
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}`
        }, {
            name: "3D Cone",
            description: "Cone; c = sin/cos of angle, h = height",
            author: "Inigo Quilez (iquilezles.org)",
            code: `float sd_cone(vec3 p, vec2 c, float h)
{
    vec2 q = h * vec2(c.x / c.y, -1.0);
    vec2 w = vec2(length(p.xz), p.y);
    vec2 a = w - q * clamp(dot(w, q) / dot(q, q), 0.0, 1.0);
    vec2 b = w - q * vec2(clamp(w.x / q.x, 0.0, 1.0), 1.0);
    float k = sign(q.y);
    float d = min(dot(a, a), dot(b, b));
    float s = max(k * (w.x * q.y - w.y * q.x), k * (w.y - q.y));
    return sqrt(d) * sign(s);
}`
        }, {
            name: "3D Plane",
            description: "Infinite plane; n = normal (normalized), h = distance from origin",
            author: "Inigo Quilez (iquilezles.org)",
            code: `float sd_plane(vec3 p, vec3 n, float h)
{
    return dot(p, n) + h;
}`
        }, {
            name: "3D Round Cone",
            description: "Cone with rounded ends (r1 at base, r2 at tip)",
            author: "Inigo Quilez (iquilezles.org)",
            code: `float sd_round_cone(vec3 p, float r1, float r2, float h)
{
    float b = (r1 - r2) / h;
    float a = sqrt(1.0 - b * b);
    vec2 q = vec2(length(p.xz), p.y);
    float k = dot(q, vec2(-b, a));
    if (k < 0.0) return length(q) - r1;
    if (k > a * h) return length(q - vec2(0.0, h)) - r2;
    return dot(q, vec2(a, b)) - r1;
}`
        }, {
            name: "3D Hexagonal Prism",
            description: "Hexagonal prism; h.x = half-width, h.y = half-height",
            author: "Inigo Quilez (iquilezles.org)",
            code: `float sd_hex_prism(vec3 p, vec2 h)
{
    const vec3 k = vec3(-0.8660254, 0.5, 0.57735);
    p = abs(p);
    p.xy -= 2.0 * min(dot(k.xy, p.xy), 0.0) * k.xy;
    vec2 d = vec2(
        length(p.xy - vec2(clamp(p.x, -k.z * h.x, k.z * h.x), h.x)) * sign(p.y - h.x),
        p.z - h.y
    );
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}`
        }, {
            name: "3D Round Box",
            description: "Box with rounded edges",
            author: "Inigo Quilez (iquilezles.org)",
            code: `float sd_round_box(vec3 p, vec3 b, float r)
{
    vec3 q = abs(p) - b + r;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}`
        }]
    }, {
        name: "Ray Marching",
        icon: "Globe",
        snippets: [{
            name: "Ray March Loop",
            description: "Basic sphere-tracing ray march loop",
            code: `#define MAX_STEPS 128
#define MAX_DIST  100.0
#define SURF_DIST 0.001

float ray_march(vec3 ro, vec3 rd)
{
    float t = 0.0;
    for (int i = 0; i < MAX_STEPS; i++)
    {
        vec3 p = ro + rd * t;
        float d = scene(p); // Replace with your scene SDF
        if (d < SURF_DIST) break;
        t += d;
        if (t > MAX_DIST) break;
    }
    return t;
}`
        }, {
            name: "Normal Estimation",
            description: "Compute surface normal from SDF gradient",
            code: `#define EPSILON 0.0001

vec3 get_normal(vec3 p)
{
    vec2 e = vec2(EPSILON, 0.0);
    return normalize(vec3(
        scene(p + e.xyy) - scene(p - e.xyy),
        scene(p + e.yxy) - scene(p - e.yxy),
        scene(p + e.yyx) - scene(p - e.yyx)
    ));
}`
        }, {
            name: "Basic Lighting",
            description: "Diffuse + specular Phong-style lighting",
            code: `vec3 lighting(vec3 p, vec3 normal, vec3 rd, vec3 light_dir, vec3 col)
{
    float diff = max(dot(normal, light_dir), 0.0);
    vec3 half_dir = normalize(light_dir - rd);
    float spec = pow(max(dot(normal, half_dir), 0.0), 32.0);
    vec3 ambient = 0.05 * col;
    return ambient + col * diff + vec3(1.0) * spec * 0.5;
}`
        }, {
            name: "Camera Setup",
            description: "Look-at camera matrix for ray marching",
            code: `mat3 camera(vec3 eye, vec3 target, float roll)
{
    vec3 cw = normalize(target - eye);
    vec3 cp = vec3(sin(roll), cos(roll), 0.0);
    vec3 cu = normalize(cross(cw, cp));
    vec3 cv = cross(cu, cw);
    return mat3(cu, cv, cw);
}`
        }]
    }];
export const GLSL_SHADER_CONSTANTS = [{
    name: "PI",
    value: "3.14159265359",
    detail: "π — ratio of circumference to diameter"
}, {
    name: "TAU",
    value: "6.28318530718",
    detail: "2π — full circle in radians"
}, {
    name: "HPI",
    value: "1.57079632679",
    detail: "π/2 — quarter turn"
}, {
    name: "SQRT_HALF",
    value: "0.70710678118",
    detail: "√(1/2)"
}, {
    name: "SQRT2",
    value: "1.41421356237",
    detail: "√2"
}, {
    name: "SQRT_THIRD",
    value: "0.57735026919",
    detail: "√(1/3)"
}, {
    name: "PHI",
    value: "1.61803398875",
    detail: "Golden ratio (φ)"
}, {
    name: "GOLDEN_ANGLE",
    value: "2.39996322972",
    detail: "Golden angle in radians ≈ 137.508°"
}, {
    name: "SQRT_3_4",
    value: "0.86602540378",
    detail: "√(3/4) — simplex height factor"
}, {
    name: "F2",
    value: "0.36602540378",
    detail: "2D simplex skew: (√3−1)/2"
}, {
    name: "G2",
    value: "0.21132486541",
    detail: "2D simplex unskew: (3−√3)/6"
}, {
    name: "F3",
    value: "0.33333333333",
    detail: "3D simplex skew: 1/3"
}, {
    name: "G3",
    value: "0.16666666667",
    detail: "3D simplex unskew: 1/6"
}, {
    name: "F4",
    value: "0.30901699437",
    detail: "4D simplex skew: (√5−1)/4"
}, {
    name: "G4",
    value: "0.13819660113",
    detail: "4D simplex unskew: (1−1/√5)/4"
}]
export const GLSL_SHADER_FUNCTIONS = [{
    name: "sin",
    signature: "sin(x)",
    description: "Sine of angle in radians"
}, {
    name: "cos",
    signature: "cos(x)",
    description: "Cosine of angle in radians"
}, {
    name: "tan",
    signature: "tan(x)",
    description: "Tangent of angle in radians"
}, {
    name: "asin",
    signature: "asin(x)",
    description: "Arc sine, returns radians"
}, {
    name: "acos",
    signature: "acos(x)",
    description: "Arc cosine, returns radians"
}, {
    name: "atan",
    signature: "atan(y, x)",
    description: "Arc tangent of y/x"
}, {
    name: "radians",
    signature: "radians(degrees)",
    description: "Convert degrees to radians"
}, {
    name: "degrees",
    signature: "degrees(radians)",
    description: "Convert radians to degrees"
}, {
    name: "pow",
    signature: "pow(x, y)",
    description: "x raised to the power y"
}, {
    name: "exp",
    signature: "exp(x)",
    description: "e raised to the power x"
}, {
    name: "exp2",
    signature: "exp2(x)",
    description: "2 raised to the power x"
}, {
    name: "log",
    signature: "log(x)",
    description: "Natural logarithm"
}, {
    name: "log2",
    signature: "log2(x)",
    description: "Base-2 logarithm"
}, {
    name: "sqrt",
    signature: "sqrt(x)",
    description: "Square root"
}, {
    name: "inversesqrt",
    signature: "inversesqrt(x)",
    description: "1 / sqrt(x)"
}, {
    name: "abs",
    signature: "abs(x)",
    description: "Absolute value"
}, {
    name: "sign",
    signature: "sign(x)",
    description: "Returns -1, 0, or 1"
}, {
    name: "floor",
    signature: "floor(x)",
    description: "Round down to integer"
}, {
    name: "ceil",
    signature: "ceil(x)",
    description: "Round up to integer"
}, {
    name: "fract",
    signature: "fract(x)",
    description: "Fractional part (x - floor(x))"
}, {
    name: "mod",
    signature: "mod(x, y)",
    description: "Modulo (x - y * floor(x/y))"
}, {
    name: "min",
    signature: "min(x, y)",
    description: "Minimum of x and y"
}, {
    name: "max",
    signature: "max(x, y)",
    description: "Maximum of x and y"
}, {
    name: "clamp",
    signature: "clamp(x, min, max)",
    description: "Clamp x between min and max"
}, {
    name: "mix",
    signature: "mix(a, b, t)",
    description: "Linear interpolation: a*(1-t) + b*t"
}, {
    name: "step",
    signature: "step(edge, x)",
    description: "0.0 if x < edge, else 1.0"
}, {
    name: "smoothstep",
    signature: "smoothstep(a, b, x)",
    description: "Hermite interpolation between a and b"
}, {
    name: "length",
    signature: "length(v)",
    description: "Length of vector"
}, {
    name: "distance",
    signature: "distance(a, b)",
    description: "Distance between two points"
}, {
    name: "dot",
    signature: "dot(a, b)",
    description: "Dot product"
}, {
    name: "cross",
    signature: "cross(a, b)",
    description: "Cross product (vec3)"
}, {
    name: "normalize",
    signature: "normalize(v)",
    description: "Unit vector in same direction"
}, {
    name: "reflect",
    signature: "reflect(I, N)",
    description: "Reflection of incident vector"
}, {
    name: "refract",
    signature: "refract(I, N, eta)",
    description: "Refraction of incident vector"
}, {
    name: "faceforward",
    signature: "faceforward(N, I, Nref)",
    description: "Flip N if facing away from I"
}, {
    name: "texture",
    signature: "texture(sampler, uv)",
    description: "Sample a texture at UV coordinates"
}, {
    name: "dFdx",
    signature: "dFdx(p)",
    description: "Partial derivative in x"
}, {
    name: "dFdy",
    signature: "dFdy(p)",
    description: "Partial derivative in y"
}, {
    name: "fwidth",
    signature: "fwidth(p)",
    description: "abs(dFdx(p)) + abs(dFdy(p))"
}];