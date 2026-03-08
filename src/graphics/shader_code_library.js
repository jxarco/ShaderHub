// snippets collected from Xor's fragcoord.xyz website

const CONSTANTS_LIBRARY_SNIPPETS = [
    {
        name: "PI",
        description: "π — ratio of circumference to diameter",
        code: `#define PI         3.14159265359`
    }, {
        name: "TAU",
        description: "2π — full circle in radians",
        code: `#define TAU        6.28318530718`
    }, {
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
export const COMMON_SHADER_CONSTANTS = [{
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

export const WGSL_CODE_UTILS = {
    "hash_u": `fn hash_u(x: u32) -> u32
{
    var n: u32 = (x ^ 61u) ^ (x >> 16u);
    n *= 9u;
    n = n ^ (n >> 4u);
    n *= 0x27d4eb2du;
    return n ^ (n >> 15u);
}`,
    "mod289_f": `fn mod289_f(x: f32) -> f32 { return x - floor(x * (1.0 / 289.0)) * 289.0; }`,
    "mod289_v2": `fn mod289_v2(x: vec2f) -> vec2f { return x - floor(x * (1.0 / 289.0)) * 289.0; }`,
    "mod289_v3": `fn mod289_v3(x: vec3f) -> vec3f { return x - floor(x * (1.0 / 289.0)) * 289.0; }`,
    "mod289_v4": `fn mod289_v4(x: vec4f) -> vec4f { return x - floor(x * (1.0 / 289.0)) * 289.0; }`,
    "permute_f": `fn permute_f(x: f32) -> f32 { return mod289_f(((x * 34.0) + 1.0) * x); }`,
    "permute_v3": `fn permute_v3(x: vec3f) -> vec3f { return mod289_v3(((x * 34.0) + 1.0) * x); }`,
    "permute_v4": `fn permute_v4(x: vec4f) -> vec4f { return mod289_v4(((x * 34.0) + 1.0) * x); }`,
    "taylorInvSqrt_f": `fn taylorInvSqrt_f(r: f32) -> f32 { return 1.79284291400159 - 0.85373472095314 * r; }`,
    "taylorInvSqrt_v4": `fn taylorInvSqrt_v4(r: vec4f) -> vec4f { return 1.79284291400159 - 0.85373472095314 * r; }`,
};
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

const GOLDEN_ROT2 = mat2x2f(
    -0.73736887808,  0.67549029426,
    -0.67549029426, -0.73736887808);

const GOLDEN_ROT3 = mat3x3f(
    -0.571464913,  0.814921382,  0.096597072,
    -0.278044873, -0.303026659,  0.911518454,
     0.772087367,  0.494042493,  0.399753815);`
            }
        ]
    }, {
        name: "Noise",
        icon: "Hash",
        snippets: [{
            name: "Random",
            description: "Pseudorandom number from float/vec2/vec3/vec4 seed",
            options: [{
                label: "1D → 1D",
                code: `fn rand1(p: f32) -> f32
{
    var p1: f32 = fract(p * 0.1031);
    p1 *= p1 + 33.33;
    p1 *= p1 + p1;
    return fract(p1);
}`
            }, {
                label: "2D → 1D",
                code: `fn rand1(p: vec2f) -> f32
{
    var p3: vec3f = fract(vec3f(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}`
            }, {
                label: "2D → 2D",
                code: `fn rand2(p: vec2f) -> vec2f
{
    var p3 = fract(vec3f(p.xyx) * vec3f(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}`
            }, {
                label: "2D → 3D",
                code: `fn rand3(p: vec2f) -> vec3f
{
    var p4 = fract(vec4f(p.xyx, p.y) * vec4f(0.1031, 0.1030, 0.0973, 0.1099));
    p4 += dot(p4, p4.wzxy + 33.33);
    return fract((p4.xxz + p4.yww) * p4.zyx);
}`
            }, {
                label: "3D → 1D",
                code: `fn rand1(p: vec3f) -> f32
{
    var p4 = fract(vec4f(p.xyzx) * vec4f(0.1031, 0.1030, 0.0973, 0.1099));
    p4 += dot(p4, p4.wzxy + 33.33);
    return fract((p4.x + p4.y + p4.z) * p4.w);
}`
            }, {
                label: "3D → 2D",
                code: `fn rand2(p: vec3f) -> vec2f
{
    var p4 = fract(vec4f(p.xyzx) * vec4f(0.1031, 0.1030, 0.0973, 0.1099));
    p4 += dot(p4, p4.wzxy + 33.33);
    return fract((p4.xx + p4.yz) * p4.zy);
}`
            }, {
                label: "3D → 3D",
                code: `fn rand3(p: vec3f) -> vec3f
{
    var p4 = fract(vec4f(p.xyzx) * vec4f(0.1031, 0.1030, 0.0973, 0.1099));
    p4 += dot(p4, p4.wzxy + 33.33);
    return fract((p4.xxz + p4.yww) * p4.zyx);
}`
            }, {
                label: "1D → 4D",
                code: `fn rand4(p: f32) -> vec4f
{
    var p4 = fract(vec4f(p) * vec4f(0.1031, 0.1030, 0.0973, 0.1099));
    p4 += dot(p4, p4.wzxy + 33.33);
    return fract(vec4f(
        (p4.x + p4.y) * p4.z,
        (p4.y + p4.z) * p4.w,
        (p4.z + p4.w) * p4.x,
        (p4.w + p4.x) * p4.y
    ));
}`
            }, {
                label: "4D → 4D",
                code: `fn rand4(p: vec4f) -> vec4f
{
    var p4 = fract(p * vec4f(0.1031, 0.1030, 0.0973, 0.1099));
    p4 += dot(p4, p4.wzxy + 33.33);
    return fract(vec4f(
        (p4.x + p4.y) * p4.z,
        (p4.y + p4.z) * p4.w,
        (p4.z + p4.w) * p4.x,
        (p4.w + p4.x) * p4.y
    ));
}`
            }]
        }, {
            name: "Integer Hash",
            description: "Pseudorandom from i32/vec2i/vec3i seed",
            options: [{
                label: "i32 → f32",
                dependencies: ["hash_u"],
                code: `fn hash_i(x: i32) -> f32
{
    return f32(hash_u(u32(x))) / 4294967295.0;
}`
            }, {
                label: "vec2i → f32",
                dependencies: ["hash_u"],
                code: `fn hash_i(p: vec2i) -> f32
{
    let n: u32 = hash_u(u32(p.x)) + hash_u(u32(p.y)) * 57u;
    return f32(hash_u(n)) / 4294967295.0;
}`
            }, {
                label: "vec2i → vec2f",
                dependencies: ["hash_u"],
                code: `fn hash_i2(p: vec2i) -> vec2f
{
    let n: u32 = hash_u(u32(p.x)) + hash_u(u32(p.y)) * 57u;
    return vec2f(
        f32(hash_u(n)) / 4294967295.0,
        f32(hash_u(n + 1u)) / 4294967295.0
    );
}`
            }, {
                label: "vec3i → f32",
                dependencies: ["hash_u"],
                code: `fn hash_i(p: vec3i) -> f32
{
    let n: u32 = hash_u(u32(p.x)) + hash_u(u32(p.y)) * 57u + hash_u(u32(p.z)) * 131u;
    return f32(hash_u(n)) / 4294967295.0;
}`
            }, {
                label: "vec3i → vec2f",
                dependencies: ["hash_u"],
                code: `fn hash_i2(p: vec3i) -> vec2f
{
    let n: u32 = hash_u(u32(p.x)) + hash_u(u32(p.y)) * 57u + hash_u(u32(p.z)) * 131u;
    return vec2f(
        f32(hash_u(n)) / 4294967295.0,
        f32(hash_u(n + 1u)) / 4294967295.0
    );
}`
            }, {
                label: "vec3i → vec3f",
                dependencies: ["hash_u"],
                code: `fn hash_i3(p: vec3i) -> vec3f
{
    let n: u32 = hash_u(u32(p.x)) + hash_u(u32(p.y)) * 57u + hash_u(u32(p.z)) * 131u;
    return vec3f(
        f32(hash_u(n)) / 4294967295.0,
        f32(hash_u(n + 1u)) / 4294967295.0,
        f32(hash_u(n + 2u)) / 4294967295.0
    );
}`
            }]
        }, {
            name: "Value Noise",
            description: "Smooth interpolated value noise",
            options: [{
                label: "2D",
                dependencies: ["Noise/Random/2D → 1D"],
                code: `fn value_noise(p: vec2f) -> f32
{
    let i: vec2f = floor(p);
    var f: vec2f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    let a: f32 = rand1(i);
    let b: f32 = rand1(i + vec2(1.0, 0.0));
    let c: f32 = rand1(i + vec2(0.0, 1.0));
    let d: f32 = rand1(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}`
            }, {
                label: "3D",
                dependencies: ["Noise/Random/3D → 1D"],
                code: `fn value_noise(p: vec3f) -> f32
{
    let i: vec3f = floor(p);
    var f: vec3f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    let a: f32 = rand1(i);
    let b: f32 = rand1(i + vec3(1.0, 0.0, 0.0));
    let c: f32 = rand1(i + vec3(0.0, 1.0, 0.0));
    let d: f32 = rand1(i + vec3(1.0, 1.0, 0.0));
    let e: f32 = rand1(i + vec3(0.0, 0.0, 1.0));
    let f0: f32 = rand1(i + vec3(1.0, 0.0, 1.0));
    let g: f32 = rand1(i + vec3(0.0, 1.0, 1.0));
    let h: f32 = rand1(i + vec3(1.0, 1.0, 1.0));
    return mix(
        mix(mix(a, b, f.x), mix(c, d, f.x), f.y),
        mix(mix(e, f0, f.x), mix(g, h, f.x), f.y),
        f.z
    );
}`
            }, {
                label: "4D",
                dependencies: ["Noise/Random/4D → 4D"],
                code: `fn value_noise(p: vec4f) -> f32
{
    let i: vec4f = floor(p);
    var f: vec4f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    let n0000: f32 = rand4(i).x;
    let n1000: f32 = rand4(i + vec4(1, 0, 0, 0)).x;
    let n0100: f32 = rand4(i + vec4(0, 1, 0, 0)).x;
    let n1100: f32 = rand4(i + vec4(1, 1, 0, 0)).x;
    let n0010: f32 = rand4(i + vec4(0, 0, 1, 0)).x;
    let n1010: f32 = rand4(i + vec4(1, 0, 1, 0)).x;
    let n0110: f32 = rand4(i + vec4(0, 1, 1, 0)).x;
    let n1110: f32 = rand4(i + vec4(1, 1, 1, 0)).x;
    let n0001: f32 = rand4(i + vec4(0, 0, 0, 1)).x;
    let n1001: f32 = rand4(i + vec4(1, 0, 0, 1)).x;
    let n0101: f32 = rand4(i + vec4(0, 1, 0, 1)).x;
    let n1101: f32 = rand4(i + vec4(1, 1, 0, 1)).x;
    let n0011: f32 = rand4(i + vec4(0, 0, 1, 1)).x;
    let n1011: f32 = rand4(i + vec4(1, 0, 1, 1)).x;
    let n0111: f32 = rand4(i + vec4(0, 1, 1, 1)).x;
    let n1111: f32 = rand4(i + vec4(1, 1, 1, 1)).x;
    return mix(
        mix(mix(mix(n0000, n1000, f.x), mix(n0100, n1100, f.x), f.y),
            mix(mix(n0010, n1010, f.x), mix(n0110, n1110, f.x), f.y), f.z),
        mix(mix(mix(n0001, n1001, f.x), mix(n0101, n1101, f.x), f.y),
            mix(mix(n0011, n1011, f.x), mix(n0111, n1111, f.x), f.y), f.z),
        f.w
    );
}`
            }]
        }, {
            name: "Simplex Noise",
            description: "Classic simplex noise",
            author: "Ashima Arts, Stefan Gustavson (webgl-noise)",
            options: [{
                label: "2D",
                dependencies: ["mod289_v2", "mod289_v3", "permute_v3"],
                code: `fn simplex_noise(v: vec2f) -> f32
{
    const C: vec4f = vec4f(0.211324865405187, 0.366025403784439,
                          -0.577350269189626, 0.024390243902439);
    var i = floor(v + dot(v, C.yy));
    let x0 = v - i + dot(i, C.xx);
    let i1 = select(vec2f(0.0, 1.0), vec2f(1.0, 0.0), x0.x > x0.y);
    var x12 = x0.xyxy + C.xxzz;
    x12.x -= i1.x;
    x12.y -= i1.y;
    i = mod289_v2(i);
    let p = permute_v3(permute_v3(i.y + vec3f(0.0, i1.y, 1.0))
                                 + i.x + vec3f(0.0, i1.x, 1.0));
    var m = max(0.5 - vec3f(dot(x0, x0), dot(x12.xy, x12.xy),
                            dot(x12.zw, x12.zw)), vec3f(0.0));
    m = m * m;
    m = m * m;
    let x = 2.0 * fract(p * C.www) - 1.0;
    let h = abs(x) - 0.5;
    let ox = floor(x + 0.5);
    let a0 = x - ox;
    m *= 1.79284291400159 - 0.85373472095314 * (a0 * a0 + h * h);
    var g = vec3f(0.0);
    g.x = a0.x * x0.x + h.x * x0.y;
    g.y = a0.y * x12.x + h.y * x12.y;
    g.z = a0.z * x12.z + h.z * x12.w;
    return 130.0 * dot(m, g);
}`
            }, {
                label: "3D",
                dependencies: ["mod289_v3", "mod289_v4", "permute_v4", "taylorInvSqrt_v4"],
                code: `fn simplex_noise(v: vec3f) -> f32
{
    const C: vec2f = vec2f(1.0 / 6.0, 1.0 / 3.0);
    const D: vec4f = vec4f(0.0, 0.5, 1.0, 2.0);
    var i = floor(v + dot(v, C.yyy));
    let x0 = v - i + dot(i, C.xxx);
    let g = step(x0.yzx, x0.xyz);
    let l = 1.0 - g;
    let i1 = min(g.xyz, l.zxy);
    let i2 = max(g.xyz, l.zxy);
    let x1 = x0 - i1 + C.xxx;
    let x2 = x0 - i2 + C.yyy;
    let x3 = x0 - D.yyy;
    i = mod289_v3(i);
    let p = permute_v4(permute_v4(permute_v4(
        i.z + vec4f(0.0, i1.z, i2.z, 1.0))
        + i.y + vec4f(0.0, i1.y, i2.y, 1.0))
        + i.x + vec4f(0.0, i1.x, i2.x, 1.0));
    let n_ = 0.142857142857;
    let ns = n_ * D.wyz - D.xzx;
    let j = p - 49.0 * floor(p * ns.z * ns.z);
    let x_ = floor(j * ns.z);
    let y_ = floor(j - 7.0 * x_);
    let x = x_ * ns.x + ns.yyyy;
    let y = y_ * ns.x + ns.yyyy;
    let h = 1.0 - abs(x) - abs(y);
    let b0 = vec4f(x.xy, y.xy);
    let b1 = vec4f(x.zw, y.zw);
    let s0 = floor(b0) * 2.0 + 1.0;
    let s1 = floor(b1) * 2.0 + 1.0;
    let sh = -step(h, vec4f(0.0));
    let a0 = b0.xzyw + s0.xzyw * sh.xxyy;
    let a1 = b1.xzyw + s1.xzyw * sh.zzww;
    var p0 = vec3f(a0.xy, h.x);
    var p1 = vec3f(a0.zw, h.y);
    var p2 = vec3f(a1.xy, h.z);
    var p3 = vec3f(a1.zw, h.w);
    let norm = taylorInvSqrt_v4(vec4f(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;
    var m = max(0.5 - vec4f(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), vec4f(0.0));
    m = m * m;
    return 105.0 * dot(m * m, vec4f(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));
}`
            }, {
                label: "4D",
                dependencies: ["mod289_v4", "mod289_f", "permute_v4", "permute_f", "taylorInvSqrt_v4"],
                code: `fn grad4(j: f32, ip: vec4f) -> vec4f
{
    const ones: vec4f = vec4f(1.0, 1.0, 1.0, -1.0);
    var p = vec4f(0.0);
    let pxyz = floor(fract(vec3f(j) * ip.xyz) * 7.0) * ip.z - 1.0;
    p.x = pxyz.x; p.y = pxyz.y; p.z = pxyz.z;
    p.w = 1.5 - dot(abs(pxyz), ones.xyz);
    let s = select(vec4f(0.0), vec4f(1.0), p < vec4f(0.0));
    let new_xyz = p.xyz + (s.xyz * 2.0 - 1.0) * s.www;
    p.x = new_xyz.x; p.y = new_xyz.y; p.z = new_xyz.z;
    return p;
}

fn simplex_noise(v: vec4f) -> f32
{
    const F4: f32 = 0.309016994374947451;
    const C: vec4f = vec4f(0.138196601125011, 0.276393202250021, 0.414589803375032, -0.447213595499958);
    var i = floor(v + dot(v, vec4f(F4)));
    let x0 = v - i + dot(i, C.xxxx);
    var i0 = vec4f(0.0);
    let isX = step(x0.yzw, x0.xxx);
    let isYZ = step(x0.zww, x0.yyz);
    i0.x = isX.x + isX.y + isX.z;
    i0.y = 1.0 - isX.x;
    i0.z = 1.0 - isX.y;
    i0.w = 1.0 - isX.z;
    i0.y += isYZ.x + isYZ.y;
    i0.z += 1.0 - isYZ.x;
    i0.w += 1.0 - isYZ.y;
    i0.z += isYZ.z;
    i0.w += 1.0 - isYZ.z;
    let i3 = clamp(i0, vec4f(0.0), vec4f(1.0));
    let i2 = clamp(i0 - 1.0, vec4f(0.0), vec4f(1.0));
    let i1 = clamp(i0 - 2.0, vec4f(0.0), vec4f(1.0));
    let x1 = x0 - i1 + C.xxxx;
    let x2 = x0 - i2 + C.yyyy;
    let x3 = x0 - i3 + C.zzzz;
    let x4 = x0 + C.wwww;
    i = mod289_v4(i);
    let j0 = permute_f(permute_f(permute_f(permute_f(i.w) + i.z) + i.y) + i.x);
    let j1 = permute_v4(permute_v4(permute_v4(permute_v4(
        i.w + vec4f(i1.w, i2.w, i3.w, 1.0))
        + i.z + vec4f(i1.z, i2.z, i3.z, 1.0))
        + i.y + vec4f(i1.y, i2.y, i3.y, 1.0))
        + i.x + vec4f(i1.x, i2.x, i3.x, 1.0));
    let ip = vec4f(1.0 / 294.0, 1.0 / 49.0, 1.0 / 7.0, 0.0);
    var p0 = grad4(j0, ip);
    var p1 = grad4(j1.x, ip);
    var p2 = grad4(j1.y, ip);
    var p3 = grad4(j1.z, ip);
    var p4 = grad4(j1.w, ip);
    let norm = taylorInvSqrt_v4(vec4f(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;
    p4 *= taylorInvSqrt_f(dot(p4, p4));
    var m0 = max(0.6 - vec3f(dot(x0, x0), dot(x1, x1), dot(x2, x2)), vec3f(0.0));
    var m1 = max(0.6 - vec2f(dot(x3, x3), dot(x4, x4)), vec2f(0.0));
    m0 = m0 * m0;
    m1 = m1 * m1;
    return 49.0 * (dot(m0 * m0, vec3f(dot(p0, x0), dot(p1, x1), dot(p2, x2)))
        + dot(m1 * m1, vec2f(dot(p3, x3), dot(p4, x4))));
}`
            }]
        }, {
            name: "FBM (Fractal Brownian Motion)",
            description: "Layered noise with configurable octaves",
            options: [{
                label: "2D",
                dependencies: ["Noise/Value Noise/2D"],
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
                label: "3D",
                dependencies: ["Noise/Value Noise/3D"],
                code: `fn fbm(p: vec3f) -> f32
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
            }]
        }, {
            name: "Voronoi",
            description: "Cell noise returning F1 distance, F2 distance, and cell ID (includes rand)",
            author: "Georgy Voronoi (Voronoi diagrams)",
            options: [{
                label: "2D",
                dependencies: ["Noise/Random/2D → 1D", "Noise/Random/2D → 2D"],
                code: `fn voronoi(p: vec2f) -> vec3f
{
    let i = floor(p);
    let f = fract(p);
    var f1 = 1.0; var f2 = 1.0;
    var cell_id = vec2f(0.0);
    for (var y: i32 = -1; y <= 1; y++)
    {
        for (var x: i32 = -1; x <= 1; x++)
        {
            let neighbor = vec2f(f32(x), f32(y));
            let point = rand2(i + neighbor);
            let dist = length(neighbor + point - f);
            if (dist < f1) { f2 = f1; f1 = dist; cell_id = i + neighbor; }
            else if (dist < f2) { f2 = dist; }
        }
    }
    return vec3f(f1, f2, rand1(cell_id));
}`
            }, {
                label: "3D",
                dependencies: ["Noise/Random/2D → 1D", "Noise/Random/3D → 3D"],
                code: `fn voronoi(p: vec3f) -> vec3f
{
    let i = floor(p);
    let f = fract(p);
    var f1 = 1.0; var f2 = 1.0;
    var cell_id = vec3f(0.0);
    for (var z: i32 = -1; z <= 1; z++)
    {
        for (var y: i32 = -1; y <= 1; y++)
        {
            for (var x: i32 = -1; x <= 1; x++)
            {
                let neighbor = vec3f(f32(x), f32(y), f32(z));
                let point = rand3(i + neighbor);
                let dist = length(neighbor + point - f);
                if (dist < f1) { f2 = f1; f1 = dist; cell_id = i + neighbor; }
                else if (dist < f2) { f2 = dist; }
            }
        }
    }
    return vec3f(f1, f2, rand1(cell_id.xy + vec2f(cell_id.z * 31.0)));
}`
            }]
        }, {
            name: "Worley Noise",
            description: "F1 distance to nearest feature point (includes rand)",
            author: "Steven Worley (SIGGRAPH 1996)",
            options: [{
                label: "2D",
                dependencies: ["Noise/Random/2D → 2D"],
                code: `fn worley(p: vec2f) -> f32
{
    let i: vec2f = floor(p);
    let f: vec2f = fract(p);
    var min_dist: f32 = 1.0;
    for (var y: i32 = -1; y <= 1; y++)
    {
        for (var x: i32 = -1; x <= 1; x++)
        {
            let neighbor: vec2f = vec2f(f32(x), f32(y));
            let point: vec2f = rand2(i + neighbor);
            min_dist = min(min_dist, length(neighbor + point - f));
        }
    }
    return min_dist;
}`
            }, {
                label: "3D",
                dependencies: ["Noise/Random/3D → 3D"],
                code: `fn worley(p: vec3f) -> f32
{
    let i: vec3f = floor(p);
    let f: vec3f = fract(p);
    var min_dist: f32 = 1.0;
    for (var z: i32 = -1; z <= 1; z++)
    {
        for (var y: i32 = -1; y <= 1; y++)
        {
            for (var x: i32 = -1; x <= 1; x++)
            {
                let neighbor: vec3f = vec3f(f32(x), f32(y), f32(z));
                let point: vec3f = rand3(i + neighbor);
                min_dist = min(min_dist, length(neighbor + point - f));
            }
        }
    }
    return min_dist;
}`
            }]
        }, {
            name: "Turbulence",
            description: "Fake fluid dynamics via layered rotated sine waves",
            author: "Xor (mini.gmshaders.com/p/turbulence)",
            options: [{
                label: "2D",
                code: `// Turbulence — 2D | Xor (mini.gmshaders.com/p/turbulence)
// Layered sine-wave displacements with golden-ratio rotation.
#define TURB_NUM   10.0
#define TURB_AMP   0.7
#define TURB_SPEED 0.3
#define TURB_FREQ  2.0
#define TURB_EXP   1.4

const GOLDEN_ROT2: mat2x2f = mat2x2f(
    -0.73736887808,  0.67549029426,
    -0.67549029426, -0.73736887808);

fn turbulence(pos: vec2f, time: f32) -> vec2f
{
    var p = pos;
    var freq: f32 = TURB_FREQ;
    var rot = GOLDEN_ROT2;
    for (var i: f32 = 0.0; i < TURB_NUM; i += 1.0)
    {
        let phase = freq * (p * rot).y + TURB_SPEED * time + i;
        p += TURB_AMP * rot[0] * sin(phase) / freq;
        rot *= GOLDEN_ROT2;
        freq *= TURB_EXP;
    }
    return p;
}`
            }, {
                label: "3D",
                code: `// Turbulence — 3D | Xor (mini.gmshaders.com/p/turbulence)
// Layered sine-wave displacements with golden-ratio rotation.
#define TURB_NUM   10.0
#define TURB_AMP   0.7
#define TURB_SPEED 0.3
#define TURB_FREQ  2.0
#define TURB_EXP   1.4

const GOLDEN_ROT3: mat3x3f = mat3x3f(
    -0.571464913,  0.814921382,  0.096597072,
    -0.278044873, -0.303026659,  0.911518454,
     0.772087367,  0.494042493,  0.399753815);

fn turbulence(pos: vec3f, time: f32) -> vec3f
{
    var p = pos;
    var freq: f32 = TURB_FREQ;
    var rot = GOLDEN_ROT3;
    for (var i: f32 = 0.0; i < TURB_NUM; i += 1.0)
    {
        let phase = freq * (p * rot).y + TURB_SPEED * time + i;
        p += TURB_AMP * rot[0] * sin(phase) / freq;
        rot *= GOLDEN_ROT3;
        freq *= TURB_EXP;
    }
    return p;
}`
            }]
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
    var p1: vec3f = p;
    for (var i: i32 = 0; i < 5; i++)
    {
        value += amplitude * dot(cos(GOLD * p1), sin(PHI * p1 * GOLD)) / 3.0;
        p1 = GOLD * p1 * 2.0;
        amplitude *= 0.5;
    }
    return value;
}`
        }]
    }, {
        name: "Color",
        icon: "Contrast",
        snippets: [{
            name: "HSV ↔ RGB",
            description: "Convert between HSV and RGB color spaces",
            author: "Sam Hocevar (RGB→HSV)",
            options: [{
                label: "HSV → RGB",
                code: `fn hsv_to_rgb(c: vec3f) -> vec3f
{
    let p = abs(fract(c.xxx + vec3f(1.0, 2.0/3.0, 1.0/3.0)) * 6.0 - 3.0);
    return c.z * mix(vec3f(1.0), clamp(p - 1.0, vec3f(0.0), vec3f(1.0)), c.y);
}`
            }, {
                label: "RGB → HSV",
                code: `fn rgb_to_hsv(c: vec3f) -> vec3f
{
    let K = vec4f(0.0, -1.0/3.0, 2.0/3.0, -1.0);
    let p = mix(vec4f(c.bg, K.wz), vec4f(c.gb, K.xy), vec4f(step(c.b, c.g)));
    let q = mix(vec4f(p.xyw, c.r), vec4f(c.r, p.yzx), vec4f(step(p.x, c.r)));
    let d = q.x - min(q.w, q.y);
    let e = 1.0e-10;
    return vec3f(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}`
            }]
        }, {
            name: "Cosine Palette",
            description: "Cosine gradient palette (4 vec3 parameters)",
            author: "Inigo Quilez (iquilezles.org/articles/palettes)",
            dependencies: ["TAU"],
            code: `fn palette(t: f32, a: vec3f, b: vec3f, c: vec3f, d: vec3f) -> vec3f
{
    return a + b * cos(TAU * (c * t + d));
}

// Example presets:
// Rainbow:   palette(t, vec3f(0.5), vec3f(0.5), vec3f(1.0), vec3f(0.0, 0.33, 0.67))
// Warm:      palette(t, vec3f(0.5), vec3f(0.5), vec3f(1.0), vec3f(0.0, 0.1, 0.2))
// Cool:      palette(t, vec3f(0.5), vec3f(0.5), vec3f(1.0, 1.0, 0.5), vec3f(0.8, 0.9, 0.3))`
        }, {
            name: "Linear ↔ sRGB",
            description: "Linear ↔ sRGB with IEC 61966-2-1 transfer curve",
            options: [{
                label: "Linear → sRGB",
                code: `fn linear_to_srgb(c: vec3f) -> vec3f
{
    let lo = 12.92 * c;
    let hi = 1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055;
    return mix(lo, hi, step(vec3f(0.0031308), c));
}`
            }, {
                label: "sRGB → Linear",
                code: `fn srgb_to_linear(c: vec3f) -> vec3f
{
    let lo = c / 12.92;
    let hi = pow((c + vec3f(0.055)) / 1.055, vec3f(2.4));
    return mix(lo, hi, step(vec3f(0.04045), c));
}`
            }]
        }, {
            name: "Tonemapping",
            description: "HDR → LDR tonemapping operators",
            author: "Krzysztof Narkowicz (ACES), Erik Reinhard et al.",
            options: [{
                label: "ACES",
                code: `fn tonemap_aces(x: vec3f) -> vec3f
{
    return clamp(
        (x * (2.51 * x + vec3f(0.03))) / (x * (2.43 * x + vec3f(0.59)) + vec3f(0.14)),
        vec3f(0.0), vec3f(1.0));
}`
            }, {
                label: "Reinhard",
                code: `fn tonemap_reinhard(x: vec3f) -> vec3f
{
    return x / (vec3f(1.0) + x);
}`
            }, {
                label: "Reinhard (white point)",
                code: `fn tonemap_reinhard_ext(x: vec3f, white: f32) -> vec3f
{
    let num = x * (vec3f(1.0) + x / (white * white));
    return num / (vec3f(1.0) + x);
}`
            }]
        }, {
            name: "Hue Rotation",
            description: "Rotate the hue of an RGB color",
            author: "W3C CSS Filters (hue-rotate matrix)",
            code: `fn hue_rotate(col: vec3f, angle: f32) -> vec3f
{
    let s = sin(angle);
    let c = cos(angle);
    let m = mat3x3f(
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
            code: `fn luminance(col: vec3f) -> f32
{
    return dot(col, vec3f(0.2126, 0.7152, 0.0722));
}`
        }, {
            name: "OKLab - OKLch",
            description: "Perceptually uniform color space; OKLch = cylindrical OKLab",
            author: "Björn Ottosson (bottosson.github.io/posts/oklab)",
            options: [{
                label: "RGB → OKLab",
                code: `fn ok_cbrt(x: f32) -> f32 { return sign(x) * pow(abs(x), 1.0 / 3.0); }

fn rgb_to_oklab(c: vec3f) -> vec3f
{
    let l = 0.4122214708 * c.r + 0.5363325363 * c.g + 0.0514459929 * c.b;
    let m = 0.2119034982 * c.r + 0.6806995451 * c.g + 0.1073969566 * c.b;
    let s = 0.0883024619 * c.r + 0.2817188376 * c.g + 0.6299787005 * c.b;
    let l_ = ok_cbrt(l); let m_ = ok_cbrt(m); let s_ = ok_cbrt(s);
    return vec3f(
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    );
}`
            }, {
                label: "OKLab → RGB",
                code: `fn oklab_to_rgb(c: vec3f) -> vec3f
{
    let l_ = c.r + 0.3963377774 * c.g + 0.2158037573 * c.b;
    let m_ = c.r - 0.1055613458 * c.g - 0.0638541728 * c.b;
    let s_ = c.r - 0.0894841775 * c.g - 1.2914855480 * c.b;
    let l = l_ * l_ * l_; let m = m_ * m_ * m_; let s = s_ * s_ * s_;
    return vec3f(
        4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    );
}`
            }, {
                label: "OKLab → OKLch",
                code: `fn oklab_to_oklch(ok: vec3f) -> vec3f
{
    let L = ok.r;
    let C = sqrt(ok.g * ok.g + ok.b * ok.b);
    let H = atan2(ok.b, ok.g);
    return vec3f(L, C, H);
}`
            }, {
                label: "OKLch → OKLab",
                code: `fn oklch_to_oklab(lch: vec3f) -> vec3f
{
    let L = lch.r; let C = lch.g; let H = lch.b;
    return vec3f(L, C * cos(H), C * sin(H));
}`
            }, {
                label: "OKLab Mix",
                author: "Inigo Quilez",
                code: `// Perceptually uniform mix via LMS cone space. Compact, no separate conversion helpers.
// https://bottosson.github.io/posts/oklab

fn oklab_mix(colA: vec3f, colB: vec3f, h: f32) -> vec3f
{
    const kCONEtoLMS: mat3x3f = mat3x3f(
         0.4121656120,  0.5362752080,  0.0514575653,
         0.2118591070,  0.6807189584,  0.1074065790,
         0.0883097947,  0.2818474174,  0.6302613616);
    const kLMStoCONE: mat3x3f = mat3x3f(
         4.0767245293, -3.3072168827,  0.2307590544,
        -1.2681437731,  2.6093323231, -0.3411344290,
        -0.0041119885, -0.7034763098,  1.7068625689);
    let lmsA = pow(kCONEtoLMS * colA, vec3f(1.0 / 3.0));
    let lmsB = pow(kCONEtoLMS * colB, vec3f(1.0 / 3.0));
    let lms = mix(lmsA, lmsB, h);
    return kLMStoCONE * (lms * lms * lms);
}`
            }]
        }, {
            name: "Saturation",
            description: "Adjust color saturation; s=0 grayscale, s=1 original, s>1 oversaturated",
            code: `fn saturation(col: vec3f, sat: f32) -> vec3f
{
    let lum = dot(col, vec3f(0.2126, 0.7152, 0.0722));
    return mix(vec3f(lum), col, sat);
}`
        }]
    }, {
        name: "Coordinates",
        icon: "Grid3x2",
        snippets: [{
            name: "Rotation",
            description: "Rotate vec2 or vec3 by angle(s)",
            options: [{
                label: "2D",
                code: `// Usage: p.xy *= rotate_2d(angle);
fn rotate_2d(angle: f32) -> mat2x2f
{
    let s: f32 = sin(angle);
    let c: f32 = cos(angle);
    return mat2x2f(c, -s, s, c);
}`
            }, {
                label: "3D (Euler)",
                dependencies: ["Coordinates/Rotation/2D"],
                code: `fn euler_rotate(v: vec3f, roll: f32, pitch: f32, yaw: f32) -> vec3f
{
    var p = v;
    let yz = rotate_2d(roll)   * vec2f(p.y, p.z); p = vec3f(p.x,  yz.x, yz.y);
    let xz = rotate_2d(pitch)  * vec2f(p.x, p.z); p = vec3f(xz.x, p.y,  xz.y);
    let xy = rotate_2d(yaw)    * vec2f(p.x, p.y); p = vec3f(xy.x, xy.y, p.z );
    return p;
}`
            }, {
                label: "3D (Axis-Angle)",
                author: "Xor / Fabrice Neyret (mini.gmshaders.com/p/3d-rotation)",
                code: `fn axis_rotate(v: vec3f, axis: vec3f, angle: f32) -> vec3f
{
    return mix(dot(v, axis) * axis, v, cos(angle))
         + sin(angle) * cross(v, axis);
}`
            }]
        }, {
            name: "Quaternions",
            description: "Quaternion rotation (vec4: x,y,z,w)",
            options: [{
                label: "Multiply",
                code: `fn qmul(a: vec4f, b: vec4f) -> vec4f
{
    return vec4f(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
    );
}`
            }, {
                label: "Slerp",
                code: `fn qslerp(a: vec4f, b: vec4f, t: f32) -> vec4f
{
    var bb: vec4f = b;
    var d: f32 = dot(a, bb);
    if (d < 0.0) { bb = -bb; d = -d; }
    if (d > 0.9995) { return normalize(mix(a, bb, t)); }
    let theta: f32 = acos(d);
    let st: f32 = sin(theta);
    return (sin((1.0 - t) * theta) / st) * a + (sin(t * theta) / st) * bb;
}`
            }, {
                label: "From Axis-Angle",
                code: `fn qfrom_axis_angle(axis: vec3f, angle: f32) -> vec4f
{
    let ha: f32 = angle * 0.5;
    let s: f32 = sin(ha);
    return vec4f(normalize(axis) * s, cos(ha));
}`
            }, {
                label: "To Mat3",
                code: `fn qto_mat3(q: vec4f) -> mat3x3f
{
    let x: f32 = q.x; let y: f32 = q.y; let z: f32 = q.z; let w: f32 = q.w;
    return mat3x3f(
        1.0 - 2.0*(y*y + z*z), 2.0*(x*y - z*w), 2.0*(x*z + y*w),
        2.0*(x*y + z*w), 1.0 - 2.0*(x*x + z*z), 2.0*(y*z - x*w),
        2.0*(x*z - y*w), 2.0*(y*z + x*w), 1.0 - 2.0*(x*x + y*y)
    );
}`
            }, {
                label: "Rotate Vec3",
                code: `fn qrotate(q: vec4f, v: vec3f) -> vec3f
{
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}`
            }]
        }, {
            name: "Polar Coordinates",
            description: "Convert between cartesian and polar/spherical coords",
            options: [{
                label: "2D: Cartesian → Polar",
                code: `fn to_polar(p: vec2f) -> vec2f
{
    return vec2f(length(p), atan2(p.y, p.x));
}`
            }, {
                label: "2D: Polar → Cartesian",
                code: `fn to_cartesian(polar: vec2f) -> vec2f
{
    return vec2f(polar.x * cos(polar.y), polar.x * sin(polar.y));
}`
            }, {
                label: "3D: Cartesian → Spherical",
                code: `// spherical = (r, theta, phi): r=radius, theta=azimuth (xy), phi=polar from +z.
fn to_spherical(p: vec3f) -> vec3f
{
    let r: f32 = length(p);
    let theta: f32 = atan2(p.y, p.x);
    let phi: f32 = select(0.0, acos(clamp(p.z / r, -1.0, 1.0)), r > 0.0);
    return vec3f(r, theta, phi);
}`
            }, {
                label: "3D: Spherical → Cartesian",
                code: `fn to_cartesian(spherical: vec3f) -> vec3f
{
    let r: f32 = spherical.x;
    let theta: f32 = spherical.y;
    let phi: f32 = spherical.z;
    return r * vec3f(sin(phi) * cos(theta), sin(phi) * sin(theta), cos(phi));
}`
            }]
        }, {
            name: "Repeat - Tile",
            description: "Repeat space for infinite tiling",
            options: [{
                label: "Cartesian",
                code: `fn repeat(p: vec2f, size: vec2f) -> vec2f
{
    return (p + size * 0.5) % size - size * 0.5;
}`
            }, {
                label: "Mirror",
                code: `fn mirror_repeat(p: vec2f, size: vec2f) -> vec2f
{
    let half = size * 0.5;
    let q = (p + half) % (size * 2.0) - size;
    return half - abs(q);
}`
            }, {
                label: "Angular",
                code: `// Repeat by angle; n = number of segments (e.g. 6 for hexagon).
fn angular_repeat(p: vec2f, n: f32) -> vec2f
{
    var a: f32 = atan2(p.y, p.x);
    let r: f32 = length(p);
    a = a % (6.28318530718 / n) - 0.5 * 6.28318530718 / n;
    return vec2f(cos(a), sin(a)) * r;
}`
            }]
        }, {
            name: "UV Setup",
            description: "Centered UV coordinate helpers",
            author: "Xor (mini.gmshaders.com/p/gm-shaders-mini-scaling)",
            options: [{
                label: "Stretched",
                code: `// Stretched: full resolution to 0–1, non-square screens stretched.
fn uv_stretched(pixel: vec2f, res: vec2f) -> vec2f
{
    return (pixel - res * 0.5) / res + 0.5;
}`
            }, {
                label: "Fit",
                code: `// Fit: uniform scale, full area visible (letterboxing).
fn uv_fit(pixel: vec2f, res: vec2f) -> vec2f
{
    let center = pixel - res * 0.5;
    return center / max(res.x, res.y) + 0.5;
}`
            }, {
                label: "Fill",
                code: `// Fill: uniform scale, screen fully covered (edges may crop).
fn uv_fill(pixel: vec2f, res: vec2f) -> vec2f
{
    let center = pixel - res * 0.5;
    return center / min(res.x, res.y) + 0.5;
}`
            }, {
                label: "Custom Origin",
                code: `// Custom origin (e.g. vec2f(0.5, 1.0) for middle-bottom).
fn uv_center_origin(pixel: vec2f, res: vec2f, origin: vec2f) -> vec2f
{
    let center = pixel - res * origin;
    return center / min(res.x, res.y) + origin;
}`
            }, {
                label: "Aspect Ratio",
                code: `// Correct aspect ratio so circles stay circular.
fn uv_aspect(uv_in: vec2f, res: vec2f) -> vec2f
{
    var uv = uv_in - 0.5;
    uv.x *= res.x / res.y;
    return uv + 0.5;
}`
            }, {
                label: "Aspect Centered",
                code: `// Aspect correction, origin at 0,0.
fn uv_aspect_centered(uv_in: vec2f, res: vec2f) -> vec2f
{
    var uv = uv_in - 0.5;
    uv.x *= res.x / res.y;
    return uv;
}`
            }, {
                label: "Fit Texture",
                code: `// Fit texture with ratio (e.g. vec2f(2,1)) to screen, no cropping.
fn uv_fit_ratio(pixel: vec2f, res: vec2f, ratio: vec2f) -> vec2f
{
    let res_ratio = res / ratio;
    let origin = vec2f(0.5);
    let center = pixel - res * origin;
    return center / max(res_ratio.x, res_ratio.y) / ratio + origin;
}`
            }]
        }]
    }, {
        name: "Math",
        icon: "Function",
        snippets: [{
            name: "Remap",
            description: "Map a value from one range to another",
            options: [{
                label: "Remap",
                code: `fn remap(value: f32, in_min: f32, in_max: f32, out_min: f32, out_max: f32) -> f32
{
    return out_min + (out_max - out_min) * (value - in_min) / (in_max - in_min);
}`
            }, {
                label: "To Unit",
                code: `// Linear map [in_min, in_max] → [0, 1]; no clamp (values outside range extrapolate).
fn to_unit(value: f32, in_min: f32, in_max: f32) -> f32
{
    return (value - in_min) / (in_max - in_min);
}`
            }, {
                label: "From Unit",
                code: `// Linear map [0, 1] → [out_min, out_max]; no clamp (values outside range extrapolate).
fn from_unit(value: f32, out_min: f32, out_max: f32) -> f32
{
    return out_min + value * (out_max - out_min);
}`
            }]
        }, {
            name: "Smooth Min / Max",
            description: "Smooth minimum and maximum blending",
            author: "Inigo Quilez (iquilezles.org/articles/smin)",
            options: [{
                label: "Smooth Min",
                code: `fn smin(a: f32, b: f32, k: f32) -> f32
{
    let h: f32 = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}`
            }, {
                label: "Smooth Max",
                code: `fn smax(a: f32, b: f32, k: f32) -> f32
{
    let h: f32 = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(a, b, h) + k * h * (1.0 - h);
}`
            }]
        }, {
            name: "Easing",
            description: "Easing curves for animations (t in [0,1])",
            options: [{
                label: "Quadratic",
                code: `fn ease_in_quad(t: f32) -> f32 { return t * t; }
fn ease_out_quad(t: f32) -> f32 { return t * (2.0 - t); }
fn ease_in_out_quad(t: f32) -> f32 { return select(-1.0 + (4.0 - 2.0 * t) * t, 2.0 * t * t, t < 0.5); }`
            }, {
                label: "Cubic",
                code: `fn ease_in_cubic(t: f32) -> f32 { return t * t * t; }
fn ease_out_cubic(t: f32) -> f32 { let u = 1.0 - t; return 1.0 - u * u * u; }
fn ease_in_out_cubic(t: f32) -> f32 { return select(1.0 - pow(-2.0 * t + 2.0, 3.0) * 0.5, 4.0 * t * t * t, t < 0.5); }`
            }, {
                label: "Quartic",
                code: `fn ease_in_quart(t: f32) -> f32 { return t * t * t * t; }
fn ease_out_quart(t: f32) -> f32 { let u = 1.0 - t; return 1.0 - u * u * u * u; }
fn ease_in_out_quart(t: f32) -> f32 { return select(1.0 - pow(-2.0 * t + 2.0, 4.0) * 0.5, 8.0 * t * t * t * t, t < 0.5); }`
            }, {
                label: "Quintic",
                code: `fn ease_in_quint(t: f32) -> f32 { return t * t * t * t * t; }
fn ease_out_quint(t: f32) -> f32 { let u = 1.0 - t; return 1.0 - u * u * u * u * u; }
fn ease_in_out_quint(t: f32) -> f32 { return select(1.0 - pow(-2.0 * t + 2.0, 5.0) * 0.5, 16.0 * t * t * t * t * t, t < 0.5); }`
            }, {
                label: "Exponential",
                code: `fn ease_in_expo(t: f32) -> f32 { return select(pow(2.0, 10.0 * t - 10.0), 0.0, t <= 0.0); }
fn ease_out_expo(t: f32) -> f32 { return select(1.0 - pow(2.0, -10.0 * t), 1.0, t >= 1.0); }
fn ease_in_out_expo(t: f32) -> f32
{
    if (t <= 0.0) { return 0.0; }
    if (t >= 1.0) { return 1.0; }
    return select(1.0 - pow(2.0, -20.0 * t + 10.0) * 0.5, pow(2.0, 20.0 * t - 10.0) * 0.5, t < 0.5);
}`
            }, {
                label: "Sine",
                code: `fn ease_in_sine(t: f32) -> f32 { return 1.0 - cos(1.57079632679 * t); }
fn ease_out_sine(t: f32) -> f32 { return sin(1.57079632679 * t); }
fn ease_in_out_sine(t: f32) -> f32 { return -0.5 * (cos(3.14159265359 * t) - 1.0); }`
            }, {
                label: "Circular",
                code: `fn ease_in_circ(t: f32) -> f32 { return 1.0 - sqrt(1.0 - t * t); }
fn ease_out_circ(t: f32) -> f32 { return sqrt(1.0 - (t - 1.0) * (t - 1.0)); }
fn ease_in_out_circ(t: f32) -> f32
{
    return select(
        0.5 * (sqrt(1.0 - (2.0 * t - 2.0) * (2.0 * t - 2.0)) + 1.0),
        0.5 * (1.0 - sqrt(1.0 - 4.0 * t * t)),
        t < 0.5
    );
}`
            }, {
                label: "Back",
                code: `const c1: f32 = 1.70158;
const c3: f32 = c1 + 1.0;

fn ease_in_back(t: f32) -> f32 { return c3 * t * t * t - c1 * t * t; }
fn ease_out_back(t: f32) -> f32 { let u = 1.0 - t; return 1.0 + c3 * u * u * u + c1 * u * u; }
fn ease_in_out_back(t: f32) -> f32
{
    let c2 = c1 * 1.525;
    return select(
        0.5 * (pow(2.0 * t - 2.0, 2.0) * ((c2 + 1.0) * (2.0 * t - 2.0) + c2) + 2.0),
        (pow(2.0 * t, 2.0) * ((c2 + 1.0) * 2.0 * t - c2)) * 0.5,
        t < 0.5
    );
}`
            }]
        }, {
            name: "Smootherstep",
            description: "Improved smoothstep (C2 continuous)",
            author: "Ken Perlin",
            code: `fn smootherstep(edge0: f32, edge1: f32, x: f32) -> f32
{
    let v: f32 = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return v * v * v * (v * (v * 6.0 - 15.0) + 10.0);
}`
        }, {
            name: "Anti-Aliasing",
            description: "Smooth edges for SDFs or any continuous function",
            author: "Xor (mini.gmshaders.com/p/antialiasing)",
            options: [{
                label: "SDF",
                code: `// For SDFs with known pixel scale (texel size).

fn antialias_sdf(dist: f32, texel: f32) -> f32
{
    return clamp(dist / texel + 0.5, 0.0, 1.0);
}`
            }, {
                label: "Derivative (fwidth)",
                code: `fn antialias(d: f32) -> f32
{
    let w = fwidth(d);
    let scale = select(1e7, 1.0 / w, w > 0.0);
    return clamp(0.5 + 0.5 * scale * d, 0.0, 1.0);
}`
            }, {
                label: "Derivative (L2)",
                code: `fn antialias_l2(d: f32) -> f32
{
    let dxy = vec2f(dpdx(d), dpdy(d));
    let w = length(dxy);
    let scale = select(1e7, 1.0 / w, w > 0.0);
    return clamp(0.5 + 0.7 * scale * d, 0.0, 1.0);
}`
            }, {
                label: "Derivative (L2 manual)",
                code: `fn antialias_l2_dxy(d: f32, dxy: vec2f) -> f32
{
    let w = length(dxy);
    let scale = select(1e7, 1.0 / w, w > 0.0);
    return clamp(0.5 + 0.7 * scale * d, 0.0, 1.0);
}`
            }]
        }, {
            name: "Complex",
            description: "Complex number arithmetic (vec2 = real + imaginary)",
            options: [{
                label: "Multiply / Divide",
                code: `vec2 cmul(vec2 a, vec2 b)
{
    return vec2(a.x * b.x - a.y * b.y,
                a.x * b.y + a.y * b.x);
}

vec2 cdiv(vec2 a, vec2 b)
{
    float d = dot(b, b);
    return vec2(dot(a, b), a.y * b.x - a.x * b.y) / d;
}

vec2 cconj(vec2 z)
{
    return vec2(z.x, -z.y);
}`
            }, {
                label: "Exp / Log / Power",
                code: `float cabs(vec2 z) { return length(z); }
float carg(vec2 z) { return atan(z.y, z.x); }

vec2 cexp(vec2 z)
{
    return exp(z.x) * vec2(cos(z.y), sin(z.y));
}

vec2 clog(vec2 z)
{
    return vec2(log(length(z)), atan(z.y, z.x));
}

vec2 cpow(vec2 z, float n)
{
    float r = length(z);
    float theta = atan(z.y, z.x);
    return pow(r, n) * vec2(cos(n * theta), sin(n * theta));
}

vec2 csqrt(vec2 z)
{
    float r = length(z);
    float theta = atan(z.y, z.x);
    return sqrt(r) * vec2(cos(theta * 0.5), sin(theta * 0.5));
}`
            }, {
                label: "Trigonometry",
                code: `vec2 csin(vec2 z)
{
    return vec2(sin(z.x) * cosh(z.y), cos(z.x) * sinh(z.y));
}

vec2 ccos(vec2 z)
{
    return vec2(cos(z.x) * cosh(z.y), -sin(z.x) * sinh(z.y));
}`
            }]
        }]
    }, {
        name: "(TODO) SDF",
        icon: "Hexagon",
        snippets: [{
            name: "2D Shapes",
            description: "Signed distance functions for 2D primitives",
            author: "Inigo Quilez (iquilezles.org)",
            options: [{
                label: "Circle",
                code: `float sd_circle(vec2 p, float r)
{
    return length(p) - r;
}`
            }, {
                label: "Box",
                code: `float sd_box(vec2 p, vec2 b)
{
    vec2 d = abs(p) - b;
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}`
            }, {
                label: "Rounded Box",
                code: `float sd_rounded_box(vec2 p, vec2 b, vec4 r)
{
    r.xy = (p.x > 0.0) ? r.xy : r.zw;
    r.x  = (p.y > 0.0) ? r.x  : r.y;
    vec2 q = abs(p) - b + r.x;
    return min(max(q.x, q.y), 0.0) + length(max(q, 0.0)) - r.x;
}`
            }, {
                label: "Line Segment",
                code: `float sd_segment(vec2 p, vec2 a, vec2 b)
{
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}`
            }, {
                label: "Triangle",
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
                label: "Ring",
                code: `float sd_ring(vec2 p, float r, float thickness)
{
    return abs(length(p) - r) - thickness;
}`
            }, {
                label: "Capsule",
                code: `float sd_capsule(vec2 p, vec2 a, vec2 b, float r)
{
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}`
            }, {
                label: "Vesica",
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
                label: "Hexagon",
                code: `float sd_hexagon(vec2 p, float r)
{
    const vec3 k = vec3(-0.866025404, 0.5, 0.577350269);
    p = abs(p);
    p -= 2.0 * min(dot(k.xy, p), 0.0) * k.xy;
    p -= vec2(clamp(p.x, -k.z * r, k.z * r), r);
    return length(p) * sign(p.y);
}`
            }, {
                label: "Pentagon",
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
                label: "Star",
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
                label: "Arc",
                code: `float sd_arc(vec2 p, vec2 sc, float ra, float rb)
{
    p.x = abs(p.x);
    return ((sc.y * p.x > sc.x * p.y)
        ? length(p - sc * ra)
        : abs(length(p) - ra)) - rb;
}`
            }, {
                label: "Ellipse",
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
            }]
        }, {
            name: "3D Shapes",
            description: "Signed distance functions for 3D primitives",
            author: "Inigo Quilez (iquilezles.org)",
            options: [{
                label: "Sphere",
                code: `float sd_sphere(vec3 p, float r)
{
    return length(p) - r;
}`
            }, {
                label: "Box",
                code: `float sd_box_3d(vec3 p, vec3 b)
{
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}`
            }, {
                label: "Torus",
                code: `float sd_torus(vec3 p, vec2 t)
{
    vec2 q = vec2(length(p.xz) - t.x, p.y);
    return length(q) - t.y;
}`
            }, {
                label: "Capsule",
                code: `float sd_capsule(vec3 p, vec3 a, vec3 b, float r)
{
    vec3 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}`
            }, {
                label: "Cylinder",
                code: `float sd_capped_cylinder(vec3 p, float r, float h)
{
    vec2 d = abs(vec2(length(p.xz), p.y)) - vec2(r, h);
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}`
            }, {
                label: "Cone",
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
                label: "Plane",
                code: `float sd_plane(vec3 p, vec3 n, float h)
{
    return dot(p, n) + h;
}`
            }, {
                label: "Round Cone",
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
                label: "Hex Prism",
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
                label: "Round Box",
                code: `float sd_round_box(vec3 p, vec3 b, float r)
{
    vec3 q = abs(p) - b + r;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}`
            }]
        }, {
            name: "Operations",
            description: "Combine SDFs: union, intersection, subtraction, and smooth variants",
            author: "Inigo Quilez (iquilezles.org)",
            options: [{
                label: "Union",
                code: "float op_union(float d1, float d2) { return min(d1, d2); }"
            }, {
                label: "Intersect",
                code: "float op_intersect(float d1, float d2) { return max(d1, d2); }"
            }, {
                label: "Subtract",
                code: "float op_subtract(float d1, float d2) { return max(-d1, d2); }"
            }, {
                label: "Smooth Union",
                code: `float op_smooth_union(float d1, float d2, float k)
{
    float h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return mix(d2, d1, h) - k * h * (1.0 - h);
}`
            }, {
                label: "Smooth Intersect",
                code: `float op_smooth_intersect(float d1, float d2, float k)
{
    float h = clamp(0.5 - 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return mix(d2, d1, h) + k * h * (1.0 - h);
}`
            }, {
                label: "Smooth Subtract",
                code: `float op_smooth_subtract(float d1, float d2, float k)
{
    float h = clamp(0.5 - 0.5 * (d2 + d1) / k, 0.0, 1.0);
    return mix(d2, -d1, h) + k * h * (1.0 - h);
}`
            }]
        }, {
            name: "Ray Marching",
            description: "Sphere-tracing loop, normals, lighting, camera",
            options: [{
                label: "Ray March Loop",
                code: `#define MAX_STEPS 128
#define MAX_DIST  100.0
#define SURF_DIST 0.001

float ray_march(vec3 ro, vec3 rd)
{
    float t = 0.0;
    for (int i = 0; i < MAX_STEPS; i++)
    {
        vec3 p = ro + rd * t;
        float d = scene(p);
        if (d < SURF_DIST) break;
        t += d;
        if (t > MAX_DIST) break;
    }
    return t;
}`
            }, {
                label: "Normal",
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
                label: "Lighting",
                code: `vec3 lighting(vec3 p, vec3 normal, vec3 rd, vec3 light_dir, vec3 col)
{
    float diff = max(dot(normal, light_dir), 0.0);
    vec3 half_dir = normalize(light_dir - rd);
    float spec = pow(max(dot(normal, half_dir), 0.0), 32.0);
    vec3 ambient = 0.05 * col;
    return ambient + col * diff + vec3(1.0) * spec * 0.5;
}`
            }, {
                label: "Camera",
                code: `mat3 camera(vec3 eye, vec3 target, float roll)
{
    vec3 cw = normalize(target - eye);
    vec3 cp = vec3(sin(roll), cos(roll), 0.0);
    vec3 cu = normalize(cross(cw, cp));
    vec3 cv = cross(cu, cw);
    return mat3(cu, cv, cw);
}`
            }]
        }]
    }, {
        name: "(TODO) Raytracing",
        icon: "Globe",
        snippets: [{
            name: "Ray Setup",
            description: "Camera ray generation for raytracing",
            options: [{
                label: "Perspective",
                code: `// ro: ray origin, rd: ray direction (normalized).
// uv: fragCoord.xy / resolution.xy in [0,1].

vec3 ray_dir(vec2 uv, vec3 ro, vec3 target, float fov)
{
    vec3 fwd = normalize(target - ro);
    vec3 right = normalize(cross(vec3(0.0, 1.0, 0.0), fwd));
    vec3 up = cross(fwd, right);
    vec2 ndc = uv * 2.0 - 1.0;
    float aspect = iResolution.x / iResolution.y;
    return normalize(fwd + (right * ndc.x * aspect + up * ndc.y) * tan(fov * 0.5));
}`
            }, {
                label: "Orthographic",
                code: `// Orthographic: parallel rays. rd = normalize(target - ro) for all pixels.
// Ray origin for pixel: ro + ortho_offset(uv, ro, target, scale).

vec3 ortho_offset(vec2 uv, vec3 ro, vec3 target, float scale)
{
    vec3 fwd = normalize(target - ro);
    vec3 right = normalize(cross(vec3(0.0, 1.0, 0.0), fwd));
    vec3 up = cross(fwd, right);
    vec2 ndc = uv * 2.0 - 1.0;
    float aspect = iResolution.x / iResolution.y;
    return (right * ndc.x * aspect + up * ndc.y) * scale;
}`
            }]
        }, {
            name: "Ray Shapes",
            description: "Ray-primitive intersections (analytic)",
            options: [{
                label: "Sphere",
                code: `// Returns (t, 1) if hit, (t_max, 0) if miss. t is distance along ray.

vec2 ray_sphere(vec3 ro, vec3 rd, vec3 center, float radius)
{
    vec3 oc = ro - center;
    float b = dot(oc, rd);
    float c = dot(oc, oc) - radius * radius;
    float d = b * b - c;
    if (d < 0.0) return vec2(1e10, 0.0);
    float t = -b - sqrt(d);
    if (t < 0.0) t = -b + sqrt(d);
    if (t < 0.0) return vec2(1e10, 0.0);
    return vec2(t, 1.0);
}

vec3 sphere_normal(vec3 p, vec3 center) { return normalize(p - center); }`
            }, {
                label: "Plane",
                code: `// Plane: dot(p, n) + d = 0. n is unit normal, d is offset from origin.

vec2 ray_plane(vec3 ro, vec3 rd, vec3 n, float d)
{
    float denom = dot(rd, n);
    if (abs(denom) < 1e-6) return vec2(1e10, 0.0);
    float t = -(dot(ro, n) + d) / denom;
    if (t < 0.0) return vec2(1e10, 0.0);
    return vec2(t, 1.0);
}`
            }, {
                label: "Box (AABB)",
                code: `// Slab method. box_min, box_max = AABB bounds.

vec2 ray_box(vec3 ro, vec3 rd, vec3 box_min, vec3 box_max)
{
    vec3 inv_rd = 1.0 / rd;
    vec3 t0 = (box_min - ro) * inv_rd;
    vec3 t1 = (box_max - ro) * inv_rd;
    vec3 tmin = min(t0, t1);
    vec3 tmax = max(t0, t1);
    float t_near = max(max(tmin.x, tmin.y), tmin.z);
    float t_far  = min(min(tmax.x, tmax.y), tmax.z);
    if (t_near > t_far || t_far < 0.0) return vec2(1e10, 0.0);
    float t = t_near < 0.0 ? t_far : t_near;
    return vec2(t, 1.0);
}`
            }, {
                label: "Triangle",
                code: `// Möller–Trumbore. Returns (t, u, v) — t = hit distance, (u,v) barycentrics. Miss: t < 0.

vec3 ray_triangle(vec3 ro, vec3 rd, vec3 v0, vec3 v1, vec3 v2)
{
    vec3 e1 = v1 - v0;
    vec3 e2 = v2 - v0;
    vec3 h = cross(rd, e2);
    float a = dot(e1, h);
    if (abs(a) < 1e-8) return vec3(-1.0, 0.0, 0.0);
    float f = 1.0 / a;
    vec3 s = ro - v0;
    float u = f * dot(s, h);
    if (u < 0.0 || u > 1.0) return vec3(-1.0, 0.0, 0.0);
    vec3 q = cross(s, e1);
    float v = f * dot(rd, q);
    if (v < 0.0 || u + v > 1.0) return vec3(-1.0, 0.0, 0.0);
    float t = f * dot(e2, q);
    if (t < 1e-6) return vec3(-1.0, 0.0, 0.0);
    return vec3(t, u, v);
}

vec3 triangle_normal(vec3 v0, vec3 v1, vec3 v2) { return normalize(cross(v1 - v0, v2 - v0)); }`
            }]
        }]
    }];
export const WGSL_SHADER_FUNCTIONS = [{
    name: "abs",
    signature: "abs(x)",
    description: "Absolute value"
}, {
    name: "acos",
    signature: "acos(x)",
    description: "Arc cosine, returns radians"
}, {
    name: "acosh",
    signature: "acosh(x)",
    description: "Hyperbolic arc cosine"
}, {
    name: "all",
    signature: "all(e)",
    description: "Returns true if all components of a boolean vector are true"
}, {
    name: "any",
    signature: "any(e)",
    description: "Returns true if any component of a boolean vector is true"
}, {
    name: "arrayLength",
    signature: "arrayLength(a)",
    description: "Returns the number of elements in a runtime-sized array"
}, {
    name: "asin",
    signature: "asin(x)",
    description: "Arc sine, returns radians"
}, {
    name: "asinh",
    signature: "asinh(x)",
    description: "Hyperbolic arc sine"
}, {
    name: "atan",
    signature: "atan(x)",
    description: "Arc tangent, returns radians"
}, {
    name: "atan2",
    signature: "atan2(y, x)",
    description: "Arc tangent of y/x, handles quadrants"
}, {
    name: "atanh",
    signature: "atanh(x)",
    description: "Hyperbolic arc tangent"
}, {
    name: "atomicAdd",
    signature: "atomicAdd(atomic_ptr, v)",
    description: "Atomically add v and return old value"
}, {
    name: "atomicLoad",
    signature: "atomicLoad(atomic_ptr)",
    description: "Atomically load the value of an atomic"
}, {
    name: "atomicStore",
    signature: "atomicStore(atomic_ptr, v)",
    description: "Atomically store v to an atomic"
}, {
    name: "bitcast",
    signature: "bitcast<T>(v)",
    description: "Reinterpret the bits of v as type T"
}, {
    name: "ceil",
    signature: "ceil(x)",
    description: "Round up to integer"
}, {
    name: "clamp",
    signature: "clamp(x, min, max)",
    description: "Clamp x between min and max"
}, {
    name: "cos",
    signature: "cos(x)",
    description: "Cosine of angle in radians"
}, {
    name: "cosh",
    signature: "cosh(x)",
    description: "Hyperbolic cosine"
}, {
    name: "countLeadingZeros",
    signature: "countLeadingZeros(x)",
    description: "Number of leading 0 bits"
}, {
    name: "countOneBits",
    signature: "countOneBits(x)",
    description: "Number of 1 bits in the binary representation (popcount)"
}, {
    name: "countTrailingZeros",
    signature: "countTrailingZeros(x)",
    description: "Number of trailing 0 bits (index of lowest set bit)"
}, {
    name: "cross",
    signature: "cross(a, b)",
    description: "Cross product (vec3)"
}, {
    name: "degrees",
    signature: "degrees(radians)",
    description: "Convert radians to degrees"
}, {
    name: "determinant",
    signature: "determinant(m)",
    description: "Determinant of a square matrix"
}, {
    name: "distance",
    signature: "distance(a, b)",
    description: "Distance between two points"
}, {
    name: "dot",
    signature: "dot(a, b)",
    description: "Dot product"
}, {
    name: "dpdx",
    signature: "dpdx(e)",
    description: "Partial derivative in x (screen-space)"
}, {
    name: "dpdxCoarse",
    signature: "dpdxCoarse(e)",
    description: "Coarse partial derivative in x"
}, {
    name: "dpdxFine",
    signature: "dpdxFine(e)",
    description: "Fine partial derivative in x"
}, {
    name: "dpdy",
    signature: "dpdy(e)",
    description: "Partial derivative in y (screen-space)"
}, {
    name: "dpdyCoarse",
    signature: "dpdyCoarse(e)",
    description: "Coarse partial derivative in y"
}, {
    name: "dpdyFine",
    signature: "dpdyFine(e)",
    description: "Fine partial derivative in y"
}, {
    name: "exp",
    signature: "exp(x)",
    description: "e raised to the power x"
}, {
    name: "exp2",
    signature: "exp2(x)",
    description: "2 raised to the power x"
}, {
    name: "extractBits",
    signature: "extractBits(e, offset, count)",
    description: "Extract count bits starting at offset from integer e"
}, {
    name: "faceForward",
    signature: "faceForward(N, I, Nref)",
    description: "Flip N if facing away from I"
}, {
    name: "firstLeadingBit",
    signature: "firstLeadingBit(x)",
    description: "Index of the most significant 1 bit"
}, {
    name: "firstTrailingBit",
    signature: "firstTrailingBit(x)",
    description: "Index of the least significant 1 bit"
}, {
    name: "floor",
    signature: "floor(x)",
    description: "Round down to integer"
}, {
    name: "fma",
    signature: "fma(a, b, c)",
    description: "Fused multiply-add: a*b + c"
}, {
    name: "fract",
    signature: "fract(x)",
    description: "Fractional part (x - floor(x))"
}, {
    name: "frexp",
    signature: "frexp(x)",
    description: "Splits x into fractional mantissa and integer exponent"
}, {
    name: "fwidth",
    signature: "fwidth(e)",
    description: "abs(dpdx(e)) + abs(dpdy(e))"
}, {
    name: "fwidthCoarse",
    signature: "fwidthCoarse(e)",
    description: "abs(dpdxCoarse(e)) + abs(dpdyCoarse(e))"
}, {
    name: "fwidthFine",
    signature: "fwidthFine(e)",
    description: "abs(dpdxFine(e)) + abs(dpdyFine(e))"
}, {
    name: "insertBits",
    signature: "insertBits(e, newbits, offset, count)",
    description: "Replace count bits in e starting at offset with newbits"
}, {
    name: "inverseSqrt",
    signature: "inverseSqrt(x)",
    description: "1 / sqrt(x)"
}, {
    name: "ldexp",
    signature: "ldexp(e, exp)",
    description: "e * 2^exp (construct float from mantissa and exponent)"
}, {
    name: "length",
    signature: "length(v)",
    description: "Length of vector"
}, {
    name: "log",
    signature: "log(x)",
    description: "Natural logarithm"
}, {
    name: "log2",
    signature: "log2(x)",
    description: "Base-2 logarithm"
}, {
    name: "max",
    signature: "max(x, y)",
    description: "Maximum of x and y"
}, {
    name: "min",
    signature: "min(x, y)",
    description: "Minimum of x and y"
}, {
    name: "mix",
    signature: "mix(a, b, t)",
    description: "Linear interpolation: a*(1-t) + b*t"
}, {
    name: "modf",
    signature: "modf(x)",
    description: "Returns struct with fract and whole parts of x"
}, {
    name: "normalize",
    signature: "normalize(v)",
    description: "Unit vector in same direction"
}, {
    name: "pack2x16float",
    signature: "pack2x16float(v)",
    description: "Pack vec2f to u32 as two f16 values"
}, {
    name: "pack2x16snorm",
    signature: "pack2x16snorm(v)",
    description: "Pack vec2f to u32 as two snorm16 values"
}, {
    name: "pack2x16unorm",
    signature: "pack2x16unorm(v)",
    description: "Pack vec2f to u32 as two unorm16 values"
}, {
    name: "pack4x8snorm",
    signature: "pack4x8snorm(v)",
    description: "Pack vec4f to u32 as four snorm8 values"
}, {
    name: "pack4x8unorm",
    signature: "pack4x8unorm(v)",
    description: "Pack vec4f to u32 as four unorm8 values"
}, {
    name: "pow",
    signature: "pow(x, y)",
    description: "x raised to the power y"
}, {
    name: "quantizeToF16",
    signature: "quantizeToF16(x)",
    description: "Quantize f32 to f16 precision"
}, {
    name: "radians",
    signature: "radians(degrees)",
    description: "Convert degrees to radians"
}, {
    name: "reflect",
    signature: "reflect(I, N)",
    description: "Reflection of incident vector"
}, {
    name: "refract",
    signature: "refract(I, N, eta)",
    description: "Refraction of incident vector"
}, {
    name: "reverseBits",
    signature: "reverseBits(x)",
    description: "Reverse the bits of an integer"
}, {
    name: "round",
    signature: "round(x)",
    description: "Round to nearest integer (ties to even)"
}, {
    name: "saturate",
    signature: "saturate(x)",
    description: "Clamp x to [0, 1]"
}, {
    name: "select",
    signature: "select(f, t, cond)",
    description: "Returns t if cond is true, else f"
}, {
    name: "sign",
    signature: "sign(x)",
    description: "Returns -1, 0, or 1"
}, {
    name: "sin",
    signature: "sin(x)",
    description: "Sine of angle in radians"
}, {
    name: "sinh",
    signature: "sinh(x)",
    description: "Hyperbolic sine"
}, {
    name: "smoothstep",
    signature: "smoothstep(a, b, x)",
    description: "Hermite interpolation between a and b"
}, {
    name: "sqrt",
    signature: "sqrt(x)",
    description: "Square root"
}, {
    name: "step",
    signature: "step(edge, x)",
    description: "0.0 if x < edge, else 1.0"
}, {
    name: "storageBarrier",
    signature: "storageBarrier()",
    description: "Memory barrier for storage buffer access"
}, {
    name: "tan",
    signature: "tan(x)",
    description: "Tangent of angle in radians"
}, {
    name: "tanh",
    signature: "tanh(x)",
    description: "Hyperbolic tangent"
}, {
    name: "textureDimensions",
    signature: "textureDimensions(t)",
    description: "Returns the dimensions of a texture"
}, {
    name: "textureGather",
    signature: "textureGather(component, t, s, uv)",
    description: "Gather one component from the four texels used in bilinear filtering"
}, {
    name: "textureGatherCompare",
    signature: "textureGatherCompare(t, s, uv, depth_ref)",
    description: "Gather depth comparison results from four texels"
}, {
    name: "textureLoad",
    signature: "textureLoad(t, coords, level)",
    description: "Load a texel without a sampler"
}, {
    name: "textureNumLayers",
    signature: "textureNumLayers(t)",
    description: "Number of array layers in a texture"
}, {
    name: "textureNumLevels",
    signature: "textureNumLevels(t)",
    description: "Number of mip levels in a texture"
}, {
    name: "textureNumSamples",
    signature: "textureNumSamples(t)",
    description: "Number of samples per texel in a multisampled texture"
}, {
    name: "textureSample",
    signature: "textureSample(t, s, uv)",
    description: "Sample a texture at UV coordinates"
}, {
    name: "textureSampleBaseClampToEdge",
    signature: "textureSampleBaseClampToEdge(t, s, uv)",
    description: "Sample base mip level clamped to edge"
}, {
    name: "textureSampleBias",
    signature: "textureSampleBias(t, s, uv, bias)",
    description: "Sample a texture with LOD bias"
}, {
    name: "textureSampleCompare",
    signature: "textureSampleCompare(t, s, uv, depth_ref)",
    description: "Sample a depth texture with comparison"
}, {
    name: "textureSampleCompareLevel",
    signature: "textureSampleCompareLevel(t, s, uv, depth_ref, level)",
    description: "Sample a depth texture with comparison at explicit mip level"
}, {
    name: "textureSampleGrad",
    signature: "textureSampleGrad(t, s, uv, ddx, ddy)",
    description: "Sample a texture with explicit gradients"
}, {
    name: "textureSampleLevel",
    signature: "textureSampleLevel(t, s, uv, level)",
    description: "Sample a texture at an explicit mip level"
}, {
    name: "textureStore",
    signature: "textureStore(t, coords, value)",
    description: "Write a texel to a storage texture"
}, {
    name: "transpose",
    signature: "transpose(m)",
    description: "Transpose of a matrix"
}, {
    name: "trunc",
    signature: "trunc(x)",
    description: "Round toward zero"
}, {
    name: "unpack2x16float",
    signature: "unpack2x16float(v)",
    description: "Unpack u32 to vec2f from two f16 values"
}, {
    name: "unpack2x16snorm",
    signature: "unpack2x16snorm(v)",
    description: "Unpack u32 to vec2f from two snorm16 values"
}, {
    name: "unpack2x16unorm",
    signature: "unpack2x16unorm(v)",
    description: "Unpack u32 to vec2f from two unorm16 values"
}, {
    name: "unpack4x8snorm",
    signature: "unpack4x8snorm(v)",
    description: "Unpack u32 to vec4f from four snorm8 values"
}, {
    name: "unpack4x8unorm",
    signature: "unpack4x8unorm(v)",
    description: "Unpack u32 to vec4f from four unorm8 values"
}, {
    name: "workgroupBarrier",
    signature: "workgroupBarrier()",
    description: "Memory barrier for workgroup memory access"
}, {
    name: "workgroupUniformLoad",
    signature: "workgroupUniformLoad(p)",
    description: "Load a uniform value from workgroup memory with implicit barrier"
}];

export const GLSL_CODE_UTILS = {
    "hash_u": ``
};
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
            name: "Random",
            description: "Pseudorandom number from float/vec2/vec3/vec4 seed",
            options: [{
                label: "1D → 1D",
                code: `float rand1(float p)
{
    p = fract(p * 0.1031);
    p *= p + 33.33;
    p *= p + p;
    return fract(p);
}`
            }, {
                label: "2D → 1D",
                code: `float rand1(vec2 p)
{
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}`
            }, {
                label: "2D → 2D",
                code: `fn rand2(p: vec2f) -> vec2f
{
    var p3 = fract(vec3f(p.xyx) * vec3f(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}`
            }, {
                label: "2D → 3D",
                code: `fn rand3(p: vec2f) -> vec3f
{
    var p4 = fract(vec4f(p.xyx, p.y) * vec4f(0.1031, 0.1030, 0.0973, 0.1099));
    p4 += dot(p4, p4.wzxy + 33.33);
    return fract((p4.xxz + p4.yww) * p4.zyx);
}`
            }, {
                label: "3D → 1D",
                code: `fn rand1(p: vec3f) -> f32
{
    var p4 = fract(vec4f(p.xyzx) * vec4f(0.1031, 0.1030, 0.0973, 0.1099));
    p4 += dot(p4, p4.wzxy + 33.33);
    return fract((p4.x + p4.y + p4.z) * p4.w);
}`
            }, {
                label: "3D → 2D",
                code: `fn rand2(p: vec3f) -> vec2f
{
    var p4 = fract(vec4f(p.xyzx) * vec4f(0.1031, 0.1030, 0.0973, 0.1099));
    p4 += dot(p4, p4.wzxy + 33.33);
    return fract((p4.xx + p4.yz) * p4.zy);
}`
            }, {
                label: "3D → 3D",
                code: `fn rand3(p: vec3f) -> vec3f
{
    var p4 = fract(vec4f(p.xyzx) * vec4f(0.1031, 0.1030, 0.0973, 0.1099));
    p4 += dot(p4, p4.wzxy + 33.33);
    return fract((p4.xxz + p4.yww) * p4.zyx);
}`
            }, {
                label: "1D → 4D",
                code: `fn rand4(p: f32) -> vec4f
{
    var p4 = fract(vec4f(p) * vec4f(0.1031, 0.1030, 0.0973, 0.1099));
    p4 += dot(p4, p4.wzxy + 33.33);
    return fract(vec4f(
        (p4.x + p4.y) * p4.z,
        (p4.y + p4.z) * p4.w,
        (p4.z + p4.w) * p4.x,
        (p4.w + p4.x) * p4.y
    ));
}`
            }, {
                label: "4D → 4D",
                code: `fn rand4(p: vec4f) -> vec4f
{
    var p4 = fract(p * vec4f(0.1031, 0.1030, 0.0973, 0.1099));
    p4 += dot(p4, p4.wzxy + 33.33);
    return fract(vec4f(
        (p4.x + p4.y) * p4.z,
        (p4.y + p4.z) * p4.w,
        (p4.z + p4.w) * p4.x,
        (p4.w + p4.x) * p4.y
    ));
}`
            }]
        }, {
            name: "Integer Hash",
            description: "Pseudorandom from int/ivec2/ivec3 seed (GLSL 300 es)",
            options: [{
                label: "int → float",
                code: `uint hash_u(uint n)
{
    n = (n ^ 61u) ^ (n >> 16u);
    n *= 9u;
    n = n ^ (n >> 4u);
    n *= 0x27d4eb2du;
    return n ^ (n >> 15u);
}

float hash_i(int n)
{
    return float(hash_u(uint(n))) / 4294967295.0;
}`
            }, {
                label: "ivec2 → float",
                code: `uint hash_u(uint n)
{
    n = (n ^ 61u) ^ (n >> 16u);
    n *= 9u;
    n = n ^ (n >> 4u);
    n *= 0x27d4eb2du;
    return n ^ (n >> 15u);
}

float hash_i(ivec2 p)
{
    uint n = hash_u(uint(p.x)) + hash_u(uint(p.y)) * 57u;
    return float(hash_u(n)) / 4294967295.0;
}`
            }, {
                label: "ivec2 → vec2",
                code: `uint hash_u(uint n)
{
    n = (n ^ 61u) ^ (n >> 16u);
    n *= 9u;
    n = n ^ (n >> 4u);
    n *= 0x27d4eb2du;
    return n ^ (n >> 15u);
}

vec2 hash_i2(ivec2 p)
{
    uint n = hash_u(uint(p.x)) + hash_u(uint(p.y)) * 57u;
    return vec2(
        float(hash_u(n)) / 4294967295.0,
        float(hash_u(n + 1u)) / 4294967295.0
    );
}`
            }, {
                label: "ivec3 → float",
                code: `uint hash_u(uint n)
{
    n = (n ^ 61u) ^ (n >> 16u);
    n *= 9u;
    n = n ^ (n >> 4u);
    n *= 0x27d4eb2du;
    return n ^ (n >> 15u);
}

float hash_i(ivec3 p)
{
    uint n = hash_u(uint(p.x)) + hash_u(uint(p.y)) * 57u + hash_u(uint(p.z)) * 131u;
    return float(hash_u(n)) / 4294967295.0;
}`
            }, {
                label: "ivec3 → vec2",
                code: `uint hash_u(uint n)
{
    n = (n ^ 61u) ^ (n >> 16u);
    n *= 9u;
    n = n ^ (n >> 4u);
    n *= 0x27d4eb2du;
    return n ^ (n >> 15u);
}

vec2 hash_i2(ivec3 p)
{
    uint n = hash_u(uint(p.x)) + hash_u(uint(p.y)) * 57u + hash_u(uint(p.z)) * 131u;
    return vec2(
        float(hash_u(n)) / 4294967295.0,
        float(hash_u(n + 1u)) / 4294967295.0
    );
}`
            }, {
                label: "ivec3 → vec3",
                code: `uint hash_u(uint n)
{
    n = (n ^ 61u) ^ (n >> 16u);
    n *= 9u;
    n = n ^ (n >> 4u);
    n *= 0x27d4eb2du;
    return n ^ (n >> 15u);
}

vec3 hash_i3(ivec3 p)
{
    uint n = hash_u(uint(p.x)) + hash_u(uint(p.y)) * 57u + hash_u(uint(p.z)) * 131u;
    return vec3(
        float(hash_u(n)) / 4294967295.0,
        float(hash_u(n + 1u)) / 4294967295.0,
        float(hash_u(n + 2u)) / 4294967295.0
    );
}`
            }]
        }, {
            name: "Value Noise",
            description: "Smooth interpolated value noise",
            options: [{
                label: "2D",
                code: `float rand1(vec2 p)
{
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float value_noise(vec2 p)
{
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = rand1(i);
    float b = rand1(i + vec2(1.0, 0.0));
    float c = rand1(i + vec2(0.0, 1.0));
    float d = rand1(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}`
            }, {
                label: "3D",
                code: `float rand1(vec3 p)
{
    vec4 p4 = fract(vec4(p.xyzx) * vec4(0.1031, 0.1030, 0.0973, 0.1099));
    p4 += dot(p4, p4.wzxy + 33.33);
    return fract((p4.x + p4.y + p4.z) * p4.w);
}

float value_noise(vec3 p)
{
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = rand1(i);
    float b = rand1(i + vec3(1.0, 0.0, 0.0));
    float c = rand1(i + vec3(0.0, 1.0, 0.0));
    float d = rand1(i + vec3(1.0, 1.0, 0.0));
    float e = rand1(i + vec3(0.0, 0.0, 1.0));
    float f0 = rand1(i + vec3(1.0, 0.0, 1.0));
    float g = rand1(i + vec3(0.0, 1.0, 1.0));
    float h = rand1(i + vec3(1.0, 1.0, 1.0));
    return mix(
        mix(mix(a, b, f.x), mix(c, d, f.x), f.y),
        mix(mix(e, f0, f.x), mix(g, h, f.x), f.y),
        f.z
    );
}`
            }, {
                label: "4D",
                code: `vec4 rand4(vec4 p)
{
    vec4 p4 = fract(p * vec4(0.1031, 0.1030, 0.0973, 0.1099));
    p4 += dot(p4, p4.wzxy + 33.33);
    return fract(vec4(
        (p4.x + p4.y) * p4.z,
        (p4.y + p4.z) * p4.w,
        (p4.z + p4.w) * p4.x,
        (p4.w + p4.x) * p4.y
    ));
}

float value_noise(vec4 p)
{
    vec4 i = floor(p);
    vec4 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float n0000 = rand4(i).x;
    float n1000 = rand4(i + vec4(1, 0, 0, 0)).x;
    float n0100 = rand4(i + vec4(0, 1, 0, 0)).x;
    float n1100 = rand4(i + vec4(1, 1, 0, 0)).x;
    float n0010 = rand4(i + vec4(0, 0, 1, 0)).x;
    float n1010 = rand4(i + vec4(1, 0, 1, 0)).x;
    float n0110 = rand4(i + vec4(0, 1, 1, 0)).x;
    float n1110 = rand4(i + vec4(1, 1, 1, 0)).x;
    float n0001 = rand4(i + vec4(0, 0, 0, 1)).x;
    float n1001 = rand4(i + vec4(1, 0, 0, 1)).x;
    float n0101 = rand4(i + vec4(0, 1, 0, 1)).x;
    float n1101 = rand4(i + vec4(1, 1, 0, 1)).x;
    float n0011 = rand4(i + vec4(0, 0, 1, 1)).x;
    float n1011 = rand4(i + vec4(1, 0, 1, 1)).x;
    float n0111 = rand4(i + vec4(0, 1, 1, 1)).x;
    float n1111 = rand4(i + vec4(1, 1, 1, 1)).x;
    return mix(
        mix(mix(mix(n0000, n1000, f.x), mix(n0100, n1100, f.x), f.y),
            mix(mix(n0010, n1010, f.x), mix(n0110, n1110, f.x), f.y), f.z),
        mix(mix(mix(n0001, n1001, f.x), mix(n0101, n1101, f.x), f.y),
            mix(mix(n0011, n1011, f.x), mix(n0111, n1111, f.x), f.y), f.z),
        f.w
    );
}`
            }]
        }, {
            name: "Simplex Noise",
            description: "Classic simplex noise",
            author: "Ashima Arts, Stefan Gustavson (webgl-noise)",
            options: [{
                label: "2D",
                code: `vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec2 mod289(vec2 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec3 permute(vec3 x) { return mod289(((x * 34.0) + 1.0) * x); }

float simplex_noise(vec2 v)
{
    const vec4 C = vec4(0.211324865405187, 0.366025403784439,
                       -0.577350269189626, 0.024390243902439);
    vec2 i = floor(v + dot(v, C.yy));
    vec2 x0 = v - i + dot(i, C.xx);
    vec2 i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec4 x12 = x0.xyxy + C.xxzz;
    x12.xy -= i1;
    i = mod289(i);
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
                label: "3D",
                code: `vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 permute(vec4 x) { return mod289(((x * 34.0) + 1.0) * x); }
vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

float simplex_noise(vec3 v)
{
    const vec2 C = vec2(1.0 / 6.0, 1.0 / 3.0);
    const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
    vec3 i = floor(v + dot(v, C.yyy));
    vec3 x0 = v - i + dot(i, C.xxx);
    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min(g.xyz, l.zxy);
    vec3 i2 = max(g.xyz, l.zxy);
    vec3 x1 = x0 - i1 + C.xxx;
    vec3 x2 = x0 - i2 + C.yyy;
    vec3 x3 = x0 - D.yyy;
    i = mod289(i);
    vec4 p = permute(permute(permute(
        i.z + vec4(0.0, i1.z, i2.z, 1.0))
        + i.y + vec4(0.0, i1.y, i2.y, 1.0))
        + i.x + vec4(0.0, i1.x, i2.x, 1.0));
    float n_ = 0.142857142857;
    vec3 ns = n_ * D.wyz - D.xzx;
    vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_);
    vec4 x = x_ * ns.x + ns.yyyy;
    vec4 y = y_ * ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);
    vec4 b0 = vec4(x.xy, y.xy);
    vec4 b1 = vec4(x.zw, y.zw);
    vec4 s0 = floor(b0) * 2.0 + 1.0;
    vec4 s1 = floor(b1) * 2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));
    vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
    vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;
    vec3 p0 = vec3(a0.xy, h.x);
    vec3 p1 = vec3(a0.zw, h.y);
    vec3 p2 = vec3(a1.xy, h.z);
    vec3 p3 = vec3(a1.zw, h.w);
    vec4 norm = taylorInvSqrt(vec4(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;
    vec4 m = max(0.5 - vec4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.0);
    m = m * m;
    return 105.0 * dot(m * m, vec4(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));
}`
            }, {
                label: "4D",
                code: `vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
float mod289(float x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 permute(vec4 x) { return mod289(((x * 34.0) + 1.0) * x); }
float permute(float x) { return mod289(((x * 34.0) + 1.0) * x); }
vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }
float taylorInvSqrt(float r) { return 1.79284291400159 - 0.85373472095314 * r; }

vec4 grad4(float j, vec4 ip)
{
    const vec4 ones = vec4(1.0, 1.0, 1.0, -1.0);
    vec4 p, s;
    p.xyz = floor(fract(vec3(j) * ip.xyz) * 7.0) * ip.z - 1.0;
    p.w = 1.5 - dot(abs(p.xyz), ones.xyz);
    s = vec4(lessThan(p, vec4(0.0)));
    p.xyz = p.xyz + (s.xyz * 2.0 - 1.0) * s.www;
    return p;
}

float simplex_noise(vec4 v)
{
    const float F4 = 0.309016994374947451;
    const vec4 C = vec4(0.138196601125011, 0.276393202250021, 0.414589803375032, -0.447213595499958);
    vec4 i = floor(v + dot(v, vec4(F4)));
    vec4 x0 = v - i + dot(i, C.xxxx);
    vec4 i0;
    vec3 isX = step(x0.yzw, x0.xxx);
    vec3 isYZ = step(x0.zww, x0.yyz);
    i0.x = isX.x + isX.y + isX.z;
    i0.yzw = 1.0 - isX;
    i0.y += isYZ.x + isYZ.y;
    i0.zw += 1.0 - isYZ.xy;
    i0.z += isYZ.z;
    i0.w += 1.0 - isYZ.z;
    vec4 i3 = clamp(i0, 0.0, 1.0);
    vec4 i2 = clamp(i0 - 1.0, 0.0, 1.0);
    vec4 i1 = clamp(i0 - 2.0, 0.0, 1.0);
    vec4 x1 = x0 - i1 + C.xxxx;
    vec4 x2 = x0 - i2 + C.yyyy;
    vec4 x3 = x0 - i3 + C.zzzz;
    vec4 x4 = x0 + C.wwww;
    i = mod289(i);
    float j0 = permute(permute(permute(permute(i.w) + i.z) + i.y) + i.x);
    vec4 j1 = permute(permute(permute(permute(
        i.w + vec4(i1.w, i2.w, i3.w, 1.0))
        + i.z + vec4(i1.z, i2.z, i3.z, 1.0))
        + i.y + vec4(i1.y, i2.y, i3.y, 1.0))
        + i.x + vec4(i1.x, i2.x, i3.x, 1.0));
    vec4 ip = vec4(1.0 / 294.0, 1.0 / 49.0, 1.0 / 7.0, 0.0);
    vec4 p0 = grad4(j0, ip);
    vec4 p1 = grad4(j1.x, ip);
    vec4 p2 = grad4(j1.y, ip);
    vec4 p3 = grad4(j1.z, ip);
    vec4 p4 = grad4(j1.w, ip);
    vec4 norm = taylorInvSqrt(vec4(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;
    p4 *= taylorInvSqrt(dot(p4, p4));
    vec3 m0 = max(0.6 - vec3(dot(x0, x0), dot(x1, x1), dot(x2, x2)), 0.0);
    vec2 m1 = max(0.6 - vec2(dot(x3, x3), dot(x4, x4)), 0.0);
    m0 = m0 * m0;
    m1 = m1 * m1;
    return 49.0 * (dot(m0 * m0, vec3(dot(p0, x0), dot(p1, x1), dot(p2, x2)))
        + dot(m1 * m1, vec2(dot(p3, x3), dot(p4, x4))));
}`
            }]
        }, {
            name: "FBM (Fractal Brownian Motion)",
            description: "Layered noise with configurable octaves",
            options: [{
                label: "2D",
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
                label: "3D",
                code: `float fbm(vec3 p)
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
            }]
        }, {
            name: "Voronoi",
            description: "Cell noise returning F1 distance, F2 distance, and cell ID (includes rand)",
            author: "Georgy Voronoi (Voronoi diagrams)",
            options: [{
                label: "2D",
                code: `vec3 voronoi(vec2 p)
{
    vec2 i = floor(p);
    vec2 f = fract(p);
    float f1 = 1.0, f2 = 1.0;
    vec2 cell_id = vec2(0.0);
    for (int y = -1; y <= 1; y++)
    {
        for (int x = -1; x <= 1; x++)
        {
            vec2 neighbor = vec2(float(x), float(y));
            vec2 point = rand2(i + neighbor);
            float dist = length(neighbor + point - f);
            if (dist < f1) { f2 = f1; f1 = dist; cell_id = i + neighbor; }
            else if (dist < f2) { f2 = dist; }
        }
    }
    return vec3(f1, f2, rand1(cell_id));
}`
            }, {
                label: "3D",
                code: `vec3 voronoi(vec3 p)
{
    vec3 i = floor(p);
    vec3 f = fract(p);
    float f1 = 1.0, f2 = 1.0;
    vec3 cell_id = vec3(0.0);
    for (int z = -1; z <= 1; z++)
    {
        for (int y = -1; y <= 1; y++)
        {
            for (int x = -1; x <= 1; x++)
            {
                vec3 neighbor = vec3(float(x), float(y), float(z));
                vec3 point = rand3(i + neighbor);
                float dist = length(neighbor + point - f);
                if (dist < f1) { f2 = f1; f1 = dist; cell_id = i + neighbor; }
                else if (dist < f2) { f2 = dist; }
            }
        }
    }
    return vec3(f1, f2, rand1(cell_id.xy + cell_id.z * 31.0));
}`
            }]
        }, {
            name: "Worley Noise",
            description: "F1 distance to nearest feature point (includes rand)",
            author: "Steven Worley (SIGGRAPH 1996)",
            options: [{
                label: "2D",
                code: `float worley(vec2 p)
{
    vec2 i = floor(p);
    vec2 f = fract(p);
    float min_dist = 1.0;
    for (int y = -1; y <= 1; y++)
    {
        for (int x = -1; x <= 1; x++)
        {
            vec2 neighbor = vec2(float(x), float(y));
            vec2 point = rand2(i + neighbor);
            min_dist = min(min_dist, length(neighbor + point - f));
        }
    }
    return min_dist;
}`
            }, {
                label: "3D",
                code: `float worley(vec3 p)
{
    vec3 i = floor(p);
    vec3 f = fract(p);
    float min_dist = 1.0;
    for (int z = -1; z <= 1; z++)
    {
        for (int y = -1; y <= 1; y++)
        {
            for (int x = -1; x <= 1; x++)
            {
                vec3 neighbor = vec3(float(x), float(y), float(z));
                vec3 point = rand3(i + neighbor);
                min_dist = min(min_dist, length(neighbor + point - f));
            }
        }
    }
    return min_dist;
}`
            }]
        }, {
            name: "Turbulence",
            description: "Fake fluid dynamics via layered rotated sine waves",
            author: "Xor (mini.gmshaders.com/p/turbulence)",
            options: [{
                label: "2D",
                code: `// Turbulence — 2D | Xor (mini.gmshaders.com/p/turbulence)
// Turbulence: layered sine-wave displacements with golden-ratio rotation.
// TURB_NUM, TURB_AMP, TURB_SPEED, TURB_FREQ, TURB_EXP – see Constants.
#define TURB_NUM   10.0
#define TURB_AMP   0.7
#define TURB_SPEED 0.3
#define TURB_FREQ  2.0
#define TURB_EXP   1.4

const mat2 GOLDEN_ROT2 = mat2(
    -0.73736887808,  0.67549029426,
    -0.67549029426, -0.73736887808);

vec2 turbulence(vec2 pos, float time)
{
    float freq = TURB_FREQ;
    mat2 rot = GOLDEN_ROT2;
    for (float i = 0.0; i < TURB_NUM; i++)
    {
        float phase = freq * (pos * rot).y + TURB_SPEED * time + i;
        pos += TURB_AMP * rot[0] * sin(phase) / freq;
        rot *= GOLDEN_ROT2;
        freq *= TURB_EXP;
    }
    return pos;
}`
            }, {
                label: "3D",
                code: `// Turbulence — 3D | Xor (mini.gmshaders.com/p/turbulence)
// Turbulence 3D: layered sine-wave displacements with golden-ratio rotation.
#define TURB_NUM   10.0
#define TURB_AMP   0.7
#define TURB_SPEED 0.3
#define TURB_FREQ  2.0
#define TURB_EXP   1.4

const mat3 GOLDEN_ROT3 = mat3(
    -0.571464913, +0.814921382, +0.096597072,
    -0.278044873, -0.303026659, +0.911518454,
    +0.772087367, +0.494042493, +0.399753815);

vec3 turbulence(vec3 pos, float time)
{
    float freq = TURB_FREQ;
    mat3 rot = GOLDEN_ROT3;
    for (float i = 0.0; i < TURB_NUM; i++)
    {
        float phase = freq * (pos * rot).y + TURB_SPEED * time + i;
        pos += TURB_AMP * rot[0] * sin(phase) / freq;
        rot *= GOLDEN_ROT3;
        freq *= TURB_EXP;
    }
    return pos;
}`
            }]
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
            name: "HSV ↔ RGB",
            description: "Convert between HSV and RGB color spaces",
            author: "Sam Hocevar (RGB→HSV)",
            options: [{
                label: "HSV → RGB",
                code: `vec3 hsv_to_rgb(vec3 c)
{
    vec3 p = abs(fract(c.xxx + vec3(1.0, 2.0/3.0, 1.0/3.0)) * 6.0 - 3.0);
    return c.z * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), c.y);
}`
            }, {
                label: "RGB → HSV",
                code: `vec3 rgb_to_hsv(vec3 c)
{
    vec4 K = vec4(0.0, -1.0/3.0, 2.0/3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}`
            }]
        }, {
            name: "Cosine Palette",
            description: "Cosine gradient palette (4 vec3 parameters)",
            author: "Inigo Quilez (iquilezles.org/articles/palettes)",
            dependencies: ["TAU"],
            code: `vec3 palette(float t, vec3 a, vec3 b, vec3 c, vec3 d)
{
    return a + b * cos(TAU * (c * t + d));
}

// Example presets:
// Rainbow:   palette(t, vec3(0.5), vec3(0.5), vec3(1.0), vec3(0.0, 0.33, 0.67))
// Warm:      palette(t, vec3(0.5), vec3(0.5), vec3(1.0), vec3(0.0, 0.1, 0.2))
// Cool:      palette(t, vec3(0.5), vec3(0.5), vec3(1.0, 1.0, 0.5), vec3(0.8, 0.9, 0.3))`
        }, {
            name: "Linear ↔ sRGB",
            description: "Linear ↔ sRGB with IEC 61966-2-1 transfer curve",
            options: [{
                label: "Linear → sRGB",
                code: `vec3 linear_to_srgb(vec3 c)
{
    vec3 lo = 12.92 * c;
    vec3 hi = 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055;
    return mix(lo, hi, step(vec3(0.0031308), c));
}`
            }, {
                label: "sRGB → Linear",
                code: `vec3 srgb_to_linear(vec3 c)
{
    vec3 lo = c / 12.92;
    vec3 hi = pow((c + 0.055) / 1.055, vec3(2.4));
    return mix(lo, hi, step(vec3(0.04045), c));
}`
            }]
        }, {
            name: "Tonemapping",
            description: "HDR → LDR tonemapping operators",
            author: "Krzysztof Narkowicz (ACES), Erik Reinhard et al.",
            options: [{
                label: "ACES",
                code: `vec3 tonemap_aces(vec3 x)
{
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}`
            }, {
                label: "Reinhard",
                code: `vec3 tonemap_reinhard(vec3 x)
{
    return x / (1.0 + x);
}`
            }, {
                label: "Reinhard (white point)",
                code: `vec3 tonemap_reinhard_ext(vec3 x, float white)
{
    vec3 num = x * (1.0 + x / (white * white));
    return num / (1.0 + x);
}`
            }]
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
            name: "OKLab - OKLch",
            description: "Perceptually uniform color space; OKLch = cylindrical OKLab",
            author: "Björn Ottosson (bottosson.github.io/posts/oklab)",
            options: [{
                label: "RGB → OKLab",
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
}`
            }, {
                label: "OKLab → RGB",
                code: `vec3 oklab_to_rgb(vec3 c)
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
                label: "OKLab → OKLch",
                code: `vec3 oklab_to_oklch(vec3 ok)
{
    float L = ok.r;
    float C = sqrt(ok.g * ok.g + ok.b * ok.b);
    float H = atan(ok.b, ok.g);
    return vec3(L, C, H);
}`
            }, {
                label: "OKLch → OKLab",
                code: `vec3 oklch_to_oklab(vec3 lch)
{
    float L = lch.r, C = lch.g, H = lch.b;
    return vec3(L, C * cos(H), C * sin(H));
}`
            }, {
                label: "OKLab Mix",
                author: "Inigo Quilez",
                code: `// Perceptually uniform mix via LMS cone space. Compact, no separate conversion helpers.
// https://bottosson.github.io/posts/oklab

vec3 oklab_mix(vec3 colA, vec3 colB, float h)
{
    const mat3 kCONEtoLMS = mat3(
         0.4121656120,  0.5362752080,  0.0514575653,
         0.2118591070,  0.6807189584,  0.1074065790,
         0.0883097947,  0.2818474174,  0.6302613616);
    const mat3 kLMStoCONE = mat3(
         4.0767245293, -3.3072168827,  0.2307590544,
        -1.2681437731,  2.6093323231, -0.3411344290,
        -0.0041119885, -0.7034763098,  1.7068625689);
    vec3 lmsA = pow(kCONEtoLMS * colA, vec3(1.0 / 3.0));
    vec3 lmsB = pow(kCONEtoLMS * colB, vec3(1.0 / 3.0));
    vec3 lms = mix(lmsA, lmsB, h);
    return kLMStoCONE * (lms * lms * lms);
}`
            }]
        }, {
            name: "Saturation",
            description: "Adjust color saturation; s=0 grayscale, s=1 original, s>1 oversaturated",
            code: `vec3 saturation(vec3 col, float sat)
{
    float luminance = dot(col, vec3(0.2126, 0.7152, 0.0722));
    return mix(vec3(luminance), col, sat);
}`
        }]
    }, {
        name: "Coordinates",
        icon: "Grid3x2",
        snippets: [{
            name: "Rotation",
            description: "Rotate vec2 or vec3 by angle(s)",
            options: [{
                label: "2D",
                code: `mat2 rotate_2d(float angle)
{
    float s = sin(angle);
    float c = cos(angle);
    return mat2(c, -s, s, c);
}

// Usage: p.xy *= rotate_2d(angle);`
            }, {
                label: "3D (Euler)",
                code: `mat2 rotate_2d(float a)
{
    return mat2(cos(a), -sin(a), sin(a), cos(a));
}
vec3 euler_rotate(vec3 v, float roll, float pitch, float yaw)
{
    v.yz *= rotate_2d(roll);
    v.xz *= rotate_2d(pitch);
    v.xy *= rotate_2d(yaw);
    return v;
}`
            }, {
                label: "3D (Axis-Angle)",
                author: "Xor / Fabrice Neyret (mini.gmshaders.com/p/3d-rotation)",
                code: `vec3 axis_rotate(vec3 v, vec3 axis, float angle)
{
    return mix(dot(v, axis) * axis, v, cos(angle))
         + sin(angle) * cross(v, axis);
}`
            }]
        }, {
            name: "Quaternions",
            description: "Quaternion rotation (vec4: x,y,z,w)",
            options: [{
                label: "Multiply",
                code: `vec4 qmul(vec4 a, vec4 b)
{
    return vec4(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
    );
}`
            }, {
                label: "Slerp",
                code: `vec4 qslerp(vec4 a, vec4 b, float t)
{
    float d = dot(a, b);
    if (d < 0.0) { b = -b; d = -d; }
    if (d > 0.9995) return normalize(mix(a, b, t));
    float theta = acos(d);
    float st = sin(theta);
    return (sin((1.0 - t) * theta) / st) * a + (sin(t * theta) / st) * b;
}`
            }, {
                label: "From Axis-Angle",
                code: `vec4 qfrom_axis_angle(vec3 axis, float angle)
{
    float ha = angle * 0.5;
    float s = sin(ha);
    return vec4(normalize(axis) * s, cos(ha));
}`
            }, {
                label: "To Mat3",
                code: `mat3 qto_mat3(vec4 q)
{
    float x = q.x, y = q.y, z = q.z, w = q.w;
    return mat3(
        1.0 - 2.0*(y*y + z*z), 2.0*(x*y - z*w), 2.0*(x*z + y*w),
        2.0*(x*y + z*w), 1.0 - 2.0*(x*x + z*z), 2.0*(y*z - x*w),
        2.0*(x*z - y*w), 2.0*(y*z + x*w), 1.0 - 2.0*(x*x + y*y)
    );
}`
            }, {
                label: "Rotate Vec3",
                code: `vec3 qrotate(vec4 q, vec3 v)
{
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}`
            }]
        }, {
            name: "Polar Coordinates",
            description: "Convert between cartesian and polar/spherical coords",
            options: [{
                label: "2D: Cartesian → Polar",
                code: `vec2 to_polar(vec2 p)
{
    return vec2(length(p), atan(p.y, p.x));
}`
            }, {
                label: "2D: Polar → Cartesian",
                code: `vec2 to_cartesian(vec2 polar)
{
    return vec2(polar.x * cos(polar.y), polar.x * sin(polar.y));
}`
            }, {
                label: "3D: Cartesian → Spherical",
                code: `// spherical = (r, theta, phi): r=radius, theta=azimuth (xy), phi=polar from +z.

vec3 to_spherical(vec3 p)
{
    float r = length(p);
    float theta = atan(p.y, p.x);
    float phi = (r > 0.0) ? acos(clamp(p.z / r, -1.0, 1.0)) : 0.0;
    return vec3(r, theta, phi);
}`
            }, {
                label: "3D: Spherical → Cartesian",
                code: `vec3 to_cartesian(vec3 spherical)
{
    float r = spherical.x;
    float theta = spherical.y;
    float phi = spherical.z;
    return r * vec3(sin(phi) * cos(theta), sin(phi) * sin(theta), cos(phi));
}`
            }]
        }, {
            name: "Repeat - Tile",
            description: "Repeat space for infinite tiling",
            options: [{
                label: "Cartesian",
                code: `vec2 repeat(vec2 p, vec2 size)
{
    return mod(p + size * 0.5, size) - size * 0.5;
}`
            }, {
                label: "Mirror",
                code: `vec2 mirror_repeat(vec2 p, vec2 size)
{
    vec2 half = size * 0.5;
    vec2 q = mod(p + half, size * 2.0) - size;
    return half - abs(q);
}`
            }, {
                label: "Angular",
                code: `// Repeat by angle; n = number of segments (e.g. 6 for hexagon).

vec2 angular_repeat(vec2 p, float n)
{
    float a = atan(p.y, p.x);
    float r = length(p);
    a = mod(a, 6.28318530718 / n) - 0.5 * 6.28318530718 / n;
    return vec2(cos(a), sin(a)) * r;
}`
            }]
        }, {
            name: "UV Setup",
            description: "Centered UV coordinate helpers",
            author: "Xor (mini.gmshaders.com/p/gm-shaders-mini-scaling)",
            options: [{
                label: "Stretched",
                code: `// Stretched: full resolution to 0–1, non-square screens stretched.

vec2 uv_stretched(vec2 pixel, vec2 res)
{
    return (pixel - res * 0.5) / res + 0.5;
}`
            }, {
                label: "Fit",
                code: `// Fit: uniform scale, full area visible (letterboxing).

vec2 uv_fit(vec2 pixel, vec2 res)
{
    vec2 center = pixel - res * 0.5;
    return center / max(res.x, res.y) + 0.5;
}`
            }, {
                label: "Fill",
                code: `// Fill: uniform scale, screen fully covered (edges may crop).

vec2 uv_fill(vec2 pixel, vec2 res)
{
    vec2 center = pixel - res * 0.5;
    return center / min(res.x, res.y) + 0.5;
}`
            }, {
                label: "Custom Origin",
                code: `// Custom origin (e.g. vec2(0.5, 1.0) for middle-bottom).

vec2 uv_center_origin(vec2 pixel, vec2 res, vec2 origin)
{
    vec2 center = pixel - res * origin;
    return center / min(res.x, res.y) + origin;
}`
            }, {
                label: "Aspect Ratio",
                code: `// Correct aspect ratio so circles stay circular.

vec2 uv_aspect(vec2 uv, vec2 res)
{
    uv -= 0.5;
    uv.x *= res.x / res.y;
    return uv + 0.5;
}`
            }, {
                label: "Aspect Centered",
                code: `// Aspect correction, origin at 0,0.

vec2 uv_aspect_centered(vec2 uv, vec2 res)
{
    uv -= 0.5;
    uv.x *= res.x / res.y;
    return uv;
}`
            }, {
                label: "Fit Texture",
                code: `// Fit texture with ratio (e.g. vec2(2,1)) to screen, no cropping.

vec2 uv_fit_ratio(vec2 pixel, vec2 res, vec2 ratio)
{
    vec2 res_ratio = res / ratio;
    vec2 origin = vec2(0.5);
    vec2 center = pixel - res * origin;
    return center / max(res_ratio.x, res_ratio.y) / ratio + origin;
}`
            }]
        }]
    }, {
        name: "Math",
        icon: "Function",
        snippets: [{
            name: "Remap",
            description: "Map a value from one range to another",
            options: [{
                label: "Remap",
                code: `float remap(float value, float in_min, float in_max, float out_min, float out_max)
{
    return out_min + (out_max - out_min) * (value - in_min) / (in_max - in_min);
}`
            }, {
                label: "To Unit",
                code: `// Linear map [in_min, in_max] → [0, 1]; no clamp (values outside range extrapolate).

float to_unit(float value, float in_min, float in_max)
{
    return (value - in_min) / (in_max - in_min);
}`
            }, {
                label: "From Unit",
                code: `// Linear map [0, 1] → [out_min, out_max]; no clamp (values outside range extrapolate).

float from_unit(float value, float out_min, float out_max)
{
    return out_min + value * (out_max - out_min);
}`
            }]
        }, {
            name: "Smooth Min / Max",
            description: "Smooth minimum and maximum blending",
            author: "Inigo Quilez (iquilezles.org/articles/smin)",
            options: [{
                label: "Smooth Min",
                code: `float smin(float a, float b, float k)
{
    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}`
            }, {
                label: "Smooth Max",
                code: `float smax(float a, float b, float k)
{
    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(a, b, h) + k * h * (1.0 - h);
}`
            }]
        }, {
            name: "Easing",
            description: "Easing curves for animations (t in [0,1])",
            options: [{
                label: "Quadratic",
                code: `float ease_in_quad(float t) { return t * t; }
float ease_out_quad(float t) { return t * (2.0 - t); }
float ease_in_out_quad(float t) { return t < 0.5 ? 2.0 * t * t : -1.0 + (4.0 - 2.0 * t) * t; }`
            }, {
                label: "Cubic",
                code: `float ease_in_cubic(float t) { return t * t * t; }
float ease_out_cubic(float t) { float u = 1.0 - t; return 1.0 - u * u * u; }
float ease_in_out_cubic(float t) { return t < 0.5 ? 4.0 * t * t * t : 1.0 - pow(-2.0 * t + 2.0, 3.0) * 0.5; }`
            }, {
                label: "Quartic",
                code: `float ease_in_quart(float t) { return t * t * t * t; }
float ease_out_quart(float t) { float u = 1.0 - t; return 1.0 - u * u * u * u; }
float ease_in_out_quart(float t) { return t < 0.5 ? 8.0 * t * t * t * t : 1.0 - pow(-2.0 * t + 2.0, 4.0) * 0.5; }`
            }, {
                label: "Quintic",
                code: `float ease_in_quint(float t) { return t * t * t * t * t; }
float ease_out_quint(float t) { float u = 1.0 - t; return 1.0 - u * u * u * u * u; }
float ease_in_out_quint(float t) { return t < 0.5 ? 16.0 * t * t * t * t * t : 1.0 - pow(-2.0 * t + 2.0, 5.0) * 0.5; }`
            }, {
                label: "Exponential",
                code: `float ease_in_expo(float t) { return t <= 0.0 ? 0.0 : pow(2.0, 10.0 * t - 10.0); }
float ease_out_expo(float t) { return t >= 1.0 ? 1.0 : 1.0 - pow(2.0, -10.0 * t); }
float ease_in_out_expo(float t)
{
    if (t <= 0.0) return 0.0;
    if (t >= 1.0) return 1.0;
    return t < 0.5 ? pow(2.0, 20.0 * t - 10.0) * 0.5 : 1.0 - pow(2.0, -20.0 * t + 10.0) * 0.5;
}`
            }, {
                label: "Sine",
                code: `float ease_in_sine(float t) { return 1.0 - cos(1.57079632679 * t); }
float ease_out_sine(float t) { return sin(1.57079632679 * t); }
float ease_in_out_sine(float t) { return -0.5 * (cos(3.14159265359 * t) - 1.0); }`
            }, {
                label: "Circular",
                code: `float ease_in_circ(float t) { return 1.0 - sqrt(1.0 - t * t); }
float ease_out_circ(float t) { return sqrt(1.0 - (t - 1.0) * (t - 1.0)); }
float ease_in_out_circ(float t)
{
    return t < 0.5
        ? 0.5 * (1.0 - sqrt(1.0 - 4.0 * t * t))
        : 0.5 * (sqrt(1.0 - (2.0 * t - 2.0) * (2.0 * t - 2.0)) + 1.0);
}`
            }, {
                label: "Back",
                code: `float c1 = 1.70158;
float c3 = c1 + 1.0;

float ease_in_back(float t) { return c3 * t * t * t - c1 * t * t; }
float ease_out_back(float t) { float u = 1.0 - t; return 1.0 + c3 * u * u * u + c1 * u * u; }
float ease_in_out_back(float t)
{
    float c2 = c1 * 1.525;
    return t < 0.5
        ? (pow(2.0 * t, 2.0) * ((c2 + 1.0) * 2.0 * t - c2)) * 0.5
        : 0.5 * (pow(2.0 * t - 2.0, 2.0) * ((c2 + 1.0) * (2.0 * t - 2.0) + c2) + 2.0);
}`
            }]
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
            name: "Anti-Aliasing",
            description: "Smooth edges for SDFs or any continuous function",
            author: "Xor (mini.gmshaders.com/p/antialiasing)",
            options: [{
                label: "SDF",
                code: `// For SDFs with known pixel scale (texel size).

float antialias_sdf(float dist, float texel)
{
    return clamp(dist / texel + 0.5, 0.0, 1.0);
}`
            }, {
                label: "Derivative (fwidth)",
                code: `float antialias(float d)
{
    float w = fwidth(d);
    float scale = w > 0.0 ? 1.0 / w : 1e7;
    return clamp(0.5 + 0.5 * scale * d, 0.0, 1.0);
}`
            }, {
                label: "Derivative (L2)",
                code: `float antialias_l2(float d)
{
    vec2 dxy = vec2(dFdx(d), dFdy(d));
    float w = length(dxy);
    float scale = w > 0.0 ? 1.0 / w : 1e7;
    return clamp(0.5 + 0.7 * scale * d, 0.0, 1.0);
}`
            }, {
                label: "Derivative (L2 manual)",
                code: `float antialias_l2_dxy(float d, vec2 dxy)
{
    float w = length(dxy);
    float scale = w > 0.0 ? 1.0 / w : 1e7;
    return clamp(0.5 + 0.7 * scale * d, 0.0, 1.0);
}`
            }]
        }, {
            name: "Complex",
            description: "Complex number arithmetic (vec2 = real + imaginary)",
            options: [{
                label: "Multiply / Divide",
                code: `fn cmul(a: vec2f, b: vec2f) -> vec2f
{
    return vec2f(a.x * b.x - a.y * b.y,
                 a.x * b.y + a.y * b.x);
}

fn cdiv(a: vec2f, b: vec2f) -> vec2f
{
    let d = dot(b, b);
    return vec2f(dot(a, b), a.y * b.x - a.x * b.y) / d;
}

fn cconj(z: vec2f) -> vec2f
{
    return vec2f(z.x, -z.y);
}`
            }, {
                label: "Exp / Log / Power",
                code: `fn cabs(z: vec2f) -> f32 { return length(z); }
fn carg(z: vec2f) -> f32 { return atan2(z.y, z.x); }

fn cexp(z: vec2f) -> vec2f
{
    return exp(z.x) * vec2f(cos(z.y), sin(z.y));
}

fn clog(z: vec2f) -> vec2f
{
    return vec2f(log(length(z)), atan2(z.y, z.x));
}

fn cpow(z: vec2f, n: f32) -> vec2f
{
    let r = length(z);
    let theta = atan2(z.y, z.x);
    return pow(r, n) * vec2f(cos(n * theta), sin(n * theta));
}

fn csqrt(z: vec2f) -> vec2f
{
    let r = length(z);
    let theta = atan2(z.y, z.x);
    return sqrt(r) * vec2f(cos(theta * 0.5), sin(theta * 0.5));
}`
            }, {
                label: "Trigonometry",
                code: `fn csin(z: vec2f) -> vec2f
{
    return vec2f(sin(z.x) * cosh(z.y), cos(z.x) * sinh(z.y));
}

fn ccos(z: vec2f) -> vec2f
{
    return vec2f(cos(z.x) * cosh(z.y), -sin(z.x) * sinh(z.y));
}`
            }]
        }]
    }, {
        name: "SDF",
        icon: "Hexagon",
        snippets: [{
            name: "2D Shapes",
            description: "Signed distance functions for 2D primitives",
            author: "Inigo Quilez (iquilezles.org)",
            options: [{
                label: "Circle",
                code: `float sd_circle(vec2 p, float r)
{
    return length(p) - r;
}`
            }, {
                label: "Box",
                code: `float sd_box(vec2 p, vec2 b)
{
    vec2 d = abs(p) - b;
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}`
            }, {
                label: "Rounded Box",
                code: `float sd_rounded_box(vec2 p, vec2 b, vec4 r)
{
    r.xy = (p.x > 0.0) ? r.xy : r.zw;
    r.x  = (p.y > 0.0) ? r.x  : r.y;
    vec2 q = abs(p) - b + r.x;
    return min(max(q.x, q.y), 0.0) + length(max(q, 0.0)) - r.x;
}`
            }, {
                label: "Line Segment",
                code: `float sd_segment(vec2 p, vec2 a, vec2 b)
{
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}`
            }, {
                label: "Triangle",
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
                label: "Ring",
                code: `float sd_ring(vec2 p, float r, float thickness)
{
    return abs(length(p) - r) - thickness;
}`
            }, {
                label: "Capsule",
                code: `float sd_capsule(vec2 p, vec2 a, vec2 b, float r)
{
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}`
            }, {
                label: "Vesica",
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
                label: "Hexagon",
                code: `float sd_hexagon(vec2 p, float r)
{
    const vec3 k = vec3(-0.866025404, 0.5, 0.577350269);
    p = abs(p);
    p -= 2.0 * min(dot(k.xy, p), 0.0) * k.xy;
    p -= vec2(clamp(p.x, -k.z * r, k.z * r), r);
    return length(p) * sign(p.y);
}`
            }, {
                label: "Pentagon",
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
                label: "Star",
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
                label: "Arc",
                code: `float sd_arc(vec2 p, vec2 sc, float ra, float rb)
{
    p.x = abs(p.x);
    return ((sc.y * p.x > sc.x * p.y)
        ? length(p - sc * ra)
        : abs(length(p) - ra)) - rb;
}`
            }, {
                label: "Ellipse",
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
            }]
        }, {
            name: "3D Shapes",
            description: "Signed distance functions for 3D primitives",
            author: "Inigo Quilez (iquilezles.org)",
            options: [{
                label: "Sphere",
                code: `float sd_sphere(vec3 p, float r)
{
    return length(p) - r;
}`
            }, {
                label: "Box",
                code: `float sd_box_3d(vec3 p, vec3 b)
{
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}`
            }, {
                label: "Torus",
                code: `float sd_torus(vec3 p, vec2 t)
{
    vec2 q = vec2(length(p.xz) - t.x, p.y);
    return length(q) - t.y;
}`
            }, {
                label: "Capsule",
                code: `float sd_capsule(vec3 p, vec3 a, vec3 b, float r)
{
    vec3 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}`
            }, {
                label: "Cylinder",
                code: `float sd_capped_cylinder(vec3 p, float r, float h)
{
    vec2 d = abs(vec2(length(p.xz), p.y)) - vec2(r, h);
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}`
            }, {
                label: "Cone",
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
                label: "Plane",
                code: `float sd_plane(vec3 p, vec3 n, float h)
{
    return dot(p, n) + h;
}`
            }, {
                label: "Round Cone",
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
                label: "Hex Prism",
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
                label: "Round Box",
                code: `float sd_round_box(vec3 p, vec3 b, float r)
{
    vec3 q = abs(p) - b + r;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}`
            }]
        }, {
            name: "Operations",
            description: "Combine SDFs: union, intersection, subtraction, and smooth variants",
            author: "Inigo Quilez (iquilezles.org)",
            options: [{
                label: "Union",
                code: "float op_union(float d1, float d2) { return min(d1, d2); }"
            }, {
                label: "Intersect",
                code: "float op_intersect(float d1, float d2) { return max(d1, d2); }"
            }, {
                label: "Subtract",
                code: "float op_subtract(float d1, float d2) { return max(-d1, d2); }"
            }, {
                label: "Smooth Union",
                code: `float op_smooth_union(float d1, float d2, float k)
{
    float h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return mix(d2, d1, h) - k * h * (1.0 - h);
}`
            }, {
                label: "Smooth Intersect",
                code: `float op_smooth_intersect(float d1, float d2, float k)
{
    float h = clamp(0.5 - 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return mix(d2, d1, h) + k * h * (1.0 - h);
}`
            }, {
                label: "Smooth Subtract",
                code: `float op_smooth_subtract(float d1, float d2, float k)
{
    float h = clamp(0.5 - 0.5 * (d2 + d1) / k, 0.0, 1.0);
    return mix(d2, -d1, h) + k * h * (1.0 - h);
}`
            }]
        }, {
            name: "Ray Marching",
            description: "Sphere-tracing loop, normals, lighting, camera",
            options: [{
                label: "Ray March Loop",
                code: `#define MAX_STEPS 128
#define MAX_DIST  100.0
#define SURF_DIST 0.001

float ray_march(vec3 ro, vec3 rd)
{
    float t = 0.0;
    for (int i = 0; i < MAX_STEPS; i++)
    {
        vec3 p = ro + rd * t;
        float d = scene(p);
        if (d < SURF_DIST) break;
        t += d;
        if (t > MAX_DIST) break;
    }
    return t;
}`
            }, {
                label: "Normal",
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
                label: "Lighting",
                code: `vec3 lighting(vec3 p, vec3 normal, vec3 rd, vec3 light_dir, vec3 col)
{
    float diff = max(dot(normal, light_dir), 0.0);
    vec3 half_dir = normalize(light_dir - rd);
    float spec = pow(max(dot(normal, half_dir), 0.0), 32.0);
    vec3 ambient = 0.05 * col;
    return ambient + col * diff + vec3(1.0) * spec * 0.5;
}`
            }, {
                label: "Camera",
                code: `mat3 camera(vec3 eye, vec3 target, float roll)
{
    vec3 cw = normalize(target - eye);
    vec3 cp = vec3(sin(roll), cos(roll), 0.0);
    vec3 cu = normalize(cross(cw, cp));
    vec3 cv = cross(cu, cw);
    return mat3(cu, cv, cw);
}`
            }]
        }]
    }, {
        name: "Raytracing",
        icon: "Globe",
        snippets: [{
            name: "Ray Setup",
            description: "Camera ray generation for raytracing",
            options: [{
                label: "Perspective",
                code: `// ro: ray origin, rd: ray direction (normalized).
// uv: fragCoord.xy / resolution.xy in [0,1].

vec3 ray_dir(vec2 uv, vec3 ro, vec3 target, float fov)
{
    vec3 fwd = normalize(target - ro);
    vec3 right = normalize(cross(vec3(0.0, 1.0, 0.0), fwd));
    vec3 up = cross(fwd, right);
    vec2 ndc = uv * 2.0 - 1.0;
    float aspect = iResolution.x / iResolution.y;
    return normalize(fwd + (right * ndc.x * aspect + up * ndc.y) * tan(fov * 0.5));
}`
            }, {
                label: "Orthographic",
                code: `// Orthographic: parallel rays. rd = normalize(target - ro) for all pixels.
// Ray origin for pixel: ro + ortho_offset(uv, ro, target, scale).

vec3 ortho_offset(vec2 uv, vec3 ro, vec3 target, float scale)
{
    vec3 fwd = normalize(target - ro);
    vec3 right = normalize(cross(vec3(0.0, 1.0, 0.0), fwd));
    vec3 up = cross(fwd, right);
    vec2 ndc = uv * 2.0 - 1.0;
    float aspect = iResolution.x / iResolution.y;
    return (right * ndc.x * aspect + up * ndc.y) * scale;
}`
            }]
        }, {
            name: "Ray Shapes",
            description: "Ray-primitive intersections (analytic)",
            options: [{
                label: "Sphere",
                code: `// Returns (t, 1) if hit, (t_max, 0) if miss. t is distance along ray.

vec2 ray_sphere(vec3 ro, vec3 rd, vec3 center, float radius)
{
    vec3 oc = ro - center;
    float b = dot(oc, rd);
    float c = dot(oc, oc) - radius * radius;
    float d = b * b - c;
    if (d < 0.0) return vec2(1e10, 0.0);
    float t = -b - sqrt(d);
    if (t < 0.0) t = -b + sqrt(d);
    if (t < 0.0) return vec2(1e10, 0.0);
    return vec2(t, 1.0);
}

vec3 sphere_normal(vec3 p, vec3 center) { return normalize(p - center); }`
            }, {
                label: "Plane",
                code: `// Plane: dot(p, n) + d = 0. n is unit normal, d is offset from origin.

vec2 ray_plane(vec3 ro, vec3 rd, vec3 n, float d)
{
    float denom = dot(rd, n);
    if (abs(denom) < 1e-6) return vec2(1e10, 0.0);
    float t = -(dot(ro, n) + d) / denom;
    if (t < 0.0) return vec2(1e10, 0.0);
    return vec2(t, 1.0);
}`
            }, {
                label: "Box (AABB)",
                code: `// Slab method. box_min, box_max = AABB bounds.

vec2 ray_box(vec3 ro, vec3 rd, vec3 box_min, vec3 box_max)
{
    vec3 inv_rd = 1.0 / rd;
    vec3 t0 = (box_min - ro) * inv_rd;
    vec3 t1 = (box_max - ro) * inv_rd;
    vec3 tmin = min(t0, t1);
    vec3 tmax = max(t0, t1);
    float t_near = max(max(tmin.x, tmin.y), tmin.z);
    float t_far  = min(min(tmax.x, tmax.y), tmax.z);
    if (t_near > t_far || t_far < 0.0) return vec2(1e10, 0.0);
    float t = t_near < 0.0 ? t_far : t_near;
    return vec2(t, 1.0);
}`
            }, {
                label: "Triangle",
                code: `// Möller–Trumbore. Returns (t, u, v) — t = hit distance, (u,v) barycentrics. Miss: t < 0.

vec3 ray_triangle(vec3 ro, vec3 rd, vec3 v0, vec3 v1, vec3 v2)
{
    vec3 e1 = v1 - v0;
    vec3 e2 = v2 - v0;
    vec3 h = cross(rd, e2);
    float a = dot(e1, h);
    if (abs(a) < 1e-8) return vec3(-1.0, 0.0, 0.0);
    float f = 1.0 / a;
    vec3 s = ro - v0;
    float u = f * dot(s, h);
    if (u < 0.0 || u > 1.0) return vec3(-1.0, 0.0, 0.0);
    vec3 q = cross(s, e1);
    float v = f * dot(rd, q);
    if (v < 0.0 || u + v > 1.0) return vec3(-1.0, 0.0, 0.0);
    float t = f * dot(e2, q);
    if (t < 1e-6) return vec3(-1.0, 0.0, 0.0);
    return vec3(t, u, v);
}

vec3 triangle_normal(vec3 v0, vec3 v1, vec3 v2) { return normalize(cross(v1 - v0, v2 - v0)); }`
            }]
        }]
    }];
export const GLSL_SHADER_FUNCTIONS = [{
    name: "abs",
    signature: "abs(x)",
    description: "Absolute value"
}, {
    name: "acos",
    signature: "acos(x)",
    description: "Arc cosine, returns radians"
}, {
    name: "acosh",
    signature: "acosh(x)",
    description: "Hyperbolic arc cosine"
}, {
    name: "all",
    signature: "all(bvec)",
    description: "Returns true if all components of a boolean vector are true"
}, {
    name: "any",
    signature: "any(bvec)",
    description: "Returns true if any component of a boolean vector is true"
}, {
    name: "asin",
    signature: "asin(x)",
    description: "Arc sine, returns radians"
}, {
    name: "asinh",
    signature: "asinh(x)",
    description: "Hyperbolic arc sine"
}, {
    name: "atan",
    signature: "atan(x)",
    description: "Arc tangent, returns radians"
}, {
    name: "atan",
    signature: "atan(y, x)",
    description: "Arc tangent of y/x, handles quadrants"
}, {
    name: "atanh",
    signature: "atanh(x)",
    description: "Hyperbolic arc tangent"
}, {
    name: "bitCount",
    signature: "bitCount(x)",
    description: "Number of 1 bits in the binary representation (popcount)"
}, {
    name: "bitfieldExtract",
    signature: "bitfieldExtract(value, offset, bits)",
    description: "Extract bits from value starting at offset"
}, {
    name: "bitfieldInsert",
    signature: "bitfieldInsert(base, insert, offset, bits)",
    description: "Insert bits into base at offset"
}, {
    name: "bitfieldReverse",
    signature: "bitfieldReverse(x)",
    description: "Reverse the bits of an integer"
}, {
    name: "ceil",
    signature: "ceil(x)",
    description: "Round up to integer"
}, {
    name: "clamp",
    signature: "clamp(x, min, max)",
    description: "Clamp x between min and max"
}, {
    name: "cos",
    signature: "cos(x)",
    description: "Cosine of angle in radians"
}, {
    name: "cosh",
    signature: "cosh(x)",
    description: "Hyperbolic cosine"
}, {
    name: "cross",
    signature: "cross(a, b)",
    description: "Cross product (vec3)"
}, {
    name: "dFdx",
    signature: "dFdx(p)",
    description: "Partial derivative in x"
}, {
    name: "dFdy",
    signature: "dFdy(p)",
    description: "Partial derivative in y"
}, {
    name: "degrees",
    signature: "degrees(radians)",
    description: "Convert radians to degrees"
}, {
    name: "determinant",
    signature: "determinant(m)",
    description: "Determinant of a square matrix"
}, {
    name: "distance",
    signature: "distance(a, b)",
    description: "Distance between two points"
}, {
    name: "dot",
    signature: "dot(a, b)",
    description: "Dot product"
}, {
    name: "equal",
    signature: "equal(a, b)",
    description: "Component-wise equality, returns bvec"
}, {
    name: "exp",
    signature: "exp(x)",
    description: "e raised to the power x"
}, {
    name: "exp2",
    signature: "exp2(x)",
    description: "2 raised to the power x"
}, {
    name: "faceforward",
    signature: "faceforward(N, I, Nref)",
    description: "Flip N if facing away from I"
}, {
    name: "findLSB",
    signature: "findLSB(x)",
    description: "Index of the least significant 1 bit"
}, {
    name: "findMSB",
    signature: "findMSB(x)",
    description: "Index of the most significant 1 bit"
}, {
    name: "floatBitsToInt",
    signature: "floatBitsToInt(x)",
    description: "Reinterpret float bits as int"
}, {
    name: "floatBitsToUint",
    signature: "floatBitsToUint(x)",
    description: "Reinterpret float bits as uint"
}, {
    name: "floor",
    signature: "floor(x)",
    description: "Round down to integer"
}, {
    name: "fma",
    signature: "fma(a, b, c)",
    description: "Fused multiply-add: a*b + c"
}, {
    name: "fract",
    signature: "fract(x)",
    description: "Fractional part (x - floor(x))"
}, {
    name: "frexp",
    signature: "frexp(x, out exp)",
    description: "Splits x into mantissa and exponent"
}, {
    name: "fwidth",
    signature: "fwidth(p)",
    description: "abs(dFdx(p)) + abs(dFdy(p))"
}, {
    name: "greaterThan",
    signature: "greaterThan(a, b)",
    description: "Component-wise greater-than, returns bvec"
}, {
    name: "greaterThanEqual",
    signature: "greaterThanEqual(a, b)",
    description: "Component-wise greater-than-or-equal, returns bvec"
}, {
    name: "intBitsToFloat",
    signature: "intBitsToFloat(x)",
    description: "Reinterpret int bits as float"
}, {
    name: "inverse",
    signature: "inverse(m)",
    description: "Inverse of a square matrix"
}, {
    name: "inversesqrt",
    signature: "inversesqrt(x)",
    description: "1 / sqrt(x)"
}, {
    name: "isinf",
    signature: "isinf(x)",
    description: "Returns true if x is positive or negative infinity"
}, {
    name: "isnan",
    signature: "isnan(x)",
    description: "Returns true if x is NaN"
}, {
    name: "ldexp",
    signature: "ldexp(x, exp)",
    description: "x * 2^exp"
}, {
    name: "length",
    signature: "length(v)",
    description: "Length of vector"
}, {
    name: "lessThan",
    signature: "lessThan(a, b)",
    description: "Component-wise less-than, returns bvec"
}, {
    name: "lessThanEqual",
    signature: "lessThanEqual(a, b)",
    description: "Component-wise less-than-or-equal, returns bvec"
}, {
    name: "log",
    signature: "log(x)",
    description: "Natural logarithm"
}, {
    name: "log2",
    signature: "log2(x)",
    description: "Base-2 logarithm"
}, {
    name: "matrixCompMult",
    signature: "matrixCompMult(a, b)",
    description: "Component-wise matrix multiplication"
}, {
    name: "max",
    signature: "max(x, y)",
    description: "Maximum of x and y"
}, {
    name: "min",
    signature: "min(x, y)",
    description: "Minimum of x and y"
}, {
    name: "mix",
    signature: "mix(a, b, t)",
    description: "Linear interpolation: a*(1-t) + b*t"
}, {
    name: "mod",
    signature: "mod(x, y)",
    description: "Modulo (x - y * floor(x/y))"
}, {
    name: "modf",
    signature: "modf(x, out i)",
    description: "Returns fractional part, stores whole part in i"
}, {
    name: "normalize",
    signature: "normalize(v)",
    description: "Unit vector in same direction"
}, {
    name: "not",
    signature: "not(bvec)",
    description: "Component-wise boolean NOT"
}, {
    name: "notEqual",
    signature: "notEqual(a, b)",
    description: "Component-wise inequality, returns bvec"
}, {
    name: "outerProduct",
    signature: "outerProduct(c, r)",
    description: "Outer product of two vectors, returns matrix"
}, {
    name: "packHalf2x16",
    signature: "packHalf2x16(v)",
    description: "Pack vec2 to uint as two f16 values"
}, {
    name: "packSnorm2x16",
    signature: "packSnorm2x16(v)",
    description: "Pack vec2 to uint as two snorm16 values"
}, {
    name: "packSnorm4x8",
    signature: "packSnorm4x8(v)",
    description: "Pack vec4 to uint as four snorm8 values"
}, {
    name: "packUnorm2x16",
    signature: "packUnorm2x16(v)",
    description: "Pack vec2 to uint as two unorm16 values"
}, {
    name: "packUnorm4x8",
    signature: "packUnorm4x8(v)",
    description: "Pack vec4 to uint as four unorm8 values"
}, {
    name: "pow",
    signature: "pow(x, y)",
    description: "x raised to the power y"
}, {
    name: "radians",
    signature: "radians(degrees)",
    description: "Convert degrees to radians"
}, {
    name: "reflect",
    signature: "reflect(I, N)",
    description: "Reflection of incident vector"
}, {
    name: "refract",
    signature: "refract(I, N, eta)",
    description: "Refraction of incident vector"
}, {
    name: "round",
    signature: "round(x)",
    description: "Round to nearest integer"
}, {
    name: "roundEven",
    signature: "roundEven(x)",
    description: "Round to nearest even integer (banker's rounding)"
}, {
    name: "sign",
    signature: "sign(x)",
    description: "Returns -1, 0, or 1"
}, {
    name: "sin",
    signature: "sin(x)",
    description: "Sine of angle in radians"
}, {
    name: "sinh",
    signature: "sinh(x)",
    description: "Hyperbolic sine"
}, {
    name: "smoothstep",
    signature: "smoothstep(a, b, x)",
    description: "Hermite interpolation between a and b"
}, {
    name: "sqrt",
    signature: "sqrt(x)",
    description: "Square root"
}, {
    name: "step",
    signature: "step(edge, x)",
    description: "0.0 if x < edge, else 1.0"
}, {
    name: "tan",
    signature: "tan(x)",
    description: "Tangent of angle in radians"
}, {
    name: "tanh",
    signature: "tanh(x)",
    description: "Hyperbolic tangent"
}, {
    name: "texelFetch",
    signature: "texelFetch(sampler, coords, lod)",
    description: "Fetch a texel using integer coordinates without filtering"
}, {
    name: "texelFetchOffset",
    signature: "texelFetchOffset(sampler, coords, lod, offset)",
    description: "Fetch a texel with integer coordinates and offset"
}, {
    name: "texture",
    signature: "texture(sampler, uv)",
    description: "Sample a texture at UV coordinates"
}, {
    name: "textureGather",
    signature: "textureGather(sampler, uv)",
    description: "Gather one component from the four texels used in bilinear filtering"
}, {
    name: "textureGatherOffset",
    signature: "textureGatherOffset(sampler, uv, offset)",
    description: "Gather with texel offset"
}, {
    name: "textureLod",
    signature: "textureLod(sampler, uv, lod)",
    description: "Sample a texture at an explicit mip level"
}, {
    name: "textureLodOffset",
    signature: "textureLodOffset(sampler, uv, lod, offset)",
    description: "Sample at explicit mip level with texel offset"
}, {
    name: "textureOffset",
    signature: "textureOffset(sampler, uv, offset)",
    description: "Sample a texture with a texel offset"
}, {
    name: "textureProj",
    signature: "textureProj(sampler, P)",
    description: "Project the texture coordinate before sampling"
}, {
    name: "textureProjLod",
    signature: "textureProjLod(sampler, P, lod)",
    description: "Projective sample at explicit mip level"
}, {
    name: "textureProjLodOffset",
    signature: "textureProjLodOffset(sampler, P, lod, offset)",
    description: "Projective sample at explicit mip level with offset"
}, {
    name: "textureQueryLevels",
    signature: "textureQueryLevels(sampler)",
    description: "Number of accessible mip levels"
}, {
    name: "textureQueryLod",
    signature: "textureQueryLod(sampler, uv)",
    description: "Returns the computed LOD for given UV coordinates"
}, {
    name: "textureSize",
    signature: "textureSize(sampler, lod)",
    description: "Returns the dimensions of a texture at a given mip level"
}, {
    name: "transpose",
    signature: "transpose(m)",
    description: "Transpose of a matrix"
}, {
    name: "trunc",
    signature: "trunc(x)",
    description: "Round toward zero"
}, {
    name: "uintBitsToFloat",
    signature: "uintBitsToFloat(x)",
    description: "Reinterpret uint bits as float"
}, {
    name: "unpackHalf2x16",
    signature: "unpackHalf2x16(v)",
    description: "Unpack uint to vec2 from two f16 values"
}, {
    name: "unpackSnorm2x16",
    signature: "unpackSnorm2x16(v)",
    description: "Unpack uint to vec2 from two snorm16 values"
}, {
    name: "unpackSnorm4x8",
    signature: "unpackSnorm4x8(v)",
    description: "Unpack uint to vec4 from four snorm8 values"
}, {
    name: "unpackUnorm2x16",
    signature: "unpackUnorm2x16(v)",
    description: "Unpack uint to vec2 from two unorm16 values"
}, {
    name: "unpackUnorm4x8",
    signature: "unpackUnorm4x8(v)",
    description: "Unpack uint to vec4 from four unorm8 values"
}];