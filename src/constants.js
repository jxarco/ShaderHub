// General
export const IMAGE_EMPTY_SRC =
    'data:image/gif;base64,R0lGODlhAQABAPcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAABAAEAAAgEAP8FBAA7';

// Errors
export const WEBGPU_OK = 0;
export const WEBGPU_ERROR = 1;
export const WEBGL_OK = 0;
export const WEBGL_ERROR = 1;

// Auth
export const USERNAME_MIN_LENGTH = 3;
export const PASSWORD_MIN_LENGTH = 8;

// Graphics
export const BUFFER_PASS_TEXTURE_A_INDEX = 0;
export const BUFFER_PASS_TEXTURE_B_INDEX = 1;

export const FEATURES = [ 'All', 'Multipass', 'Compute', 'Keyboard', 'Sound' ];
export const ORDER_BY_NAMES = [ 'Popular', 'Trending', 'Recent' ];
export const ORDER_BY_MAPPING = {
    'popular': { field: 'like_count', direction: 'desc' },
    'trending': { field: 'view_count', direction: 'desc' },
    'recent': { field: '$createdAt', direction: 'desc' }
};

export const UNIFORM_CHANNELS_COUNT = 4;
export const DEFAULT_UNIFORMS_LIST = [
    { name: 'iTime', type: { webgpu: 'f32', webgl: 'float' }, info: 'Shader playback time (s)' },
    { name: 'iTimeDelta', type: { webgpu: 'f32', webgl: 'float' }, info: 'Render time (s)' },
    { name: 'iFrame', type: { webgpu: 'i32', webgl: 'int' }, info: 'Shader playback frame' },
    { name: 'iResolution', type: { webgpu: 'vec2f', webgl: 'vec2' }, info: 'Viewport resolution (px)' },
    { name: 'iDate', type: { webgpu: 'vec4f', webgl: 'vec4' }, info: 'Year, month, day, seconds' },
    { name: 'iMouse', type: { webgpu: 'MouseData', webgl: 'MouseData' }, info: '{ pos, start, delta, press, click }' },
    { name: 'iChannel0..3', type: { webgpu: 'texture_2d<f32>', webgl: 'sampler2D' }, info: 'Texture input channel', skipBindings: true }
];
export const DEFAULT_UNIFORM_NAMES = DEFAULT_UNIFORMS_LIST.map( ( u ) => u.name );
export const DEFAULT_UNIFORM_FIELDS = {
    iMouse: [
        { name: 'pos', size: 2 }, // Current mouse position (x, y)
        { name: 'start', size: 2 }, // Initial/last mouse position (x, y)
        { name: 'delta', size: 2 }, // Delta movement (dx, dy)
        { name: 'press', size: 1 }, // Mouse down state (-1 or button)
        { name: 'click', size: 1 } // Mouse pressed/clicked state
    ]
};

// WebGL Renderer
export const WGSL_TO_GLSL = {
    // Scalars
    f32: 'float',
    i32: 'int',
    u32: 'uint',
    // Vectors (float)
    vec2f: 'vec2',
    vec3f: 'vec3',
    vec4f: 'vec4',
    // Vectors (int)
    vec2i: 'ivec2',
    vec3i: 'ivec3',
    vec4i: 'ivec4',
    // Vectors (uint)
    vec2u: 'uvec2',
    vec3u: 'uvec3',
    vec4u: 'uvec4',
    // Matrices (float only in WGSL)
    mat2x2f: 'mat2',
    mat2x3f: 'mat2x3',
    mat2x4f: 'mat2x4',

    mat3x2f: 'mat3x2',
    mat3x3f: 'mat3',
    mat3x4f: 'mat3x4',

    mat4x2f: 'mat4x2',
    mat4x3f: 'mat4x3',
    mat4x4f: 'mat4',
    // Samplers & textures (handled separately, but listed for completeness)
    sampler: 'sampler',
    sampler_comparison: 'samplerShadow',
    texture_2d: 'sampler2D',
    texture_2d_array: 'sampler2DArray',
    texture_cube: 'samplerCube'
};
