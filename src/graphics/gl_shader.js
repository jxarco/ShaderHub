import * as Constants from '../constants.js';
import { Shader, ShaderPass } from './shader.js';

class GLShaderPass extends ShaderPass
{
    constructor( shader, renderer, data )
    {
        super( shader, renderer, data );

        this.uniformLocations = {};
    }

    async execute( renderer )
    {
        if ( this.type === 'common' )
        {
            return;
        }

        if ( this.mustCompile || !this.program )
        {
            const r = await this.compile( renderer );
            if ( r !== Constants.WEBGPU_OK )
            {
                return;
            }
        }

        const gl = renderer.gl;

        if ( this.type === 'image' )
        {
            // default swapchain
            gl.bindFramebuffer( gl.FRAMEBUFFER, null );

            gl.viewport( 0, 0, renderer.canvas.width, renderer.canvas.height );
            gl.clearColor( 0.0, 0.0, 0.0, 1.0 );
            gl.clear( gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT );

            gl.useProgram( this.program );

            // bind textures
            for ( let i = 0; i < this.channelTextures.length; i++ )
            {
                const channel = this.channels[i];
                if ( !channel ) continue;
                const name = channel.id;
                let texture = this.channelTextures[i];
                if ( texture.constructor === Array )
                {
                    texture = texture[this.frameCount % 2].texture;
                }
                gl.activeTexture( gl.TEXTURE0 + i );
                gl.bindTexture( channel.category === 'cubemap' ? gl.TEXTURE_CUBE_MAP : gl.TEXTURE_2D, texture );
                gl.uniform1i( this.uniformLocations[name], i );
            }

            gl.bindBuffer( gl.ARRAY_BUFFER, renderer.fullscreenVBO );
            gl.enableVertexAttribArray( 0 );
            gl.vertexAttribPointer(
                0, // location
                2, // vec2
                gl.FLOAT,
                false,
                0,
                0
            );

            gl.drawArrays( gl.TRIANGLES, 0, 3 );

            this.frameCount++;
        }
        else if ( this.type === 'buffer' )
        {
            if ( !this.textures[0] || !this.textures[1] )
            {
                return;
            }

            const renderTarget = this.textures[( this.frameCount + 1 ) % 2];
            gl.bindFramebuffer( gl.FRAMEBUFFER, renderTarget.framebuffer );

            gl.viewport( 0, 0, renderer.canvas.width, renderer.canvas.height );
            gl.clearColor( 0.0, 0.0, 0.0, 1.0 );
            gl.clear( gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT );

            gl.useProgram( this.program );

            // bind textures
            for ( let i = 0; i < this.channelTextures.length; i++ )
            {
                const channel = this.channels[i];
                if ( !channel ) continue;
                const name = channel.id;
                let texture = this.channelTextures[i];
                if ( texture.constructor === Array )
                {
                    texture = texture[this.frameCount % 2].texture;
                }
                gl.activeTexture( gl.TEXTURE0 + i );
                gl.bindTexture( channel.category === 'cubemap' ? gl.TEXTURE_CUBE_MAP : gl.TEXTURE_2D, texture );
                gl.uniform1i( this.uniformLocations[name], i );
            }

            gl.bindBuffer( gl.ARRAY_BUFFER, renderer.fullscreenVBO );
            gl.enableVertexAttribArray( 0 );
            gl.vertexAttribPointer(
                0, // location
                2, // vec2
                gl.FLOAT,
                false,
                0,
                0
            );

            gl.drawArrays( gl.TRIANGLES, 0, 3 );

            this.frameCount++;
        }
    }

    compileShaderCode( gl, type, source )
    {
        const shader = gl.createShader( type );
        gl.shaderSource( shader, source );
        gl.compileShader( shader );

        const log = [];

        if ( !gl.getShaderParameter( shader, gl.COMPILE_STATUS ) )
        {
            const err = gl.getShaderInfoLog( shader );
            console.error( err );
            gl.deleteShader( shader );

            const log = err
                .replace( '\x00', '' )
                .split( '\n' )
                .map( ( line ) => line.trim() )
                .filter( Boolean )
                .map( ( line ) => {
                    // Attempt to parse line number
                    const match = line.match( /0:(\d+):\s*(.*)/ );
                    if ( match )
                    {
                        const lineNum = parseInt( match[1], 10 );
                        const message = match[2];
                        return { linePos: 0, lineNum, message, type: 'error' };
                    }
                    // fallback if parsing fails
                    return { linePos: 0, lineNum: 0, message: line, type: 'error' };
                } );

            return { log };
        }

        return { shader, log };
    }

    async createProgram( gl )
    {
        const result = await this.validate();
        if ( !result.valid )
        {
            return result;
        }

        const vs = result.vs;
        const fs = result.fs;

        if ( !vs || !fs )
        {
            return null;
        }

        const program = gl.createProgram();
        gl.attachShader( program, vs.shader );
        gl.attachShader( program, fs.shader );
        gl.linkProgram( program );
        gl.deleteShader( vs.shader );
        gl.deleteShader( fs.shader );

        if ( !gl.getProgramParameter( program, gl.LINK_STATUS ) )
        {
            console.error(
                'Program link error:',
                gl.getProgramInfoLog( program )
            );
            gl.deleteProgram( program );
            return null;
        }

        this.defaultBindings = result.defaultBindings;
        this.customBindings = result.customBindings;
        this.textureBindings = result.textureBindings;
        this.codeContent = result.code.replace( '$fs$', '\n' );
        this.uniformLocations = {};

        const bindings = {
            ...this.defaultBindings,
            ...this.customBindings,
            ...this.textureBindings
        };

        for ( const name in bindings )
        {
            const texChannelName = bindings[name];
            const loc = gl.getUniformLocation( program, texChannelName );
            this.uniformLocations[name] = loc;

            // For struct bindings, also cache field locations (e.g., iMouse.pos)
            const fields = Constants.DEFAULT_UNIFORM_FIELDS[name];
            if ( fields?.length )
            {
                for ( const field of fields )
                {
                    const loc = gl.getUniformLocation( program, `${name}.${field.name}` );
                    if ( loc ) this.uniformLocations[`${name}.${field.name}`] = loc;
                }
            }
        }

        return program;
    }

    async compile( renderer )
    {
        const gl = renderer.gl;

        this.defines = {
            'SCREEN_WIDTH': this.resolution[0],
            'SCREEN_HEIGHT': this.resolution[1]
        };

        const program = await this.createProgram( gl );
        if ( program?.constructor !== WebGLProgram )
        {
            // This is an error..
            return program;
        }

        // Cache locations (minimal for now)
        this.attribs = {
            position: gl.getAttribLocation( program, 'a_position' )
        };

        this.program = program;
        this.frameCount = 0;
        this.mustCompile = false;

        return Constants.WEBGL_OK;
    }

    async validate( entryName, entryCode )
    {
        const r = this.getShaderCode( true, entryName, entryCode );
        const gl = this.renderer.gl;
        const vs = this.compileShaderCode( gl, gl.VERTEX_SHADER, r.vs_code );
        const fs = this.compileShaderCode( gl, gl.FRAGMENT_SHADER, r.fs_code );
        const errorLog = [ ...vs.log, ...fs.log ];

        if ( errorLog.length > 0 )
        {
            console.log( entryCode ?? '' );
            return { valid: false, code: r.fs_code, messages: errorLog }; // only fs is important log-wise
        }

        return { valid: true, ...r, vs, fs };
    }

    getShaderCode( includeBindings = true, entryName, entryCode )
    {
        const templateCodeLines = [ ...GLShader.RENDER_SHADER_TEMPLATE ];
        const shaderLines = [ ...( entryCode ? entryCode.split( '\n' ) : this.codeLines ) ];
        const defaultBindings = {};
        const customBindings = {};
        const textureBindings = {};
        const samplerBindings = {};

        // Flip Y or not the shader if it's a render target
        {
            const flipYIndex = templateCodeLines.indexOf( '$vs_flip_y' );
            if ( flipYIndex > -1 )
            {
                const utils = [
                    this.type === 'buffer' ? `  v_uv.y = 1.0 - v_uv.y;` : ``
                ];
                templateCodeLines.splice( flipYIndex, 1, ...utils );
            }
        }

        // Add shader utils depending on bind group
        {
            const features = this.shader.getFeatures();
            const glslUtilsIndex = templateCodeLines.indexOf( '$glsl_utils' );
            if ( glslUtilsIndex > -1 )
            {
                const utils = [
                    ...( features.includes( 'keyboard' ) ? GLShader.GLSL_KEYBOARD_UTILS : [] )
                ];
                templateCodeLines.splice( glslUtilsIndex, 1, ...utils );
            }
        }

        // Add common block
        {
            const commonPass = this.shader.passes.find( ( p ) => p.type === 'common' );
            const allCommon = commonPass?.codeLines ?? [];
            const commonIndex = templateCodeLines.indexOf( '$common' );
            console.assert( commonIndex > -1 );
            templateCodeLines.splice( commonIndex, 1, ...allCommon );
        }

        // Add main lines
        {
            const mainImageIndex = templateCodeLines.indexOf( '$main_entry' );
            console.assert( mainImageIndex > -1 );
            templateCodeLines.splice( mainImageIndex, 1, ...shaderLines );
        }

        // Parse general preprocessor lines
        // This has to be the last step before the bindings, to replace every define appearance!
        {
            this._pLine = 0;

            while ( this._pLine < templateCodeLines.length )
            {
                this._parseShaderLine( templateCodeLines );
            }

            delete this._pLine;
        }

        const noBindingsShaderCode = templateCodeLines.join( '\n' );

        if ( includeBindings )
        {
            // Default Uniform bindings
            {
                const defaultBindingsIndex = templateCodeLines.indexOf( '$default_bindings' );
                console.assert( defaultBindingsIndex > -1 );
                templateCodeLines.splice( defaultBindingsIndex, 1, ...Constants.DEFAULT_UNIFORMS_LIST.map( ( u, index ) => {
                    if ( u.skipBindings ?? false ) return;
                    if ( !this.isBindingUsed( u.name, noBindingsShaderCode ) ) return;
                    defaultBindings[u.name] = u.name;
                    const type = u.type[this.renderer.backend] ?? 'float';
                    return `uniform ${type} ${u.name};`;
                } ).filter( ( u ) => u !== undefined ) );
            }

            // Custom Uniform bindings
            {
                const customBindingsIndex = templateCodeLines.indexOf( '$custom_bindings' );
                console.assert( customBindingsIndex > -1 );
                templateCodeLines.splice( customBindingsIndex, 1, ...this.uniforms.map( ( u, index ) => {
                    if ( !u ) return;
                    if ( !this.isBindingUsed( u.name, noBindingsShaderCode ) ) return;
                    customBindings[u.name] = u.name;
                    const type = Constants.WGSL_TO_GLSL[u.type[this.renderer.backend] ?? 'f32'];
                    return `uniform ${type} ${u.name};`;
                } ).filter( ( u ) => u !== undefined ) );
            }

            // Process texture bindings
            {
                const textureBindingsIndex = templateCodeLines.indexOf( '$texture_bindings' );
                console.assert( textureBindingsIndex > -1 );
                const bindings = this.channels.map( ( channel, index ) => {
                    if ( !channel ) return;
                    const channelIndexName = `iChannel${index}`;
                    if ( !this.isBindingUsed( channelIndexName, noBindingsShaderCode ) ) return;
                    textureBindings[channel.id] = channelIndexName; // store channel name instead of binding, for the uniform location
                    const texture = this.channelTextures[index];
                    return `uniform ${texture.depthOrArrayLayers > 1 ? 'samplerCube' : 'sampler2D'} ${channelIndexName};`;
                } ).filter( ( u ) => u !== undefined );
                templateCodeLines.splice( textureBindingsIndex, 1, ...bindings );
            }
        }

        const code = templateCodeLines.join( '\n' );
        const codes = code.split( '$fs$' );

        const shaderResult = {
            code,
            vs_code: codes[0].trim(),
            fs_code: codes[1].trim(),
            defaultBindings,
            customBindings,
            textureBindings,
            samplerBindings,
            executeOnce: this.executeOnce
        };

        // delete tmp context
        delete this.structs;

        return shaderResult;
    }

    updateUniforms()
    {
        if ( this.uniforms.length === 0 )
        {
            return;
        }

        this.uniforms.map( ( u, index ) => {
            this.setUniform( null, u.name, u.value );
        } );

        this.uniformsDirty = false;
    }

    setUniform( gl, name, value )
    {
        if ( this.type === 'common' ) return;

        gl = gl ?? this.renderer.gl;
        gl.useProgram( this.program );

        // Handle struct/object data (e.g., MouseData struct)
        if ( typeof value === 'object' && !Array.isArray( value ) && value?.length === undefined )
        {
            for ( const field in value )
            {
                this.setUniform( gl, `${name}.${field}`, value[field] );
            }
            return;
        }

        const loc = this.uniformLocations[name];
        if ( !loc ) return;

        // Handle scalar and vector types
        if ( typeof value === 'number' ) gl.uniform1f( loc, value );
        else if ( value.length === 2 ) gl.uniform2fv( loc, value );
        else if ( value.length === 3 ) gl.uniform3fv( loc, value );
        else if ( value.length === 4 ) gl.uniform4fv( loc, value );
    }
}

class GLShader extends Shader
{
    constructor( data )
    {
        super( data );
    }

    static GetUniformSize = function( type )
    {
        if ( type === 'float' || type === 'int' || type === 'uint' || type === 'bool' || type.startsWith( 'sampler' ) ) return 4;

        if ( type.includes( 'vec2' ) ) return 8;
        if ( type.includes( 'vec3' ) ) return 12;
        if ( type.includes( 'vec4' ) ) return 16;

        // matCxR
        if ( type.startsWith( 'mat' ) )
        {
            const match = type.match( /mat(\d)(?:x(\d))?/ );
            if ( match )
            {
                const cols = parseInt( match[1] );
                const rows = match[2] ? parseInt( match[2] ) : cols; // mat3 = mat3x3
                // In std140: vec2 columns = 8 bytes, vec3/vec4 columns = 16 bytes (padded)
                const colSize = rows === 2 ? 8 : 16;
                return cols * colSize;
            }
        }

        return 0;
    };

    static GetUniformAlign = function( type )
    {
        if ( type === 'float' || type === 'int' || type === 'uint' || type === 'bool' ) return 4;
        if ( type.startsWith( 'sampler' ) ) return 4;
        if ( type.includes( 'vec2' ) ) return 8;
        if ( type.includes( 'vec3' ) || type.includes( 'vec4' ) || type.startsWith( 'mat' ) ) return 16;
        return 0;
    };

    getDefaultCode( pass )
    {
        return ( pass.type === 'buffer' ? GLShader.RENDER_BUFFER_TEMPLATE : GLShader.RENDER_COMMON_TEMPLATE );
    }

    getFeatures()
    {
        const features = [];

        const buffers = this.passes.filter( ( p ) => p.type === 'buffer' );
        if ( buffers.length ) features.push( 'multipass' );

        this.passes.some( ( p ) => {
            if ( p.channels.filter( ( u ) => u?.id === 'Keyboard' ).length )
            {
                features.push( 'keyboard' );
                return true;
            }
            if ( p.channels.filter( ( u ) => u?.category === 'sound' ).length )
            {
                features.push( 'sound' );
                return true;
            }
        } );

        return features.join( ',' );
    }
}

GLShader.GLSL_KEYBOARD_UTILS = ``.split( '\n' );

GLShader.COMMON = `struct MouseData {
    vec2 pos;
    vec2 start;
    vec2 delta;
    float press;
    float click;
};`;

GLShader.RENDER_VS_SHADER_TEMPLATE = `#version 300 es

layout(location = 0) in vec2 a_position;

out vec2 v_uv;

void main()
{
    gl_Position = vec4(a_position, 0.0, 1.0);
    v_uv = a_position * 0.5 + 0.5;
$vs_flip_y
}`.split( '\n' );

GLShader.RENDER_FS_SHADER_TEMPLATE = `$fs$#version 300 es
precision highp float;

${GLShader.COMMON}

$default_bindings
$custom_bindings
$texture_bindings

in vec2 v_uv;
out vec4 fragColor;

$glsl_utils
// Common pass code
$common
$main_entry

void main()
{
    vec2 fragCoord = v_uv * iResolution + vec2(0.5);
    fragColor = mainImage(v_uv, fragCoord);
}`.split( '\n' );

GLShader.RENDER_SHADER_TEMPLATE = [ ...GLShader.RENDER_VS_SHADER_TEMPLATE, ...GLShader.RENDER_FS_SHADER_TEMPLATE ];

GLShader.RENDER_MAIN_TEMPLATE = `vec4 mainImage(vec2 fragUV, vec2 fragCoord) {
    // Normalized pixel coordinates (from 0 to 1)
    vec2 uv = fragUV; // The same as: fragCoord/iResolution.xy;

    // Time varying pixel color
    vec3 color = 0.5 + 0.5 * cos(iTime + uv.xyx + vec3(0,2,4));

    // Output to screen
    return vec4(color, 1.0);
}`.split( '\n' );

GLShader.RENDER_TEXTURE_TEMPLATE = `vec4 mainImage(vec2 fragUV, vec2 fragCoord) {
    // Normalized pixel coordinates (from 0 to 1)
    vec2 uv = fragUV; // The same as: fragCoord/iResolution.xy;

    // Sample from texture channel 0
    vec4 color = texture(iChannel0, uv);

    // Output to screen
    return vec4(color.rgb, 1.0);
}`.split( '\n' );

GLShader.RENDER_MOUSE_TEMPLATE = `vec4 mainImage(vec2 fragUV, vec2 fragCoord) {
    // Normalized pixel coordinates (from 0 to 1)
    vec2 uv = fragCoord/iResolution.y;

    // Get mouse position in normalized coordinates
    vec2 mousePos = iMouse.pos / iResolution.y;

    // Calculate distance from mouse
    float dist = length(uv - mousePos);

    // Create a smooth circle around the mouse
    float radius = 0.2;
    float circle = smoothstep(radius, radius - 0.05, dist);

    // Add a glow effect
    float glow = exp(-dist * 5.0) * 0.5;

    // Color based on mouse click state
    vec3 color = mix(
        vec3(0.2, 0.5, 1.0),  // Default blue
        vec3(1.0, 0.3, 0.5),  // Pink when clicked
        iMouse.click
    );

    // Combine circle and glow
    vec3 finalColor = color * (circle + glow);

    // Output to screen
    return vec4(finalColor, 1.0);
}`.split( '\n' );

GLShader.RENDER_ANIMATED_TEMPLATE = `vec4 mainImage(vec2 fragUV, vec2 fragCoord) {
    // Centered coordinates (from -0.5 to 0.5)
    vec2 uv = fragUV - 0.5;

    // Make aspect ratio square
    uv.x *= iResolution.x / iResolution.y;

    // Convert to polar coordinates
    float angle = atan(uv.y, uv.x);
    float radius = length(uv);

    // Rotating pattern
    float pattern = sin(angle * 6.0 - iTime * 2.0) * 0.5 + 0.5;

    // Pulsing rings
    float rings = sin(radius * 20.0 - iTime * 3.0) * 0.5 + 0.5;

    // Combine patterns
    float combined = pattern * rings;

    // Create color gradient based on angle and time
    vec3 color1 = vec3(1.0, 0.3, 0.5);
    vec3 color2 = vec3(0.2, 0.8, 1.0);
    vec3 color = mix(color1, color2, sin(angle + iTime) * 0.5 + 0.5);

    // Apply pattern
    color *= combined * 2.0;

    // Add vignette
    color *= 1.0 - radius * 0.5;

    // Output to screen
    return vec4(color, 1.0);
}`.split( '\n' );

GLShader.RENDER_KEYBOARD_TEMPLATE = `vec4 mainImage(vec2 fragUV, vec2 fragCoord) {
    // Connect iChannel0 to: Keyboard
    // Keyboard texture layout (256 x 3):
    //   Row 0 — state  : is the key currently held down
    //   Row 1 — press  : one-frame pulse when the key is pressed
    //   Row 2 — toggle : flips each time the key is pressed

    const int KEY_LEFT  = 37, KEY_RIGHT = 39, KEY_UP = 38, KEY_DOWN = 40;
    const int KEY_W = 87, KEY_A = 65, KEY_S = 83, KEY_D = 68;
    const int KEY_SPACE = 32;

    // State (held): is the key currently held down?
    float l = clamp(
        texelFetch(iChannel0, ivec2(KEY_LEFT, 0), 0).r +
        texelFetch(iChannel0, ivec2(KEY_A, 0), 0).r, 0.0, 1.0);
    float r = clamp(
        texelFetch(iChannel0, ivec2(KEY_RIGHT, 0), 0).r +
        texelFetch(iChannel0, ivec2(KEY_D, 0), 0).r, 0.0, 1.0);
    float u = clamp(
        texelFetch(iChannel0, ivec2(KEY_UP, 0), 0).r +
        texelFetch(iChannel0, ivec2(KEY_W, 0), 0).r, 0.0, 1.0);
    float d = clamp(
        texelFetch(iChannel0, ivec2(KEY_DOWN, 0), 0).r +
        texelFetch(iChannel0, ivec2(KEY_S, 0), 0).r, 0.0, 1.0);

    // Toggle: flips each time the key is pressed
    float toggle  = texelFetch(iChannel0, ivec2(KEY_SPACE, 1), 0).r;
    
    // Press (one-frame): fires once the moment the key is pressed
    float press = texelFetch(iChannel0, ivec2(KEY_SPACE, 2), 0).r;

    vec2 uv = fragCoord / iResolution.xy;
    float aspect = iResolution.x / iResolution.y;
    vec2 dir = vec2(r - l, u - d);

    // Scrolling grid driven by held arrow keys / WASD
    vec2 gv = (uv * vec2(aspect, 1.0) + dir * iTime * 2.0) * 10.0;
    float grid = smoothstep(0.05, 0.0, min(fract(gv.x), fract(gv.y)));

    // Space toggles color palette
    vec3 colA = vec3(0.15, 0.45, 1.0);
    vec3 colB = vec3(1.0, 0.30, 0.60);
    vec3 pal  = mix(colA, colB, toggle);

    vec3 col = vec3(0.04, 0.04, 0.08) + grid * pal * 0.6;

    // Space press flashes the screen
    col += press * 0.4;

    return vec4(col, 1.0);
}`.split( '\n' );

GLShader.RENDER_COMMON_TEMPLATE = `float someFunc(float a, float b) {
    return a + b;
}`.split( '\n' );

GLShader.RENDER_BUFFER_TEMPLATE = `vec4 mainImage(vec2 fragUV, vec2 fragCoord) {
    // Output to screen
    return vec4(0.0, 0.0, 1.0, 1.0);
}`.split( '\n' );

/*
    End of Shader code
*/

export { GLShader, GLShaderPass };
