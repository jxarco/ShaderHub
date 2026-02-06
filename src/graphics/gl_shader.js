import * as Constants from "../constants.js";
import { Shader, ShaderPass } from "./shader.js";

const WGSL_TO_GLSL = {
    // Scalars
    f32: "float",
    i32: "int",
    u32: "uint",

    // Vectors (float)
    vec2f: "vec2",
    vec3f: "vec3",
    vec4f: "vec4",

    // Vectors (int)
    vec2i: "ivec2",
    vec3i: "ivec3",
    vec4i: "ivec4",

    // Vectors (uint)
    vec2u: "uvec2",
    vec3u: "uvec3",
    vec4u: "uvec4",

    // Matrices (float only in WGSL)
    mat2x2f: "mat2",
    mat2x3f: "mat2x3",
    mat2x4f: "mat2x4",

    mat3x2f: "mat3x2",
    mat3x3f: "mat3",
    mat3x4f: "mat3x4",

    mat4x2f: "mat4x2",
    mat4x3f: "mat4x3",
    mat4x4f: "mat4",

    // Samplers & textures (handled separately, but listed for completeness)
    sampler: "sampler",
    sampler_comparison: "samplerShadow",

    texture_2d: "sampler2D",
    texture_2d_array: "sampler2DArray",
    texture_cube: "samplerCube",
};

class GLShaderPass extends ShaderPass
{
    constructor( shader, renderer, data )
    {
        super( shader, renderer, data );

        this.uniformLocations = {};
    }

    async execute( renderer )
    {
        if( this.type === "common" )
        {
            return;
        }

        if( this.mustCompile || !this.program )
        {
            const r = await this.compile( renderer );
            if( r !== Constants.WEBGPU_OK )
            {
                return;
            }
        }

        const gl = renderer.gl;

        if( this.type === "image" )
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
                if( !channel ) continue;
                const name = channel.id;
                const tex = this.channelTextures[i];
                gl.activeTexture( gl.TEXTURE0 + i );
                gl.bindTexture( channel.category === 'cubemap' ? gl.TEXTURE_CUBE_MAP : gl.TEXTURE_2D, tex );
                gl.uniform1i( this.uniformLocations[name], i );
            }

            gl.bindBuffer( gl.ARRAY_BUFFER, renderer.fullscreenVBO );
            gl.enableVertexAttribArray( 0 );
            gl.vertexAttribPointer(
                0,          // location
                2,          // vec2
                gl.FLOAT,
                false,
                0,
                0
            );

            gl.drawArrays( gl.TRIANGLES, 0, 3 );

        }
        // else if( this.type === "buffer" )
        // {
        //     if( !this.textures[ 0 ] || !this.textures[ 1 ] )
        //     {
        //         return;
        //     }

        //     const commandEncoder = this.device.createCommandEncoder();
        //     const renderTarget = this.textures[( this.frameCount + 1 ) % 2];
        //     const textureView = renderTarget.createView();

        //     const renderPassDescriptor = {
        //         colorAttachments: [
        //             {
        //                 view: textureView,
        //                 clearValue: [0, 0, 0, 1],
        //                 loadOp: 'clear',
        //                 // loadOp: 'load',
        //                 storeOp: 'store'
        //             },
        //         ],
        //     };

        //     const passEncoder = commandEncoder.beginRenderPass( renderPassDescriptor );
        //     passEncoder.setPipeline( this.pipeline );

        //     const bindGroup = ( this.frameCount % 2 === 0 ) ? this.bindGroup : this.bindGroupB;
        //     if( bindGroup )
        //     {
        //         passEncoder.setBindGroup( 0, bindGroup );
        //     }

        //     passEncoder.draw( 6 );
        //     passEncoder.end();

        //     this.device.queue.submit( [ commandEncoder.finish() ] );

        //     this.frameCount++;
        // }
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
                .map( line => line.trim() )
                .filter( Boolean )
                .map( line => {
                    // Attempt to parse line number
                    const match = line.match(/0:(\d+):\s*(.*)/);
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
        if( !result.valid )
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
        }

        for ( const name in bindings )
        {
            const loc = gl.getUniformLocation( program, name );
            this.uniformLocations[name] = loc;
        }

        return program;
    }

    async compile( renderer )
    {
        const gl = renderer.gl;

        this.defines = {
            "SCREEN_WIDTH": this.resolution[ 0 ],
            "SCREEN_HEIGHT": this.resolution[ 1 ],
        };

        const program = await this.createProgram( gl );
        if ( program?.constructor !== WebGLProgram )
        {
            // This is an error..
            return program;
        }

        // Cache locations (minimal for now)
        this.attribs = {
            position: gl.getAttribLocation( program, 'a_position' ),
        };

        this.program        = program;
        this.frameCount     = 0;
        this.mustCompile    = false;

        return Constants.WEBGL_OK;
    }

    async validate( entryName, entryCode )
    {
        const r = this.getShaderCode( true, entryName, entryCode );

        // Close all toasts
        document.querySelectorAll( ".lextoast" ).forEach( t => t.close() );

        const gl = this.renderer.gl;
        const vs = this.compileShaderCode( gl, gl.VERTEX_SHADER, r.vs_code );
        const fs = this.compileShaderCode( gl, gl.FRAGMENT_SHADER, r.fs_code );
        const errorLog = [ ...vs.log, ...fs.log ];

        if( errorLog.length > 0 )
        {
            console.log( entryCode ?? "" );
            return { valid: false, code: r.fs_code, messages: errorLog }; // only fs is important log-wise
        }

        return { valid: true, ...r, vs, fs };
    }

    getShaderCode( includeBindings = true, entryName, entryCode )
    {
        const templateCodeLines = [ ...GLShader.RENDER_SHADER_TEMPLATE ];
        const shaderLines       = [ ...( entryCode ? entryCode.split( "\n" ) : this.codeLines ) ];
        const defaultBindings   = {};
        const customBindings    = {};
        const textureBindings   = {};
        const samplerBindings   = {};

        // Flip Y or not the shader if it's a render target
        {
            const flipYIndex = templateCodeLines.indexOf( "$vs_flip_y" );
            if( flipYIndex > -1 )
            {
                const utils = [
                    this.type === 'buffer' ? `  v_uv.y = 1.0 - v_uv.y;` : ``
                ]
                templateCodeLines.splice( flipYIndex, 1, ...utils );
            }
        }

        // Add shader utils depending on bind group
        {
            const features = this.shader.getFeatures();
            const glslUtilsIndex = templateCodeLines.indexOf( "$glsl_utils" );
            if( glslUtilsIndex > -1 )
            {
                const utils = [
                    ...( features.includes( "keyboard" ) ? GLShader.GLSL_KEYBOARD_UTILS : [] ),
                ]
                templateCodeLines.splice( glslUtilsIndex, 1, ...utils );
            }
        }

        // Add common block
        {
            const commonPass = this.shader.passes.find( p => p.type === "common" );
            const allCommon = commonPass?.codeLines ?? [];
            const commonIndex = templateCodeLines.indexOf( "$common" );
            console.assert( commonIndex > -1 );
            templateCodeLines.splice( commonIndex, 1, ...allCommon );
        }

        // Add main lines
        {
            const mainImageIndex = templateCodeLines.indexOf( "$main_entry" );
            console.assert( mainImageIndex > -1 );
            templateCodeLines.splice( mainImageIndex, 1, ...shaderLines );
        }

        // Parse general preprocessor lines
        // This has to be the last step before the bindings, to replace every define appearance!
        {
            this._pLine = 0;

            while( this._pLine < templateCodeLines.length )
            {
                this._parseShaderLine( templateCodeLines );
            }

            delete this._pLine;
        }

        const noBindingsShaderCode = templateCodeLines.join( "\n" );

        if( includeBindings )
        {
            let bindingIndex = 0;

            // Default Uniform bindings
            {
                const defaultBindingsIndex = templateCodeLines.indexOf( "$default_bindings" );
                console.assert( defaultBindingsIndex > -1 );
                templateCodeLines.splice( defaultBindingsIndex, 1, ...Constants.DEFAULT_UNIFORMS_LIST.map( ( u, index ) => {
                    if( u.skipBindings ?? false ) return;
                    if( !this.isBindingUsed( u.name, noBindingsShaderCode ) ) return;
                    const binding = bindingIndex++;
                    defaultBindings[ u.name ] = binding;
                    const type = u.type[ this.renderer.backend ] ?? "float";
                    return `uniform ${ type } ${ u.name };`;
                } ).filter( u => u !== undefined ) );
            }

            // Custom Uniform bindings
            {
                const customBindingsIndex = templateCodeLines.indexOf( "$custom_bindings" );
                console.assert( customBindingsIndex > -1 );
                templateCodeLines.splice( customBindingsIndex, 1, ...this.uniforms.map( ( u, index ) => {
                    if( !u ) return;
                    if( !this.isBindingUsed( u.name, noBindingsShaderCode ) ) return;
                    const binding = bindingIndex++;
                    customBindings[ u.name ] = binding;
                    const type = WGSL_TO_GLSL[u.type[ this.renderer.backend ] ?? "f32"];
                    return `uniform ${ type } ${ u.name };`;
                } ).filter( u => u !== undefined ) );
            }

            // Process texture bindings
            {
                const textureBindingsIndex = templateCodeLines.indexOf( "$texture_bindings" );
                console.assert( textureBindingsIndex > -1 );
                const bindings = this.channels.map( ( channel, index ) => {
                    if( !channel ) return;
                    const channelIndexName = `iChannel${ index }`;
                    if( !this.isBindingUsed( channelIndexName, noBindingsShaderCode ) ) return;
                    const binding = bindingIndex++;
                    textureBindings[ channel.id ] = binding;
                    const texture = this.channelTextures[ index ];
                    return `uniform ${ texture.depthOrArrayLayers > 1 ? "samplerCube" : "sampler2D" } ${ channelIndexName };`;
                } ).filter( u => u !== undefined );
                templateCodeLines.splice( textureBindingsIndex, 1, ...bindings );
            }
        }

        const code = templateCodeLines.join( "\n" );
        const codes = code.split( '$fs$' );

        const shaderResult = {
            code,
            vs_code: codes[ 0 ].trim(),
            fs_code: codes[ 1 ].trim(),
            defaultBindings,
            customBindings,
            textureBindings,
            samplerBindings,
            executeOnce: this.executeOnce,
        };

        // delete tmp context
        delete this.structs;

        return shaderResult;
    }

    updateUniforms()
    {
        if( this.uniforms.length === 0 )
            return;

        this.uniforms.map( ( u, index ) => {
            this.setUniform( null, u.name, u.value );
        } );

        this.uniformsDirty = false;
    }

    setUniform( gl, name, value )
    {
        if( this.type === "common" ) return;

        const loc = this.uniformLocations[ name ];
        if ( !loc ) return;

        gl = gl ?? this.renderer.gl;

        gl.useProgram( this.program );

        if ( typeof value === "number" ) gl.uniform1f( loc, value );
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

    static GetUniformSize = function( type ) {
        switch( type )
        {
            case "f32":
            case "i32":
            case "u32":
            return 4;
            case "vec2f":
            case "vec2i":
            case "vec2u":
            return 8;
            case "vec3f":
            case "vec3i":
            case "vec3u":
            return 12;
            case "vec4f":
            case "vec4i":
            case "vec4u":
            return 16;
            case "mat4x4f":
            return 64;
        }
        return 0;
    }

    static GetUniformAlign = function( type ) {
        switch( type )
        {
            case "f32":
            case "i32":
            case "u32":
            return 4;
            case "vec2f":
            case "vec2i":
            case "vec2u":
            return 8;
            case "vec3f":
            case "vec3i":
            case "vec3u":
            case "vec4f":
            case "vec4i":
            case "vec4u":
            return 16;
        }
        return 0;
    }

    getDefaultCode( pass )
    {
        return ( pass.type === "buffer" ? GLShader.RENDER_BUFFER_TEMPLATE : GLShader.RENDER_COMMON_TEMPLATE );
    }

    getFeatures()
    {
        const features = [];

        const buffers = this.passes.filter( p => p.type === "buffer" );
        if( buffers.length ) features.push( "multipass" );

        this.passes.some( p => {
            if( p.channels.filter( u => u?.id === "Keyboard" ).length )
            {
                features.push( "keyboard" );
                return true;
            }
            if( p.channels.filter( u => u?.category === "sound" ).length )
            {
                features.push( "sound" );
                return true;
            }
        } )

        return features.join( "," );
    }
}

GLShader.GLSL_KEYBOARD_UTILS = ``.split( "\n" );

GLShader.COMMON = ``;

GLShader.RENDER_VS_SHADER_TEMPLATE = `#version 300 es

layout(location = 0) in vec2 a_position;

out vec2 v_uv;

void main()
{
    gl_Position = vec4(a_position, 0.0, 1.0);
    v_uv = a_position * 0.5 + 0.5;
$vs_flip_y
}`.split( "\n" );

GLShader.RENDER_FS_SHADER_TEMPLATE = `$fs$#version 300 es
precision highp float;

$default_bindings
$custom_bindings
$texture_bindings

in vec2 v_uv;

out vec4 fragColor;

$glsl_utils
$common
$main_entry

void main()
{
    vec2 fragCoord = v_uv * iResolution + vec2(0.5);
    fragColor = mainImage(v_uv, fragCoord);
}`.split( "\n" );

GLShader.RENDER_SHADER_TEMPLATE = [ ...GLShader.RENDER_VS_SHADER_TEMPLATE, ...GLShader.RENDER_FS_SHADER_TEMPLATE ];

GLShader.RENDER_MAIN_TEMPLATE = `vec4 mainImage(vec2 fragUV, vec2 fragCoord) {
    // Normalized pixel coordinates (from 0 to 1)
    vec2 uv = fragUV; // The same as: fragCoord/iResolution.xy;

    // Time varying pixel color
    vec3 color = 0.5 + 0.5 * cos(iTime + uv.xyx + vec3(0,2,4));

    // Output to screen
    return vec4(color, 1.0);
}`.split( "\n" );

GLShader.RENDER_COMMON_TEMPLATE = `float someFunc(float a, float b) {
    return a + b;
}`.split( "\n" );

GLShader.RENDER_BUFFER_TEMPLATE = `fn mainImage(fragUV : vec2f, fragCoord : vec2f) -> vec4f {
    // Output to screen
    return vec4f(0.0, 0.0, 1.0, 1.0);
}`.split( "\n" );

/*
    End of Shader code
*/

export { GLShader, GLShaderPass };