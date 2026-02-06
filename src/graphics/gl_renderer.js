import * as Constants from '../constants.js';
import { Renderer } from './renderer.js';

const GL_SAMPLER_MAP = {
    magFilter: {
        'nearest': WebGL2RenderingContext.NEAREST,
        'linear': WebGL2RenderingContext.LINEAR
    },
    minFilter: {
        'nearest': WebGL2RenderingContext.NEAREST,
        'linear': WebGL2RenderingContext.LINEAR,
        'nearest-mipmap-nearest': WebGL2RenderingContext.NEAREST_MIPMAP_NEAREST,
        'linear-mipmap-nearest': WebGL2RenderingContext.LINEAR_MIPMAP_NEAREST,
        'nearest-mipmap-linear': WebGL2RenderingContext.NEAREST_MIPMAP_LINEAR,
        'linear-mipmap-linear': WebGL2RenderingContext.LINEAR_MIPMAP_LINEAR
    },
    addressMode: {
        'clamp-to-edge': WebGL2RenderingContext.CLAMP_TO_EDGE,
        'repeat': WebGL2RenderingContext.REPEAT,
        'mirror-repeat': WebGL2RenderingContext.MIRRORED_REPEAT
    }
};

class GLRenderer extends Renderer
{
    constructor( canvas, backend )
    {
        super( canvas, backend );

        this.lang = 'GLSL';
    }

    async init()
    {
        this.gl = this.canvas.getContext( 'webgl2', {
            alpha: false,
            antialias: true,
            depth: true,
            stencil: false,
            preserveDrawingBuffer: false
        } );

        if ( this.quitIfWebGLNotAvailable() === Constants.WEBGL_ERROR )
        {
            return;
        }

        const gl = this.gl;

        const devicePixelRatio = window.devicePixelRatio;
        this.canvas.width = this.canvas.clientWidth * devicePixelRatio;
        this.canvas.height = this.canvas.clientHeight * devicePixelRatio;
        gl.viewport( 0, 0, this.canvas.width, this.canvas.height );

        // Default webgl state
        gl.enable( gl.DEPTH_TEST );
        gl.depthFunc( gl.LEQUAL );

        gl.enable( gl.CULL_FACE );
        gl.cullFace( gl.BACK );
        gl.frontFace( gl.CCW );

        gl.disable( gl.BLEND );

        // Create vertex data
        const FULLSCREEN_VERTICES = new Float32Array( [
            // x,  y,
            -1,
            -1,
            3,
            -1,
            -1,
            3
        ] );

        this.fullscreenVBO = gl.createBuffer();
        gl.bindBuffer( gl.ARRAY_BUFFER, this.fullscreenVBO );
        gl.bufferData( gl.ARRAY_BUFFER, FULLSCREEN_VERTICES, gl.STATIC_DRAW );
        gl.bindBuffer( gl.ARRAY_BUFFER, null );

        super.init();
    }

    createBuffer( desc = {} )
    {
        // Not using desc.usage in WebGL, already using gl.UNIFORM_BUFFER

        const gl = this.gl;
        const buffer = gl.createBuffer();

        gl.bindBuffer( gl.UNIFORM_BUFFER, buffer );
        gl.bufferData( gl.UNIFORM_BUFFER, desc.size ?? 0, gl.DYNAMIC_DRAW );
        gl.bindBuffer( gl.UNIFORM_BUFFER, null );

        return buffer;
    }

    createSampler( desc = {} )
    {
        const gl = this.gl;
        const sampler = gl.createSampler();

        const mag = GL_SAMPLER_MAP.magFilter[desc.magFilter ?? 'nearest'];
        const min = GL_SAMPLER_MAP.minFilter[desc.minFilter ?? 'nearest'];
        const wrapU = GL_SAMPLER_MAP.addressMode[desc.addressModeU ?? 'clamp-to-edge'];
        const wrapV = GL_SAMPLER_MAP.addressMode[desc.addressModeV ?? 'clamp-to-edge'];

        gl.samplerParameteri( sampler, gl.TEXTURE_MAG_FILTER, mag );
        gl.samplerParameteri( sampler, gl.TEXTURE_MIN_FILTER, min );
        gl.samplerParameteri( sampler, gl.TEXTURE_WRAP_S, wrapU );
        gl.samplerParameteri( sampler, gl.TEXTURE_WRAP_T, wrapV );

        // WebGL has no W wrap unless using 3D textures
        if ( desc.addressModeW && gl.TEXTURE_WRAP_R !== undefined )
        {
            const wrapW = GL_SAMPLER_MAP.addressMode[desc.addressModeW];
            gl.samplerParameteri( sampler, gl.TEXTURE_WRAP_R, wrapW );
        }

        return sampler;
    }

    createTexture( desc = {} )
    {
        const gl = this.gl;

        const width = desc.size?.[0] ?? 1;
        const height = desc.size?.[1] ?? 1;
        const format = desc.format ?? 'rgba8unorm';

        // Map WebGPU format -> GL format
        const glFormat = gl.RGBA8;
        const glType = gl.UNSIGNED_BYTE;
        const glBase = gl.RGBA;

        // Create texture
        const texture = gl.createTexture();
        gl.bindTexture( gl.TEXTURE_2D, texture );

        gl.texImage2D(
            gl.TEXTURE_2D,
            0,
            glFormat,
            width,
            height,
            0,
            glBase,
            glType,
            null
        );

        let framebuffer = null;

        if ( desc.usage & GPUTextureUsage.RENDER_ATTACHMENT )
        {
            // required defaults in case of render target
            gl.texParameteri( gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR );
            gl.texParameteri( gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR );
            gl.texParameteri( gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE );
            gl.texParameteri( gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE );

            framebuffer = gl.createFramebuffer();

            gl.bindFramebuffer( gl.FRAMEBUFFER, framebuffer );
            gl.framebufferTexture2D(
                gl.FRAMEBUFFER,
                gl.COLOR_ATTACHMENT0,
                gl.TEXTURE_2D,
                texture,
                0
            );

            const status = gl.checkFramebufferStatus( gl.FRAMEBUFFER );
            if ( status !== gl.FRAMEBUFFER_COMPLETE )
            {
                console.error( 'Framebuffer incomplete:', status );
            }

            gl.bindFramebuffer( gl.FRAMEBUFFER, null );
        }

        gl.bindTexture( gl.TEXTURE_2D, null );

        return {
            texture,
            framebuffer,
            width,
            height,
            depthOrArrayLayers: 1,
            format
        };
    }

    updateTexture( texture, bitmap )
    {
        const gl = this.gl;

        gl.bindTexture( gl.TEXTURE_2D, texture );

        gl.texImage2D(
            gl.TEXTURE_2D,
            0,
            gl.RGBA,
            gl.RGBA,
            gl.UNSIGNED_BYTE,
            bitmap
        );

        gl.bindTexture( gl.TEXTURE_2D, null );

        return texture;
    }

    async createTextureFromImage( data, id, label = '', options = {} )
    {
        options.flipY = options.flipY ?? true;
        options.useMipmaps = options.useMipmaps ?? true;

        const gl = this.gl;
        const imageBitmap = await createImageBitmap( await new Blob( [ data ] ), {
            imageOrientation: options.flipY ? 'flipY' : 'none'
        } );

        const texture = gl.createTexture();
        texture.depthOrArrayLayers = 1;

        gl.bindTexture( gl.TEXTURE_2D, texture );

        // gl.pixelStorei( gl.UNPACK_FLIP_Y_WEBGL, options.flipY );

        // Upload level 0
        gl.texImage2D(
            gl.TEXTURE_2D,
            0,
            gl.RGBA,
            gl.RGBA,
            gl.UNSIGNED_BYTE,
            imageBitmap
        );

        // Allocate mip levels if needed
        if ( options.useMipmaps )
        {
            gl.generateMipmap( gl.TEXTURE_2D );
        }
        else
        {
            // WebGL requires this if no mipmaps
            gl.texParameteri( gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR );
        }

        // Sensible defaults, samplers will override
        gl.texParameteri( gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE );
        gl.texParameteri( gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE );

        // unbind once done
        gl.bindTexture( gl.TEXTURE_2D, null );

        this.gpuTextures[id] = texture;

        return texture;
    }

    async createCubemapTextureFromImage( arrayBuffer, id, label = '', options = {} )
    {
        options.flipY = options.flipY ?? false;
        options.useMipmaps = options.useMipmaps ?? true;

        const gl = this.gl;
        const zip = await JSZip.loadAsync( arrayBuffer );
        const faceNames = [ 'px', 'nx', 'ny', 'py', 'pz', 'nz' ];
        const faceImages = [];

        for ( const face of faceNames )
        {
            const file = zip.file( `${face}.png` ) || zip.file( `${face}.jpg` );
            if ( !file ) throw new Error( `Missing cubemap face: ${face}` );
            const blob = await file.async( 'blob' );
            const imageBitmap = await createImageBitmap( blob, {
                imageOrientation: options.flipY ? 'flipY' : 'none'
            } );
            faceImages.push( imageBitmap );
        }

        const texture = gl.createTexture();
        texture.depthOrArrayLayers = 6;

        gl.bindTexture( gl.TEXTURE_CUBE_MAP, texture );

        for ( let i = 0; i < faceImages.length; ++i )
        {
            const bitmap = faceImages[i];

            gl.texImage2D(
                gl.TEXTURE_CUBE_MAP_POSITIVE_X + i,
                0,
                gl.RGBA,
                gl.RGBA,
                gl.UNSIGNED_BYTE,
                bitmap
            );
        }

        if ( options.useMipmaps )
        {
            gl.generateMipmap( gl.TEXTURE_CUBE_MAP );
        }
        else
        {
            // WebGL requires this if no mipmaps
            gl.texParameteri( gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MIN_FILTER, gl.LINEAR );
        }

        // Sensible defaults, samplers will override
        gl.texParameteri( gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE );
        gl.texParameteri( gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE );
        gl.texParameteri( gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE );
        gl.texParameteri( gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MIN_FILTER, gl.LINEAR );
        gl.texParameteri( gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MAG_FILTER, gl.LINEAR );

        // unbind once done
        gl.bindTexture( gl.TEXTURE_CUBE_MAP, null );

        this.gpuTextures[id] = texture;

        return texture;
    }

    updateFrame( timeDelta, elapsedTime, frameCount, shader )
    {
        const gl = this.gl;
        if ( !gl )
        {
            return;
        }

        for ( const pass of shader.passes )
        {
            pass.setUniform( gl, 'iTime', elapsedTime );
            pass.setUniform( gl, 'iTimeDelta', timeDelta );
            pass.setUniform( gl, 'iFrame', frameCount );
        }
    }

    updateResolution( resolutionX, resolutionY, shader )
    {
        const gl = this.gl;
        if ( !gl )
        {
            return;
        }

        for ( const pass of shader.passes )
        {
            pass.setUniform( gl, 'iResolution', [
                resolutionX ?? this.canvas.offsetWidth,
                resolutionY ?? this.canvas.offsetHeight
            ] );
        }
    }

    updateMouse( data, shader )
    {
        if ( !this.device )
        {
            return;
        }

        for ( const pass of shader.passes )
        {
            // pass.setUniform( gl, "iMouse", data );
        }
    }

    generateMipmaps( texture, mipLevelCount )
    {
        const encoder = this.device.createCommandEncoder();

        for ( let i = 0; i < mipLevelCount - 1; i++ )
        {
            const srcView = texture.createView( { baseMipLevel: i, mipLevelCount: 1 } );
            const dstView = texture.createView( { baseMipLevel: i + 1, mipLevelCount: 1 } );

            const bindGroup = this.device.createBindGroup( {
                layout: this.mipmapPipeline.getBindGroupLayout( 0 ),
                entries: [
                    { binding: 0, resource: srcView },
                    { binding: 1, resource: dstView }
                ]
            } );

            const pass = encoder.beginComputePass();
            pass.setPipeline( this.mipmapPipeline );
            pass.setBindGroup( 0, bindGroup );

            const w = Math.max( 1, texture.width >> ( i + 1 ) );
            const h = Math.max( 1, texture.height >> ( i + 1 ) );

            pass.dispatchWorkgroups( Math.ceil( w / 8 ), Math.ceil( h / 8 ) );
            pass.end();
        }

        this.device.queue.submit( [ encoder.finish() ] );
    }

    quitIfWebGLNotAvailable()
    {
        if ( !this.gl )
        {
            this.fail( 'WebGL2 not available' );
            return Constants.WEBGL_ERROR;
        }

        // this.device.lost.then( reason => {
        //     this.fail(`Device lost ("${ reason.reason }"):\n${ reason.message }`);
        // });

        // device.addEventListener('uncapturederror', (ev) => {
        //     this.fail(`Uncaptured error:\n${ev.error.message}`);
        // });

        return Constants.WEBGL_OK;
    }
}

export { GLRenderer };
