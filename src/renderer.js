import { LX } from 'lexgui';
import * as Constants from "./constants.js";
import { Shader } from './shader.js';

class Renderer
{
    constructor( canvas, backend )
    {
        this.backend = backend;
        this.canvas = canvas;

        this.gpuTextures    = {};
        this.gpuBuffers     = {};
    }

    async init()
    {
        // Input Parameters
        {
            this.gpuBuffers[ "iTime" ] = this.createBuffer({
                size: 4,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
            });

            this.gpuBuffers[ "iTimeDelta" ] = this.createBuffer({
                size: 4,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
            });

            this.gpuBuffers[ "iFrame" ] = this.createBuffer({
                size: 4,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
            });

            this.gpuBuffers[ "iResolution" ] = this.createBuffer({
                size: 8,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
            });

            this.gpuBuffers[ "iMouse" ] = this.createBuffer({
                size: 32,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
            });
        }

        // Create default texture samplers
        {
            // clamp-to-edge samplers
            Renderer.nearestSampler = this.createSampler();
            Renderer.bilinearSampler = this.createSampler({ magFilter: 'linear', minFilter: 'linear' });
            Renderer.trilinearSampler = this.createSampler({ magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear' });

            // repeat samplers
            Renderer.nearestRepeatSampler = this.createSampler({ addressModeU: "repeat", addressModeV: "repeat", addressModeW: "repeat" });
            Renderer.bilinearRepeatSampler = this.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: "repeat", addressModeV: "repeat", addressModeW: "repeat" });
            Renderer.trilinearRepeatSampler = this.createSampler({ magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear', addressModeU: "repeat", addressModeV: "repeat", addressModeW: "repeat" });
        }
    }

    createBuffer( desc = {} ) {}
    createSampler( desc = {} ) {}
    async createTexture( data, id, label = "", options = {} ) { return null; }
    async createCubemapTexture( arrayBuffer, id, label = "", options = {} ) { return null; }

    updateFrame( timeDelta, elapsedTime, frameCount ) {}
    updateResolution( resolutionX, resolutionY ) {}
    updateMouse( data ) {}

    generateMipmaps( texture, mipLevelCount ) {}

    fail( msg, msgTitle )
    {
        new LX.Dialog( msgTitle ?? `❌ ${this.backend} Error`, p => {
            p.root.classList.add( "p-4" );
            p.root.innerHTML = msg;
        }, { modal: true } );
    }
}

class GPURenderer extends Renderer
{
    constructor( canvas, backend )
    {
        super( canvas, backend );
    }

    async init()
    {
        this.adapter = await navigator.gpu?.requestAdapter({
            featureLevel: 'compatibility',
        });

        this.device = await this.adapter?.requestDevice();
        if( this.quitIfWebGPUNotAvailable() === Constants.WEBGPU_ERROR )
        {
            return;
        }

        this.webGPUContext = this.canvas.getContext( 'webgpu' );

        const devicePixelRatio = window.devicePixelRatio;
        this.canvas.width = this.canvas.clientWidth * devicePixelRatio;
        this.canvas.height = this.canvas.clientHeight * devicePixelRatio;

        this.presentationFormat = navigator.gpu.getPreferredCanvasFormat();

        this.webGPUContext.configure({
            device: this.device,
            format: this.presentationFormat,
        });

        // Only for WebGPU renderer
        this.mipmapPipeline = this.device.createComputePipeline({
            layout: "auto",
            compute: {
                module: this.device.createShaderModule({
                    code: Shader.MIMAP_GENERATION_WGSL
                }),
                entryPoint: "main"
            }
        });

        super.init();
    }

    createBuffer( desc = {} )
    {
        return this.device.createBuffer( desc );
    }

    createSampler( desc = {} )
    {
        return this.device.createSampler( desc );
    }

    async createTexture( data, id, label = "", options = {} )
    {
        options.flipY = options.flipY ?? true;
        options.useMipmaps = options.useMipmaps ?? true;

        const imageBitmap = await createImageBitmap( await new Blob( [ data ] ) );
        const mipLevelCount = options.useMipmaps ? 
            ( Math.floor( Math.log2( Math.max( imageBitmap.width, imageBitmap.height ) ) ) + 1 ) : undefined;
        const dimensions = [ imageBitmap.width, imageBitmap.height ];
        const texture = this.device.createTexture({
            label,
            size: [ imageBitmap.width, imageBitmap.height, 1 ],
            mipLevelCount,
            format: 'rgba8unorm',
            usage:
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.COPY_DST |
                GPUTextureUsage.STORAGE_BINDING |
                GPUTextureUsage.RENDER_ATTACHMENT,
        });

        this.device.queue.copyExternalImageToTexture(
            { source: imageBitmap, ...options },
            { texture: texture, mipLevel: 0 },
            dimensions
        );

        if( options.useMipmaps )
        {
            this.generateMipmaps( texture, mipLevelCount );
        }

        this.gpuTextures[ id ] = texture;

        return texture;
    }

    async createCubemapTexture( arrayBuffer, id, label = "", options = {} )
    {
        options.flipY = options.flipY ?? true;

        const zip = await JSZip.loadAsync( arrayBuffer );
        const faceNames = [ "px", "nx", "ny", "py", "pz", "nz" ];
        const faceImages = [];

        for( const face of faceNames )
        {
            const file = zip.file( `${ face }.png` ) || zip.file( `${ face }.jpg` );
            if( !file ) throw new Error( `Missing cubemap face: ${ face }` );
            const blob = await file.async( "blob" );
            const imageBitmap = await createImageBitmap( blob );
            faceImages.push( imageBitmap );
        }

        const { width, height } = faceImages[ 0 ];

        const texture = this.device.createTexture({
            label,
            size: [ width, height, 6 ],
            format: "rgba8unorm",
            usage:
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.COPY_DST |
                GPUTextureUsage.RENDER_ATTACHMENT,
            dimension: "2d",
        });

        for( let i = 0; i < 6; i++ )
        {
            this.device.queue.copyExternalImageToTexture(
                { source: faceImages[ i ], ...options },
                { texture, origin: [ 0, 0, i ] },
                [ width, height ]
            );
        }

        this.gpuTextures[ id ] = texture;

        return texture;
    }

    updateFrame( timeDelta, elapsedTime, frameCount )
    {
        if( !this.device )
        {
            return;
        }

        this.device.queue.writeBuffer(
            this.gpuBuffers[ "iTimeDelta" ],
            0,
            new Float32Array([ timeDelta ])
        );

        this.device.queue.writeBuffer(
            this.gpuBuffers[ "iTime" ],
            0,
            new Float32Array([ elapsedTime ])
        );

        this.device.queue.writeBuffer(
            this.gpuBuffers[ "iFrame" ],
            0,
            new Int32Array([ frameCount ])
        );
    }

    updateResolution( resolutionX, resolutionY )
    {
        if( !this.device )
        {
            return;
        }

        this.device.queue.writeBuffer(
            this.gpuBuffers[ "iResolution" ],
            0,
            new Float32Array([
                resolutionX ?? this.canvas.offsetWidth,
                resolutionY ?? this.canvas.offsetHeight
            ])
        );
    }

    updateMouse( data )
    {
        if( !this.device )
        {
            return;
        }

        this.device.queue.writeBuffer(
            this.gpuBuffers[ "iMouse" ],
            0,
            new Float32Array( data )
        );
    }

    generateMipmaps( texture, mipLevelCount )
    {
        const encoder = this.device.createCommandEncoder();

        for( let i = 0; i < mipLevelCount - 1; i++ )
        {
            const srcView = texture.createView({ baseMipLevel: i, mipLevelCount: 1 });
            const dstView = texture.createView({ baseMipLevel: i + 1, mipLevelCount: 1 });

            const bindGroup = this.device.createBindGroup({
                layout: this.mipmapPipeline.getBindGroupLayout( 0 ),
                entries: [
                    { binding: 0, resource: srcView },
                    { binding: 1, resource: dstView }
                ]
            });

            const pass = encoder.beginComputePass();
            pass.setPipeline( this.mipmapPipeline );
            pass.setBindGroup( 0, bindGroup );

            const w = Math.max( 1, texture.width  >> ( i + 1 ) );
            const h = Math.max( 1, texture.height >> ( i + 1 ) );

            pass.dispatchWorkgroups( Math.ceil( w / 8 ), Math.ceil( h / 8 ) );
            pass.end();
        }

        this.device.queue.submit( [ encoder.finish() ] );
    }

    quitIfWebGPUNotAvailable()
    {
        if( !this.device )
        {
            return this.quitIfAdapterNotAvailable();
        }

        this.device.lost.then( reason => {
            this.fail(`Device lost ("${ reason.reason }"):\n${ reason.message }`);
        });

        // device.addEventListener('uncapturederror', (ev) => {
        //     this.fail(`Uncaptured error:\n${ev.error.message}`);
        // });

        return Constants.WEBGPU_OK;
    }

    quitIfAdapterNotAvailable()
    {
        if( !( "gpu" in navigator ) )
        {
            this.fail( "'navigator.gpu' is not defined - WebGPU not available in this browser" );
        }
        else if( !this.adapter )
        {
            this.fail( "No adapter found after calling 'requestAdapter'." );
        }
        else
        {
            this.fail( "Unable to get WebGPU device for an unknown reason." );
        }

        return Constants.WEBGPU_ERROR;
    }
}

const GL_SAMPLER_MAP = {
    magFilter: {
        'nearest': WebGL2RenderingContext.NEAREST,
        'linear': WebGL2RenderingContext.LINEAR,
    },
    minFilter: {
        'nearest': WebGL2RenderingContext.NEAREST,
        'linear': WebGL2RenderingContext.LINEAR,
        'nearest-mipmap-nearest': WebGL2RenderingContext.NEAREST_MIPMAP_NEAREST,
        'linear-mipmap-nearest': WebGL2RenderingContext.LINEAR_MIPMAP_NEAREST,
        'nearest-mipmap-linear': WebGL2RenderingContext.NEAREST_MIPMAP_LINEAR,
        'linear-mipmap-linear': WebGL2RenderingContext.LINEAR_MIPMAP_LINEAR,
    },
    addressMode: {
        'clamp-to-edge': WebGL2RenderingContext.CLAMP_TO_EDGE,
        'repeat': WebGL2RenderingContext.REPEAT,
        'mirror-repeat': WebGL2RenderingContext.MIRRORED_REPEAT,
    },
};

class GLRenderer extends Renderer
{
    constructor( canvas, backend )
    {
        super( canvas, backend );
    }

    async init()
    {
        this.gl = this.canvas.getContext( 'webgl2', {
            alpha: false,
            antialias: true,
            depth: true,
            stencil: false,
            preserveDrawingBuffer: false,
        } );

        if( this.quitIfWebGLNotAvailable() === Constants.WEBGL_ERROR )
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

        super.init();

        // bind buffers (UBOs) to fixed binding points
        gl.bindBufferBase( gl.UNIFORM_BUFFER, 0, this.gpuBuffers.iTime );
        gl.bindBufferBase( gl.UNIFORM_BUFFER, 1, this.gpuBuffers.iTimeDelta );
        gl.bindBufferBase( gl.UNIFORM_BUFFER, 2, this.gpuBuffers.iFrame );
        gl.bindBufferBase( gl.UNIFORM_BUFFER, 3, this.gpuBuffers.iResolution );
        gl.bindBufferBase( gl.UNIFORM_BUFFER, 4, this.gpuBuffers.iMouse );
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
            const wrapW = GL_SAMPLER_MAP.addressMode[ desc.addressModeW ];
            gl.samplerParameteri( sampler, gl.TEXTURE_WRAP_R, wrapW );
        }

        return sampler;
    }

    async createTexture( data, id, label = "", options = {} )
    {
        options.flipY = options.flipY ?? true;
        options.useMipmaps = options.useMipmaps ?? true;

        const gl = this.gl;
        const imageBitmap = await createImageBitmap( await new Blob( [ data ] ) );
        const texture = gl.createTexture();

        gl.bindTexture( gl.TEXTURE_2D, texture );

        // WebGPU-style Y flip
        gl.pixelStorei( gl.UNPACK_FLIP_Y_WEBGL, options.flipY );

        // Upload level 0
        gl.texImage2D(
            gl.TEXTURE_2D,
            0,
            gl.RGBA8,
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

        this.gpuTextures[ id ] = texture;

        return texture;
    }

    async createCubemapTexture( arrayBuffer, id, label = "", options = {} )
    {
        options.flipY = options.flipY ?? true;

        const zip = await JSZip.loadAsync( arrayBuffer );
        const faceNames = [ "px", "nx", "ny", "py", "pz", "nz" ];
        const faceImages = [];

        for( const face of faceNames )
        {
            const file = zip.file( `${ face }.png` ) || zip.file( `${ face }.jpg` );
            if( !file ) throw new Error( `Missing cubemap face: ${ face }` );
            const blob = await file.async( "blob" );
            const imageBitmap = await createImageBitmap( blob );
            faceImages.push( imageBitmap );
        }

        const { width, height } = faceImages[ 0 ];

        const texture = this.device.createTexture({
            label,
            size: [ width, height, 6 ],
            format: "rgba8unorm",
            usage:
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.COPY_DST |
                GPUTextureUsage.RENDER_ATTACHMENT,
            dimension: "2d",
        });

        for( let i = 0; i < 6; i++ )
        {
            this.device.queue.copyExternalImageToTexture(
                { source: faceImages[ i ], ...options },
                { texture, origin: [ 0, 0, i ] },
                [ width, height ]
            );
        }

        this.gpuTextures[ id ] = texture;

        return texture;
    }

    updateFrame( timeDelta, elapsedTime, frameCount )
    {
        if( !this.device )
        {
            return;
        }

        this.device.queue.writeBuffer(
            this.gpuBuffers[ "iTimeDelta" ],
            0,
            new Float32Array([ timeDelta ])
        );

        this.device.queue.writeBuffer(
            this.gpuBuffers[ "iTime" ],
            0,
            new Float32Array([ elapsedTime ])
        );

        this.device.queue.writeBuffer(
            this.gpuBuffers[ "iFrame" ],
            0,
            new Int32Array([ frameCount ])
        );
    }

    updateResolution( resolutionX, resolutionY )
    {
        if( !this.device )
        {
            return;
        }

        this.device.queue.writeBuffer(
            this.gpuBuffers[ "iResolution" ],
            0,
            new Float32Array([
                resolutionX ?? this.canvas.offsetWidth,
                resolutionY ?? this.canvas.offsetHeight
            ])
        );
    }

    updateMouse( data )
    {
        if( !this.device )
        {
            return;
        }

        this.device.queue.writeBuffer(
            this.gpuBuffers[ "iMouse" ],
            0,
            new Float32Array( data )
        );
    }

    generateMipmaps( texture, mipLevelCount )
    {
        const encoder = this.device.createCommandEncoder();

        for( let i = 0; i < mipLevelCount - 1; i++ )
        {
            const srcView = texture.createView({ baseMipLevel: i, mipLevelCount: 1 });
            const dstView = texture.createView({ baseMipLevel: i + 1, mipLevelCount: 1 });

            const bindGroup = this.device.createBindGroup({
                layout: this.mipmapPipeline.getBindGroupLayout( 0 ),
                entries: [
                    { binding: 0, resource: srcView },
                    { binding: 1, resource: dstView }
                ]
            });

            const pass = encoder.beginComputePass();
            pass.setPipeline( this.mipmapPipeline );
            pass.setBindGroup( 0, bindGroup );

            const w = Math.max( 1, texture.width  >> ( i + 1 ) );
            const h = Math.max( 1, texture.height >> ( i + 1 ) );

            pass.dispatchWorkgroups( Math.ceil( w / 8 ), Math.ceil( h / 8 ) );
            pass.end();
        }

        this.device.queue.submit( [ encoder.finish() ] );
    }

    quitIfWebGLNotAvailable()
    {
        if( !this.gl )
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

export { Renderer, GPURenderer, GLRenderer };