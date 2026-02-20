import * as Constants from '../constants.js';
import { Renderer } from './renderer.js';
import { Shader } from './shader.js';

class GPURenderer extends Renderer
{
    constructor( canvas, backend )
    {
        super( canvas, backend );
    }

    async init()
    {
        this.adapter = await navigator.gpu?.requestAdapter( {
            featureLevel: 'compatibility'
        } );

        this.device = await this.adapter?.requestDevice();
        if ( this.quitIfWebGPUNotAvailable() === Constants.WEBGPU_ERROR )
        {
            return;
        }

        this.webGPUContext = this.canvas.getContext( 'webgpu' );

        const devicePixelRatio = window.devicePixelRatio;
        const clientW = this.canvas.clientWidth * devicePixelRatio;
        const clientH = this.canvas.clientHeight * devicePixelRatio;
        if ( clientW > 0 && clientH > 0 )
        {
            this.canvas.width = clientW;
            this.canvas.height = clientH;
        }

        this.presentationFormat = navigator.gpu.getPreferredCanvasFormat();

        this.webGPUContext.configure( {
            device: this.device,
            format: this.presentationFormat
        } );

        // Only for WebGPU renderer
        this.mipmapPipeline = this.device.createComputePipeline( {
            layout: 'auto',
            compute: {
                module: this.device.createShaderModule( {
                    code: Shader.MIMAP_GENERATION_WGSL
                } ),
                entryPoint: 'main'
            }
        } );

        // Default Input Parameters buffers (webgpu only)
        {
            this.gpuBuffers['iTime'] = this.createBuffer( {
                size: 4,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM
            } );

            this.gpuBuffers['iTimeDelta'] = this.createBuffer( {
                size: 4,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM
            } );

            this.gpuBuffers['iFrame'] = this.createBuffer( {
                size: 4,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM
            } );

            this.gpuBuffers['iResolution'] = this.createBuffer( {
                size: 8,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM
            } );

            this.gpuBuffers['iMouse'] = this.createBuffer( {
                size: 32,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM
            } );
        }

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

    createTexture( desc = {} )
    {
        const texture = this.device.createTexture( desc );

        if ( desc.bitmap )
        {
            this.device.queue.copyExternalImageToTexture(
                { source: desc.bitmap },
                { texture },
                [ desc.bitmap.width, desc.bitmap.height ]
            );
        }

        return texture;
    }

    updateTexture( texture, bitmap )
    {
        this.device.queue.copyExternalImageToTexture(
            { source: bitmap },
            { texture },
            [ bitmap.width, bitmap.height ]
        );

        return texture;
    }

    async createTextureFromImage( data, id, label = '', options = {} )
    {
        options.flipY = options.flipY ?? true;
        options.useMipmaps = options.useMipmaps ?? true;

        const imageBitmap = await createImageBitmap( await new Blob( [ data ] ) );
        const mipLevelCount = options.useMipmaps
            ? ( Math.floor( Math.log2( Math.max( imageBitmap.width, imageBitmap.height ) ) ) + 1 )
            : undefined;
        const dimensions = [ imageBitmap.width, imageBitmap.height ];
        const texture = this.createTexture( {
            label,
            size: [ imageBitmap.width, imageBitmap.height, 1 ],
            mipLevelCount,
            format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING
                | GPUTextureUsage.COPY_DST
                | GPUTextureUsage.STORAGE_BINDING
                | GPUTextureUsage.RENDER_ATTACHMENT
        } );

        this.device.queue.copyExternalImageToTexture(
            { source: imageBitmap, ...options },
            { texture: texture, mipLevel: 0 },
            dimensions
        );

        if ( options.useMipmaps )
        {
            this.generateMipmaps( texture, mipLevelCount );
        }

        this.gpuTextures[id] = texture;

        return texture;
    }

    async createCubemapTextureFromImage( arrayBuffer, id, label = '', options = {} )
    {
        options.flipY = options.flipY ?? true;

        const zip = await JSZip.loadAsync( arrayBuffer );
        const faceNames = [ 'px', 'nx', 'ny', 'py', 'pz', 'nz' ];
        const faceImages = [];

        for ( const face of faceNames )
        {
            const file = zip.file( `${face}.png` ) || zip.file( `${face}.jpg` );
            if ( !file ) throw new Error( `Missing cubemap face: ${face}` );
            const blob = await file.async( 'blob' );
            const imageBitmap = await createImageBitmap( blob );
            faceImages.push( imageBitmap );
        }

        const { width, height } = faceImages[0];

        const texture = this.device.createTexture( {
            label,
            size: [ width, height, 6 ],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING
                | GPUTextureUsage.COPY_DST
                | GPUTextureUsage.RENDER_ATTACHMENT,
            dimension: '2d'
        } );

        for ( let i = 0; i < 6; i++ )
        {
            this.device.queue.copyExternalImageToTexture(
                { source: faceImages[i], ...options },
                { texture, origin: [ 0, 0, i ] },
                [ width, height ]
            );
        }

        this.gpuTextures[id] = texture;

        return texture;
    }

    updateFrame( timeDelta, elapsedTime, frameCount, shader )
    {
        if ( !this.device )
        {
            return;
        }

        this.device.queue.writeBuffer(
            this.gpuBuffers['iTimeDelta'],
            0,
            new Float32Array( [ timeDelta ] )
        );

        this.device.queue.writeBuffer(
            this.gpuBuffers['iTime'],
            0,
            new Float32Array( [ elapsedTime ] )
        );

        this.device.queue.writeBuffer(
            this.gpuBuffers['iFrame'],
            0,
            new Int32Array( [ frameCount ] )
        );
    }

    updateResolution( resolutionX, resolutionY, shader )
    {
        if ( !this.device )
        {
            return;
        }

        this.device.queue.writeBuffer(
            this.gpuBuffers['iResolution'],
            0,
            new Float32Array( [
                resolutionX ?? this.canvas.offsetWidth,
                resolutionY ?? this.canvas.offsetHeight
            ] )
        );
    }

    updateMouse( data, shader )
    {
        if ( !this.device )
        {
            return;
        }

        this.device.queue.writeBuffer(
            this.gpuBuffers['iMouse'],
            0,
            new Float32Array( data )
        );
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

    quitIfWebGPUNotAvailable()
    {
        if ( !this.device )
        {
            return this.quitIfAdapterNotAvailable();
        }

        this.device.lost.then( ( reason ) => {
            this.fail( `Device lost ("${reason.reason}"):\n${reason.message}` );
        } );

        // device.addEventListener('uncapturederror', (ev) => {
        //     this.fail(`Uncaptured error:\n${ev.error.message}`);
        // });

        return Constants.WEBGPU_OK;
    }

    quitIfAdapterNotAvailable()
    {
        if ( !( 'gpu' in navigator ) )
        {
            this.fail( "'navigator.gpu' is not defined - WebGPU not available in this browser" );
        }
        else if ( !this.adapter )
        {
            this.fail( "No adapter found after calling 'requestAdapter'." );
        }
        else
        {
            this.fail( 'Unable to get WebGPU device for an unknown reason.' );
        }

        return Constants.WEBGPU_ERROR;
    }
}

export { GPURenderer };
