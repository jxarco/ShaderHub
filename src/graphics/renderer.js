import { LX } from 'lexgui';

class Renderer
{
    constructor( canvas, backend )
    {
        this.backend = backend;
        this.canvas = canvas;
        this.lang = 'WGSL';

        this.gpuTextures = {};
        this.gpuBuffers = {};
    }

    async init()
    {
        // Instance-level samplers — each renderer (main + previews) creates its own
        // so they're always tied to the correct GPU device and never cross-contaminate
        this.nearestSampler = this.createSampler();
        this.bilinearSampler = this.createSampler( { magFilter: 'linear', minFilter: 'linear' } );
        this.trilinearSampler = this.createSampler( { magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear' } );
        this.nearestRepeatSampler = this.createSampler( { addressModeU: 'repeat', addressModeV: 'repeat', addressModeW: 'repeat' } );
        this.bilinearRepeatSampler = this.createSampler( { magFilter: 'linear', minFilter: 'linear', addressModeU: 'repeat', addressModeV: 'repeat', addressModeW: 'repeat' } );
        this.trilinearRepeatSampler = this.createSampler( { magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear', addressModeU: 'repeat', addressModeV: 'repeat',
            addressModeW: 'repeat' } );
    }

    createBuffer( desc = {} )
    {}
    createSampler( desc = {} )
    {}
    createTexture( desc = {} )
    {}
    async createTextureFromImage( data, id, label = '', options = {} )
    {
        return null;
    }
    async createCubemapTextureFromImage( arrayBuffer, id, label = '', options = {} )
    {
        return null;
    }

    updateTexture( texture, bitmap )
    {}
    updateFrame( timeDelta, elapsedTime, frameCount, shader )
    {}
    updateResolution( resolutionX, resolutionY, shader )
    {}
    updateMouse( data, shader )
    {}

    generateMipmaps( texture, mipLevelCount )
    {}

    fail( msg, msgTitle )
    {
        new LX.Dialog( msgTitle ?? `❌ ${this.backend} Error`, ( p ) => {
            p.root.classList.add( 'p-4' );
            p.root.innerHTML = msg;
        }, { modal: true } );
    }
}

export { Renderer };
