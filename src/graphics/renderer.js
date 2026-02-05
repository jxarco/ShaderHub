import { LX } from 'lexgui';

class Renderer
{
    constructor( canvas, backend )
    {
        this.backend = backend;
        this.canvas = canvas;
        this.lang = 'WGSL';

        this.gpuTextures    = {};
        this.gpuBuffers     = {};
    }

    async init()
    {
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
    async createTextureFromImage( data, id, label = "", options = {} ) { return null; }
    async createCubemapTextureFromImage( arrayBuffer, id, label = "", options = {} ) { return null; }

    updateFrame( timeDelta, elapsedTime, frameCount, shader ) {}
    updateResolution( resolutionX, resolutionY, shader ) {}
    updateMouse( data, shader ) {}

    generateMipmaps( texture, mipLevelCount ) {}

    fail( msg, msgTitle )
    {
        new LX.Dialog( msgTitle ?? `❌ ${this.backend} Error`, p => {
            p.root.classList.add( "p-4" );
            p.root.innerHTML = msg;
        }, { modal: true } );
    }
}

export { Renderer };